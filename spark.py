"""Resident FP8 runtime for GB10 and discrete Blackwell, preserving Turbo."""

import importlib.metadata
import os
from pathlib import Path

MODEL_REVISION = "42ed227ee7df40d41602854ae760620d6eb651fe"
ENCODER_REVISION = "4bf2c2f39c37c0fede78bede4056e1f18cdf8109"


def release_checkpoint_pages():
    """Evict this runtime's file cache; CUDA on GB10 cannot reclaim it itself.

    POSIX_FADV_DONTNEED is advisory and does not delete files or evict live
    tensors. This avoids a privileged, system-wide drop_caches operation.
    """
    root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "hf"))
    files = []
    for repo in ("MiniMaxAI--MiniMax-H3", "Qwen--Qwen3-VL-32B-Instruct-FP8"):
        files.extend((root / "hub" / f"models--{repo}" / "blobs").glob("*"))
    files.extend((root / "derived" / "h3-fp8").rglob("*.safetensors"))
    for path in files:
        try:
            if path.stat().st_size < 16 * 1024**2:
                continue
            with path.open("rb") as handle:
                os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        except OSError:
            pass


def is_spark() -> bool:
    import torch

    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 1)


def use_spark() -> bool:
    profile = os.environ.get("MINIMAX_H3_PROFILE", "auto")
    if profile not in ("auto", "spark", "default", "resident-fp8"):
        raise ValueError("MINIMAX_H3_PROFILE must be auto, spark, default, or resident-fp8")
    return profile == "spark" or (profile == "auto" and is_spark())


def use_resident_fp8() -> bool:
    # Keep Spark's shared-memory preflight separate from discrete GPU checks.
    return use_spark() or os.environ.get("MINIMAX_H3_PROFILE") == "resident-fp8"


def retain_conditioning_layer(encoder, layer=50):
    """H3 only reads the pre-norm hidden state entering decoder layer 50.

    Avoid Transformers' output_hidden_states collection retaining all 51 large
    multimodal activations. This adapter is specific to H3's conditioner.
    """
    import functools
    model = encoder.model
    original = model.forward

    @functools.wraps(original)
    def forward(*args, **kwargs):
        if not kwargs.get("output_hidden_states", False):
            return original(*args, **kwargs)
        captured = []

        def capture(_module, inputs, named):
            captured.append(inputs[0] if inputs else named["hidden_states"])

        handle = model.language_model.layers[layer].register_forward_pre_hook(capture, with_kwargs=True)
        try:
            result = original(*args, **{**kwargs, "output_hidden_states": False})
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError("Expected exactly one H3 conditioning activation")
        states = [None] * (encoder.config.text_config.num_hidden_layers + 1)
        states[layer] = captured[0]
        result.hidden_states = tuple(states)
        return result

    model.forward = forward


SHARED_COMPONENTS = ("text_encoder", "vae", "audio_vae", "tokenizer", "processor",
                     "scheduler", "audio_scheduler")


def resident_shared_components(pipe):
    """Keep shared objects alive while the old pipeline/manager is collected."""
    if not getattr(pipe, "_h3_resident", False):
        return None
    shared = {name: getattr(pipe, name) for name in SHARED_COMPONENTS}
    if any(value is None for value in shared.values()):
        raise RuntimeError("Cannot reuse an incomplete resident pipeline")
    return shared


def build_spark_pipeline(model_id: str, task: str, bf16_text_encoder: bool = False,
                         shared_components=None):
    import gc
    import torch
    from diffusers import ComponentsManager, MiniMaxH3Transformer3DModel, ModularPipeline, TorchAoConfig
    from torchao.quantization import Float8DynamicActivationFloat8WeightConfig
    from torchao.quantization.granularity import PerTensor
    from transformers import AutoConfig, Qwen3VLForConditionalGeneration
    from huggingface_hub import snapshot_download

    if task not in ("fl2va", "ref2va"):
        raise ValueError(f"Unknown task: {task}")
    if shared_components is not None and (
        set(shared_components) != set(SHARED_COMPONENTS)
        or any(value is None for value in shared_components.values())
    ):
        raise ValueError("Expected all shared resident components")
    if bf16_text_encoder:
        raise ValueError("The resident profile requires a quantized conditioner for activation headroom.")
    torch.set_num_threads(int(os.environ.get("MINIMAX_H3_CPU_THREADS", "10")))
    component = "transformer_ref" if task == "ref2va" else "transformer"
    cache = (
        Path(os.environ["HF_HOME"]) / "derived" / "h3-fp8" / MODEL_REVISION
        / f"torch-{torch.__version__}-ao-{importlib.metadata.version('torchao')}-tensorwise"
        / component
    )
    cached = (cache / "READY").is_file()
    if cached:
        source = str(cache)
    else:
        # Complete disk reads/downloads before CUDA's allocator warmup: downloads
        # can fill the unified-memory page cache after the API's preflight.
        source = snapshot_download(
            model_id, revision=MODEL_REVISION, allow_patterns=[f"{component}/*"],
            max_workers=4,
        )
    release_checkpoint_pages()
    load_options = {} if cached else {
        "subfolder": component,
        "revision": MODEL_REVISION,
        "quantization_config": TorchAoConfig(
            Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor()),
        ),
    }
    print(f"[spark] Loading {component} directly to CUDA with tensorwise FP8 (cached={cached})", flush=True)
    denoiser = MiniMaxH3Transformer3DModel.from_pretrained(
        source,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda"},
        **load_options,
    )
    denoiser.requires_grad_(False)
    release_checkpoint_pages()
    if not cached:
        print(f"[spark] Saving FP8 cache for subsequent starts: {cache}", flush=True)
        denoiser.save_pretrained(cache, max_shard_size="5GB")
        (cache / "READY").write_text(MODEL_REVISION + "\n")
        release_checkpoint_pages()

    if shared_components is None:
        discrete = os.environ.get("MINIMAX_H3_PROFILE") == "resident-fp8"
        encoder_id = model_id if discrete else "Qwen/Qwen3-VL-32B-Instruct-FP8"
        encoder_source = snapshot_download(
            encoder_id, revision=MODEL_REVISION if discrete else ENCODER_REVISION,
            allow_patterns=["text_encoder/*"] if discrete else None, max_workers=4,
        )
        if discrete:
            encoder_source = str(Path(encoder_source) / "text_encoder")
        release_checkpoint_pages()
        config = AutoConfig.from_pretrained(encoder_source)
        # The published checkpoint uses the vLLM spelling for its BF16 exclusions.
        if not discrete:
            config.quantization_config["modules_to_not_convert"] = config.quantization_config["ignored_layers"]
        # H3 consumes hidden_states[50]. Keep 51 layers so that entry is still
        # pre-norm; the final hidden state is normalized by Qwen's forward.
        config.text_config.num_hidden_layers = 51
        encoder_options = {}
        if discrete:
            from torchao.quantization import Int8WeightOnlyConfig
            from transformers import TorchAoConfig as EncoderTorchAoConfig
            encoder_options["quantization_config"] = EncoderTorchAoConfig(
                Int8WeightOnlyConfig(version=2),
                modules_to_not_convert=["model.visual", "model.language_model.embed_tokens",
                                        "model.language_model.norm", "lm_head"],
            )
        print(f"[resident] Loading {'INT8' if discrete else 'FP8'} conditioner (51 decoder layers)", flush=True)
        encoder = Qwen3VLForConditionalGeneration.from_pretrained(
            encoder_source, config=config, dtype=torch.bfloat16, device_map={"": "cuda"},
            attn_implementation="sdpa",
            **encoder_options,
        )
        # H3 calls encoder.model directly; its vocabulary projection is unused.
        encoder.lm_head = torch.nn.Identity()
        encoder.requires_grad_(False)
        retain_conditioning_layer(encoder)
        release_checkpoint_pages()

    else:
        encoder = shared_components["text_encoder"]
        print("[resident] Reusing CUDA conditioner, video/audio VAEs and shared components", flush=True)

    snapshot = snapshot_download(
        model_id, revision=MODEL_REVISION, max_workers=4,
        allow_patterns=[
            "modular_model_index.json", "tokenizer/*", "processor/*", "vae/*",
            "audio_vae/*", "scheduler/*", "audio_scheduler/*",
        ],
    )
    manager = ComponentsManager()
    pipe = ModularPipeline.from_pretrained(snapshot, components_manager=manager)
    pipe.update_components(**{component: denoiser, "text_encoder": encoder})
    if shared_components is not None:
        pipe.update_components(**shared_components)
    # Direct component directories also avoid an AutoProcessor subfolder/revision
    # lookup bug and Diffusers' sharded-checkpoint Hub metadata request offline.
    pipe.load_components(
        workflow=task, dtype=torch.bfloat16, local_files_only=True, subfolder="",
        pretrained_model_name_or_path={
            name: str(Path(snapshot) / name) for name in pipe._component_specs
        },
    )
    for name in ("vae", "audio_vae", "tokenizer", "processor", "scheduler", "audio_scheduler"):
        if getattr(pipe, name, None) is None:
            raise RuntimeError(f"Spark component {name} failed to load; see the component error above")
    pipe.vae.to("cuda")
    pipe.audio_vae.to("cuda")
    pipe._h3_resident = True
    gc.collect()
    torch.cuda.empty_cache()
    release_checkpoint_pages()
    print(f"[spark] Resident CUDA models: {torch.cuda.memory_allocated() / 1024**3:.1f} GiB", flush=True)
    return pipe


def prepare_spark_transformer(pipe):
    """Compile repeated blocks after the LoRA is installed; keep adapters toggleable."""
    if not getattr(pipe, "_h3_resident", False):
        return
    denoiser = getattr(pipe, "transformer_ref", None) or pipe.transformer
    if getattr(denoiser, "_h3_compiled", False):
        return
    if os.environ.get("MINIMAX_H3_COMPILE", "1") == "1":
        denoiser.compile_repeated_blocks(fullgraph=True)
        denoiser._h3_compiled = True
        print("[spark] Repeated transformer blocks compiled (first render warms kernels)", flush=True)
