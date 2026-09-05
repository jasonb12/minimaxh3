"""GB10 runtime: resident FP8 models, with the original Turbo layer layout."""

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
    if profile not in ("auto", "spark", "default"):
        raise ValueError("MINIMAX_H3_PROFILE must be auto, spark, or default")
    return profile == "spark" or (profile == "auto" and is_spark())


def build_spark_pipeline(model_id: str, task: str, bf16_text_encoder: bool = False):
    import gc
    import torch
    from diffusers import ComponentsManager, MiniMaxH3Transformer3DModel, ModularPipeline, TorchAoConfig
    from torchao.quantization import Float8DynamicActivationFloat8WeightConfig
    from torchao.quantization.granularity import PerTensor
    from transformers import AutoConfig, Qwen3VLForConditionalGeneration
    from huggingface_hub import snapshot_download

    if bf16_text_encoder:
        raise ValueError("The Spark profile needs an FP8 conditioner to preserve shared-memory headroom.")
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

    encoder_id = "Qwen/Qwen3-VL-32B-Instruct-FP8"
    encoder_source = snapshot_download(encoder_id, revision=ENCODER_REVISION, max_workers=4)
    release_checkpoint_pages()
    config = AutoConfig.from_pretrained(encoder_source)
    # The published checkpoint uses the vLLM spelling for its BF16 exclusions.
    config.quantization_config["modules_to_not_convert"] = config.quantization_config["ignored_layers"]
    # H3 consumes hidden_states[50]. Keep 51 layers so that entry is still
    # pre-norm; the final hidden state is normalized by Qwen's forward.
    config.text_config.num_hidden_layers = 51
    print("[spark] Loading prequantized FP8 conditioner (51 decoder layers)", flush=True)
    encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        encoder_source, config=config, dtype=torch.bfloat16, device_map={"": "cuda"},
        attn_implementation="sdpa",
    )
    # H3 calls encoder.model directly; its vocabulary projection is unused.
    encoder.lm_head = torch.nn.Identity()
    encoder.requires_grad_(False)
    release_checkpoint_pages()

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
