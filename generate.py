#!/usr/bin/env python
"""Generate video + stereo audio with MiniMax-H3.

Supports:
  - t2va / fl2va — text, and optional first/last keyframes (transformer/)
  - ref2va — ordered image/video/audio references (transformer_ref/)

Examples:
    # Text to video+audio (defaults: 16:9 canvas, ~5s)
    python generate.py 'A red fox trotting through a snowy pine forest, snow crunching underfoot'

    # First-frame conditioned
    python generate.py 'The astronaut waves at the camera' --image astronaut.jpg

    # Omni-reference (furniture / subject identity). Order is semantic.
    python generate.py 'Use <Picture 1> as the product; orbit slowly around it' \\
        --ref image:sofa_front.jpg --ref image:sofa_side.jpg

    # Faster smoke test on a smaller canvas
    python generate.py 'Ocean waves at sunset' --width 960 --height 544 --num-frames 124
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Default HF cache: ~/.cache/huggingface is root-owned on this machine, so use a
# user-owned location. An explicit HF_HOME in the environment still wins.
os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "hf"))

MODEL_ID = "MiniMaxAI/MiniMax-H3"
REF_KINDS = ("image", "video", "audio")


def _parse_ref(value: str) -> tuple[str, str]:
    """Parse `kind:path` (kind = image|video|audio). Bare paths default to image."""
    if ":" in value:
        kind, path = value.split(":", 1)
        kind = kind.lower().strip()
        if kind in REF_KINDS and path:
            return kind, path
    # Allow Windows-style absolute paths and bare image paths without a kind.
    return "image", value


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MiniMax-H3 video+audio generation")
    p.add_argument("prompt", help="Text prompt (see prompting guide on the model card)")
    p.add_argument("--image", help="Optional first-frame image (path or URL); fl2va")
    p.add_argument("--last-image", help="Optional last-frame image (path or URL); fl2va")
    p.add_argument(
        "--ref",
        action="append",
        default=[],
        metavar="KIND:PATH",
        help=(
            "Omni-reference for ref2va. Repeatable; order is semantic. "
            "KIND is image|video|audio (default image if omitted), e.g. "
            "--ref image:chair.jpg --ref video:orbit.mp4 --ref audio:voice.wav"
        ),
    )
    p.add_argument("--height", type=int, help="Canvas height, multiple of 32 (default: model's 16:9 canvas)")
    p.add_argument("--width", type=int, help="Canvas width, multiple of 32")
    p.add_argument("--num-frames", type=int, help="Frame count; snapped up to 17*n+5, 24 fps, 5-15s")
    p.add_argument("--steps", type=int, help="num_inference_steps (default: 7 with Turbo / 6 evals; pipeline default with --no-turbo)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--turbo",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply the community Turbo distillation LoRA (larryvrh/MiniMax-H3-Turbo-Lora, "
        "v4-600 EMA): 6 model evaluations instead of ~50, roughly 5–10x faster sampling. "
        "Preview quality. Ref2VA use is experimental and may reduce reference identity "
        "fidelity. Enabled by default; use --no-turbo for the full-quality sampler.",
    )
    p.add_argument(
        "--turbo-strength",
        type=float,
        default=1.0,
        help="Turbo LoRA scale. Nudge up (1.05-1.2) against blurry ghosting, "
        "down (0.8-0.95) against over-sharp grain.",
    )
    p.add_argument("--output", default=None, help="Output .mp4 path (default: outputs/<timestamp>.mp4)")
    p.add_argument(
        "--bf16-text-encoder",
        action="store_true",
        help="Load the Qwen3-VL conditioner in bf16 (~62GB) instead of int8 (~32GB). "
        "Needs more free host RAM during loading.",
    )
    args = p.parse_args()
    args.refs = [_parse_ref(r) for r in args.ref]
    if args.refs and (args.image or args.last_image):
        p.error("--ref (ref2va) cannot be combined with --image/--last-image (fl2va)")
    return args


def build_pipeline(bf16_text_encoder: bool, task: str = "fl2va"):
    """Build the FL2VA (`t2va`/`fl2va`) or Ref2VA (`ref2va`) modular pipeline.

    The two tasks load different transformer partitions from the same repo
    (`transformer/` vs `transformer_ref/`); shared components (VAE, conditioner,
    schedulers) are the same.
    """
    import torch
    from diffusers import ComponentsManager, ModularPipeline

    if task not in ("fl2va", "ref2va"):
        raise ValueError(f"Unknown task {task!r}; expected 'fl2va' or 'ref2va'")

    from spark import build_spark_pipeline, use_spark

    if use_spark():
        return build_spark_pipeline(MODEL_ID, task, bf16_text_encoder)

    manager = ComponentsManager()
    # diffusers 0.40 ships one auto blockset that dispatches t2va / fl2va /
    # ref2va on the inputs. Keep the full blocks (so a prompt-only request still
    # runs as t2va) and load only the components the task's workflow uses;
    # otherwise `load_components` pulls both 62GB transformer partitions.
    pipe = ModularPipeline.from_pretrained(MODEL_ID, components_manager=manager)

    if not bf16_text_encoder:
        # Official int8 recipe for the conditioner: halves its footprint with
        # negligible conditioning quality loss. Keeps peak host RAM under this
        # machine's 125GB (bf16 transformer 61.7GB + int8 conditioner ~32GB).
        from torchao.quantization import Int8WeightOnlyConfig
        from transformers import Qwen3VLForConditionalGeneration
        from transformers import TorchAoConfig as TransformersTorchAoConfig

        pipe.update_components(
            text_encoder=Qwen3VLForConditionalGeneration.from_pretrained(
                MODEL_ID,
                subfolder="text_encoder",
                dtype=torch.bfloat16,
                quantization_config=TransformersTorchAoConfig(
                    Int8WeightOnlyConfig(version=2),
                    modules_to_not_convert=[
                        "model.visual",
                        "model.language_model.embed_tokens",
                        "model.language_model.norm",
                        "lm_head",
                    ],
                ),
            ),
        )

    pipe.load_components(workflow=task, dtype=torch.bfloat16)
    # Ref2VA denoises with `transformer_ref`; t2va/fl2va with `transformer`.
    denoiser = getattr(pipe, "transformer_ref", None) or pipe.transformer
    denoiser.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)

    # 96GB card: transformer (61.7GB bf16) and conditioner (~32GB int8) cannot
    # both stay resident at any margin; the manager swaps them between GPU and
    # host RAM on demand. The margin is only activation headroom for whichever
    # model is on the card. Tunable for other GPUs.
    margin_gb = os.environ.get("MINIMAX_H3_RESERVE_MARGIN_GB", "12")
    manager.enable_auto_cpu_offload(device="cuda", memory_reserve_margin=f"{margin_gb}GB")
    return pipe


def build_references(refs: list[tuple[str, str]]):
    """Turn `(kind, path_or_url)` pairs into reference dataclasses (order preserved).

    `from_file` decodes the media and keeps its native fps / sample rate, which
    is what places a video or audio reference on the model's clock correctly.
    """
    from diffusers.modular_pipelines.minimax_h3 import (
        MiniMaxH3AudioReference,
        MiniMaxH3ImageReference,
        MiniMaxH3VideoReference,
    )

    loaders = {
        "image": MiniMaxH3ImageReference.from_file,
        "video": MiniMaxH3VideoReference.from_file,
        "audio": MiniMaxH3AudioReference.from_file,
    }
    out = []
    for kind, path in refs:
        if kind not in loaders:
            raise ValueError(f"Unknown reference kind {kind!r}")
        out.append(loaders[kind](path))
    return out


def main() -> None:
    args = parse_args()

    import torch
    from diffusers.utils import load_image
    from diffusers.utils.export_utils import encode_video

    task = "ref2va" if args.refs else "fl2va"
    output = args.output or f"outputs/h3_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    pipe = build_pipeline(args.bf16_text_encoder, task=task)
    print(f"[+] Pipeline loaded ({task}) in {time.time() - t0:.0f}s", flush=True)

    if args.turbo:
        from turbo import TURBO_NUM_INFERENCE_STEPS, load_turbo_lora

        denoiser = getattr(pipe, "transformer_ref", None) or pipe.transformer
        load_turbo_lora(denoiser, strength=args.turbo_strength)
        if args.steps is None:
            args.steps = TURBO_NUM_INFERENCE_STEPS
        print(f"[+] Turbo LoRA applied (strength {args.turbo_strength}, steps {args.steps})", flush=True)

    from spark import prepare_spark_transformer

    prepare_spark_transformer(pipe)
    call_kwargs = {"prompt": args.prompt, "generator": torch.Generator().manual_seed(args.seed)}
    if task == "ref2va":
        call_kwargs["references"] = build_references(args.refs)
    else:
        if args.image:
            call_kwargs["image"] = load_image(args.image)
        if args.last_image:
            call_kwargs["last_image"] = load_image(args.last_image)
    for key, value in (
        ("height", args.height),
        ("width", args.width),
        ("num_frames", args.num_frames),
        ("num_inference_steps", args.steps),
    ):
        if value is not None:
            call_kwargs[key] = value

    t0 = time.time()
    state = pipe(**call_kwargs)
    print(f"[+] Generated in {time.time() - t0:.0f}s", flush=True)

    encode_video(
        state.get("videos")[0],
        fps=24,
        output_path=output,
        audio=state.get("audio")[0],
        audio_sample_rate=state.get("sampling_rate"),
    )
    print(f"[+] Wrote {output}")


if __name__ == "__main__":
    sys.exit(main())
