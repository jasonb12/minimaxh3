#!/usr/bin/env python
"""Generate video + stereo audio with MiniMax-H3 (text-to-video or first/last-frame conditioning).

Examples:
    # Text to video+audio (defaults: 16:9 canvas, ~5s)
    python generate.py 'A red fox trotting through a snowy pine forest, snow crunching underfoot'

    # First-frame conditioned
    python generate.py 'The astronaut waves at the camera' --image astronaut.jpg

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MiniMax-H3 video+audio generation")
    p.add_argument("prompt", help="Text prompt (see prompting guide on the model card)")
    p.add_argument("--image", help="Optional first-frame image (path or URL)")
    p.add_argument("--last-image", help="Optional last-frame image (path or URL)")
    p.add_argument("--height", type=int, help="Canvas height, multiple of 32 (default: model's 16:9 canvas)")
    p.add_argument("--width", type=int, help="Canvas width, multiple of 32")
    p.add_argument("--num-frames", type=int, help="Frame count; snapped up to 17*n+5, 24 fps, 5-15s")
    p.add_argument("--steps", type=int, help="num_inference_steps (default: pipeline default)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default=None, help="Output .mp4 path (default: outputs/<timestamp>.mp4)")
    p.add_argument(
        "--bf16-text-encoder",
        action="store_true",
        help="Load the Qwen3-VL conditioner in bf16 (~62GB) instead of int8 (~32GB). "
        "Needs more free host RAM during loading.",
    )
    return p.parse_args()


def build_pipeline(bf16_text_encoder: bool):
    import torch
    from diffusers import ComponentsManager, ModularPipeline

    manager = ComponentsManager()
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

    pipe.load_components(dtype=torch.bfloat16)
    pipe.transformer.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)

    # 96GB card: transformer (61.7GB bf16) and conditioner cannot both stay
    # resident; the manager swaps them between GPU and host RAM on demand.
    manager.enable_auto_cpu_offload(device="cuda", memory_reserve_margin="12GB")
    return pipe


def main() -> None:
    args = parse_args()

    import torch
    from diffusers.utils import load_image
    from diffusers.utils.export_utils import encode_video

    output = args.output or f"outputs/h3_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    pipe = build_pipeline(args.bf16_text_encoder)
    print(f"[+] Pipeline loaded in {time.time() - t0:.0f}s", flush=True)

    call_kwargs = {"prompt": args.prompt, "generator": torch.Generator().manual_seed(args.seed)}
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
