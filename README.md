# MiniMax-H3 Local Inference

Local video + stereo audio generation with [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3),
MiniMax's omni-modal model (open weights released 2026-08-03). It jointly denoises video and
32kHz stereo audio in one transformer pass: 24fps, 5–15 seconds, 768px short edge.

This repo runs the **H3-Base FL2VA** checkpoint via the official
[diffusers integration](https://github.com/huggingface/diffusers/pull/14355)
(Modular Diffusers blocks), covering:

- **t2va** — text to video+audio
- **fl2va** — first-frame and/or last-frame conditioned video+audio

Not set up here (yet): **ref2va** (omni-reference, `transformer_ref/` subfolder, extra ~62GB) and the
hosted **H3-Context-IR** / **H3-Regenerate-2K** API stages used for full 2K output.

## Hardware fit (this machine)

Single RTX PRO 6000 Blackwell 96GB, 125GB host RAM.

- Transformer: 61.7GB bf16 — full precision on GPU during denoising.
- Qwen3-VL-32B conditioner: quantized to int8 (~32GB) at load using the official torchao recipe,
  because bf16 for both components (~124GB) exceeds host RAM.
- A diffusers `ComponentsManager` auto-swaps components between GPU and CPU
  (`memory_reserve_margin="12GB"`).
- `HF_HOME` defaults to `~/.cache/hf` (the standard `~/.cache/huggingface` is root-owned on this box).

## Setup

```bash
uv venv --python 3.12
uv pip install -r requirements.txt --torch-backend=auto

# Weights (~130GB, diffusers layout only; skips the original FL2VA/Ref2VA checkpoints):
export HF_HOME=~/.cache/hf
hf download MiniMaxAI/MiniMax-H3 --include 'modular_model_index.json' 'transformer/*' \
  'text_encoder/*' 'tokenizer/*' 'processor/*' 'vae/*' 'audio_vae/*' \
  'scheduler/*' 'audio_scheduler/*'
```

> The GPU must be free first — stop the vLLM server if it's running:
> `systemctl --user stop vllm.service` (restart later with `start`).

## Usage

```bash
source .venv/bin/activate

# Text to video+audio
python generate.py 'A red fox trotting through a snowy pine forest, snow crunching underfoot'

# First-frame conditioned (canvas follows the image's aspect ratio)
python generate.py 'The astronaut waves at the camera' --image astronaut.jpg

# Smaller canvas ≈ 2.3x faster per step than the default 1344x768
python generate.py 'Ocean waves at sunset' --width 960 --height 544 --num-frames 124
```

Output lands in `outputs/` as an mp4 with the soundtrack muxed in.

Constraints to keep in mind:

- `--num-frames` snaps up to the next `17*n + 5`; duration must stay within 5–15s at 24fps.
- `--height`/`--width` must be multiples of 32; short edge is designed for 768.
- The checkpoint is CFG-distilled: no negative prompt, no guidance scale.
- Prompt quality matters a lot — the hosted pipeline uses a rewriting stage (H3-Context-IR).
  See the [prompting guide](https://huggingface.co/MiniMaxAI/MiniMax-H3) on the model card;
  detailed, shot-by-shot descriptions with an `overall_soundscape` section work best.

## License

Model weights are under the MiniMax H3 Community License Agreement (see model card).
