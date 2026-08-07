# MiniMax-H3 Local Inference

Local video + stereo audio generation with [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3),
MiniMax's omni-modal model (open weights released 2026-08-03). It jointly denoises video and
32kHz stereo audio in one transformer pass: 24fps, 5–15 seconds, 768px short edge.

This repo runs the official
[diffusers integration](https://github.com/huggingface/diffusers/pull/14355)
(Modular Diffusers blocks), covering:

- **t2va** — text to video+audio (`transformer/`)
- **fl2va** — first-frame and/or last-frame conditioned video+audio (`transformer/`)
- **ref2va** — omni-reference images/videos/audio (`transformer_ref/`, extra ~62GB)

Not set up here: the hosted **H3-Context-IR** / **H3-Regenerate-2K** API stages used for full 2K output.

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

# Weights (diffusers layout only; skips the original FL2VA/Ref2VA folders):
export HF_HOME=~/.cache/hf
hf download MiniMaxAI/MiniMax-H3 --include 'modular_model_index.json' 'transformer/*' \
  'transformer_ref/*' 'text_encoder/*' 'tokenizer/*' 'processor/*' 'vae/*' 'audio_vae/*' \
  'scheduler/*' 'audio_scheduler/*'
```

> The GPU must be free first — stop the vLLM server if it's running:
> `systemctl --user stop vllm.service` (restart later with `start`).

## Web UI

```bash
.venv/bin/python app.py
```

Serves Gradio on `http://0.0.0.0:7860` (LAN-reachable, no auth — don't expose past the LAN).
Pick **FL2VA** (text / first-last frame) or **Ref2VA** (reference images/video/audio). The matching
transformer partition loads lazily on first use; switching modes reloads ~62GB of weights.
Requests queue one at a time. Outputs land in `outputs/gradio/`.

## CLI usage

```bash
source .venv/bin/activate

# Text to video+audio
python generate.py 'A red fox trotting through a snowy pine forest, snow crunching underfoot'

# First-frame conditioned (canvas follows the image's aspect ratio)
python generate.py 'The astronaut waves at the camera' --image astronaut.jpg

# Omni-reference (subject / product identity). Order is semantic — name each ref in the prompt.
python generate.py 'Use <Picture 1> as the product; slow orbit around it in a bright showroom' \
  --ref image:sofa_front.jpg --ref image:sofa_side.jpg

# Mixed refs: image + motion video + voice
python generate.py 'The character from <Picture 1> walks as in <Video 1>, speaking with <Audio 1>' \
  --ref image:subject.jpg --ref video:walk.mp4 --ref audio:line.wav

# Smaller canvas ≈ 2.3x faster per step than the default 1344x768
python generate.py 'Ocean waves at sunset' --width 960 --height 544 --num-frames 124
```

Output lands in `outputs/` as an mp4 with the soundtrack muxed in.

## Turbo (~5x faster sampling)

`--turbo` (CLI) or the Turbo checkbox (web UI) applies the community
[4-step distillation LoRA](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora):
4 model evaluations instead of ~49, measured 248s → 47s end-to-end at 960×544/5.2s.
Preview quality (can show plastic skin / over-sharp grain); FL2VA only.
See [docs/TURBO.md](docs/TURBO.md) for the key-remapping details and tuning.

Constraints to keep in mind:

- `--num-frames` snaps up to the next `17*n + 5`; duration must stay within 5–15s at 24fps.
- `--height`/`--width` must be multiples of 32; short edge is designed for 768.
- The checkpoint is CFG-distilled: no negative prompt, no guidance scale.
- Ref2VA limits: ≤9 images, ≤3 videos, ≤3 audios, ≤12 total; audio cannot be the sole input.
- `--ref` cannot be combined with `--image` / `--last-image` (different transformer partitions).
- Prompt quality matters a lot — the hosted pipeline uses a rewriting stage (H3-Context-IR).
  See the [prompting guide](https://huggingface.co/MiniMaxAI/MiniMax-H3) and the
  [ref prompt guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md).

## License

Model weights are under the MiniMax H3 Community License Agreement (see model card).
