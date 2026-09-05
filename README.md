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

**DGX Spark / GB10:** use the [Spark setup](docs/SPARK.md). The `spark` profile
uses resident FP8 models, compiled transformer blocks, and Turbo sampling. The
configuration below describes the original RTX PRO 6000 host.

Single RTX PRO 6000 Blackwell 96GB, 125GB host RAM.

- Transformer: 61.7GB bf16 — full precision on GPU during denoising.
- Qwen3-VL-32B conditioner: quantized to int8 (~32GB) at load using the official torchao recipe,
  because bf16 for both components (~124GB) exceeds host RAM.
- A diffusers `ComponentsManager` auto-swaps components between GPU and CPU
  (`memory_reserve_margin` 12GB, override with `MINIMAX_H3_RESERVE_MARGIN_GB`). No margin
  lets the transformer and conditioner share the card, so the margin is activation
  headroom only; the swap itself is avoided by staged scheduling (below).
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
transformer partition loads on first use and stays resident for later jobs of
the same task. Switching modes reloads ~62GB of weights. Requests run one at a
time. Outputs land in `outputs/gradio/`.

Ref2VA accepts up to **9 graphic/image references**, 3 video references, and
3 audio references, with 12 references total. The UI shows a live graphic
reference counter; upload order maps to `<Picture 1>`, `<Picture 2>`, and so on.

## REST API

The same server exposes a job-based REST API on port 7860 ([docs/API.md](docs/API.md),
Swagger at `/api/docs`). Submit, poll, download:

```bash
JOB=$(curl -s http://localhost:7860/api/generate -H 'Content-Type: application/json' \
  -d '{"prompt": "A red panda waves. overall_soundscape: forest ambience.", "turbo": true}' | jq -r .job_id)
curl -s http://localhost:7860/api/jobs/$JOB | jq          # queued -> running -> done
curl -o out.mp4 http://localhost:7860/api/jobs/$JOB/video
```

The REST API supports FL2VA first/last frames and Ref2VA image references
(`reference_image_urls`). Videos persist in `outputs/api/`.

Queued jobs are scheduled in two stages: while the conditioner is on the GPU for
one job, the prompts of the next few queued jobs (same task) are encoded too, so
they later denoise without swapping the 62GB transformer out and back. Tune with
`MINIMAX_H3_PREFETCH_JOBS` (default 3). See [docs/API.md](docs/API.md#staged-scheduling).

For an interactive test that should run immediately after the current render,
use the queued CLI. Its default priority is 0; normal API jobs use 100:

```bash
.venv/bin/python 'queue_generate.py' 'A red panda waves at the camera' \
  --priority '0' \
  --output 'outputs/priority_test.mp4'
```

Priority does not interrupt a running generation. It moves the job ahead of
normal jobs that are still queued. `test_turbo.sh` uses this priority path.

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

Turbo is enabled by default in the CLI, REST API, and web UI. It applies the community
[v4-600 distillation LoRA](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora)
at 6 model evaluations (our `steps=7`) and strength 1.0 instead of ~49 steps.
v1-850 at 4 evals measured 248s → 47s end-to-end at 960×544/5.2s; v4 at 6 evals
is a bit slower and should look better (less plastic / over-sharp). Ref2VA can
use the same structurally compatible adapter experimentally, but it was trained
on FL2VA and may weaken reference identity.
Use `--no-turbo`, `"turbo": false`, or uncheck Turbo for full-quality sampling.
See [docs/TURBO.md](docs/TURBO.md) for the key-remapping details, step mapping,
and when to fall back to the older v1-850 file.

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
