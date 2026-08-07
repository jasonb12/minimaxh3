# Product Context

## Why

MiniMax released H3's open weights on 2026-08-03. Jason wants it running locally instead of
through MiniMax's paid API. The GPU normally serves a Qwen LLM via vLLM; H3 is used on demand.

## How it should work

- `python generate.py '<prompt>' [--image first.jpg] [--last-image last.jpg]` produces a
  768p-class mp4 (5–15s, 24fps) with native stereo audio in `outputs/` (t2va / fl2va).
- `python generate.py '<prompt>' --ref image:sofa.jpg` (repeatable `--ref KIND:PATH`) runs
  Ref2VA for subject/product consistency (furniture, characters, style, etc.).
- Generation quality depends heavily on prompt detail; the model card's prompting guide
  (shot-by-shot description + `overall_soundscape` + `non_diegetic_music`) is the reference.
  Ref2VA prompts should name each reference (`<Picture 1>`, …).

## Model overview

H3 is a 33B dense single-stream diffusion transformer that jointly denoises video and audio
latents in one packed sequence, conditioned on Qwen3-VL-32B hidden states (layer 50, unnormalized).
Separate video VAE (f16t4d24) and stereo audio VAE (40Hz latents @ 32kHz). Two schedulers
(video shift=12.0, audio shift=3.0). Released checkpoints are CFG-distilled.
