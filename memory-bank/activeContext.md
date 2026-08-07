# Active Context

## Current focus (2026-08-07)

REST API added to `app.py`: FastAPI routes mounted alongside Gradio on the
same port (7860) via `gr.mount_gradio_app` + uvicorn. `POST /api/generate`
returns a job id; `GET /api/jobs/{id}` polls; `GET /api/jobs/{id}/video`
downloads the mp4 (persisted in `outputs/api/`). Single worker thread; UI and
API share `_gen_lock` (turbo adapter toggle mutates shared transformer state,
so generations must serialize). FL2VA only over REST — Ref2VA stays on the UI
/ `/gradio_api`. Swagger at `/api/docs`. Documented in docs/API.md. Verified
end-to-end (turbo job: 50s, h264+aac output, validation/404 paths checked).
Server runs via nohup, log at `logs/app.log`.

## Previous focus: Turbo LoRA (2026-08-07)

Turbo LoRA integrated: community 4-step distillation LoRA
(larryvrh/MiniMax-H3-Turbo-Lora) applied via `turbo.py` key remapping +
PEFT adapter. Measured 248s -> 47s end-to-end at 960x544/5.2s. `--turbo` on
the CLI, Turbo checkbox in the web UI (both verified end-to-end). FL2VA only.
Full solution documented in docs/TURBO.md — read it before touching the
remapping (the fc1 half-swap and the "no custom sampler needed" reasoning are
non-obvious).

## Previous focus: Ref2VA (2026-08-04)

Ref2VA wired up (2026-08-04). `generate.py` and `app.py` support both FL2VA
(`transformer/`) and Ref2VA (`transformer_ref/` via `MiniMaxH3Ref2VABlocks`).
CLI: `--ref image:path` (repeatable, order semantic). Gradio: mode radio that
swaps the resident transformer partition on demand.

`transformer_ref/*` download was started to `HF_HOME=~/.cache/hf` (~62GB); check
`/tmp/h3_transformer_ref_download.log` until complete before the first Ref2VA run.

Web UI (FL2VA) was already verified end-to-end earlier the same day. Gradio runs
via nohup, log at `/tmp/h3_gradio.log`. Restart after the Ref2VA code change.

## Recent changes

- Added Ref2VA path: `build_pipeline(..., task="ref2va")`, `--ref KIND:PATH`,
  Gradio mode + gallery/video/audio inputs.
- Stopped `vllm.service` to free the GPU. **Still stopped** — restart with
  `systemctl --user start vllm.service` when H3 isn't needed.
- Downloaded ~150GB of FL2VA-side weights to `~/.cache/hf`.
- Fixed missing `torchvision` (Qwen3VL video processor requires it).

## Next steps

1. Finish `transformer_ref/*` download; smoke-test a single-image Ref2VA run
   (furniture / product) at 960x544.
2. Default-canvas (1344x768) and fl2va (image-conditioned) runs.
3. Unpin diffusers once MiniMax-H3 lands in a release.
4. Set HF_TOKEN for faster future downloads.
