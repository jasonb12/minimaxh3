# Active Context

## Current focus (2026-08-08)

Ref2VA turbo at 544×960 / 10.4s / 4 pictures OOMed during denoise:
92.75 GiB already allocated (resident denoiser plus conditioner) and a
6.52 GiB step failed. The denoiser forward now offloads every other
managed component first. `run_turbo.sh` generates at 416×736 and
normalizes to 1080×1920.

The server keeps the loaded pipeline resident after each job of the same task.
Discard-after-every-job dropped the Python handle without returning ~89GB of
VRAM, then `_check_gpu_free()` treated that leftover as a foreign occupant and
rejected the next queued load. Reuse applies Turbo on the weights already on
CUDA. The free-VRAM preflight now subtracts only other processes. Mode
switches still replace the partition. The systemd unit uses `Restart=always`,
a five-second delay, and unlimited start attempts.

REST jobs now use a stable priority queue: normal callers default to priority
100, while `queue_generate.py` and `test_turbo.sh` submit at priority 0. Lower
values run first and FIFO order is preserved within a priority. Priority is
non-preemptive, so an urgent CLI test becomes the next job without interrupting
the render already on the GPU.

Brightify's production `local_h3` engine is now linked to canonical
`generation_jobs`. REST jobs accept normalized output width/height/duration;
the hosted parity run upscaled H3 to 1920×1080, padded the hero hold to exactly
10.0s, uploaded through the leased worker, finalized an `asset_version`, and
passed compliance.

REST jobs now accept `include_audio` (default true for backward compatibility).
When false, H3 still performs joint video/audio inference but `encode_video`
writes a silent MP4. Brightify's corrected Home Banner acceptance used
`turbo=true` and `include_audio=false`; ffprobe confirmed one H.264 stream and
no audio stream.

The Ref2VA Gradio UI now labels image inputs as graphic references and shows a
live `selected / 9` counter with multi-upload/order guidance. Existing backend
limits remain authoritative: 9 images, 3 videos, 3 audios, 12 references total.

The REST API now accepts up to nine `reference_image_urls` and routes those jobs
through Ref2VA. References cannot be mixed with FL2VA keyframes; signed URLs
are omitted from public job status. Turbo can be attached to `transformer_ref`
experimentally because its module layout matches FL2VA, but the LoRA was not
trained for Ref2VA and may weaken identity fidelity. Local job `1faa27936158`
validated the path at 960×544/124 frames/5 steps in 84.9 seconds with a
recognizable product across the generated turn. Brightify job `bea6178a…`
validated a packshot as identity-only reference: generated campaign art appears
at frame zero, with no white source canvas.

Turbo is now the default for Gradio, REST, CLI, and Brightify H3 queue jobs.
Callers can explicitly opt out (`turbo=false`, `--no-turbo`, or uncheck Turbo)
to use the full-quality sampler.

Brightify's fixture-level loop experiment validated a Ref2VA front→back v1
followed by an FL2VA back→front v2 keyed from `v1:last` to `v1:first`. The
assembled 10.292s loop measured ~3.5 MAE at the segment join and ~4.0 after
encoding across the playback seam. General worker integration is still deferred.

## Previous focus (2026-08-07)

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
