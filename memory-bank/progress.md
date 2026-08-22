# Progress

## Works

- Ref2VA denoise offloads the conditioner (and other siblings) before each
  transformer forward so a resident pipe cannot keep ~93GB occupied. The
  Cake Box turbo script uses 416×736 plus 1080×1920 output normalization.
- Priority REST scheduling is available: normal jobs default to 100, queued CLI
  tests use 0 and become next after the active render, with FIFO ordering among
  equal priorities. `queue_generate.py` submits, polls and downloads output.
- Queued REST/UI jobs reuse the resident pipeline instead of discarding it
  after every encode. The previous discard left ~89GB allocated, then the
  70GB free-VRAM check refused the next load. The preflight now ignores this
  process's own VRAM. The service is configured to restart after any unexpected
  exit.
- Brightify production jobs can request deterministic output normalization
  (dimensions and exact duration); 1920×1080/24fps/10.0s H3 output passed the
  canonical asset/compliance pipeline.
- REST API output audio is selectable with `include_audio`; false produces a
  validated silent H.264 MP4 while preserving true as the direct-API default.
- Ref2VA's Gradio gallery supports multiple graphic references and displays a
  live count against the model limit (9 images; 12 mixed references total).
- REST jobs support Ref2VA image-reference URLs without treating them as first
  frames; status responses expose only the reference count, never the URLs.
- Experimental Turbo Ref2VA is functional: the FL2VA-trained LoRA attaches to
  `transformer_ref`, and job `1faa27936158` rendered 124 frames at 960×544 in
  84.9 seconds. The output retained the Doritos product identity across a turn,
  but the UI/docs warn that Ref2VA fidelity is not guaranteed.
- Turbo defaults on across the UI, REST API, CLI, and Brightify requests;
  full-quality sampling remains an explicit opt-out.
- A two-segment loop test passed using Ref2VA v1 plus reversed-boundary FL2VA v2;
  the assembled loop has low measured join/playback seam error.
- REST API (docs/API.md) on the same port as the UI: job-based
  submit/poll/download under `/api/*`, FL2VA only, first/last frame via
  base64 or URL. Jobs in memory, mp4s in `outputs/api/`. Verified end-to-end
  with a turbo job (50s gen) plus validation/404 paths.

- Turbo LoRA (docs/TURBO.md): 4-eval sampling via `--turbo` / web UI checkbox,
  ~5.3x wall-clock speedup, verified on CLI and through the Gradio API.
  Preview-quality caveats (plastic skin, grain) tunable via strength.

- Gradio web UI (`app.py`) on port 7860: FL2VA (prompt + optional first/last
  frames) and Ref2VA (reference images/videos/audio), canvas presets,
  duration/steps/seed, queued single-slot generation, lazy pipeline load with a
  friendly error if vLLM holds the GPU. Mode switch reloads the transformer
  partition.

- CLI (`generate.py`): t2va / fl2va via `--image`/`--last-image`; ref2va via
  `--ref KIND:PATH` (image|video|audio). Same int8 conditioner + auto offload.

- Full t2va pipeline on the single 96GB Blackwell card: int8 text encoder + bf16
  transformer + ComponentsManager auto CPU offload.
- Smoke test (FL2VA): 124 frames @ 960x544 with stereo audio in 248s of
  denoising (21s load). Output quality matches prompt.
- Weights cached at `~/.cache/hf/hub/models--MiniMaxAI--MiniMax-H3`
  (FL2VA side ~150GB; `transformer_ref/` download in progress).

## Not built / not verified

- Ref2VA end-to-end smoke test (blocked on finishing `transformer_ref/*`).
- fl2va tested only implicitly (same blockset as t2va); no image-conditioned run yet.
- 2K workflow (hosted API) and H3-Context-IR prompt rewriting.

## Known issues

- `~/.cache/huggingface` root-owned → using `HF_HOME=~/.cache/hf` everywhere.
- vLLM service (`vllm.service`) left stopped; Qwen endpoint on port 8000 down until restarted.
  H3 and the vLLM server cannot share the 96GB GPU.
- WiFi-only connectivity caps downloads at ~7-12 MB/s; no HF_TOKEN configured.
  FL2VA and Ref2VA cannot both stay resident — mode switches unload/reload.
