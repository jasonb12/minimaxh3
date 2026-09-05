# Progress

## Works

- diffusers 0.40.0 upgrade (2026-09-03): `ModularPipeline.from_pretrained(workflow=...)`,
  dataclass references, `_CONDITIONING_BLOCKS = ("before_encode", "text_encoder")`. Verified
  FL2VA Turbo (55s) and Ref2VA Turbo (97s) at 960×544/5.2s; seed 1234 reproduces the same
  composition as the `abc5e9bf` build. Two upgrade bugs fixed: `peft` unpinned, and the
  exclusive-GPU guard hid the transformer signature (needs `functools.wraps`).
- Staged REST scheduling: the conditioner stage for up to 3 queued same-task
  jobs runs while the conditioner is already on the GPU; those jobs then
  denoise with the transformer resident. Pre-encoded Turbo 960×544 jobs run in
  ~37s versus 56–68s with a per-job swap. Verified end-to-end 2026-09-02.
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

- Turbo LoRA (docs/TURBO.md): v4-600 EMA via `--turbo` / web UI checkbox,
  default 6 evals (`steps=7`) at strength 1.0. v1-850 at 4 evals measured
  ~5.3x wall-clock (248s → 47s). Preview-quality caveats remain; v4 is meant
  to drop the plastic/over-sharp look of v1.
- LightX2V's Ref2VA-trained 4-eval Turbo LoRA was integrated, A/B tested
  (faster but poor output quality) and **removed** — a direction chosen
  against. See docs/TURBO.md "Rejected alternative".

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
