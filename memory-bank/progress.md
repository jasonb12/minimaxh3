# Progress

## Works

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
