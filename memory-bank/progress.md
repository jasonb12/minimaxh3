# Progress

## Works

- Gradio web UI (`app.py`) on port 7860: prompt + optional first/last frame images,
  canvas presets, duration/steps/seed controls, queued single-slot generation,
  lazy pipeline load with a friendly error if vLLM holds the GPU.

- Full t2va pipeline on the single 96GB Blackwell card: int8 text encoder + bf16 transformer
  + ComponentsManager auto CPU offload.
- Smoke test: 124 frames @ 960x544 with stereo audio in 248s of denoising (21s load).
  Output quality matches prompt (verified frame + audio RMS).
- Weights cached at `~/.cache/hf/hub/models--MiniMaxAI--MiniMax-H3` (~150GB).

## Not built

- ref2va (needs `transformer_ref/*` download + `MiniMaxH3Ref2VABlocks` script path).
- 2K workflow (hosted API) and H3-Context-IR prompt rewriting.
- fl2va tested only implicitly (same blockset as t2va); no image-conditioned run yet.

## Known issues

- `~/.cache/huggingface` root-owned → using `HF_HOME=~/.cache/hf` everywhere.
- vLLM service (`vllm.service`) left stopped; Qwen endpoint on port 8000 down until restarted.
  H3 and the vLLM server cannot share the 96GB GPU.
- WiFi-only connectivity caps downloads at ~7-12 MB/s; no HF_TOKEN configured.
