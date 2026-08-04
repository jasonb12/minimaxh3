# Active Context

## Current focus

Web UI added (2026-08-04). `app.py` serves Gradio on 0.0.0.0:7860 for remote use;
verified end-to-end via the Gradio API (5.2s clip in 2.6 min including lazy model load).
Running in the background via nohup, log at /tmp/h3_gradio.log. Not a systemd service
(deliberately — it would fight vllm.service for the GPU on boot).

Bring-up completed 2026-08-03: first verified clip `outputs/smoke_test.mp4`
(960x544, 5.2s, stereo audio) generated in ~4 min + 21s pipeline load.

## Recent changes

- Stopped `vllm.service` to free the GPU. **Still stopped** — restart with
  `systemctl --user start vllm.service` when H3 isn't needed.
- Downloaded ~150GB of weights to `~/.cache/hf` (took ~4h over WiFi at 7-12 MB/s).
- Fixed missing `torchvision` (Qwen3VL video processor requires it; without it the
  processor component fails to load and the text encoder step crashes with
  `'NoneType' object has no attribute 'create_mm_token_type_ids'`).

## Next steps

Possible follow-ups, none started:
1. Default-canvas (1344x768) and fl2va (image-conditioned) runs.
2. ref2va support: download `transformer_ref/*` (~62GB), script path via `MiniMaxH3Ref2VABlocks`.
3. Unpin diffusers once MiniMax-H3 lands in a release.
4. Set HF_TOKEN for faster future downloads.
