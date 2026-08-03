# Tech Context

## Machine

- GPU: NVIDIA RTX PRO 6000 Blackwell, 96GB VRAM, sm_120, driver 595.71.05 (CUDA 13.2).
- Host: 125GB RAM, 8GB swap. ~767GB free disk at setup time.
- The GPU is shared with a vLLM server (`vllm.service`, user systemd unit, Qwen3.6-35B-A3B on
  port 8000, ~93GB VRAM). **Stop it before generating**: `systemctl --user stop vllm.service`;
  restart with `start` when done.

## Environment

- Python 3.12 venv at `.venv/`, managed with `uv`.
- torch 2.13.0+cu130 (CUDA 13 build required for Blackwell; installed with `--torch-backend=auto`).
- diffusers pinned to PR #14355 commit `abc5e9bf` (MiniMax-H3 not in a release yet — check for a
  proper release and unpin later).
- transformers 5.14.1, torchao 0.17.0 (int8 quant), av 18.0.0 (video/audio mux).

## Gotchas

- `~/.cache/huggingface` is **root-owned** on this machine and sudo needs a password, so
  `HF_HOME=~/.cache/hf` is used instead. `generate.py` sets this default itself.
- Weights live in `~/.cache/hf/hub/models--MiniMaxAI--MiniMax-H3` (~130GB, diffusers layout only).
- `hf` CLI and `uv` are installed globally at `~/.local/bin`.
- No ffmpeg on PATH; mp4 muxing goes through PyAV, which is fine.
