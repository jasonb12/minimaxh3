# Tech Context

## Spark checkout (2026-09-05)

The Spark changes are on branch `minimaxh3_spark`, based on `origin/master` at
`79d76fa`, on `sparky` (GB10 SM121, ARM64, 128 GB unified memory, driver
580.173.02). The RTX PRO 6000 notes below describe the original host.

See `docs/SPARK.md` for installation, service commands, cache locations, and
the measured FP8 choice. User unit: `minimax-h3-spark.service`; LAN port 7860;
linger enabled. Native `.venv` uses CUDA 13 wheels. H3 uses tensorwise FP8 with
compiled repeated blocks, prequantized Qwen FP8, and resident CUDA placement.
Do not apply the discrete-GPU exclusive-offload guard to this profile.
`spark.release_checkpoint_pages()` uses file-specific `posix_fadvise` to free
checkpoint page cache before CUDA allocation; no root cache drop is needed.

Full-model text-to-video, first-frame, and Ref2VA renders pass video/audio
decode checks and frame inspection.
Warm text generation: 93.5 seconds for 832×480 / 124 frames / Turbo 7 points.
Resident CUDA models: 66.9 GiB. Both transformer partitions and their derived
FP8 caches are complete. The service is idle with FL2VA loaded and ready.
Offline loading uses explicit local component directories plus a complete
finegrained-fp8 v4 kernel snapshot (not just the selected CUDA build).

## Machine

- GPU: NVIDIA RTX PRO 6000 Blackwell, 96GB VRAM, sm_120, driver 595.71.05 (CUDA 13.2).
- Host: 125GB RAM, 8GB swap. ~767GB free disk at setup time.
- The GPU is shared with a vLLM server (`vllm.service`, user systemd unit, Qwen3.6-35B-A3B on
  port 8000, ~93GB VRAM). **Stop it before generating**: `systemctl --user stop vllm.service`;
  restart with `start` when done.

## Environment

- Python 3.12 venv at `.venv/`, managed with `uv`.
- torch 2.13.0+cu130 (CUDA 13 build required for Blackwell; installed with `--torch-backend=auto`).
- diffusers 0.40.0 (first release with MiniMax-H3; upgraded 2026-09-03 from the PR #14355 commit
  `abc5e9bf` pin). `.venv-pinned-abc5e9bf/` is the pre-upgrade venv, kept for rollback.
- peft 0.20.0 must be pinned: diffusers' LoRA loader imports it lazily and `USE_PEFT_BACKEND` is
  evaluated at import time, so installing it into a running service still needs a restart.
- transformers 5.14.1, torchao 0.17.0 (int8 quant), av 18.0.0 (video/audio mux).

## Gotchas

- `~/.cache/huggingface` is **root-owned** on this machine and sudo needs a password, so
  `HF_HOME=~/.cache/hf` is used instead. `generate.py` sets this default itself.
- Weights live in `~/.cache/hf/hub/models--MiniMaxAI--MiniMax-H3` (FL2VA ~150GB +
  `transformer_ref/` ~62GB for Ref2VA; diffusers layout only).
- `hf` CLI and `uv` are installed globally at `~/.local/bin`.
- No ffmpeg on PATH; mp4 muxing goes through PyAV, which is fine.
