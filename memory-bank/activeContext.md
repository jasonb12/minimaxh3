# Active Context

## Current focus

Initial bring-up (2026-08-03): repo scaffolded, env installed, weights downloading (~130GB),
first smoke-test generation pending.

## Recent changes

- Stopped `vllm.service` to free the GPU (was using ~93GB since Jul 28). Not restarted.
- Created venv, installed torch 2.13+cu130 and diffusers from PR #14355.
- Wrote `generate.py` (t2va/fl2va CLI), README, requirements.
- Weight download running in background to `HF_HOME=~/.cache/hf`.

## Next steps

1. Finish weight download; run smoke test (small canvas 960x544, ~5s clip).
2. Verify audio+video mux in the output mp4.
3. Later, optionally: add ref2va support, unpin diffusers when released, try int8 transformer
   for faster loading.
