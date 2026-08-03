# Progress

## Works

- Environment: torch 2.13+cu130 sees the Blackwell GPU; diffusers PR branch imports
  `MiniMaxH3Blocks` / `MiniMaxH3Ref2VABlocks` cleanly.

## In flight

- Weight download (~130GB) to `~/.cache/hf`.
- First generation not yet run — `generate.py` is untested end to end.

## Not built

- ref2va (needs `transformer_ref/*` download + a script path using `MiniMaxH3Ref2VABlocks`).
- 2K workflow (hosted API) and H3-Context-IR prompt rewriting.

## Known issues

- `~/.cache/huggingface` root-owned → using `HF_HOME=~/.cache/hf` everywhere.
- vLLM service is stopped; Qwen LLM on port 8000 is down until restarted.
