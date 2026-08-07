# Project Brief

Run MiniMax-H3 (omni-modal video+audio generation model, open-weights released 2026-08-03)
locally on this machine's single RTX PRO 6000 Blackwell 96GB GPU.

## Goals

- Working local inference for t2va (text-to-video+audio), fl2va (first/last-frame conditioning),
  and ref2va (omni-reference images/videos/audio).
- Simple CLI (`generate.py`) that writes mp4s with muxed stereo audio.
- Documented setup that fits this machine's memory constraints.

## Out of scope (for now)

- 2K output — requires MiniMax's hosted H3-Regenerate-2K API.
- Prompt rewriting (H3-Context-IR) — hosted API; users write raw prompts per the prompting guide.
