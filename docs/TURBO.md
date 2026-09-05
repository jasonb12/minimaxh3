# Turbo LoRA: few-step sampling (~5x faster)

**What:** [larryvrh/MiniMax-H3-Turbo-Lora](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora)
(Apache-2.0) — a community distillation LoRA released Aug 5, 2026 and
[highlighted by MiniMax](https://x.com/MiniMax_AI/status/2085614043512127542) — that
lets H3-Base sample in **as few as 4 model evaluations instead of ~49**.

**Default here:** `minimax_h3_turbo_v4_step600_ema.safetensors` at **7 grid
points / 6 model evaluations**, strength **1.0**. That matches the author's
`--steps 6` recipe. v4 is the current best checkpoint: better static and
small-motion shots, better micro-detail, and the over-sharpening / plastic look
of the earlier v1 (~850) line is resolved.

**Measured here** (RTX PRO 6000 96GB, 960×544, 124 frames, v1-850 at 4 evals):
baseline 248s → turbo 47s end-to-end (~5.3x); denoising itself ~4min → ~20s.
v4 at 6 evals is a bit slower than that 4-eval number and should look better.

**Status:** still preview quality. Audio and fast, intense motion are the two
areas the author is still improving. The LoRA was trained on FL2VA, but
`transformer_ref` has the same target module layout, so this integration can
attach it to Ref2VA as an **experimental** acceleration path. Ref2VA fidelity is
not guaranteed: the LoRA author still describes native Ref2VA training as
planned, and community tests report identity loss at very low step counts.

## Which checkpoint

| File | Use |
|---|---|
| `minimax_h3_turbo_v4_step600_ema.safetensors` | **Default.** Best for almost everything, especially 6–8 sampler steps. |
| `minimax_h3_turbo_4step_ema_ckpt850.safetensors` | Older v1 line. Friendlier only for **4-step heavy / fast motion**, where v4 can smear. |

Override `turbo.TURBO_FILE` if you need the v1 file; this repo does not switch
automatically.

## Rejected alternative: LightX2V

[lightx2v/Minimax-h3-Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo)
(ModelTC) ships its own distillations, including the only **Ref2VA-trained**
Turbo checkpoint (`minimax_h3_ref2v_turbo_4step_v0.1_bf16`). We integrated it
as an opt-in variant and A/B-tested it against larryvrh v4-600 (Aug 2026, same
seed, Ref2VA cake-box scenario): output quality was judged poor, and the
integration was removed. Their 768p FL2VA files also need video shift 6, which
the server has no plumbing for. **Decision: not pursuing LightX2V** —
larryvrh remains the only Turbo family here.

## Usage

```bash
# CLI: Turbo is on by default (v4-600, 7 grid points / 6 evals, strength 1.0)
python generate.py 'prompt...'
python generate.py 'prompt...' --turbo-strength 0.9   # against grain, only if needed
python generate.py 'Use <Picture 1> as the product...' \
  --ref image:product.png  # experimental Ref2VA
python generate.py 'prompt...' --no-turbo  # full-quality sampler
# Web UI and REST API also default Turbo on; uncheck / send turbo=false to opt out.
```

Keep strength at **1.0** unless a specific clip misbehaves: **up** (1.05–1.2)
against blurry ghosting/smear, **down** (0.8–0.95) against over-sharp grain.

Author sampler-step range is 4–8 (6–8 looks best for v4). Past 8 evals it stops
helping and can over-sharpen. Our `steps` field is `num_inference_steps` (sigma
grid points including terminal 0), so **evals = steps − 1**:

| Author / ComfyUI steps (evals) | Our `steps` |
|---|---|
| 4 | 5 |
| 6 (default) | 7 |
| 8 | 9 |

## Why no custom sampler is needed (unlike ComfyUI)

H3 denoises video and audio latents on two different flow schedules (video
shift 12, audio shift 3). ComfyUI's stock samplers step both streams on one
schedule — tolerable at ~20 steps, but at few steps the audio is over-stepped and
comes out distorted, hence the companion "MiniMax-H3 Turbo Sampler" node.

The diffusers integration never had this problem: `MiniMaxH3Blocks` steps each
stream on its own `MiniMaxH3Scheduler`. Both schedulers build their sigma grid
as `linspace(1, 0, num_inference_steps)` pushed through their own shift — which
is *exactly* the grid the LoRA's reference sampler uses (`shift_sigma(1 - i/n)`
per stream, cf. `generate.py` in the LoRA repo). So the only change needed is
the LoRA plus a short `num_inference_steps` (`turbo.TURBO_NUM_INFERENCE_STEPS`).

## Key remapping (`turbo.convert_turbo_lora_to_diffusers`)

The LoRA ships in the ComfyUI checkpoint naming (`base_model:
Comfy-Org/MiniMax-H3`, 518 tensors: 50 blocks × {qkv, out, fc1, fc2, adaln} +
2 refiner blocks × {qkv, out, fc1, fc2} + final adaln). v4 uses the same layout
as v1. Our transformer is the diffusers conversion, so keys are remapped before
loading:

| ComfyUI key | diffusers key | transform |
|---|---|---|
| `blocks.N.*` | `transformer_blocks.N.*` | rename |
| `token_refiner.blocks.N.*` | `token_refiner.refiner_blocks.N.*` | rename |
| `attn.qkv_proj` | `attn.to_q` / `to_k` / `to_v` | A shared ×3; B split in row thirds (q,k,v) |
| `attn.out_proj` | `attn.to_out.0` | rename |
| `mlp.fc1` | `ff.net.0.proj` | **B row halves swapped** (see below) |
| `mlp.fc2` | `ff.net.2` | rename |
| `blocks.N.adaln_proj.linear` | same | none (row layout identical) |
| `final_layer.adaln_proj.linear` | `norm_out.linear` | rename |

The fc1 swap: ComfyUI computes `silu(x1) * x2` (gate first), diffusers' SwiGLU
computes `hidden * silu(gate)` (gate second), and the diffusers weight
conversion swapped the projection's row halves accordingly — so the LoRA's B
matrix for fc1 must swap the same way.

Every mapping was verified empirically by comparing weight slices of
`Comfy-Org/MiniMax-H3 minimax_h3_fl2va_bf16.safetensors` (HTTP range reads)
against the local diffusers shards: qkv thirds, out_proj, fc2, both adaln
layouts and the refiner matched directly; fc1 matched only half-swapped. v4-600
converts to the same 726 diffusers tensors (700 block + 24 refiner + 2 norm_out).

## Application details

- Loaded as a **PEFT adapter** (`transformer.load_lora_adapter`), i.e. applied
  at run time in activation space (`y = base(x) + B(A(x))`). This matches the
  author's guidance: merging into bf16 base weights would round most of the
  small update away. Alpha = rank in this LoRA, and diffusers defaults alpha to
  rank when the state dict carries no alpha keys, so the net scale is 1.0.
- Strength maps to `set_adapters(weights=[s])`; the web UI toggles the resident
  adapter with `enable_adapters()`/`disable_adapters()` between requests, so
  turbo and full-quality runs can be mixed without reloading anything.
- Weights file: `minimax_h3_turbo_v4_step600_ema.safetensors` (744MB bf16),
  cached under `$HF_HOME` on first use.
