# Turbo LoRA: 4-step sampling (~5x faster)

**What:** [larryvrh/MiniMax-H3-Turbo-Lora](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora)
(Apache-2.0) — a community distillation LoRA released Aug 5, 2026 and
[highlighted by MiniMax](https://x.com/MiniMax_AI/status/2085614043512127542) — that
lets H3-Base sample in **4 model evaluations instead of ~49**.

**Measured here** (RTX PRO 6000 96GB, 960×544, 124 frames, same prompt):
baseline 248s → turbo 47s end-to-end (~5.3x); denoising itself ~4min → ~20s.

**Status:** preview quality. `ckpt850` (EMA) is sharp at 4 steps but can show
plastic-looking skin and over-sharp grain. The LoRA was trained on FL2VA, but
`transformer_ref` has the same target module layout, so this integration can
attach it to Ref2VA as an **experimental** acceleration path. Ref2VA fidelity is
not guaranteed: the LoRA author still describes native Ref2VA training as
planned, and community tests report identity loss at very low step counts.

## Usage

```bash
# CLI: Turbo is on by default and uses 5 grid points (4 evals)
python generate.py 'prompt...'
python generate.py 'prompt...' --turbo --turbo-strength 0.9   # against grain
python generate.py 'Use <Picture 1> as the product...' \
  --ref image:product.png  # experimental Ref2VA
python generate.py 'prompt...' --no-turbo  # full-quality sampler
# Web UI and REST API also default Turbo on; uncheck / send turbo=false to opt out.
```

Strength is the sharpness/artifact dial: **up** (1.05–1.2) against blurry
ghosting/smear, **down** (0.8–0.95) against over-sharp grain. More steps than 4
remain valid and help a little.

## Why no custom sampler is needed (unlike ComfyUI)

H3 denoises video and audio latents on two different flow schedules (video
shift 12, audio shift 3). ComfyUI's stock samplers step both streams on one
schedule — tolerable at ~20 steps, but at 4 steps the audio is over-stepped and
comes out distorted, hence the companion "MiniMax-H3 Turbo Sampler" node.

The diffusers integration never had this problem: `MiniMaxH3Blocks` steps each
stream on its own `MiniMaxH3Scheduler`. Both schedulers build their sigma grid
as `linspace(1, 0, num_inference_steps)` pushed through their own shift — which
is *exactly* the grid the LoRA's reference sampler uses (`shift_sigma(1 - i/n)`
per stream, cf. `generate.py` in the LoRA repo). So the only change needed is
`num_inference_steps=5`: 5 sigma grid points = 4 model evaluations = the
LoRA's "4 steps" (`turbo.TURBO_NUM_INFERENCE_STEPS`).

## Key remapping (`turbo.convert_turbo_lora_to_diffusers`)

The LoRA ships in the ComfyUI checkpoint naming (`base_model:
Comfy-Org/MiniMax-H3`, 518 tensors: 50 blocks × {qkv, out, fc1, fc2, adaln} +
2 refiner blocks × {qkv, out, fc1, fc2} + final adaln). Our transformer is the
diffusers conversion, so keys are remapped before loading:

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
layouts and the refiner matched directly; fc1 matched only half-swapped.

## Application details

- Loaded as a **PEFT adapter** (`transformer.load_lora_adapter`), i.e. applied
  at run time in activation space (`y = base(x) + B(A(x))`). This matches the
  author's guidance: merging into bf16 base weights would round most of the
  small update away. Alpha = rank in this LoRA, and diffusers defaults alpha to
  rank when the state dict carries no alpha keys, so the net scale is 1.0.
- Strength maps to `set_adapters(weights=[s])`; the web UI toggles the resident
  adapter with `enable_adapters()`/`disable_adapters()` between requests, so
  turbo and full-quality runs can be mixed without reloading anything.
- Weights file: `minimax_h3_turbo_4step_ema_ckpt850.safetensors` (744MB bf16),
  cached under `$HF_HOME` on first use.
