"""MiniMax-H3 Turbo LoRA support: 4-step sampling instead of ~20 (~5x faster).

Community distillation LoRA by larryvrh (Apache-2.0):
https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora

The LoRA is published against the ComfyUI checkpoint layout, so its keys are
remapped to the diffusers `MiniMaxH3Transformer3DModel` module names before
loading as a PEFT adapter. The adapter applies at run time in activation space
(y = base(x) + B(A(x))), which the author recommends over merging: folded into
bf16 base weights, most of the small update would round away.

No sampler changes are needed in diffusers: the ComfyUI "Turbo Sampler" exists
because ComfyUI steps video and audio on a single flow schedule, which
over-steps the audio at 4 steps. The diffusers pipeline natively steps each
stream on its own shifted scheduler (video shift 12, audio shift 3) over the
same uniform grid, which is exactly the dual-clock integration the LoRA's own
reference sampler implements. `num_inference_steps=5` gives 5 sigma grid points
= 4 model evaluations = the LoRA's "4 steps".

Key remapping (verified empirically against both checkpoints, see docs/TURBO.md):
  blocks.N.*                     -> transformer_blocks.N.*
  token_refiner.blocks.N.*       -> token_refiner.refiner_blocks.N.*
  attn.qkv_proj                  -> attn.to_q / to_k / to_v  (A shared, B split in thirds)
  attn.out_proj                  -> attn.to_out.0
  mlp.fc1                        -> ff.net.0.proj  (B row halves SWAPPED: ComfyUI stores
                                    [gate; value], diffusers stores [value; gate])
  mlp.fc2                        -> ff.net.2
  final_layer.adaln_proj.linear  -> norm_out.linear
  blocks.N.adaln_proj.linear     -> transformer_blocks.N.adaln_proj.linear (unchanged)
"""

import os
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "hf"))

TURBO_REPO = "larryvrh/MiniMax-H3-Turbo-Lora"
TURBO_FILE = "minimax_h3_turbo_4step_ema_ckpt850.safetensors"
ADAPTER_NAME = "turbo"

# 5 sigma grid points -> 4 model evaluations, the schedule the LoRA was trained for.
TURBO_NUM_INFERENCE_STEPS = 5


def convert_turbo_lora_to_diffusers(state_dict):
    import torch

    out = {}
    for key, value in state_dict.items():
        name, suffix = key.rsplit(".lora_", 1)  # suffix: "A.weight" | "B.weight"
        is_b = suffix.startswith("B")

        name = name.replace("token_refiner.blocks.", "token_refiner.refiner_blocks.")
        if name.startswith("blocks."):
            name = "transformer_blocks." + name[len("blocks."):]
        if name == "final_layer.adaln_proj.linear":
            name = "norm_out.linear"

        if name.endswith("attn.qkv_proj"):
            stem = name[: -len("qkv_proj")]
            if is_b:
                third = value.shape[0] // 3
                for i, part in enumerate(("to_q", "to_k", "to_v")):
                    out[f"{stem}{part}.lora_B.weight"] = value[i * third : (i + 1) * third]
            else:
                for part in ("to_q", "to_k", "to_v"):
                    out[f"{stem}{part}.lora_A.weight"] = value
            continue

        name = name.replace("attn.out_proj", "attn.to_out.0")
        if name.endswith("mlp.fc1"):
            name = name.replace("mlp.fc1", "ff.net.0.proj")
            if is_b:
                half = value.shape[0] // 2
                value = torch.cat([value[half:], value[:half]], dim=0)
        name = name.replace("mlp.fc2", "ff.net.2")

        out[f"{name}.lora_{suffix}"] = value
    return out


def load_turbo_lora(transformer, strength: float = 1.0):
    """Download (cached), convert and attach the Turbo LoRA to the transformer."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    if ADAPTER_NAME in getattr(transformer, "peft_config", {}):
        set_turbo_strength(transformer, strength)
        return

    path = hf_hub_download(TURBO_REPO, TURBO_FILE)
    state_dict = convert_turbo_lora_to_diffusers(load_file(path))
    transformer.load_lora_adapter(state_dict, prefix=None, adapter_name=ADAPTER_NAME)
    set_turbo_strength(transformer, strength)


def set_turbo_strength(transformer, strength: float):
    transformer.set_adapters([ADAPTER_NAME], weights=[float(strength)])


def set_turbo_enabled(transformer, enabled: bool):
    """Toggle the adapter without unloading it (no-op if it was never loaded)."""
    if ADAPTER_NAME not in getattr(transformer, "peft_config", {}):
        return
    if enabled:
        transformer.enable_adapters()
    else:
        transformer.disable_adapters()
