# System Patterns

## Architecture

Single-script CLI (`generate.py`) around diffusers Modular Pipelines:

1. `ModularPipeline.from_pretrained("MiniMaxAI/MiniMax-H3", components_manager=manager)`
   resolves the t2va/fl2va blockset (`MiniMaxH3Blocks`) from `modular_model_index.json`.
2. Text encoder is replaced pre-load with an int8 torchao-quantized `Qwen3VLForConditionalGeneration`
   (official recipe, `modules_to_not_convert` preserves vision tower, embeddings, norm, lm_head).
3. `load_components(dtype=torch.bfloat16)` fetches exactly the subfolders the blockset needs —
   never `transformer_ref/` or the original `FL2VA/`/`Ref2VA/` checkpoints.
4. `ComponentsManager.enable_auto_cpu_offload(memory_reserve_margin="12GB")` swaps the 61.7GB
   transformer and ~32GB conditioner between GPU and host RAM per stage.
5. Output state carries `videos`, `audio`, `sampling_rate`; `encode_video` muxes to mp4 via PyAV.

## Key decisions

- **diffusers over SGLang/ComfyUI**: SGLang recipe assumes 4 GPUs; diffusers has official
  single-GPU offload recipes and scoped downloads.
- **int8 text encoder, bf16 transformer**: both in bf16 (~124GB) exceeds the 125GB host RAM
  during load; int8 conditioner (~32GB) keeps peak at ~99GB with minimal quality impact.
  `--bf16-text-encoder` flag exists for machines with more RAM.
- **No `_flash_3_hub` attention backend**: FA3 is Hopper-only; this GPU is Blackwell (sm_120).
- **ref2va path documented but not wired**: use `MiniMaxH3Ref2VABlocks().init_pipeline(...)` and
  download `transformer_ref/*` if needed later.
