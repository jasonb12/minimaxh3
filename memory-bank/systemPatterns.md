# System Patterns

## Architecture

Single-script CLI (`generate.py`) + Gradio UI (`app.py`) around diffusers Modular Pipelines:

1. **FL2VA** (`task="fl2va"`): `ModularPipeline.from_pretrained("MiniMaxAI/MiniMax-H3", …)`
   resolves `MiniMaxH3Blocks` from `modular_model_index.json` and loads `transformer/`.
2. **Ref2VA** (`task="ref2va"`): `MiniMaxH3Ref2VABlocks().init_pipeline(...)` resolves
   `MiniMaxH3Ref2VAModularPipeline` and loads `transformer_ref/` from the same repo.
   References are `MiniMaxH3Reference(image|video|audio=…)` in request order.
3. Text encoder is replaced pre-load with an int8 torchao-quantized `Qwen3VLForConditionalGeneration`
   (official recipe, `modules_to_not_convert` preserves vision tower, embeddings, norm, lm_head).
4. `load_components(dtype=torch.bfloat16)` fetches exactly the subfolders the blockset needs —
   never the other transformer partition, and never the original `FL2VA/`/`Ref2VA/` folders.
5. `ComponentsManager.enable_auto_cpu_offload(memory_reserve_margin="12GB")` swaps the 61.7GB
   transformer and ~32GB conditioner between GPU and host RAM per stage. The
   server keeps the loaded pipeline resident after each job. Discarding after
   every REST job dropped the Python handle without returning ~89GB of VRAM,
   so the next load's 70GB free-VRAM check failed against this process. Mode
   switches still detach hooks without CPU-offloading the old partition. The
   free-VRAM preflight counts only other processes. Each denoiser forward
   offloads sibling components first so the conditioner cannot remain on CUDA
   beside the 61.7GB transformer.
6. Output state carries `videos`, `audio`, `sampling_rate`; `encode_video` writes
   the MP4 via PyAV. REST callers may set `include_audio=false` to omit the audio
   stream without changing H3's joint inference.
7. REST `reference_image_urls` select Ref2VA and are decoded into
   `MiniMaxH3Reference` objects. They are mutually exclusive with first/last
   frames and are never echoed in job-status payloads. The FL2VA-trained Turbo
   adapter can be loaded onto the structurally identical `transformer_ref`
   modules as an experimental path; preserve the identity-fidelity warning.
   Default Turbo file is v4-600 EMA at 7 grid points (6 evals), strength 1.0.
8. REST output normalization is post-generation and deterministic: bilinear
   resize in bounded CPU chunks, then final-frame padding or truncation to the
   requested 24fps duration before PyAV encoding.
9. REST scheduling uses `PriorityQueue` entries `(priority, sequence, job_id)`.
   Lower values run first; sequence preserves FIFO ties. Normal API jobs default
   to 100 and local CLI tests use 0. Priority never preempts a running render.
10. Two-stage execution: blocks `setup` + `text_encoder` (conditioner) run
    separately from `vae/reference_encoder → … → decode` (transformer) on one
    `PipelineState`, via `pipe._blocks.sub_blocks[name](pipe, state)`. The
    worker pre-encodes the next few same-task queued jobs while the conditioner
    is resident (`job["conditioning"]`, embeds parked on CPU) so they skip the
    62GB transformer eviction/reload. Prefetch candidates are selected lazily
    inside `_gen_lock` after the running job's own encode.

## Key decisions

- **diffusers over SGLang/ComfyUI**: SGLang recipe assumes 4 GPUs; diffusers has official
  single-GPU offload recipes and scoped downloads.
- **int8 text encoder, bf16 transformer**: both in bf16 (~124GB) exceeds the 125GB host RAM
  during load; int8 conditioner (~32GB) keeps peak at ~99GB with minimal quality impact.
  `--bf16-text-encoder` flag exists for machines with more RAM.
- **No `_flash_3_hub` attention backend**: FA3 is Hopper-only; this GPU is Blackwell (sm_120).
- **One resident task at a time**: Gradio caches a single pipe; switching FL2VA ↔ Ref2VA
  deletes the old pipe and reloads (host RAM cannot hold both transformers + conditioner).
