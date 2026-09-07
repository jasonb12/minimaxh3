# Gate resident pipeline

`MINIMAX_H3_PROFILE=resident-fp8` selects a resident pipeline on Gate's
discrete Blackwell GPU. Spark's automatic SM121 selection remains unchanged.

The active H3 transformer uses tensorwise FP8 weights/activations and the
existing Turbo adapter layout. Gate reuses the original cached Qwen checkpoint
with INT8 weight-only quantization, keeping its vision tower and excluded layers
at their existing precision. Only 51 decoder layers are loaded; H3 consumes
the pre-normalization activation entering layer 50. The unused vocabulary head
is removed. A forward adapter captures that activation without collecting all
51 hidden states. A small real-Qwen test checks exact equality with the original
hidden-state collection, repeated calls, and hook cleanup.

The active transformer, conditioner, video VAE and audio VAE remain on CUDA.
No automatic CPU offload hooks or exclusive-denoiser eviction guard are installed.
Input/output tensors and prefetched conditioning may still cross PCIe; the model
weights do not swap per stage. FL2VA and Ref2VA use different transformer
partitions, so changing modes still unloads/reloads the active pipeline.

The initial Ref2VA model footprint is 66.9 GiB. Model residency alone is not a
maximum-workload guarantee: activation memory depends on reference images,
sequence length, resolution, and duration. Keep the 450 W cap fixed during
comparisons. First renders include shape-specific compilation; compare warmed
renders with identical requests and seeds, and inspect quality because FP8 is
approximate.

## Containers

Build using the repository Dockerfile and pinned AMD64 dependencies. Gate's
Compose file in `../brightify/deploy/h3/compose.yaml` accepts `H3_IMAGE_TAG`
independently of the worker image, so changing the model container preserves the
deployed heartbeat fix.

The existing `/cache/hf` mount holds source weights and derived FP8 transformer
caches. `/cache/runtime` holds compilation caches. Never copy Spark's ARM64
compiler caches into Gate's AMD64 runtime. `HF_OFFLINE=1` can be used once all
checkpoint metadata is cached. Tests run inside the image:

```bash
docker run --rm --gpus all minimax-h3:resident-fp8-20260907 \
  python -m unittest discover -s tests -v
```

Drain the Compose worker and verify the local API queue is idle before replacing
H3. Set `H3_PROFILE=resident-fp8` and `H3_IMAGE_TAG=resident-fp8-20260907` in the
Gate deployment `.env`, then recreate H3 and warm it before restarting the worker.
Rollback uses `H3_PROFILE=default` and `H3_IMAGE_TAG=docker-20260906`, with the same
drain/idle check. The previous image and source checkpoints are retained.

Each resident render logs conditioning time, denoise/decode time, peak allocated
and reserved VRAM, and parameter devices for all four model components.

## Gate validation, 2026-09-07

At 450 W, four-image Ref2VA with 544×960 generation, 243 frames, Turbo seven
scheduler points and 1080×1920 output passed full MP4 decoding and sampled-frame
inspection. The first run took 506.4 seconds including compilation; the warm
repeat took 403.0 seconds (13.56 seconds conditioning, 389.43 denoise/decode).
Warm peak allocated VRAM was 86.53 GiB; peak reserved was 87.21 GiB. All model
parameter devices remained CUDA. Host system RAM use was approximately 13 GiB.
Sampled frames were recognizable without obvious corruption, but scene choices
differed from the BF16 reference. These are operational and visual sanity checks,
not a claim of perceptual equivalence or an exhaustive quality evaluation.

The earlier 450 W production jobs took approximately 482–487 seconds, with
different prompts/seeds; that comparison is indicative, not a controlled speedup.
Validation artifacts are in `outputs/resident-validation/` and `outputs/api/`.

A subsequent FL2VA first-frame render at 512×288 / 124 frames completed in
71.7 seconds including shape compilation, with 72.56 GiB peak allocated VRAM.
Its H.264 video and AAC audio both passed a complete decode. This also verified
releasing Ref2VA before loading the other transformer partition.

Production Compose now explicitly sets `TORCHINDUCTOR_CACHE_DIR` and
`TRITON_CACHE_DIR` under `/cache/runtime`; previously TorchInductor used the
container's temporary directory despite the runtime-cache mount. Existing
compiled kernels from acceptance were copied into that persistent cache.

## Switching render modes

Resident profiles retain the conditioner, video/audio VAEs, tokenizer, processor,
and both schedulers when switching FL2VA ↔ Ref2VA. Only the transformer is
reloaded. The old pipeline and its component manager are collected before the
replacement loads, avoiding simultaneous transformer allocations and avoiding
a CUDA-to-CPU weight copy. Same-mode jobs continue using the same pipeline.

A failed replacement retains shared components for the next request. Initial
memory preflight runs only on a cold load; it must not reject the shared models
that intentionally occupy GPU memory during a switch. Nonresident profiles
retain their existing full-reload behavior. Turbo adapters and compilation are
prepared on each new transformer as before. This does not retain both compiled
transformers or eliminate transformer loading/compilation on a mode switch.

`[pipeline]` logs report task, whether shared components were reused, and load
time separately from generation time. `docker/reuse_smoke.py` is an opt-in
full-weight acceptance check for FL2VA → Ref2VA → FL2VA. Run it in the candidate
container with exclusive GPU access after draining the worker. It asserts
shared-object identity, unchanged CUDA storage pointers, collection of the old
transformer, and renders after each switch. `H3_REUSE_LOAD_ONLY=1` checks the
load lifecycle without rendering. Never run a second full pipeline alongside
the production runner on the same GPU.
