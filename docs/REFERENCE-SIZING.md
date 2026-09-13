# Experimental Ref2VA image sizing

Production behavior is unchanged. An omitted `reference_policy` retains the
upstream 2048-pixel short-edge preprocessing, including upscaling. Candidate
requests require `MINIMAX_H3_REFERENCE_EXPERIMENTS=1` on an isolated benchmark
renderer. Other servers reject them rather than silently ignoring the setting.

- `bounded-1024-v1`: only downscale image references whose short edge exceeds 1024.
- `match-output-v1`: only downscale image references whose area exceeds the output canvas.

Both preserve content without cropping and floor dimensions to the model's
32-pixel grid; rounding can slightly alter aspect ratio. Images smaller than one
grid cell are rejected rather than upscaled. Reference order, video/audio
references, first/last-frame geometry, sampler, adapter and output settings are
unchanged. Original image objects/files are not modified.

The hook runs after upstream Ref2VA setup and before text/vision/VAE encoding.
It replaces normalized image references using their original decoded pixels,
not the already-upscaled intermediate. This deliberately keeps upstream request,
canvas, modality and duration validation, at the cost of a redundant CPU-only
legacy resize. Neither Qwen nor the reference VAE encodes the legacy-size image
for a candidate request. No global pipeline configuration is mutated.

The versioned policy is stored in each job's request/params and idempotency
fingerprint. Existing legacy fingerprints remain unchanged. Per-job prefetched
conditioning carries its own prepared reference state; no cross-job image cache
is introduced. Benchmark sidecars record the effective dimensions without URLs.

Compare policy variants in the same process/model state with fixed inputs and
seeds. Smaller references can lose identity or fine text: do not promote these
policies without full video/audio comparison and an explicit release decision.
