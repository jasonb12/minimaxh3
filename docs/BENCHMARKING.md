# Opt-in render phase timing

The speedup plan begins with measurement, not a model/precision change.
`MINIMAX_H3_BENCHMARK=1` enables per-generation timing sidecars beside output MP4s
as `<job-id>.benchmark.json`. It is disabled by default. Use only on a reserved,
idle benchmark host after draining the Brightify worker; do not run a second
model server alongside production inference.

The sidecar records monotonic total time, model load/switch, compilation setup,
individual Diffusers blocks (including `text_encoder`, `denoise`, and `decode`),
output normalization and media encoding. Lazy compilation remains part of the
first block invocation; `compile_setup` alone is not the total compile cost.
CUDA synchronization brackets GPU phases only when benchmark mode is enabled.
These boundaries perturb execution, so compare instrumented and uninstrumented
runs before claiming a speed improvement. Nested/prefetched phases must not be
summed as though they were non-overlapping exclusive durations.

Only an allowlisted subset of the effective recipe is recorded; no prompts,
reference URLs, tokens or conditioning tensors are written. Sidecars are mode
0600. Error records contain the exception class, not its potentially private
message. Normal return values and exceptions are preserved. Sidecar I/O failure
does not turn a successful render into a failed business job.

This first instrumentation pass does not yet split actual CUDA kernel time from
CPU dispatch, count internal denoiser forward calls, or time remote input
downloads. Those remain Gate 1 work. No adapter, reference sizing, sampler,
precision, resolution, duration or audio default changes are included.
