# DGX Spark setup

This checkout runs on `sparky`: ARM64, NVIDIA GB10 (SM121), 128 GB unified
memory, driver 580.173.02. CPU RAM and GPU allocations share one physical pool.

## Install and start

```bash
uv venv --python '3.12'
uv pip install -r 'requirements-spark.txt' --torch-backend=cu130
.venv/bin/python download_spark.py
mkdir -p ~/.config/systemd/user
cp minimax-h3-spark.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now minimax-h3-spark.service
loginctl enable-linger "$USER"
```

Open <http://192.168.1.28:7860> on this LAN (or <http://localhost:7860> on Spark).
API documentation is at `/api/docs`. The service listens on the LAN without
authentication, as the existing application does. Linger starts the user
service at boot and keeps it running after logout.

```bash
systemctl --user status minimax-h3-spark
journalctl --user -u minimax-h3-spark -f
systemctl --user stop minimax-h3-spark
systemctl --user start minimax-h3-spark
```

One generation runs at a time. Avoid restarting during a render. Outputs are
in `outputs/api/` and `outputs/gradio/`; the job registry is in memory and resets
on service restart.

## Runtime choices

- Native ARM64 PyTorch 2.13 / CUDA 13 wheels support this installed driver and
  SM121. No emulation or additional model-serving container is needed.
- The H3 transformer uses TorchAO tensorwise FP8 weights and activations. The
  model's prescribed FP32 input/output and timestep layers stay FP32. The full
  module layout is preserved so the existing Turbo LoRA can attach unchanged.
- Qwen's prequantized FP8 conditioner avoids downloading/loading a second
  66 GB BF16 model. Its vision tower and other excluded layers retain their
  published precision. Only 51 decoder layers are loaded: H3 reads the
  **unnormalized `hidden_states[50]`**, so layer 51 preserves that convention;
  later layers and the vocabulary head are unused.
- Only the activation entering decoder layer 50 is retained during conditioning;
  the other hidden states are no longer collected. The selected tensor is
  verified against unmodified Qwen in `tests/test_conditioner.py`.
- Both models and the tiled VAEs stay on CUDA. CPU offload would move tensors
  within the same shared memory pool, adding copies without increasing capacity.
  The runtime advises Linux to release its own checkpoint file pages between
  loads: CUDA on GB10 cannot reclaim that page cache itself. This uses
  unprivileged `posix_fadvise`, without clearing other applications' file caches.
- `torch.compile` compiles repeated transformer blocks after Turbo is attached.
  The first render compiles kernels; subsequent renders reuse them. New shapes
  and toggling Turbo may compile another graph. Set `MINIMAX_H3_COMPILE=0` for
  eager debugging.
- Turbo v4 remains the default: seven scheduler points / six model evaluations.
  Full sampling remains available by disabling Turbo. Ref2VA Turbo retains its
  existing experimental identity-fidelity caveat.
- The service defaults to 832×480 and about 5.2 seconds, with 10 CPU threads and
  four compiler workers. Explicit request dimensions override the API default.

The profile selects automatically on SM121. `MINIMAX_H3_PROFILE=default` keeps
the original discrete-GPU implementation; `spark` explicitly selects this one.

## Download and cache

`HF_HOME=~/.cache/hf`. Initial downloads include the original diffusers
transformer partitions (about 66 GB each), shared VAEs (11 GB), Qwen FP8 (36 GB),
and Turbo (0.8 GB). `download_spark.py --no-ref` skips the Ref2VA partition.
Downloads reuse complete cached files.
The service sets `HF_HUB_OFFLINE=1` after this download step, so normal starts
and generation use local checkpoints without Hub network checks.

On first load of each mode, the runtime writes a derived FP8 transformer cache
under `~/.cache/hf/derived/h3-fp8/`. Later starts load that cache directly and
skip quantization. The cache key includes the model revision, Torch/TorchAO
versions, and quantization format. A `READY` marker is written only after the
entire checkpoint is saved.

Model revisions are pinned in `spark.py`; dependency pins remain in the two
requirements files. `uv pip check` can report that NVIDIA's cuSPARSELt wheel
uses the vendor `manylinux2014_sbsa` tag. Its actual library was verified as
ARM aarch64; this is not an x86 wheel.

## Measurements and alternatives

On this Spark, a representative 8192×5376 by 5376×14336 linear operation took
12.6 ms in BF16, 22.7 ms with compiled rowwise FP8, and **7.3 ms with compiled
tensorwise FP8**. These are kernel measurements, not end-to-end video timings.
The smaller FP8 model also leaves more activation headroom in unified memory.

Validated on 2026-09-05: resident models used **66.9 GiB**. A Turbo text-to-video
request at 832×480, 124 frames, seven scheduler points took **93.5 seconds**
with warmed models/kernels (55 seconds denoising). The first request took
4.4 minutes of generation including compilation, in addition to downloading,
loading, and initially quantizing the weights. Reloading the saved transformer
took about 3.2 minutes, plus conditioner/decoder loading.

Both MP4s passed a complete FFmpeg decode: H.264, 24 fps, 124 frames, AAC stereo
at 32 kHz. The inspected frame matched the requested red panda/workbench scene.
This validates operation, not a perceptual comparison with BF16.

Ref2VA with one image also passed the full decode and frame inspection at the
same output dimensions. Its first generation took 6.1 minutes including
compilation, excluding initial loading/quantization. Switching between FL2VA
and Ref2VA released the old partition before loading the next.

First-frame conditioning also passed full video/audio decoding and frame
inspection after switching back to FL2VA. This request took 3.2 minutes
including compilation for the new input shape. The final service remains
running with FL2VA loaded. These mode-switch checks ran with Hub offline mode.

The downloader caches the complete small `kernels-community/finegrained-fp8`
v4 source snapshot and its version reference. Caching only the CUDA build is
insufficient for the kernel library's offline snapshot lookup.

The CUDA integration check is `.venv/bin/python -m unittest discover -s tests -v`.
It checks FP8 serialization, compiled video/audio outputs, and adapter toggling
on a small H3 model without downloading the full checkpoint.

For the containerized deployment, run the same tests inside the native ARM64
image. Compose explicitly points TorchInductor and Triton at `/cache/runtime`,
so the mounted compiler cache survives container recreation. Keep
`H3_PROFILE=spark`: this retains the prequantized FP8 conditioner and the
shared-memory preflight. Gate's `resident-fp8` profile is a separate option.

[NVIDIA's Sol Engine GB10 recipe](https://github.com/NVlabs/Sana/tree/sol-engine/models/minimax_h3/GB10)
uses a pruned FP8 transformer, fused kernels, sparse attention, and cross-step
caching. Its rank-eight AdaLN layout is incompatible with this application's
existing Turbo AdaLN adapter weights. The current profile preserves Turbo,
first/last frames, Ref2VA, and the current API. This is a measured configuration
for this application, not a claim of the fastest possible H3 implementation.

FP8 is approximate. Validate generated content for your intended use; kernel
speed and finite-output checks alone do not establish perceptual equivalence.
