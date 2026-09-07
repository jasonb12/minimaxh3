# Docker runtime

`Dockerfile` supports native AMD64 (Gate) and ARM64 (DGX Spark) builds. It uses
CUDA 13.0.1 / Ubuntu 24.04, Python 3.12, pinned base image digests and separate
`docker/requirements-{amd64,arm64}.lock` snapshots of the verified host environments.
The Python development headers and C++ compiler support Triton/torch.compile.

```bash
docker build -t minimax-h3:docker-20260906 .
```

Production orchestration lives in the sibling Brightify repository:
`../brightify/deploy/h3/compose.yaml`. See
`../brightify/docs/H3-CONTAINER-ROLLOUT.md` for build, test, cutover, rollback and
actual deployment results. Each host builds its own native image; no emulation
or registry publication is required.

The image contains code and dependencies, not model weights or credentials.
Compose mounts the existing HF cache at `/cache/hf`, a persistent compiler cache
at `/cache/runtime`, and host outputs at `/app/outputs`. The API binds to port7860
inside the container. Host publishing and hardware profiles are configured by
Compose. Gate selects `resident-fp8` (FP8 transformer, pruned INT8 conditioner);
Spark selects `spark` FP8. The `default` BF16/INT8 offload path remains available
for rollback. See [RESIDENT.md](RESIDENT.md) for Gate validation and cutover.

Health checks verify API availability; models load on the first request. The
expanded `docker/smoke.py` verifies CUDA, model imports and actual Triton kernel
compilation without loading the full model. Full render acceptance remains
necessary to validate the complete inference pipeline.
