# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:0.10.4@sha256:4cac394b6b72846f8a85a7a0e577c6d61d4e17fe2ccee65d9451a8b3c9efb4ac AS uv
FROM nvidia/cuda:13.0.1-devel-ubuntu24.04@sha256:7d2f6a8c2071d911524f95061a0db363e24d27aa51ec831fcccf9e76eb72bc92
ARG TARGETARCH
ARG SOURCE_REVISION=unknown
LABEL org.opencontainers.image.revision=$SOURCE_REVISION
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv ffmpeg libgl1 libglib2.0-0 ca-certificates g++ && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /app
COPY docker/requirements-*.lock /tmp/locks/
RUN uv venv --python /usr/bin/python3 /opt/venv && uv pip sync --python /opt/venv/bin/python --index https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match /tmp/locks/requirements-${TARGETARCH}.lock
ENV PATH=/opt/venv/bin:$PATH HF_HOME=/cache/hf XDG_CACHE_HOME=/cache/runtime HOME=/tmp PYTHONUNBUFFERED=1 MINIMAX_H3_HOST=0.0.0.0 MINIMAX_H3_PORT=7860
COPY app.py generate.py spark.py turbo.py queue_generate.py ./
COPY docker/smoke.py /app/smoke.py
RUN python -m py_compile app.py generate.py spark.py turbo.py
RUN apt-get update && apt-get install -y --no-install-recommends python3-dev && rm -rf /var/lib/apt/lists/*
USER 1000:1000
EXPOSE 7860
STOPSIGNAL SIGINT
CMD ["python", "app.py"]
