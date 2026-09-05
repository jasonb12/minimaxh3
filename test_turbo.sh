#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    printf 'Usage: %s '\''<prompt>'\''\n' "$0" >&2
    exit 2
fi

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

exec "$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/queue_generate.py" "$1" \
    --turbo \
    --steps '7' \
    --width '768' \
    --height '448' \
    --num-frames '300' \
    --seed '42' \
    --priority '0' \
    --output "$ROOT_DIR/outputs/test_turbo.mp4"
