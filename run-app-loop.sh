#!/bin/bash
# Supervisor for the H3 generation API (added 21 Aug 2026 after a host-OOM
# kill left the queue dead for hours). Respawns app.py whenever it exits;
# remove by killing this script and app.py, or replace both with:
#   sudo systemctl enable --now minimax-h3-api
cd /home/jason/src/minimaxh3
export HF_HOME=/home/jason/.cache/hf MINIMAX_H3_HOST=127.0.0.1 MINIMAX_H3_PORT=7860
export MALLOC_ARENA_MAX=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while true; do
  echo "=== $(date -Is) supervisor: starting app.py" >> logs/app-manual.log
  .venv/bin/python app.py >> logs/app-manual.log 2>&1
  echo "=== $(date -Is) supervisor: app.py exited ($?), respawning in 10s" >> logs/app-manual.log
  sleep 10
done
