"""Download only the Spark runtime's weights; existing HF cache files are reused."""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "hf"))

from huggingface_hub import hf_hub_download, snapshot_download
from spark import ENCODER_REVISION, MODEL_REVISION
from turbo import TURBO_FILE, TURBO_REPO


def download_h3(include_ref):
    snapshot_download(
        "MiniMaxAI/MiniMax-H3", revision=MODEL_REVISION, max_workers=4,
        allow_patterns=[
            "modular_model_index.json", "transformer/*", "tokenizer/*", "processor/*",
            "vae/*", "audio_vae/*", "scheduler/*", "audio_scheduler/*",
        ],
    )
    hf_hub_download(TURBO_REPO, TURBO_FILE)
    print("FL2VA and Turbo downloaded", flush=True)
    if include_ref:
        snapshot_download(
            "MiniMaxAI/MiniMax-H3", revision=MODEL_REVISION,
            allow_patterns=["transformer_ref/*"], max_workers=4,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-ref", action="store_true", help="Skip the optional 66 GB Ref2VA partition")
    args = parser.parse_args()
    with ThreadPoolExecutor(max_workers=2) as pool:
        h3 = pool.submit(download_h3, not args.no_ref)
        encoder = pool.submit(
            snapshot_download, "Qwen/Qwen3-VL-32B-Instruct-FP8",
            revision=ENCODER_REVISION, max_workers=4,
        )
        h3.result()
        encoder.result()
    # Cache the ARM64 FP8 kernel package before the service goes offline.
    from transformers.integrations.finegrained_fp8 import load_finegrained_fp8_kernel

    # The offline resolver first requests a complete snapshot, then a build.
    # Cache the small source repo through v4 to also populate its version ref.
    snapshot_download("kernels-community/finegrained-fp8", repo_type="kernel", revision="v4")
    load_finegrained_fp8_kernel()
    print("Spark weights ready", flush=True)
