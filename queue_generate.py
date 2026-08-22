#!/usr/bin/env python
"""Submit a priority MiniMax-H3 job to the local service and download it."""

import argparse
import json
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path


def _json_request(url: str, payload: dict | None = None) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise SystemExit(f"Server returned HTTP {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise SystemExit(f"Cannot reach MiniMax-H3 service at {url}: {error.reason}") from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Queue a MiniMax-H3 generation ahead of normal backlog jobs."
    )
    parser.add_argument("prompt")
    parser.add_argument("--server", default="http://127.0.0.1:7860")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=544)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--num-frames", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--turbo", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--turbo-strength", type=float, default=0.85)
    parser.add_argument("--audio", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--priority",
        type=int,
        default=0,
        choices=range(0, 101),
        metavar="0-100",
        help="Lower runs first; CLI defaults to urgent priority 0.",
    )
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_url = args.server.rstrip("/")
    payload = {
        "prompt": args.prompt,
        "width": args.width,
        "height": args.height,
        "seconds": args.seconds,
        "seed": args.seed,
        "turbo": args.turbo,
        "turbo_strength": args.turbo_strength,
        "include_audio": args.audio,
        "priority": args.priority,
    }
    if args.steps is not None:
        payload["steps"] = args.steps
    if args.num_frames is not None:
        payload["num_frames"] = args.num_frames

    submitted = _json_request(f"{base_url}/api/generate", payload)
    job_id = submitted["job_id"]
    status_url = f"{base_url}/api/jobs/{job_id}"
    print(f"Queued {job_id} at priority {submitted['priority']}.")

    last_status = None
    while True:
        job = _json_request(status_url)
        status = job["status"]
        if status != last_status or status == "queued":
            queue_text = (
                f" (position {job['queue_position'] + 1})"
                if status == "queued" and "queue_position" in job
                else ""
            )
            print(f"{status}{queue_text}", flush=True)
            last_status = status
        if status == "done":
            break
        if status == "error":
            raise SystemExit(job.get("error", "Generation failed without an error message."))
        time.sleep(args.poll_seconds)

    output = args.output or Path("outputs") / f"queued_{job_id}.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with (
            urllib.request.urlopen(f"{base_url}{job['video_url']}", timeout=60) as response,
            output.open("wb") as destination,
        ):
            shutil.copyfileobj(response, destination)
    except urllib.error.URLError as error:
        raise SystemExit(f"Could not download completed video: {error.reason}") from error
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
