#!/usr/bin/env python
"""Web UI + REST API for MiniMax-H3 video+audio generation.

Serves on 0.0.0.0:7860, reachable from other machines on the network:
  /            Gradio UI (and its own machine API under /gradio_api)
  /api/*       job-based REST API (see docs/API.md):
                 POST /api/generate        -> {"job_id": ...}
                 GET  /api/jobs/{id}       -> status / info / error
                 GET  /api/jobs/{id}/video -> the mp4
                 GET  /api/jobs            -> recent jobs

The pipeline loads lazily on the first generation (~30s) and stays resident;
UI and API requests share one GPU lock, so they serialize. Switching between
FL2VA and Ref2VA reloads the transformer partition (~62GB) once.

Run:  .venv/bin/python app.py
"""

import base64
import gc
import io
import os
import queue
import threading
import time
import traceback
import uuid
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "hf"))

import gradio as gr
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from generate import build_pipeline, build_references

OUTPUT_DIR = Path(__file__).parent / "outputs" / "gradio"

# label -> (width, height); None means let the model pick its canvas
# (matches the first keyframe's aspect ratio, 16:9 otherwise, 768px short edge)
SIZES = {
    "960×544 landscape (fast)": (960, 544),
    "1344×768 landscape (native, ~2.3x slower)": (1344, 768),
    "544×960 portrait (fast)": (544, 960),
    "768×1344 portrait (native)": (768, 1344),
    "768×768 square": (768, 768),
    "Auto (model default for input image)": None,
}

MODE_FL2VA = "Text / first-last frame (FL2VA)"
MODE_REF2VA = "Reference images/video/audio (Ref2VA)"

_pipe = None
_pipe_task = None
_pipe_lock = threading.Lock()

# One generation at a time: the model saturates the GPU, and the turbo adapter
# toggle mutates shared transformer state. Held by both the UI and API paths.
_gen_lock = threading.Lock()


def _check_gpu_free() -> None:
    import torch

    free_bytes, _ = torch.cuda.mem_get_info()
    if free_bytes < 70 * 1024**3:
        raise gr.Error(
            "GPU does not have enough free VRAM (needs ~70GB). "
            "If the vLLM server is running, stop it first: "
            "systemctl --user stop vllm.service"
        )


def _get_pipe(task: str):
    """Return the pipeline for `task`, swapping transformers if the mode changed."""
    global _pipe, _pipe_task
    import torch

    with _pipe_lock:
        if _pipe is not None and _pipe_task != task:
            # Different transformer partition; drop the resident one so the
            # next load fits in host RAM / VRAM.
            del _pipe
            _pipe = None
            _pipe_task = None
            gc.collect()
            torch.cuda.empty_cache()
        if _pipe is None:
            _check_gpu_free()
            _pipe = build_pipeline(bf16_text_encoder=False, task=task)
            _pipe_task = task
    return _pipe


def _snap_frames(seconds: float) -> int:
    """Largest frame count of the form 17n+5 that fits in `seconds` (min 5.17s)."""
    n = (int(seconds * 24) - 5) // 17
    return 17 * max(n, 7) + 5


def _mode_visibility(mode):
    is_ref = mode == MODE_REF2VA
    return (
        gr.update(visible=not is_ref),  # fl2va_row
        gr.update(visible=is_ref),  # ref_images
        gr.update(visible=is_ref),  # ref_videos
        gr.update(visible=is_ref),  # ref_audios
        gr.update(visible=is_ref),  # ref_help
    )


def _run_generation(task, kwargs, turbo, turbo_strength, out_path):
    """Toggle turbo, run the pipeline, and encode the mp4.

    Shared by the Gradio UI and the REST API; serialized on _gen_lock because
    the turbo adapter toggle mutates shared transformer state and the model
    saturates the GPU anyway. Returns generation wall time in seconds.
    """
    from diffusers.utils.export_utils import encode_video

    with _gen_lock:
        pipe = _get_pipe(task)
        if task == "fl2va":
            from turbo import load_turbo_lora, set_turbo_enabled

            if turbo:
                load_turbo_lora(pipe.transformer, strength=float(turbo_strength))
            set_turbo_enabled(pipe.transformer, turbo)

        t0 = time.time()
        state = pipe(**kwargs)
        elapsed = time.time() - t0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    encode_video(
        state.get("videos")[0],
        fps=24,
        output_path=str(out_path),
        audio=state.get("audio")[0],
        audio_sample_rate=state.get("sampling_rate"),
    )
    return elapsed


def generate(
    mode,
    prompt,
    image,
    last_image,
    ref_images,
    ref_videos,
    ref_audios,
    size,
    seconds,
    steps,
    seed,
    turbo,
    turbo_strength,
    progress=gr.Progress(),
):
    import torch

    if not prompt or not prompt.strip():
        raise gr.Error("A prompt is required.")

    task = "ref2va" if mode == MODE_REF2VA else "fl2va"
    num_frames = _snap_frames(seconds)
    seed = int(seed)
    if seed < 0:
        seed = int(torch.seed() % 2**31)

    refs = []
    if task == "ref2va":
        # Order: gallery images (upload order), then videos, then audios.
        # Matches the usual "subject first, then motion/voice" pattern.
        for item in ref_images or []:
            # Gallery may yield a path str, (path, caption) tuple, or a dict.
            if isinstance(item, (list, tuple)):
                path = item[0]
            elif isinstance(item, dict):
                path = item.get("name") or item.get("path") or item.get("image")
            else:
                path = item
            if path:
                refs.append(("image", str(path)))
        for path in ref_videos or []:
            if path:
                refs.append(("video", str(path)))
        for path in ref_audios or []:
            if path:
                refs.append(("audio", str(path)))
        if not refs:
            raise gr.Error("Ref2VA needs at least one reference image, video, or audio.")
        if all(kind == "audio" for kind, _ in refs):
            raise gr.Error("Audio references cannot be the only inputs — add an image or video.")
        n_img = sum(1 for k, _ in refs if k == "image")
        n_vid = sum(1 for k, _ in refs if k == "video")
        n_aud = sum(1 for k, _ in refs if k == "audio")
        if n_img > 9 or n_vid > 3 or n_aud > 3 or len(refs) > 12:
            raise gr.Error("Limits: ≤9 images, ≤3 videos, ≤3 audios, ≤12 total.")

    if turbo and task == "ref2va":
        raise gr.Error("Turbo is trained against the FL2VA transformer; switch mode or disable Turbo.")

    kwargs = {
        "prompt": prompt.strip(),
        "num_frames": num_frames,
        "num_inference_steps": int(steps),
        "generator": torch.Generator().manual_seed(seed),
    }
    if SIZES[size] is not None:
        kwargs["width"], kwargs["height"] = SIZES[size]

    if task == "ref2va":
        kwargs["references"] = build_references(refs)
    else:
        if image is not None:
            kwargs["image"] = image
        if last_image is not None:
            kwargs["last_image"] = last_image

    progress(0.05, desc=f"Generating {num_frames / 24:.1f}s of video (takes minutes)")
    out_path = OUTPUT_DIR / f"h3_{task}_{time.strftime('%Y%m%d_%H%M%S')}_seed{seed}.mp4"
    elapsed = _run_generation(task, kwargs, turbo, turbo_strength, out_path)

    size_txt = "×".join(map(str, SIZES[size])) if SIZES[size] else "auto"
    ref_txt = f" · {len(refs)} refs" if task == "ref2va" else ""
    turbo_txt = f" · turbo@{float(turbo_strength):g}" if turbo and task == "fl2va" else ""
    info = (
        f"{task}{ref_txt}{turbo_txt} · seed {seed} · {num_frames} frames ({num_frames / 24:.1f}s) · "
        f"{size_txt} · {int(steps)} steps · generated in {elapsed / 60:.1f} min"
    )
    return str(out_path), info


# --------------------------------------------------------------------------
# REST API: POST /api/generate returns a job id; a single worker thread runs
# jobs one at a time (sharing _gen_lock with the UI). Jobs live in memory;
# finished videos persist under outputs/api/.
# --------------------------------------------------------------------------

API_OUTPUT_DIR = Path(__file__).parent / "outputs" / "api"
MAX_JOBS_KEPT = 200

_jobs: dict = {}
_jobs_lock = threading.Lock()
_job_queue: queue.Queue = queue.Queue()


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Prompt, ideally shot-by-shot with a soundscape")
    image_b64: str | None = Field(None, description="First frame, base64 (raw or data URL)")
    image_url: str | None = Field(None, description="First frame, fetched from URL")
    last_image_b64: str | None = None
    last_image_url: str | None = None
    width: int | None = Field(None, description="Canvas width, multiple of 32 (omit both for model default)")
    height: int | None = None
    seconds: float = Field(8.0, ge=5.2, le=14.4)
    steps: int | None = Field(None, ge=4, le=60, description="Defaults to 50, or 5 with turbo")
    seed: int = Field(-1, description="-1 = random")
    turbo: bool = False
    turbo_strength: float = Field(1.0, ge=0.5, le=1.5)


def _load_image(b64: str | None, url: str | None):
    from PIL import Image

    if b64:
        if b64.startswith("data:") and "," in b64:
            b64 = b64.split(",", 1)[1]
        return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    if url:
        import urllib.request

        with urllib.request.urlopen(url, timeout=60) as r:
            return Image.open(io.BytesIO(r.read())).convert("RGB")
    return None


def _job_public(job: dict) -> dict:
    out = {k: job[k] for k in ("job_id", "status", "created", "params") if k in job}
    for k in ("info", "error", "elapsed_seconds", "seed"):
        if job.get(k) is not None:
            out[k] = job[k]
    if job.get("status") == "done":
        out["video_url"] = f"/api/jobs/{job['job_id']}/video"
    if job.get("status") == "queued":
        with _jobs_lock:
            queued = [j for j in _jobs.values() if j["status"] == "queued"]
        queued.sort(key=lambda j: j["created"])
        out["queue_position"] = next(
            (i for i, j in enumerate(queued) if j["job_id"] == job["job_id"]), 0
        )
    return out


def _api_worker():
    import torch

    while True:
        job_id = _job_queue.get()
        with _jobs_lock:
            job = _jobs.get(job_id)
            if job is None or job["status"] != "queued":
                continue
            job["status"] = "running"
        try:
            req: GenerateRequest = job["request"]
            seed = req.seed if req.seed >= 0 else int(torch.seed() % 2**31)
            num_frames = _snap_frames(req.seconds)
            steps = req.steps if req.steps is not None else (5 if req.turbo else 50)

            kwargs = {
                "prompt": req.prompt.strip(),
                "num_frames": num_frames,
                "num_inference_steps": steps,
                "generator": torch.Generator().manual_seed(seed),
            }
            if req.width and req.height:
                kwargs["width"], kwargs["height"] = req.width, req.height
            image = _load_image(req.image_b64, req.image_url)
            if image is not None:
                kwargs["image"] = image
            last = _load_image(req.last_image_b64, req.last_image_url)
            if last is not None:
                kwargs["last_image"] = last

            out_path = API_OUTPUT_DIR / f"{job_id}.mp4"
            elapsed = _run_generation("fl2va", kwargs, req.turbo, req.turbo_strength, out_path)

            size_txt = f"{req.width}×{req.height}" if req.width and req.height else "auto"
            turbo_txt = f" · turbo@{req.turbo_strength:g}" if req.turbo else ""
            with _jobs_lock:
                job.update(
                    status="done",
                    seed=seed,
                    elapsed_seconds=round(elapsed, 1),
                    video_path=str(out_path),
                    info=(
                        f"fl2va{turbo_txt} · seed {seed} · {num_frames} frames "
                        f"({num_frames / 24:.1f}s) · {size_txt} · {steps} steps · "
                        f"generated in {elapsed / 60:.1f} min"
                    ),
                )
        except Exception:
            with _jobs_lock:
                job.update(status="error", error=traceback.format_exc(limit=8))


threading.Thread(target=_api_worker, daemon=True, name="api-worker").start()

api = FastAPI(title="MiniMax-H3 API", docs_url="/api/docs", openapi_url="/api/openapi.json")


@api.post("/api/generate")
def api_generate(req: GenerateRequest):
    if (req.width is None) != (req.height is None):
        raise HTTPException(422, "Provide both width and height, or neither.")
    if req.width and (req.width % 32 or req.height % 32):
        raise HTTPException(422, "width and height must be multiples of 32.")

    job_id = uuid.uuid4().hex[:12]
    job = {
        "job_id": job_id,
        "status": "queued",
        "created": time.time(),
        "request": req,
        "params": req.model_dump(exclude={"image_b64", "last_image_b64"}, exclude_none=True),
    }
    with _jobs_lock:
        _jobs[job_id] = job
        # Drop oldest finished jobs beyond the cap (their mp4s stay on disk).
        if len(_jobs) > MAX_JOBS_KEPT:
            for jid in sorted(_jobs, key=lambda j: _jobs[j]["created"]):
                if len(_jobs) <= MAX_JOBS_KEPT:
                    break
                if _jobs[jid]["status"] in ("done", "error"):
                    del _jobs[jid]
    _job_queue.put(job_id)
    return {"job_id": job_id, "status_url": f"/api/jobs/{job_id}"}


@api.get("/api/jobs")
def api_jobs():
    with _jobs_lock:
        jobs = sorted(_jobs.values(), key=lambda j: j["created"], reverse=True)
    return [_job_public(j) for j in jobs[:50]]


@api.get("/api/jobs/{job_id}")
def api_job(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job id.")
    return _job_public(job)


@api.get("/api/jobs/{job_id}/video")
def api_job_video(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job id.")
    if job["status"] != "done":
        raise HTTPException(409, f"Job is {job['status']}, not done.")
    return FileResponse(job["video_path"], media_type="video/mp4", filename=f"h3_{job_id}.mp4")


with gr.Blocks(title="MiniMax-H3") as demo:
    gr.Markdown(
        "# MiniMax-H3 — video + audio generation\n"
        "Joint video and stereo-audio generation, 24fps, 5–14.4s. "
        "Detailed shot-by-shot prompts with a soundscape description work best "
        "([prompting guide](https://huggingface.co/MiniMaxAI/MiniMax-H3)). "
        "Generation takes several minutes. Switching FL2VA ↔ Ref2VA reloads the "
        "transformer partition."
    )
    mode = gr.Radio(
        [MODE_FL2VA, MODE_REF2VA],
        value=MODE_FL2VA,
        label="Mode",
    )
    with gr.Row():
        with gr.Column(scale=3):
            prompt = gr.Textbox(
                label="Prompt",
                lines=6,
                placeholder=(
                    "[Shot 1] A fluffy red panda in a tiny tool belt tinkers inside an "
                    "open vending machine panel...\n"
                    "overall_soundscape: soft electrical hum, screwdriver clicks...\n"
                    "non_diegetic_music: playful pizzicato strings..."
                ),
            )
            with gr.Row(visible=True) as fl2va_row:
                image = gr.Image(label="First frame (optional)", type="pil")
                last_image = gr.Image(label="Last frame (optional)", type="pil")
            ref_images = gr.Gallery(
                label="Reference images (≤9, order = <Picture 1>…)",
                type="filepath",
                columns=3,
                height=200,
                visible=False,
            )
            ref_videos = gr.File(
                label="Reference videos (≤3, optional)",
                file_count="multiple",
                file_types=["video"],
                type="filepath",
                visible=False,
            )
            ref_audios = gr.File(
                label="Reference audio (≤3, optional; cannot be sole input)",
                file_count="multiple",
                file_types=["audio"],
                type="filepath",
                visible=False,
            )
            ref_help = gr.Markdown(
                "Name each reference in the prompt (`<Picture 1>` is the sofa, "
                "`<Picture 2>` is the room…). Order: images → videos → audios. "
                "See the [ref prompt guide]"
                "(https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/"
                "VIDEO_PROMPT_WRITING_GUIDE_ref_en.md).",
                visible=False,
            )
            with gr.Row():
                size = gr.Dropdown(list(SIZES), value="960×544 landscape (fast)", label="Canvas")
                seconds = gr.Slider(5.2, 14.4, value=8.0, step=0.1, label="Duration (seconds)")
            with gr.Row():
                steps = gr.Slider(4, 60, value=50, step=1, label="Steps")
                seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")
            with gr.Row():
                turbo = gr.Checkbox(
                    value=False,
                    label="Turbo (4-step LoRA, ~10x faster; preview quality, FL2VA only)",
                )
                turbo_strength = gr.Slider(
                    0.5,
                    1.5,
                    value=1.0,
                    step=0.05,
                    label="Turbo strength (up: fix ghosting · down: fix grain)",
                    visible=False,
                )
            btn = gr.Button("Generate", variant="primary")
        with gr.Column(scale=2):
            video = gr.Video(label="Result", autoplay=True)
            info = gr.Textbox(label="Run info", interactive=False)

    mode.change(
        _mode_visibility,
        [mode],
        [fl2va_row, ref_images, ref_videos, ref_audios, ref_help],
    )

    def _toggle_turbo(enabled):
        from turbo import TURBO_NUM_INFERENCE_STEPS

        return (
            gr.update(value=TURBO_NUM_INFERENCE_STEPS if enabled else 50),
            gr.update(visible=enabled),
        )

    turbo.change(_toggle_turbo, [turbo], [steps, turbo_strength])
    btn.click(
        generate,
        [
            mode,
            prompt,
            image,
            last_image,
            ref_images,
            ref_videos,
            ref_audios,
            size,
            seconds,
            steps,
            seed,
            turbo,
            turbo_strength,
        ],
        [video, info],
    )

if __name__ == "__main__":
    import uvicorn

    demo.queue(default_concurrency_limit=1)
    app = gr.mount_gradio_app(api, demo, path="/")
    uvicorn.run(app, host="0.0.0.0", port=7860)
