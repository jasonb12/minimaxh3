#!/usr/bin/env python
"""Web UI for MiniMax-H3 video+audio generation.

Serves Gradio on 0.0.0.0:7860 so it is reachable from other machines on the
network. The pipeline loads lazily on the first generation (~30s) and stays
resident afterwards; requests are serialized through the queue since the model
saturates the GPU. Switching between FL2VA and Ref2VA reloads the transformer
partition (~62GB) once.

Run:  .venv/bin/python app.py
"""

import gc
import os
import threading
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "hf"))

import gradio as gr

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
    from diffusers.utils.export_utils import encode_video

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

    progress(0, desc=f"Loading {task} pipeline (first run / mode switch takes a bit)")
    pipe = _get_pipe(task)

    if task == "fl2va":
        from turbo import load_turbo_lora, set_turbo_enabled

        if turbo:
            load_turbo_lora(pipe.transformer, strength=float(turbo_strength))
        set_turbo_enabled(pipe.transformer, turbo)

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
    t0 = time.time()
    state = pipe(**kwargs)
    elapsed = time.time() - t0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"h3_{task}_{time.strftime('%Y%m%d_%H%M%S')}_seed{seed}.mp4"
    progress(0.95, desc="Encoding mp4")
    encode_video(
        state.get("videos")[0],
        fps=24,
        output_path=str(out_path),
        audio=state.get("audio")[0],
        audio_sample_rate=state.get("sampling_rate"),
    )

    size_txt = "×".join(map(str, SIZES[size])) if SIZES[size] else "auto"
    ref_txt = f" · {len(refs)} refs" if task == "ref2va" else ""
    turbo_txt = f" · turbo@{float(turbo_strength):g}" if turbo and task == "fl2va" else ""
    info = (
        f"{task}{ref_txt}{turbo_txt} · seed {seed} · {num_frames} frames ({num_frames / 24:.1f}s) · "
        f"{size_txt} · {int(steps)} steps · generated in {elapsed / 60:.1f} min"
    )
    return str(out_path), info


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
    demo.queue(default_concurrency_limit=1).launch(server_name="0.0.0.0", server_port=7860)
