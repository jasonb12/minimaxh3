#!/usr/bin/env python
"""Web UI for MiniMax-H3 video+audio generation.

Serves Gradio on 0.0.0.0:7860 so it is reachable from other machines on the
network. The pipeline loads lazily on the first generation (~30s) and stays
resident afterwards; requests are serialized through the queue since the model
saturates the GPU.

Run:  .venv/bin/python app.py
"""

import os
import threading
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "hf"))

import gradio as gr

from generate import build_pipeline

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

_pipe = None
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


def _get_pipe():
    global _pipe
    with _pipe_lock:
        if _pipe is None:
            _check_gpu_free()
            _pipe = build_pipeline(bf16_text_encoder=False)
    return _pipe


def _snap_frames(seconds: float) -> int:
    """Largest frame count of the form 17n+5 that fits in `seconds` (min 5.17s)."""
    n = (int(seconds * 24) - 5) // 17
    return 17 * max(n, 7) + 5


def generate(prompt, image, last_image, size, seconds, steps, seed, progress=gr.Progress()):
    import torch
    from diffusers.utils.export_utils import encode_video

    if not prompt or not prompt.strip():
        raise gr.Error("A prompt is required.")

    num_frames = _snap_frames(seconds)
    seed = int(seed)
    if seed < 0:
        seed = int(torch.seed() % 2**31)

    progress(0, desc="Loading pipeline (first run takes ~30s)")
    pipe = _get_pipe()

    kwargs = {
        "prompt": prompt.strip(),
        "num_frames": num_frames,
        "num_inference_steps": int(steps),
        "generator": torch.Generator().manual_seed(seed),
    }
    if SIZES[size] is not None:
        kwargs["width"], kwargs["height"] = SIZES[size]
    if image is not None:
        kwargs["image"] = image
    if last_image is not None:
        kwargs["last_image"] = last_image

    progress(0.05, desc=f"Generating {num_frames / 24:.1f}s of video (takes minutes)")
    t0 = time.time()
    state = pipe(**kwargs)
    elapsed = time.time() - t0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"h3_{time.strftime('%Y%m%d_%H%M%S')}_seed{seed}.mp4"
    progress(0.95, desc="Encoding mp4")
    encode_video(
        state.get("videos")[0],
        fps=24,
        output_path=str(out_path),
        audio=state.get("audio")[0],
        audio_sample_rate=state.get("sampling_rate"),
    )

    size_txt = "×".join(map(str, SIZES[size])) if SIZES[size] else "auto"
    info = (
        f"seed {seed} · {num_frames} frames ({num_frames / 24:.1f}s) · "
        f"{size_txt} · {int(steps)} steps · generated in {elapsed / 60:.1f} min"
    )
    return str(out_path), info


with gr.Blocks(title="MiniMax-H3") as demo:
    gr.Markdown(
        "# MiniMax-H3 — video + audio generation\n"
        "Joint video and stereo-audio generation, 24fps, 5–14.4s. "
        "Detailed shot-by-shot prompts with a soundscape description work best "
        "([prompting guide](https://huggingface.co/MiniMaxAI/MiniMax-H3)). "
        "Generation takes several minutes."
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
            with gr.Row():
                image = gr.Image(label="First frame (optional)", type="pil")
                last_image = gr.Image(label="Last frame (optional)", type="pil")
            with gr.Row():
                size = gr.Dropdown(list(SIZES), value="960×544 landscape (fast)", label="Canvas")
                seconds = gr.Slider(5.2, 14.4, value=8.0, step=0.1, label="Duration (seconds)")
            with gr.Row():
                steps = gr.Slider(10, 60, value=50, step=1, label="Steps")
                seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")
            btn = gr.Button("Generate", variant="primary")
        with gr.Column(scale=2):
            video = gr.Video(label="Result", autoplay=True)
            info = gr.Textbox(label="Run info", interactive=False)

    btn.click(generate, [prompt, image, last_image, size, seconds, steps, seed], [video, info])

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(server_name="0.0.0.0", server_port=7860)
