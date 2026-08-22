# REST API

`app.py` serves a job-based REST API alongside the Gradio UI on the same port
(7860). Submit a job, poll it, download the mp4. The UI and API share a GPU
lock, so generations always run one at a time. The pipeline stays resident
after the first job of a given task so queued work reuses the loaded weights.
A FL2VA ↔ Ref2VA switch still reloads the transformer partition.

Interactive docs (Swagger): `http://<host>:7860/api/docs`

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/api/generate` | Queue a generation, returns `{job_id, status_url}` |
| GET | `/api/jobs/{id}` | Job status: `queued` / `running` / `done` / `error` |
| GET | `/api/jobs/{id}/video` | Download the mp4 (409 until done) |
| GET | `/api/jobs` | 50 most recent jobs, newest first |

## POST /api/generate

JSON body (only `prompt` is required):

| Field | Default | Notes |
| --- | --- | --- |
| `prompt` | — | Shot-by-shot prompts with a soundscape line work best |
| `image_b64` / `image_url` | none | First frame; base64 (raw or data URL) or a fetchable URL |
| `last_image_b64` / `last_image_url` | none | Last frame |
| `reference_image_urls` | none | 1–9 identity/style references for Ref2VA; not output frames |
| `width`, `height` | model default | Both or neither; multiples of 32 (e.g. 960×544) |
| `seconds` | 8.0 | 5.2–14.4; snapped down to the 17n+5 frame grid at 24fps |
| `num_frames` | none | Optional 120–360 override; the pipeline rounds to its 17n+5 grid |
| `steps` | 50 (5 with turbo) | 4–60 |
| `seed` | -1 | -1 = random; the used seed is reported on the finished job |
| `turbo` | true | 4-step LoRA, ~10x faster, preview quality; experimental with Ref2VA; set false for full quality |
| `turbo_strength` | 1.0 | 0.5–1.5; up fixes ghosting, down fixes grain |
| `include_audio` | true | Set false to write a silent MP4 |
| `priority` | 100 | Lower runs first; use 0 for an urgent CLI/test job |

`reference_image_urls` selects Ref2VA and cannot be combined with first/last
frames. Turbo can be applied to Ref2VA experimentally because the transformer
module layouts match, but the LoRA was trained on FL2VA and may weaken identity
fidelity. Reference URLs are deliberately omitted from job-status responses.
Ref2VA video/audio references remain available through the UI or Gradio's own
machine API under `/gradio_api`.

Priority is non-preemptive: priority 0 becomes the next REST job after the
current generation, ahead of normal priority-100 jobs already waiting. Jobs at
the same priority retain FIFO order. The included CLI submits at priority 0:

```bash
.venv/bin/python 'queue_generate.py' 'A red panda waves at the camera' \
  --priority '0' \
  --output 'outputs/priority_test.mp4'
```

## Example

```bash
# Submit (turbo preview at 960x544)
JOB=$(curl -s http://localhost:7860/api/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "A red panda waves at the camera. overall_soundscape: gentle forest ambience.",
       "width": 960, "height": 544, "seconds": 6, "turbo": true,
       "include_audio": false}' | jq -r .job_id)

# Poll until done (status: queued -> running -> done)
watch -n 10 "curl -s http://localhost:7860/api/jobs/$JOB | jq"

# Download
curl -o out.mp4 "http://localhost:7860/api/jobs/$JOB/video"
```

With a first frame from a local file:

```bash
curl -s http://localhost:7860/api/generate -H 'Content-Type: application/json' \
  -d "$(jq -n --arg img "$(base64 -w0 frame.png)" \
        '{prompt: "...", image_b64: $img, turbo: true}')"
```

With a product identity reference that must not become frame zero:

```bash
curl -s 'http://localhost:7860/api/generate' \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "Use <Picture 1> only for exact product identity. One continuous commercial scene.",
       "reference_image_urls": ["https://example.invalid/short-lived-signed-product.png"],
       "width": 960, "height": 544, "seconds": 10, "turbo": true,
       "include_audio": false}'
```

## Notes

- Jobs live in process memory (restart clears the list); finished videos
  persist under `outputs/api/<job_id>.mp4`.
- A queued job's response includes its priority-aware `queue_position`.
- Errors surface on the job as `status: "error"` with a traceback in `error`.
- The first job after a server start loads the pipeline (~30s extra).
