# REST API

`app.py` serves a job-based REST API alongside the Gradio UI on the same port
(7860). Submit a job, poll it, download the mp4. The UI and API share the one
resident pipeline and a GPU lock, so generations always run one at a time.

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
| `width`, `height` | model default | Both or neither; multiples of 32 (e.g. 960×544) |
| `seconds` | 8.0 | 5.2–14.4; snapped down to the 17n+5 frame grid at 24fps |
| `steps` | 50 (5 with turbo) | 4–60 |
| `seed` | -1 | -1 = random; the used seed is reported on the finished job |
| `turbo` | false | 4-step LoRA, ~10x faster, preview quality |
| `turbo_strength` | 1.0 | 0.5–1.5; up fixes ghosting, down fixes grain |

The API is FL2VA only. For Ref2VA (reference images/video/audio) use the UI or
Gradio's own machine API under `/gradio_api`.

## Example

```bash
# Submit (turbo preview at 960x544)
JOB=$(curl -s http://localhost:7860/api/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "A red panda waves at the camera. overall_soundscape: gentle forest ambience.",
       "width": 960, "height": 544, "seconds": 6, "turbo": true}' | jq -r .job_id)

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

## Notes

- Jobs live in process memory (restart clears the list); finished videos
  persist under `outputs/api/<job_id>.mp4`.
- A queued job's response includes `queue_position`.
- Errors surface on the job as `status: "error"` with a traceback in `error`.
- The first job after a server start loads the pipeline (~30s extra).
