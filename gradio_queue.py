"""Server-side Brightify queue client. This module never imports the renderer."""
import io
import mimetypes
import os
import secrets
import time
import uuid
from pathlib import Path

import requests


class BrightifyQueue:
    def __init__(self):
        self.url = os.environ.get("BRIGHTIFY_QUEUE_URL", "").rstrip("/")
        self.token = os.environ.get("BRIGHTIFY_QUEUE_TOKEN", "")
        if not self.url or not self.token:
            raise ValueError("Brightify queue is not configured. No local render was started.")

    def request(self, method, path, body=None, key=None):
        headers = {"Authorization": f"Bearer {self.token}"}
        if key:
            headers["Idempotency-Key"] = key
        for attempt in range(3):
            try:
                response = requests.request(method, self.url + path, json=body,
                                            headers=headers, timeout=30)
                if response.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                    time.sleep(attempt + 1)
                    continue
                if not response.ok:
                    raise ValueError(response.json().get("error", "Queue request failed"))
                return response.json()
            except (requests.Timeout, requests.ConnectionError):
                if attempt == 2:
                    raise ValueError("Brightify could not be reached. Check the queue before submitting again.") from None
                time.sleep(attempt + 1)

    def upload(self, value, kind, submission):
        if hasattr(value, "save"):
            data = io.BytesIO()
            value.save(data, format="PNG")
            data.seek(0)
            extension, mime = ".png", "image/png"
        else:
            path = Path(value)
            extension = path.suffix.lower()
            mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            data = path.open("rb")
        try:
            signed = self.request("POST", "/uploads", {
                "submission": submission, "extension": extension, "kind": kind,
            })
            response = requests.put(signed["url"], data=data,
                                    headers={"Content-Type": mime}, timeout=300)
            response.raise_for_status()
            return {**signed["ref"], "kind": kind}
        finally:
            data.close()

    def submit(self, mode, prompt, image, last_image, images, videos, audios,
               dimensions, seconds, steps, seed, turbo, strength):
        if not prompt or not prompt.strip():
            raise ValueError("A prompt is required.")
        refs = []
        for item in images or []:
            path = item[0] if isinstance(item, (tuple, list)) else (
                item.get("name") or item.get("path") or item.get("image")
                if isinstance(item, dict) else item)
            if path:
                refs.append(("image", path))
        refs += [("video", p) for p in videos or [] if p]
        refs += [("audio", p) for p in audios or [] if p]
        if mode == "ref2va":
            counts = {kind: sum(k == kind for k, _ in refs) for kind in ("image", "video", "audio")}
            if not counts["image"] and not counts["video"]:
                raise ValueError("Ref2VA needs an image or video reference.")
            if counts["image"] > 9 or counts["video"] > 3 or counts["audio"] > 3 or len(refs) > 12:
                raise ValueError("Limits: 9 images, 3 videos, 3 audios, 12 references total.")
        submission = str(uuid.uuid4())
        body = dict(mode=mode, prompt=prompt.strip(), durationSeconds=float(seconds),
                    steps=int(steps), seed=int(seed) if int(seed) >= 0 else secrets.randbelow(2**31),
                    turbo=bool(turbo), turboStrength=float(strength), includeAudio=True,
                    autoCanvas=dimensions is None)
        if dimensions:
            body.update(width=dimensions[0], height=dimensions[1])
        if mode == "ref2va":
            body["refs"] = [self.upload(path, kind, submission) for kind, path in refs]
        else:
            if image is not None:
                body["firstFrame"] = self.upload(image, "image", submission)
            if last_image is not None:
                body["lastFrame"] = self.upload(last_image, "image", submission)
        return self.request("POST", "/jobs", body, key=submission)


def status_text(job):
    text = f"Brightify job {job['jobId']} · {job['status']}"
    if job.get("worker"):
        text += f" · {job['worker']}"
    if job.get("error"):
        text += f" · {job['error']}"
    return text + f"\n{job['queueUrl']}"
