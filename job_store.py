"""Small crash-safe local inference receipts; Brightify owns the business queue."""
import hashlib
import json
import os
import re
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode


class JobStore:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def identity(key):
        return hashlib.sha256(key.encode()).hexdigest()[:32]

    @staticmethod
    def fingerprint(payload):
        def stable(value):
            if isinstance(value, dict):
                return {key: stable(item) for key, item in value.items()}
            if isinstance(value, list):
                return [stable(item) for item in value]
            if isinstance(value, str) and value.startswith(('http://', 'https://')):
                url = urlsplit(value)
                if '/storage/v1/object/sign/' in url.path:
                    value = urlunsplit((url.scheme, url.netloc, url.path,
                        urlencode([(k, v) for k, v in parse_qsl(url.query) if k != 'token']), url.fragment))
            return value
        return hashlib.sha256(json.dumps(stable(payload), sort_keys=True, separators=(',', ':')).encode()).hexdigest()

    def path(self, job_id):
        if not re.fullmatch(r'[a-f0-9]{12,32}', job_id):
            raise ValueError('Invalid job id')
        return self.directory / f'{job_id}.json'

    def save(self, job):
        # Requests contain expiring signed URLs and conditioning contains tensors.
        receipt = {k: v for k, v in job.items() if k not in ('request', 'conditioning')}
        path = self.path(job['job_id'])
        temporary = path.with_suffix('.tmp')
        with temporary.open('w', encoding='utf-8') as stream:
            os.chmod(temporary, 0o600)
            json.dump(receipt, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fd = os.open(self.directory, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def get(self, job_id):
        try:
            return json.loads(self.path(job_id).read_text())
        except (FileNotFoundError, ValueError):
            return None

    def recover(self):
        jobs = {}
        for path in self.directory.glob('*.json'):
            # Corrupt receipts fail startup; never silently regenerate an accepted key.
            job = json.loads(path.read_text())
            if job['status'] in ('queued', 'running', 'cancelling'):
                job.update(status='error', error='H3 process restarted before completion; submit a new attempt')
                self.save(job)
            jobs[job['job_id']] = job
        return jobs


def is_fatal_cuda(error):
    message = str(error).lower()
    return any(marker in message for marker in (
        'cuda error', 'cuda driver', 'device-side assert', 'illegal memory access',
        'unspecified launch failure', 'no cuda gpus', 'no cuda-capable device',
        'driver shutting down', 'cublas_status_execution_failed',
    ))
