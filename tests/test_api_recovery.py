import tempfile
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
import app
from job_store import JobStore


class ApiRecoveryTests(unittest.TestCase):
    def test_duplicate_submission_and_restart_reconnect_without_enqueuing_twice(self):
        with tempfile.TemporaryDirectory() as root, patch.object(app, '_jobs', {}), patch.object(app, '_job_queue') as queue, patch.object(app, '_job_store', JobStore(root)):
            req = app.GenerateRequest(prompt='Test', idempotency_key='brightify-job:1')
            first = app.api_generate(req)
            self.assertEqual(app.api_generate(req), first)
            queue.put.assert_called_once()
            app._jobs.clear()
            app._jobs.update(app._job_store.recover())
            self.assertEqual(app.api_generate(req), first)
            self.assertEqual(app.api_job(first['job_id'])['status'], 'error')
            with self.assertRaises(app.HTTPException) as conflict:
                app.api_generate(app.GenerateRequest(prompt='Changed', idempotency_key='brightify-job:1'))
            self.assertEqual(conflict.exception.status_code, 409)

    def test_auth_and_poisoned_health(self):
        with patch.dict('os.environ', {'MINIMAX_H3_TOKEN': 'test-secret'}), patch.object(app, '_fatal_error', 'Fatal CUDA failure'):
            client = TestClient(app.api)
            self.assertEqual(client.get('/api/jobs').status_code, 401)
            self.assertEqual(client.get('/api/jobs', headers={'Authorization':'Bearer test-secret'}).status_code, 200)
            self.assertEqual(client.get('/api/health').status_code, 503)
            response = client.post('/api/generate', headers={'Authorization':'Bearer test-secret'}, json={'prompt':'test'})
            self.assertEqual(response.status_code, 503)
