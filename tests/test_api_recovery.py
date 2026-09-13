import tempfile
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
import app
from job_store import JobStore


class ApiRecoveryTests(unittest.TestCase):
    def test_reference_experiments_are_explicit_and_preserve_old_fingerprints(self):
        candidate = app.GenerateRequest(prompt='Test', reference_image_urls=['https://example.com/reference.png'],
                                        reference_policy='match-output-v1')
        with patch.dict('os.environ', {'MINIMAX_H3_REFERENCE_EXPERIMENTS': '0'}):
            with self.assertRaises(app.HTTPException) as unavailable:
                app.api_generate(candidate)
            self.assertEqual(unavailable.exception.status_code, 422)
        with tempfile.TemporaryDirectory() as root, patch.object(app, '_jobs', {}), patch.object(app, '_job_queue'), patch.object(app, '_job_store', JobStore(root)):
            legacy = app.GenerateRequest(prompt='Test', idempotency_key='legacy-replay')
            receipt = app.api_generate(legacy)
            historical = legacy.model_dump(exclude={'idempotency_key', 'reference_policy'})
            self.assertEqual(app._jobs[receipt['job_id']]['fingerprint'], JobStore.fingerprint(historical))
        with patch.dict('os.environ', {'MINIMAX_H3_REFERENCE_EXPERIMENTS': '1'}):
            with self.assertRaises(app.HTTPException):
                app.api_generate(app.GenerateRequest(prompt='Test', reference_policy='match-output-v1'))

    def test_health_does_not_synchronize_an_active_render(self):
        with app._gen_lock, patch('torch.cuda.synchronize') as synchronize:
            self.assertEqual(app.api_health(), {'status': 'ok', 'inference': 'busy'})
            synchronize.assert_not_called()

    def test_idle_health_exercises_cuda_and_releases_inference_lock(self):
        with patch('torch.empty') as empty, patch('torch.cuda.synchronize') as synchronize:
            self.assertEqual(app.api_health(), {'status': 'ok'})
            empty.assert_called_once_with(1, device='cuda')
            synchronize.assert_called_once()
        with patch('torch.empty', side_effect=RuntimeError('device lost')):
            with self.assertRaises(app.HTTPException) as unavailable:
                app.api_health()
            self.assertEqual(unavailable.exception.status_code, 503)
        self.assertFalse(app._gen_lock.locked())

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
