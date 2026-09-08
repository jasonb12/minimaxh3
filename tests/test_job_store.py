import tempfile
import unittest
from pathlib import Path
from job_store import JobStore, is_fatal_cuda


class JobStoreTests(unittest.TestCase):
    def test_restart_preserves_done_and_marks_interrupted_without_replay(self):
        with tempfile.TemporaryDirectory() as root:
            store = JobStore(root)
            for key, status in [('finished', 'done'), ('running', 'running')]:
                job = dict(job_id=store.identity(key), status=status, seed=42,
                           video_path='/output/result.mp4', request='secret URL', conditioning=object())
                store.save(job)
            restored = JobStore(root).recover()
            self.assertEqual(restored[store.identity('finished')]['status'], 'done')
            self.assertEqual(restored[store.identity('running')]['status'], 'error')
            self.assertNotIn('request', restored[store.identity('finished')])
            self.assertEqual(len(list(Path(root).glob('*.json'))), 2)
            self.assertIsNone(store.get('../../passwd'))

    def test_corruption_never_silently_forgets_an_accepted_request(self):
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / 'broken.json').write_text('{')
            with self.assertRaises(ValueError):
                JobStore(root).recover()

    def test_cuda_faults_are_fatal_but_capacity_and_source_errors_are_not(self):
        for error in ['CUDA error: unspecified launch failure', 'CUDA error: device-side assert triggered', 'CUDA driver error']:
            self.assertTrue(is_fatal_cuda(error))
        for error in ['CUDA out of memory', 'HTTP Error 404', 'Invalid reference']:
            self.assertFalse(is_fatal_cuda(error))
