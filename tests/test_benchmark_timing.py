import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchmark_timing import phase, record_generation


class BenchmarkTimingTests(unittest.TestCase):
    def exercise(self, out_path, fail=False):
        @record_generation
        def generate(task, kwargs, turbo, turbo_strength, out_path):
            with phase('block:decode', gpu=True):
                if fail:
                    raise ValueError('private prompt https://private?token=secret')
            return 42
        return generate('ref2va', {'prompt': 'private', 'num_inference_steps': 7}, True, 1.0, out_path)

    def test_disabled_has_no_synchronization_or_sidecar(self):
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, {'MINIMAX_H3_BENCHMARK': '0'}), patch('benchmark_timing._synchronize') as synchronize:
            self.assertEqual(self.exercise(Path(root) / 'output.mp4'), 42)
            synchronize.assert_not_called()
            self.assertEqual(list(Path(root).iterdir()), [])

    def test_enabled_records_phases_without_prompts_or_urls(self):
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, {'MINIMAX_H3_BENCHMARK': '1'}), patch('benchmark_timing._synchronize') as synchronize:
            output = Path(root) / 'output.mp4'
            self.assertEqual(self.exercise(output), 42)
            trace_path = output.with_suffix('.benchmark.json')
            trace = json.loads(trace_path.read_text())
            self.assertEqual(trace['scheduler_points'], 7)
            self.assertEqual(trace['phases'][0]['name'], 'block:decode')
            self.assertGreaterEqual(trace['phases'][0]['seconds'], 0)
            self.assertEqual(synchronize.call_count, 2)
            self.assertEqual(trace_path.stat().st_mode & 0o777, 0o600)
            self.assertNotIn('private', trace_path.read_text())

    def test_failed_phase_keeps_original_exception_and_restores_context(self):
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, {'MINIMAX_H3_BENCHMARK': '1'}), patch('benchmark_timing._synchronize') as synchronize:
            output = Path(root) / 'output.mp4'
            with self.assertRaises(ValueError):
                self.exercise(output, fail=True)
            trace = json.loads(output.with_suffix('.benchmark.json').read_text())
            self.assertEqual(trace['status'], 'error')
            self.assertEqual(trace['phases'][0]['status'], 'error')
            self.assertNotIn('secret', json.dumps(trace))
            synchronize.reset_mock()
            with phase('outside', gpu=True):
                pass
            synchronize.assert_not_called()
