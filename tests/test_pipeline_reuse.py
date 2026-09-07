"""Lifecycle regressions: ownership, switching back, and failed-load recovery."""
import gc
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import patch

import torch
from diffusers import ComponentsManager
from spark import SHARED_COMPONENTS
import app


class PipelineReuseTest(unittest.TestCase):
    def setUp(self):
        self.saved = app._pipe, app._pipe_task, app._resident_shared
        app._pipe = app._pipe_task = app._resident_shared = None

    def tearDown(self):
        app._pipe, app._pipe_task, app._resident_shared = self.saved

    def test_switch_releases_transformer_and_preserves_shared_objects(self):
        shared = {name: torch.nn.Linear(1, 1) for name in SHARED_COMPONENTS}
        old_models = []
        builds = []

        def build(bf16_text_encoder, task, shared_components=None):
            gc.collect()
            self.assertTrue(all(ref() is None for ref in old_models))
            if builds:
                self.assertEqual(set(shared_components), set(shared))
                for name in shared:
                    self.assertIs(shared_components[name], shared[name])
            manager = ComponentsManager()
            model = torch.nn.Linear(1, 1)
            # A real manager owns models independently of pipeline attributes.
            manager.components['denoiser'] = model
            manager.components.update(shared)
            old_models.append(weakref.ref(model))
            builds.append(task)
            return SimpleNamespace(_h3_resident=True, _components_manager=manager,
                                   transformer=model if task == 'fl2va' else None,
                                   transformer_ref=model if task == 'ref2va' else None, **shared)

        with patch.object(app, 'build_pipeline', side_effect=build), patch.object(app, '_check_gpu_free') as check:
            app._get_pipe('fl2va')
            app._get_pipe('fl2va')
            app._get_pipe('ref2va')
            app._get_pipe('fl2va')
            self.assertEqual(builds, ['fl2va', 'ref2va', 'fl2va'])
            check.assert_called_once()

    def test_failed_replacement_retains_shared_for_retry(self):
        shared = {name: object() for name in SHARED_COMPONENTS}
        app._pipe = SimpleNamespace(_h3_resident=True, **shared)
        app._pipe_task = 'fl2va'
        replacement = SimpleNamespace(_h3_resident=True, **shared)
        with patch.object(app, 'build_pipeline', side_effect=[RuntimeError('load failed'), replacement]) as build, patch.object(app, '_check_gpu_free') as check:
            with self.assertRaisesRegex(RuntimeError, 'load failed'):
                app._get_pipe('ref2va')
            self.assertIsNone(app._pipe)
            self.assertIsNone(app._pipe_task)
            self.assertEqual(app._resident_shared, shared)
            self.assertIs(app._get_pipe('ref2va'), replacement)
            self.assertEqual(build.call_args.kwargs['shared_components'], shared)
            self.assertIsNone(app._resident_shared)
            check.assert_not_called()

    def test_invalid_task_keeps_loaded_pipeline(self):
        original = app._pipe = SimpleNamespace()
        app._pipe_task = 'fl2va'
        with self.assertRaises(ValueError):
            app._get_pipe('invalid')
        self.assertIs(app._pipe, original)


if __name__ == '__main__':
    unittest.main()
