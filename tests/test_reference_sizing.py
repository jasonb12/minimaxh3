from types import SimpleNamespace
import unittest

from reference_sizing import apply_reference_policy, reference_dimensions


class ReferenceSizingTests(unittest.TestCase):
    def test_output_area_policy_is_bounded_aligned_and_does_not_upscale(self):
        for width, height in [(832, 480), (2048, 2048), (512, 2048), (2048, 512), (99, 101)]:
            target = reference_dimensions(width, height, 832, 480, 'match-output-v1')
            self.assertTrue(all(value % 32 == 0 for value in target))
            self.assertLessEqual(target[0], width)
            self.assertLessEqual(target[1], height)
            self.assertLessEqual(target[0] * target[1], 832 * 480)
        self.assertEqual(reference_dimensions(832, 480, 832, 480, 'match-output-v1'), (832, 480))

    def test_short_edge_cap_preserves_native_small_images(self):
        self.assertEqual(reference_dimensions(2048, 2048, 832, 480, 'bounded-1024-v1'), (1024, 1024))
        self.assertEqual(reference_dimensions(832, 480, 832, 480, 'bounded-1024-v1'), (832, 480))

    def test_invalid_policy_geometry_and_aspect_ratio_fail(self):
        for arguments in [(0, 480, 832, 480, 'match-output-v1'), (5000, 32, 832, 480, 'match-output-v1'),
                          (832, 480, 832, 480, 'typo')]:
            with self.assertRaises(ValueError):
                reference_dimensions(*arguments)

    def test_default_is_an_exact_noop(self):
        self.assertEqual(apply_reference_policy(None, None, None), [])

    def test_real_pipeline_state_preserves_originals_order_and_other_modalities(self):
        from diffusers.image_processor import VaeImageProcessor
        from diffusers.modular_pipelines import PipelineState
        from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3ImageReference
        from PIL import Image
        picture = Image.new('RGB', (832, 480), 'red')
        original = MiniMaxH3ImageReference(image=picture)
        audio = SimpleNamespace(kind='audio')
        normalized_audio = object()
        state = PipelineState()
        state.set('references', [original, audio])
        state.set('normalized_references', [MiniMaxH3ImageReference(image=picture.resize((3552, 2048))), normalized_audio])
        state.set('width', 832)
        state.set('height', 480)
        pipe = SimpleNamespace(image_processor=VaeImageProcessor(vae_scale_factor=16), canvas_multiple=32)
        metadata = apply_reference_policy(pipe, state, 'match-output-v1')
        self.assertIs(state.get('normalized_references')[0].image, picture)
        self.assertIs(state.get('normalized_references')[1], normalized_audio)
        self.assertIs(state.get('references')[0], original)
        self.assertEqual(metadata, [{'index': 0, 'original': [832, 480], 'effective': [832, 480]}])
