import unittest

import torch
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
from spark import retain_conditioning_layer


class ConditionerCaptureTest(unittest.TestCase):
    def test_selected_hidden_state_matches_unmodified_qwen(self):
        config = Qwen3VLConfig(
            text_config=dict(vocab_size=128, hidden_size=128, intermediate_size=256,
                             num_hidden_layers=3, num_attention_heads=1,
                             num_key_value_heads=1, head_dim=128),
            vision_config=dict(depth=1, hidden_size=128, intermediate_size=256,
                               num_heads=1, out_hidden_size=128),
        )
        encoder = Qwen3VLForConditionalGeneration(config).eval()
        inputs = dict(input_ids=torch.tensor([[3, 4, 5]]), use_cache=False,
                      output_hidden_states=True)
        with torch.no_grad():
            expected = encoder.model(**inputs).hidden_states[1]
            retain_conditioning_layer(encoder, layer=1)
            actual = encoder.model(**inputs)
            torch.testing.assert_close(actual.hidden_states[1], expected, rtol=0, atol=0)
            self.assertEqual(sum(x is not None for x in actual.hidden_states), 1)
            # Repeated calls must neither accumulate hooks nor retain old activations.
            encoder.model(**inputs)
            self.assertFalse(encoder.model.language_model.layers[1]._forward_pre_hooks)
            ordinary = encoder.model(input_ids=inputs['input_ids'], use_cache=False)
            self.assertIsNone(ordinary.hidden_states)
