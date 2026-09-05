"""CUDA integration checks using a small H3 model, without downloading weights."""

import tempfile
import unittest

import torch


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class SparkRuntimeTest(unittest.TestCase):
    def test_fp8_cache_compile_and_adapter_toggle(self):
        from diffusers import MiniMaxH3Transformer3DModel, TorchAoConfig
        from peft import LoraConfig
        from torchao.quantization import Float8DynamicActivationFloat8WeightConfig
        from torchao.quantization.granularity import PerTensor

        model = MiniMaxH3Transformer3DModel(
            num_attention_heads=4, attention_head_dim=128, hidden_size=512,
            num_layers=1, ffn_dim=1024, text_dim=512,
            time_embed_dim=256, time_embed_hidden_dim=512,
        )
        with tempfile.TemporaryDirectory() as source, tempfile.TemporaryDirectory() as cache:
            model.save_pretrained(source)
            model = MiniMaxH3Transformer3DModel.from_pretrained(
                source, torch_dtype=torch.bfloat16, device_map={"": "cuda"},
                quantization_config=TorchAoConfig(
                    Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor()),
                ),
            )
            model.save_pretrained(cache)
            restored = MiniMaxH3Transformer3DModel.from_pretrained(
                cache, torch_dtype=torch.bfloat16, device_map={"": "cuda"},
            )
            with torch.no_grad():
                x = torch.randn(32, 512, device="cuda", dtype=torch.bfloat16)
                self.assertTrue(torch.equal(
                    model.transformer_blocks[0].attn.to_q(x),
                    restored.transformer_blocks[0].attn.to_q(x),
                ))
            del restored

        model.add_adapter(
            LoraConfig(r=4, lora_alpha=4, target_modules=["to_q", "adaln_proj.linear"]),
            adapter_name="turbo",
        )
        kwargs = dict(
            hidden_states=torch.randn(1, 16, 96, device="cuda"),
            audio_hidden_states=torch.randn(1, 16, 32, device="cuda"),
            encoder_hidden_states=torch.randn(1, 16, 512, device="cuda"),
            timestep=torch.tensor([0.5], device="cuda"),
            timestep_indices=torch.zeros(48, device="cuda", dtype=torch.long),
            token_tags=torch.tensor([0] * 16 + [2] * 16 + [1] * 16, device="cuda"),
            position_ids=torch.zeros(48, 3, device="cuda"),
            video_indices=torch.arange(16, device="cuda"),
            audio_indices=torch.arange(16, 32, device="cuda"),
            text_indices=torch.arange(32, 48, device="cuda"),
        )
        with torch.no_grad():
            eager = model(**kwargs)
            model.compile_repeated_blocks(fullgraph=True)
            compiled = model(**kwargs)
            self.assertEqual(compiled.sample.shape, (1, 16, 96))
            self.assertEqual(compiled.audio_sample.shape, (1, 16, 32))
            for reference, actual in (
                (eager.sample, compiled.sample), (eager.audio_sample, compiled.audio_sample),
            ):
                self.assertTrue(torch.isfinite(actual).all())
                self.assertLess(((actual - reference).norm() / reference.norm()).item(), 0.02)
            model.disable_adapters()
            output = model(**kwargs)
            self.assertTrue(torch.isfinite(output.sample).all())
            self.assertTrue(torch.isfinite(output.audio_sample).all())


if __name__ == "__main__":
    unittest.main()
