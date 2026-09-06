"""Small GPU/dependency check; no model downloads or production model allocation."""
import importlib.metadata as metadata
import json
import torch
import triton
import triton.language as tl
from spark import use_spark
from diffusers import MiniMaxH3Transformer3DModel, ModularPipeline
from transformers import Qwen3VLForConditionalGeneration
from torchao.quantization import Int8WeightOnlyConfig, Float8DynamicActivationFloat8WeightConfig
assert torch.cuda.is_available(), 'CUDA unavailable in container'
x = torch.ones((16, 16), device='cuda', dtype=torch.bfloat16)
assert float((x @ x).sum()) == 4096
print(json.dumps({'gpu': torch.cuda.get_device_name(), 'spark': use_spark(),
    'versions': {name: metadata.version(name) for name in ('torch','torchao','diffusers','transformers','gradio')}}))

@triton.jit
def copy_kernel(source, target, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    tl.store(target + offsets, tl.load(source + offsets))
source = torch.arange(128, device='cuda', dtype=torch.float32)
target = torch.empty_like(source)
copy_kernel[(1,)](source, target, BLOCK=128)
torch.testing.assert_close(source, target)
print('Triton JIT compilation and execution passed')
