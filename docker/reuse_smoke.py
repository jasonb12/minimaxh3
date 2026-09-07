"""Opt-in full-weight CUDA acceptance test; run with the production worker drained."""
import gc, json, os, time, weakref
from pathlib import Path
from PIL import Image
import torch
import app
from spark import SHARED_COMPONENTS
from generate import build_references

out = Path('/app/outputs/reuse-validation')
out.mkdir(parents=True, exist_ok=True)
Image.new('RGB', (832,480), (24,100,180)).save(out/'reference.png')
shared = None
pointers = None
previous = None
for index, task in enumerate(('fl2va', 'ref2va', 'fl2va')):
    started = time.monotonic()
    pipe = app._get_pipe(task)
    if previous is not None:
        gc.collect()
        assert previous() is None, 'Old transformer is still retained'
    actual = {name: getattr(pipe, name) for name in SHARED_COMPONENTS}
    actual_pointers = {name: next(actual[name].parameters()).data_ptr() for name in ('text_encoder','vae','audio_vae')}
    if shared is not None:
        assert all(actual[name] is shared[name] for name in shared)
        assert actual_pointers == pointers
    shared, pointers = actual, actual_pointers
    assert all(next(shared[n].parameters()).device.type == 'cuda' for n in pointers)
    previous = weakref.ref(getattr(pipe, 'transformer_ref', None) or pipe.transformer)
    pipe = None
    load_seconds = time.monotonic() - started
    kwargs = dict(prompt='A blue ceramic vase on a wooden table, stationary camera, soft daylight.',
                  width=832, height=480, num_frames=124, num_inference_steps=7,
                  generator=torch.Generator(device='cuda').manual_seed(270907))
    if task == 'ref2va':
        kwargs['references'] = build_references([('image', str(out/'reference.png'))])
        kwargs['prompt'] = 'A stationary shot of the blue colour in <Picture 1>, soft ambient sound.'
    elapsed = None
    if os.environ.get('H3_REUSE_LOAD_ONLY') != '1':
        elapsed = app._run_generation(task, kwargs, True, 1.0, out/f'{index}-{task}.mp4')
    print('REUSE_VALIDATION '+json.dumps(dict(task=task, load_seconds=load_seconds, render_seconds=elapsed,
          shared_pointers=pointers, allocated_gib=torch.cuda.memory_allocated()/1024**3)), flush=True)
print('REUSE_VALIDATION PASSED', flush=True)
