import contextvars
from contextlib import contextmanager
import functools
import inspect
import json
import os
from pathlib import Path
import time


_trace = contextvars.ContextVar('h3_benchmark_trace', default=None)


def record_reference_sizes(policy, dimensions):
    trace = _trace.get()
    if trace is not None:
        trace.setdefault('reference_preprocessing', []).append({'policy': policy, 'images': dimensions})


def _synchronize():
    import torch
    torch.cuda.synchronize()


@contextmanager
def phase(name, gpu=False):
    trace = _trace.get()
    if trace is None:
        yield
        return
    if gpu:
        _synchronize()
    started = time.perf_counter()
    status = 'ok'
    try:
        yield
        if gpu:
            _synchronize()
    except BaseException:
        status = 'error'
        raise
    finally:
        trace['phases'].append({'name': name, 'seconds': time.perf_counter() - started,
                                'status': status, 'cuda_synchronized': gpu})


def record_generation(function):
    signature = inspect.signature(function)

    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        if os.environ.get('MINIMAX_H3_BENCHMARK') != '1':
            return function(*args, **kwargs)
        arguments = signature.bind(*args, **kwargs).arguments
        settings = arguments.get('kwargs', {})
        trace = {
            'schema': 1,
            'clock': 'perf_counter',
            'benchmark_only': True,
            'task': arguments.get('task'),
            'turbo': arguments.get('turbo'),
            'turbo_strength': arguments.get('turbo_strength'),
            'scheduler_points': settings.get('num_inference_steps'),
            'width': settings.get('width'),
            'height': settings.get('height'),
            'num_frames': settings.get('num_frames'),
            'reference_policy': settings.get('_h3_reference_policy', 'legacy-2048'),
            'phases': [],
        }
        token = _trace.set(trace)
        started = time.perf_counter()
        try:
            result = function(*args, **kwargs)
            trace['status'] = 'ok'
            return result
        except BaseException as error:
            trace['status'] = 'error'
            trace['error_type'] = type(error).__name__
            raise
        finally:
            trace['total_seconds'] = time.perf_counter() - started
            _trace.reset(token)
            try:
                target = Path(arguments['out_path']).with_suffix('.benchmark.json')
                target.parent.mkdir(parents=True, exist_ok=True)
                descriptor = os.open(target, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
                with os.fdopen(descriptor, 'w') as stream:
                    json.dump(trace, stream, indent=2)
            except Exception:
                print('[benchmark] timing sidecar could not be written', flush=True)

    return wrapped
