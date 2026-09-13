import math


POLICIES = ('bounded-1024-v1', 'match-output-v1')


def reference_dimensions(width, height, canvas_width, canvas_height, policy, multiple=32):
    if policy not in POLICIES:
        raise ValueError('Unknown reference sizing policy')
    if min(width, height, canvas_width, canvas_height) < multiple:
        raise ValueError('Reference and canvas dimensions must be at least one model grid cell')
    if max(width, height) > 4 * min(width, height):
        raise ValueError('Reference aspect ratio must be between 1:4 and 4:1')
    if policy == 'bounded-1024-v1':
        scale = min(1.0, 1024 / min(width, height))
    else:
        scale = min(1.0, math.sqrt(canvas_width * canvas_height / (width * height)))
    return tuple(max(multiple, math.floor(dimension * scale / multiple) * multiple)
                 for dimension in (width, height))


def apply_reference_policy(pipe, state, policy):
    if policy is None:
        return []
    if policy not in POLICIES:
        raise ValueError('Unknown reference sizing policy')
    from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3ImageReference
    import numpy as np
    import torch
    from PIL import Image

    originals = state.get('references')
    normalized = state.get('normalized_references')
    if not originals or normalized is None or len(originals) != len(normalized):
        raise ValueError('Reference policy requires completed Ref2VA setup')
    resized = list(normalized)
    dimensions = []
    for index, entry in enumerate(originals):
        if entry.kind != 'image':
            continue
        picture = entry.image
        if isinstance(picture, torch.Tensor):
            if picture.dtype == torch.uint8:
                picture = picture.float() / 255.0
            picture = pipe.image_processor.pt_to_numpy(picture[None])[0]
        if isinstance(picture, np.ndarray):
            if picture.dtype == np.uint8:
                picture = picture.astype(np.float32) / 255.0
            picture = pipe.image_processor.numpy_to_pil(picture)[0]
        if not isinstance(picture, Image.Image):
            raise TypeError('Reference policy requires decoded image pixels')
        target = reference_dimensions(*picture.size, state.get('width'), state.get('height'), policy,
                                      multiple=pipe.canvas_multiple)
        prepared = picture if picture.size == target else pipe.image_processor.resize(
            picture, width=target[0], height=target[1])
        resized[index] = MiniMaxH3ImageReference(image=prepared)
        dimensions.append({'index': index, 'original': list(picture.size), 'effective': list(target)})
    state.set('normalized_references', resized)
    return dimensions
