"""Fixed geometric probability ensembles for the seven-channel segmentation model."""
import torch

# Four reflections; eight rotations/reflections. Identity is always first.
TTA4 = ((0, False), (0, True), (2, False), (2, True))
TTA8 = tuple((turns, flip) for turns in range(4) for flip in (False, True))


def transform_input(image, turns, flip):
    result = torch.rot90(image, turns, (-2, -1))
    return result.flip(-1) if flip else result


def invert_output(output, turns, flip):
    result = output.flip(-1) if flip else output
    return torch.rot90(result, -turns, (-2, -1))


def ensemble_probabilities(model, image, transforms=TTA8):
    """Average inverse-aligned probabilities without changing any input channels."""
    if not transforms:
        raise ValueError('At least one transform is required')
    total = None
    for turns, flip in transforms:
        p = model(transform_input(image, turns, flip).contiguous()).sigmoid()
        p = invert_output(p, turns, flip)
        total = p if total is None else total + p
    return total / len(transforms)


def inference_probabilities(model, image, geometric_ensemble="none"):
    """Opt-in deployment of the VAL-selected four-view ensemble."""
    if geometric_ensemble == "none":
        return model(image).sigmoid()
    if geometric_ensemble == "tta4":
        return ensemble_probabilities(model, image, TTA4)
    raise ValueError(f"Unknown inference.geometric_ensemble: {geometric_ensemble!r}")
