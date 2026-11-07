"""Score predictions against frozen original supervision, with optional coverage exclusion."""
import torch


def frozen_counts(probabilities, original_target, original_weights, threshold, outside=None):
    if not (probabilities.shape == original_target.shape == original_weights.shape):
        raise ValueError('Prediction and frozen supervision shapes differ')
    valid = original_weights > 0
    if outside is not None:
        if outside.shape != valid.shape:
            raise ValueError('Coverage mask shape differs')
        valid = valid & ~outside.bool()
    positive = original_target > 0.5
    predicted = probabilities >= threshold
    return dict(tp=int((predicted & positive & valid).sum()),
                fp=int((predicted & ~positive & valid).sum()),
                fn=int((~predicted & positive & valid).sum()))
