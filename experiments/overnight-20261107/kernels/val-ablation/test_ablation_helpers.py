import importlib.util
from pathlib import Path

import torch
from torch import nn

KERNEL_PATH = Path(__file__).with_name("run.py")
KERNEL_SOURCE = KERNEL_PATH.read_text(encoding="utf-8")
SPEC = importlib.util.spec_from_file_location("val_ablation_kernel", KERNEL_PATH)
KERNEL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(KERNEL)


class Capture(nn.Module):
    def forward(self, value):
        self.value = value.detach().clone()
        return value


class FakeDual(nn.Module):
    def __init__(self):
        super().__init__()
        self.optical_net = Capture()
        self.terrain_net = Capture()

    def forward(self, value):
        value = value.clone()
        value[:, :3] = (value[:, :3] - 0.5) / 0.25
        self.optical_net(value[:, :4])
        self.terrain_net(value[:, 4:7])


class FakeShared(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = Capture()

    def forward(self, value):
        value = value.clone()
        value[:, :3] = (value[:, :3] - 0.5) / 0.25
        self.network(value)


def test_kaggle_kernel_is_self_contained_in_the_uploaded_code_file():
    assert "import base_runner" not in KERNEL_SOURCE
    assert "from ablation_helpers" not in KERNEL_SOURCE


def test_kernel_prepares_a_indexes_before_hashing_them():
    prepare = KERNEL_SOURCE.index('"scripts/prepare_data.py"')
    hash_indexes = KERNEL_SOURCE.index("actual = sha256(path)")
    assert prepare < hash_indexes


def test_optical_hook_zeros_rgb_after_model_normalization_and_nir():
    model = FakeDual()
    hook = KERNEL.register_optical_zero_hook(model)
    try:
        model(torch.ones(1, 7, 2, 2))
        assert torch.count_nonzero(model.optical_net.value) == 0
        assert torch.equal(model.terrain_net.value, torch.ones(1, 3, 2, 2))
    finally:
        hook.remove()


def test_optical_hook_supports_shared_encoder_and_preserves_terrain():
    model = FakeShared()
    hook = KERNEL.register_optical_zero_hook(model)
    try:
        model(torch.ones(1, 7, 2, 2))
        assert torch.count_nonzero(model.network.value[:, :4]) == 0
        assert torch.equal(model.network.value[:, 4:7], torch.ones(1, 3, 2, 2))
    finally:
        hook.remove()


def test_terrain_neutralization_changes_only_channels_four_through_six():
    images = torch.arange(7, dtype=torch.float32).view(1, 7, 1, 1)
    result = KERNEL.neutralize_terrain(images)
    assert torch.equal(result[:, :4], images[:, :4])
    assert torch.count_nonzero(result[:, 4:7]) == 0
    assert torch.equal(images[:, 4:7], torch.tensor([4.0, 5.0, 6.0]).view(1, 3, 1, 1))


def test_coverage_mask_uses_pixel_centers_outside_bounds_only():
    from types import SimpleNamespace
    import numpy as np

    transform = SimpleNamespace(a=1.0, b=0.0, c=0.0, d=0.0, e=-1.0, f=2.0)
    outside = KERNEL.outside_footprint_mask(
        transform, width=3, height=2, bounds=(0.0, 0.0, 2.0, 2.0))
    expected = np.array([[False, False, True], [False, False, True]])
    assert np.array_equal(outside, expected)


def test_coverage_ignore_preserves_in_bounds_zero_and_existing_ignore_pixels():
    import numpy as np

    weights = np.array([[0.0, 1.0, 1.0]])
    outside = np.array([[False, False, True]])
    ignored = KERNEL.combine_ignore_masks(weights, outside)
    assert np.array_equal(ignored, np.array([[True, False, True]]))
