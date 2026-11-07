"""T4-D dual encoder: RGB/NIR Unet fused with a dedicated terrain CNN."""
import torch

from trailnet.models.unet import DualTrailUNet, TrailUNet, build_model
from trailnet.train import _split_param_groups


def test_build_model_default_is_shared_encoder():
    model = build_model(
        {
            "encoder_name": "resnet18",
            "encoder_weights": None,
            "in_channels": 7,
            "classes": 1,
        }
    )
    assert isinstance(model, TrailUNet)
    assert not isinstance(model, DualTrailUNet)


def test_build_model_dual_terrain_returns_dual_unet():
    model = build_model(
        {
            "architecture": "dual_terrain",
            "encoder_name": "resnet18",
            "terrain_encoder_name": "resnet18",
            "encoder_weights": None,
            "in_channels": 7,
            "classes": 1,
        }
    )
    assert isinstance(model, DualTrailUNet)


def test_dual_encoder_has_distinct_optical_and_terrain_encoders():
    model = DualTrailUNet(
        encoder_name="resnet18",
        encoder_weights=None,
        terrain_encoder_name="resnet18",
        in_channels=7,
        classes=1,
    )
    optical = model.optical_encoder
    terrain = model.terrain_encoder
    assert optical is not None and terrain is not None
    assert optical is not terrain
    opt_ids = {id(p) for p in optical.parameters()}
    ter_ids = {id(p) for p in terrain.parameters()}
    assert opt_ids
    assert ter_ids
    assert opt_ids.isdisjoint(ter_ids)


def test_dual_encoder_forward_shape_matches_shared_unet():
    model = DualTrailUNet(
        encoder_name="resnet18",
        encoder_weights=None,
        terrain_encoder_name="resnet18",
        in_channels=7,
        classes=1,
    ).eval()
    sample = torch.zeros(2, 7, 64, 64)
    with torch.inference_mode():
        logits = model(sample)
    assert logits.shape == (2, 1, 64, 64)


def test_dual_encoder_terrain_channels_change_logits():
    torch.manual_seed(0)
    model = DualTrailUNet(
        encoder_name="resnet18",
        encoder_weights=None,
        terrain_encoder_name="resnet18",
        in_channels=7,
        classes=1,
    ).eval()
    x = torch.rand(1, 7, 64, 64)
    with torch.inference_mode():
        full = model(x)
        no_terrain = x.clone()
        no_terrain[:, 4:7] = 0
        dropped_terrain = model(no_terrain)
        no_optical = x.clone()
        no_optical[:, :4] = 0
        dropped_optical = model(no_optical)
    assert not torch.allclose(full, dropped_terrain, atol=1e-5)
    assert not torch.allclose(full, dropped_optical, atol=1e-5)


def test_dual_encoder_gradients_flow_through_both_branches():
    model = DualTrailUNet(
        encoder_name="resnet18",
        encoder_weights=None,
        terrain_encoder_name="resnet18",
        in_channels=7,
        classes=1,
    )
    x = torch.rand(1, 7, 64, 64, requires_grad=True)
    loss = model(x).sum()
    loss.backward()
    opt_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.optical_encoder.parameters())
    ter_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.terrain_encoder.parameters())
    assert opt_grad
    assert ter_grad


def test_split_param_groups_slows_both_dual_encoders():
    model = DualTrailUNet(
        encoder_name="resnet18",
        encoder_weights=None,
        terrain_encoder_name="resnet18",
        in_channels=7,
        classes=1,
    )
    groups = _split_param_groups(model, base_lr=1e-4, weight_decay=1e-4)
    lrs = sorted(group["lr"] for group in groups)
    assert 1e-5 in lrs
    encoder_ids = {id(p) for p in list(model.optical_encoder.parameters()) + list(model.terrain_encoder.parameters())}
    slow = next(group for group in groups if group["lr"] == 1e-5)
    assert encoder_ids.issubset({id(p) for p in slow["params"]})
