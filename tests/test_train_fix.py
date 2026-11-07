"""Regression tests for the v4d collapse: empty-output loss, AMP, device, threshold."""
import math

import geopandas as gpd
import torch
from shapely.geometry import LineString

from trailnet.eval import _usable_geometries
from trailnet.models.unet import dice_bce_loss
from trailnet.train import (
    IMPROVED_PIXEL_WEIGHTS,
    _DEFAULT_THRESHOLDS,
    _cosine_warmup_lambda,
    _fit_batch_size,
    _make_pos_sampler,
    _maybe_write_threshold,
    _planned_batch_size,
    _resolve_pixel_weights,
    _split_param_groups,
    _val_thresholds,
    _write_threshold_to_config,
    confusion_counts,
    metrics_from_counts,
    select_device,
    trail_seg_loss,
)
from trailnet.models.unet import TrailUNet


def _thin_trail_batch(height=64, width=64):
    """One empty tile and one ~3% positive thin-trail tile, matching class imbalance."""
    targets = torch.zeros(2, 1, height, width)
    targets[1, 0, 30:32, :] = 1.0
    weights = torch.ones_like(targets)
    weights[1, 0, 30:32, :] = 8.0
    return targets, weights


def test_trail_seg_loss_ranks_thin_trail_above_all_negative():
    targets, weights = _thin_trail_batch()
    empty_logits = torch.full_like(targets, -6.0)
    trail_logits = empty_logits.clone()
    trail_logits[1, 0, 30:32, :] = 4.0
    empty_loss = trail_seg_loss(empty_logits, targets, weights)
    trail_loss = trail_seg_loss(trail_logits, targets, weights)
    assert torch.isfinite(empty_loss) and torch.isfinite(trail_loss)
    assert trail_loss.item() < empty_loss.item()


def test_batch_dice_and_tversky_rank_thin_trail_above_empty():
    targets = torch.zeros(4, 1, 32, 32)
    targets[0, 0, 15:17, :] = 1.0
    empty_logits = torch.full_like(targets, -8.0)
    trail_logits = empty_logits.clone()
    trail_logits[0, 0, 15:17, :] = 4.0
    weights = torch.ones_like(targets)
    empty_dice = dice_bce_loss(empty_logits, targets, weights, smooth=1.0)
    trail_dice = dice_bce_loss(trail_logits, targets, weights, smooth=1.0)
    empty_new = trail_seg_loss(empty_logits, targets, weights)
    trail_new = trail_seg_loss(trail_logits, targets, weights)
    assert trail_dice.item() < empty_dice.item()
    assert trail_new.item() < empty_new.item()
    assert empty_new.item() - trail_new.item() > 0.05


def test_trail_seg_loss_stays_finite_on_fp16_512_maps():
    logits = torch.zeros(2, 1, 512, 512, dtype=torch.float16)
    targets = torch.zeros_like(logits, dtype=torch.float32)
    targets[:, :, 10:12, :] = 1.0
    weights = torch.ones_like(targets)
    loss = trail_seg_loss(logits, targets, weights)
    assert torch.isfinite(loss)
    assert loss.dtype == torch.float32


def test_trail_seg_loss_path_pixels_get_more_gradient_than_track():
    logits = torch.zeros(1, 1, 1, 2, requires_grad=True)
    targets = torch.ones_like(logits)
    weights = torch.tensor([[[[8.0, 3.0]]]])
    loss = trail_seg_loss(logits, targets, weights)
    loss.backward()
    gradients = logits.grad[0, 0, 0].abs()
    assert gradients[0] > gradients[1]


def test_global_counts_do_not_treat_empty_tiles_as_perfect_f1():
    logits = torch.full((2, 1, 2, 2), -10.0)
    target = torch.zeros(2, 1, 2, 2)
    target[0, 0, 0, 0] = 1.0
    tp, fp, fn = confusion_counts(logits, target, threshold=0.5)
    iou, f1 = metrics_from_counts(tp, fp, fn)
    assert (tp, fp, fn) == (0, 0, 1)
    assert f1 == 0.0
    assert iou == 0.0


def test_pixel_weights_override_when_path_equals_background(capsys):
    resolved = _resolve_pixel_weights({
        "loss": {"pixel_weights": {"background": 1.0, "path": 1.0, "track": 0.30, "uncertain": 0.65}},
    })
    assert resolved == IMPROVED_PIXEL_WEIGHTS
    assert resolved["path"] > resolved["track"] > resolved["background"]
    captured = capsys.readouterr().out
    assert "PIXEL_WEIGHTS_OVERRIDE" in captured


def test_pixel_weights_keep_explicit_path_emphasis():
    configured = {"background": 1.0, "path": 8.0, "track": 3.0, "uncertain": 4.0}
    assert _resolve_pixel_weights({"loss": {"pixel_weights": configured}}) == configured


def test_warmup_then_cosine_reaches_zero_at_last_epoch():
    assert _cosine_warmup_lambda(0, 3, 25) == 1.0 / 3.0
    assert _cosine_warmup_lambda(2, 3, 25) == 1.0
    last = _cosine_warmup_lambda(24, 3, 25)
    assert last == 0.5 * (1.0 + math.cos(math.pi * 21 / 22))
    assert last < 0.02


def test_encoder_param_group_uses_one_tenth_decoder_lr():
    model = TrailUNet("resnet34", None, 7, 1, False)
    groups = _split_param_groups(model, base_lr=1e-4, weight_decay=1e-4)
    lrs = sorted(group["lr"] for group in groups)
    assert lrs == [1e-5, 1e-4]
    encoder_ids = {id(p) for p in model.network.encoder.parameters() if p.requires_grad}
    for group in groups:
        if group["lr"] == 1e-5:
            assert {id(p) for p in group["params"]} == encoder_ids


def test_select_device_uses_xpu_before_cpu():
    device = select_device()
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        assert device.type == "xpu"
    elif torch.cuda.is_available():
        assert device.type == "cuda"
    else:
        assert device.type == "cpu"


def test_fit_batch_size_keeps_cpu_request():
    model = TrailUNet("resnet34", None, 7, 1, False)
    assert _fit_batch_size(model, torch.device("cpu"), 4, 7, tile_size=32) == 4


def test_fit_batch_size_caps_probe_at_requested_not_24():
    model = TrailUNet("resnet34", None, 7, 1, False)
    chosen = _fit_batch_size(
        model, torch.device("cpu"), 12, 7, tile_size=32, max_batch=24
    )
    assert chosen == 12
    assert chosen <= 12


def test_planned_batch_size_keeps_12_on_two_cuda_gpus():
    """Kaggle T4 kernels abort if chosen != 12; 2-GPU must not bump to 16."""
    assert _planned_batch_size("cuda", 2, 12) == 12
    assert _planned_batch_size("cuda", 2, 16) == 16
    assert _planned_batch_size("cuda", 1, 12) is None
    assert _planned_batch_size("xpu", 1, 16) is None


def test_write_threshold_updates_inference_and_postprocess(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("inference:\n  threshold: 0.5\npostprocess:\n  threshold: 0.5\n", encoding="utf-8")
    _write_threshold_to_config(path, 0.35)
    text = path.read_text(encoding="utf-8")
    assert "threshold: 0.35" in text
    assert "threshold: 0.5" not in text


def test_usable_geometries_drops_empty_and_missing_rows():
    frame = gpd.GeoDataFrame(
        {"geometry": [LineString(), None, LineString([(0.0, 0.0), (1.0, 1.0)])]},
        crs="EPSG:4326",
    )
    cleaned = _usable_geometries(frame, "predicted", "kelly_smoke")
    assert cleaned is not None
    assert len(cleaned) == 1
    empty = gpd.GeoDataFrame({"geometry": [LineString(), None]}, crs="EPSG:4326")
    assert _usable_geometries(empty, "OSM", "kelly_smoke") is None


def test_default_val_thresholds_exclude_0_90():
    assert 0.90 not in _DEFAULT_THRESHOLDS
    assert _val_thresholds({}) == _DEFAULT_THRESHOLDS
    assert max(_val_thresholds({})) == 0.60


def test_experiment_val_thresholds_include_0_90():
    grid = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    vals = _val_thresholds({"val_thresholds": grid})
    assert 0.90 in vals
    assert vals[-1] == 0.90


def test_pos_sampler_print_uses_configured_oversample(capsys):
    weights = _make_pos_sampler([10, 0, 3], boost=2.0)
    assert weights == [2.0, 1.0, 2.0]
    captured = capsys.readouterr().out
    assert "oversample=2.0x" in captured
    assert "pos_tiles=2" in captured
    assert "empty_tiles=1" in captured


def test_writeback_threshold_skipped_when_disabled(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("inference:\n  threshold: 0.5\npostprocess:\n  threshold: 0.5\n", encoding="utf-8")
    _maybe_write_threshold(path, 0.80, enabled=False)
    text = path.read_text(encoding="utf-8")
    assert "threshold: 0.5" in text
    assert "threshold: 0.80" not in text


def test_writeback_threshold_runs_when_enabled(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("inference:\n  threshold: 0.5\npostprocess:\n  threshold: 0.5\n", encoding="utf-8")
    _maybe_write_threshold(path, 0.80, enabled=True)
    text = path.read_text(encoding="utf-8")
    assert "threshold: 0.80" in text


def test_confusion_counts_skip_ignore_and_legacy_treats_as_background():
    logits = torch.zeros(1, 1, 1, 3)
    logits[0, 0, 0, 0] = 8.0   # pred 1, truth 1
    logits[0, 0, 0, 1] = 8.0   # pred 1, ignore
    logits[0, 0, 0, 2] = -8.0  # pred 0, truth 0
    target = torch.tensor([[[[1.0, 0.0, 0.0]]]])
    ignore = torch.tensor([[[[False, True, False]]]])
    tp, fp, fn = confusion_counts(logits, target, threshold=0.5, ignore=ignore)
    assert (tp, fp, fn) == (1, 0, 0)
    tp_l, fp_l, fn_l = confusion_counts(logits, target, threshold=0.5, ignore=None)
    assert (tp_l, fp_l, fn_l) == (1, 1, 0)


def test_eval_parse_thresholds_cli_list():
    from trailnet.eval import parse_threshold_list, format_sweep_line
    vals = parse_threshold_list("0.60,0.65")
    assert vals == (0.60, 0.65)
    line = format_sweep_line(0.60, 10, 5, 2, iou=0.5, f1=0.7, legacy_f1=0.6)
    assert "eval_threshold_sweep t=0.60" in line
    assert "f1=0.7000" in line
    area_line = format_sweep_line(0.65, 1, 0, 0, iou=1.0, f1=1.0, area="corner_canyon")
    assert "area=corner_canyon" in area_line
    assert "t=0.65" in area_line
