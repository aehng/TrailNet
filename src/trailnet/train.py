"""Train TrailNet using geographic-block train/validation/test partitions."""
import math
import hashlib
import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from trailnet.config import load_config, resolve_path
from trailnet.data.dataset import TrailTileDataset, read_indexes, spatial_split
from trailnet.models.unet import build_model


_DEFAULT_THRESHOLDS = (0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60)
THRESHOLDS = _DEFAULT_THRESHOLDS
IMPROVED_PIXEL_WEIGHTS = {
    "background": 1.0,
    "path": 8.0,
    "track": 3.0,
    "uncertain": 4.0,
}


def _val_thresholds(train_cfg):
    raw = (train_cfg or {}).get("val_thresholds") or _DEFAULT_THRESHOLDS
    vals = tuple(float(x) for x in raw)
    if not vals or min(vals) <= 0 or max(vals) >= 1:
        raise ValueError("training.val_thresholds must be in (0, 1)")
    return vals


def _make_pos_sampler(positive_counts, boost=4.0):
    boost = float(boost)
    pos_tiles = sum(1 for count in positive_counts if count > 0)
    empty_tiles = len(positive_counts) - pos_tiles
    sample_weights = [boost if count > 0 else 1.0 for count in positive_counts]
    print(f"POS_TILE_SAMPLER pos_tiles={pos_tiles} empty_tiles={empty_tiles} oversample={boost}x")
    return sample_weights


def _maybe_write_threshold(config_path, threshold, enabled=False):
    if not enabled:
        return
    _write_threshold_to_config(config_path, threshold)


def select_device():
    """Prefer Intel XPU, then CUDA, then CPU. This box is Arc/XPU, not NVIDIA."""
    xpu = getattr(torch, "xpu", None)
    if xpu is not None and callable(getattr(xpu, "is_available", None)) and xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _accelerator_count(device):
    if device.type == "cuda":
        return torch.cuda.device_count()
    if device.type == "xpu":
        return int(torch.xpu.device_count() or 1)
    return 0


def _manual_seed_all(device, seed):
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    elif device.type == "xpu" and hasattr(torch.xpu, "manual_seed_all"):
        torch.xpu.manual_seed_all(seed)


def _peak_memory_gb(device):
    backend = {"cuda": torch.cuda, "xpu": getattr(torch, "xpu", None)}.get(device.type)
    if backend is None or not hasattr(backend, "max_memory_allocated"):
        return None
    return backend.max_memory_allocated() / (1024 ** 3)


def _device_name(device):
    backend = {"cuda": torch.cuda, "xpu": getattr(torch, "xpu", None)}.get(device.type)
    if backend is None:
        return str(device)
    try:
        return backend.get_device_name(0)
    except Exception:
        return str(device)


def _empty_cache(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "xpu":
        torch.xpu.empty_cache()


def _planned_batch_size(device_type, n_gpus, requested):
    """Fixed batch for multi-GPU CUDA; None means probe with `_fit_batch_size`.

    Never raise above `requested`. The old `max(requested, 16)` path made
    Kaggle 2xT4 sessions abort when abort_if_batch_not=12.
    """
    requested = int(requested)
    n_gpus = int(n_gpus)
    if device_type == "cuda" and n_gpus >= 2:
        aligned = (requested // n_gpus) * n_gpus
        if aligned < n_gpus:
            aligned = n_gpus
        return min(requested, aligned)
    return None


def _fit_batch_size(model, device, requested, in_channels, tile_size=512, max_batch=None):
    """Grow the batch until the device is ~90% full; never exceed `requested`."""
    requested = int(requested)
    if max_batch is None:
        max_batch = requested
    max_batch = min(int(max_batch), requested)
    if device.type not in ("cuda", "xpu"):
        return requested
    backend = torch.cuda if device.type == "cuda" else torch.xpu
    total = float(backend.get_device_properties(0).total_memory)
    was_training = model.training
    model.train()
    best = 1
    best_peak = 0.0
    start = 8 if requested >= 8 else 2
    if start > requested:
        start = requested
    for batch_size in range(start, requested + 1, 2):
        try:
            if hasattr(backend, "reset_peak_memory_stats"):
                backend.reset_peak_memory_stats()
            images = torch.zeros(batch_size, in_channels, tile_size, tile_size, device=device)
            masks = torch.zeros(batch_size, 1, tile_size, tile_size, device=device)
            weights = torch.ones_like(masks)
            model.zero_grad(set_to_none=True)
            logits = model(images)
            loss = trail_seg_loss(logits, masks, weights)
            loss.backward()
            peak = float(backend.max_memory_allocated())
            reserved = float(backend.memory_reserved())
            del images, masks, weights, logits, loss
            model.zero_grad(set_to_none=True)
            _empty_cache(device)
            print(
                f"BATCH_PROBE ok bs={batch_size} "
                f"peak_gb={peak / 1024 ** 3:.2f} reserved_gb={reserved / 1024 ** 3:.2f}"
            )
            if reserved > total * 0.90 or peak > total * 0.88:
                if best == 1:
                    best, best_peak = batch_size, peak
                else:
                    print(f"BATCH_PROBE keep bs={best} for headroom")
                break
            best, best_peak = batch_size, peak
        except Exception as exc:
            print(f"BATCH_PROBE fail bs={batch_size} {type(exc).__name__}: {exc}")
            model.zero_grad(set_to_none=True)
            _empty_cache(device)
            break
    model.train(was_training)
    chosen = max(int(best), 1)
    chosen = min(chosen, requested)
    print(
        f"BATCH_FIT requested={requested} chosen={chosen} "
        f"peak_gb={best_peak / 1024 ** 3:.2f} device={device}"
    )
    return chosen


def confusion_from_binary(pred, truth, ignore=None):
    pred = pred.bool()
    truth = truth.bool()
    if ignore is not None:
        keep = ~ignore.bool()
        pred = pred & keep
        truth = truth & keep
    tp = int((pred & truth).sum().item())
    fp = int((pred & ~truth).sum().item())
    fn = int((~pred & truth).sum().item())
    return tp, fp, fn


def confusion_counts(logits, target, threshold=0.5, ignore=None):
    pred = torch.sigmoid(logits.float()) >= threshold
    truth = target >= 0.5
    return confusion_from_binary(pred, truth, ignore=ignore)


def metrics_from_counts(tp, fp, fn):
    union = tp + fp + fn
    iou = tp / union if union else 1.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
    return iou, f1


def metrics(logits, target, threshold=0.5):
    return metrics_from_counts(*confusion_counts(logits, target, threshold))


def trail_seg_loss(logits, targets, pixel_weights=None, alpha=0.3, beta=0.7,
                   smooth=1.0, focal_gamma=2.0, tversky_gamma=1.33,
                   pos_boost=4.0):
    """Focal BCE + batch-level focal Tversky, computed in float32.

    The v4c collapse happened because:
    1. Per-image soft Dice + smooth=1 gave empty tiles a perfect score, so
       predicting nothing lowered the mean loss (seen as val_loss 0.98 -> 0.36
       while val_tp dropped to 0).
    2. Under AMP, per-image Dice sums of 512x512 maps overflow float16 early
       in training (~0.5 * 262144 > 65504), so Dice gradients were dead and
       unweighted BCE (path weight = background = 1.0) could only push
       probabilities toward the ~1.4% positive prior, i.e. below 0.5.
    3. Once probabilities shrank enough for fp16 Dice to start working, the
       empty-tile term finished the collapse (train_tp = 0 from epoch 8).

    Batch-level Tversky with beta > alpha penalizes false negatives on the
    pooled positives, and float32 math avoids the AMP overflow.
    """
    logits_f = logits.float()
    targets_f = targets.float()
    if pixel_weights is None:
        weights_f = torch.ones_like(targets_f)
    else:
        weights_f = pixel_weights.float()

    # Ignore pixels (weight==0) contribute to neither BCE nor Tversky/Dice.
    # Soft targets in (0,1) are allowed; pos_boost scales by soft confidence.
    valid = weights_f > 0
    if not bool(valid.any()):
        return logits_f.sum() * 0.0

    bce_map = nn.functional.binary_cross_entropy_with_logits(
        logits_f, targets_f.clamp(0.0, 1.0), reduction="none"
    )
    # Continuous confidence: hard positives (~1) get full pos_boost; soft band scales.
    conf = targets_f.clamp(0.0, 1.0)
    bce_w = weights_f * (1.0 + (float(pos_boost) - 1.0) * conf)
    probs = torch.sigmoid(logits_f)
    pt = (conf * probs + (1.0 - conf) * (1.0 - probs)).clamp(0.0, 1.0)
    focal_map = (1.0 - pt).pow(float(focal_gamma)) * bce_map
    denom = bce_w.sum().clamp_min(1e-6)
    focal = (focal_map * bce_w).sum() / denom

    tp = (probs * targets_f * weights_f).sum()
    fp = (probs * (1.0 - targets_f) * weights_f).sum()
    fn = ((1.0 - probs) * targets_f * weights_f).sum()
    tversky = (tp + smooth) / (tp + float(alpha) * fp + float(beta) * fn + smooth)
    tversky_loss = (1.0 - tversky).clamp(0.0, 1.0).pow(float(tversky_gamma))
    return 0.5 * focal + 0.5 * tversky_loss


def _unwrap(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def _encoder_modules(raw):
    modules = []
    shared = getattr(getattr(raw, "network", None), "encoder", None)
    if shared is not None:
        modules.append(shared)
    for name in ("optical_encoder", "terrain_encoder"):
        encoder = getattr(raw, name, None)
        if encoder is not None and encoder is not shared:
            modules.append(encoder)
    return modules


def _split_param_groups(model, base_lr, weight_decay):
    raw = _unwrap(model)
    encoder_params = []
    other_params = []
    encoder_ids = set()
    for encoder in _encoder_modules(raw):
        encoder_ids.update(id(param) for param in encoder.parameters())
    for param in raw.parameters():
        if not param.requires_grad:
            continue
        if id(param) in encoder_ids:
            encoder_params.append(param)
        else:
            other_params.append(param)
    groups = []
    if encoder_params:
        groups.append({"params": encoder_params, "lr": base_lr * 0.1, "weight_decay": weight_decay})
    if other_params:
        groups.append({"params": other_params, "lr": base_lr, "weight_decay": weight_decay})
    if not groups:
        raise RuntimeError("Model has no trainable parameters")
    return groups


def _cosine_warmup_lambda(epoch, warmup_epochs, total_epochs):
    if epoch < warmup_epochs:
        return float(epoch + 1) / float(max(1, warmup_epochs))
    span = max(1, total_epochs - warmup_epochs)
    progress = min(1.0, (epoch - warmup_epochs) / span)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _performance_core_count():
    cpu = os.cpu_count() or 1
    # Ultra 5 226V is 4P+4E. Prefer the P-core count when we can see 8 cores.
    return 4 if cpu >= 8 else max(1, cpu // 2) if cpu > 2 else cpu


def configure_runtime_threads(p_cores=None):
    p_cores = int(p_cores or _performance_core_count())
    os.environ.setdefault("OMP_NUM_THREADS", str(p_cores))
    os.environ.setdefault("MKL_NUM_THREADS", str(p_cores))
    torch.set_num_threads(p_cores)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    return p_cores


def _dataloader_kwargs(num_workers):
    kwargs = {"num_workers": int(num_workers)}
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return kwargs


def _mask_positive_counts(rows, workers=4):
    try:
        import rasterio
    except ImportError:
        return None

    def count_one(row):
        path = Path(row["index_dir"]) / row["mask"]
        if not path.exists():
            return 0
        with rasterio.open(path) as dataset:
            mask = dataset.read(1)
        return int((mask > 0).sum())

    if workers <= 1 or len(rows) < 8:
        return [count_one(row) for row in rows]
    with ThreadPoolExecutor(max_workers=int(workers)) as pool:
        return list(pool.map(count_one, rows))


def _frozen_sampler_counts(rows, path, expected_sha256):
    """Load original eligibility by identity, refusing drift or partial manifests."""
    payload = Path(path).read_bytes()
    if not expected_sha256 or hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("Frozen sampler manifest SHA-256 mismatch")
    records = json.loads(payload)["tiles"]
    identities = [(r["area"], r["image"]) for r in records]
    wanted = [(r["area"], r["image"]) for r in rows]
    if len(set(identities)) != len(identities) or len(set(wanted)) != len(wanted):
        raise ValueError("Duplicate frozen sampler tile identity")
    if set(identities) != set(wanted):
        raise ValueError("Frozen sampler membership differs from TRAIN")
    counts = {}
    for key, record in zip(identities, records):
        count = record["positive_count"]
        if type(count) is not int or count < 0:
            raise ValueError("Invalid original positive count")
        counts[key] = count
    return [counts[key] for key in wanted]


def _resolve_pixel_weights(config):
    configured = dict(config.get("loss", {}).get("pixel_weights") or {})
    path_weight = float(configured.get("path", 1.0))
    background = float(configured.get("background", 1.0))
    if path_weight <= background + 1e-6:
        print(
            "PIXEL_WEIGHTS_OVERRIDE "
            f"from {configured} to {IMPROVED_PIXEL_WEIGHTS} "
            "(path was not weighted above background)"
        )
        return dict(IMPROVED_PIXEL_WEIGHTS)
    return configured


def _write_threshold_to_config(config_path, threshold):
    if not config_path:
        return
    path = Path(config_path)
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    updated = re.sub(r"^([ \t]*threshold: )[0-9.]+$", rf"\g<1>{threshold:.2f}", text, flags=re.M)
    if updated != text:
        path.write_text(updated, encoding="utf-8")


def run_training(config_path=None, area_names=None):
    config = load_config(config_path)
    if area_names is not None:
        config["areas"] = [area for area in config["areas"] if area["name"] in area_names]
        if not config["areas"]:
            raise ValueError("No configured areas selected for training")
    train_cfg = config["training"]
    val_thresholds = _val_thresholds(train_cfg)
    pos_boost = float(config.get("loss", {}).get("pos_boost", 4.0))
    seed = int(config.get("seed", 42))
    p_cores = configure_runtime_threads()
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = select_device()
    _manual_seed_all(device, seed)
    if device.type == "cpu":
        print("WARNING: CUDA and XPU unavailable; training on CPU")
    indexes = [resolve_path(config, config["paths"]["tile_dir"]) / a["name"] / "index.csv"
               for a in config["areas"]]
    missing = [p for p in indexes if not p.exists()]
    if missing:
        raise FileNotFoundError("Prepare tiles first; missing indexes: " + ", ".join(map(str, missing)))
    rows = read_indexes(indexes)
    splits = spatial_split(rows, train_cfg["block_size_m"], train_cfg["val_fraction"],
                           train_cfg["test_fraction"], seed)
    if not splits["train"] or not splits["val"]:
        raise ValueError("Spatial split produced an empty train or validation set; adjust block size/fractions")
    n_gpus = _accelerator_count(device)
    requested_batch = int(train_cfg.get("batch_size", 4))
    model_cfg = config["model"]
    model = build_model(model_cfg).to(device)
    tile_size = int(config.get("tiling", {}).get("tile_size", 512))
    planned = _planned_batch_size(device.type, n_gpus, requested_batch)
    if planned is not None:
        batch_size = planned
        print(
            f"BATCH_FIT requested={requested_batch} chosen={batch_size} "
            f"peak_gb=0.00 device={device} n_gpus={n_gpus} mode=dataparallel"
        )
    else:
        batch_size = _fit_batch_size(
            model, device, requested_batch, int(model_cfg["in_channels"]), tile_size
        )
    required_batch = train_cfg.get("abort_if_batch_not")
    if required_batch is not None and int(batch_size) != int(required_batch):
        raise RuntimeError(
            f"BATCH_FIT abort: chosen={batch_size} required={int(required_batch)}"
        )
    allowed = train_cfg.get("abort_if_batch_not_in")
    if allowed is not None and int(batch_size) not in {int(x) for x in allowed}:
        raise RuntimeError(
            f"BATCH_FIT abort: chosen={batch_size} allowed={list(allowed)}"
        )
    pixel_weights = _resolve_pixel_weights(config)
    train_rows = splits["train"]
    sampler = None
    pos_tiles = empty_tiles = None
    io_workers = int(train_cfg.get("num_workers", _performance_core_count()))
    sampler_manifest = train_cfg.get("frozen_sampler_manifest")
    if sampler_manifest:
        positive_counts = _frozen_sampler_counts(
            train_rows, resolve_path(config, sampler_manifest),
            train_cfg.get("frozen_sampler_sha256"))
    else:
        positive_counts = _mask_positive_counts(train_rows, workers=max(1, io_workers))
    if positive_counts is not None and any(count > 0 for count in positive_counts):
        pos_boost_tiles = float(train_cfg.get("pos_oversample", 4.0))
        sample_weights = _make_pos_sampler(positive_counts, boost=pos_boost_tiles)
        pos_tiles = sum(1 for count in positive_counts if count > 0)
        empty_tiles = len(positive_counts) - pos_tiles
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    loader_kwargs = _dataloader_kwargs(io_workers)
    loaders = {
        "train": DataLoader(
            TrailTileDataset(train_rows, True, pixel_weights),
            batch_size=batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            drop_last=n_gpus >= 2,
            **loader_kwargs,
        ),
        "val": DataLoader(
            TrailTileDataset(splits["val"], pixel_weights=pixel_weights),
            batch_size=batch_size,
            shuffle=False,
            **loader_kwargs,
        ),
    }
    base_lr = float(train_cfg.get("learning_rate", 1e-4))
    if base_lr >= 2.5e-4:
        print(f"LR_OVERRIDE {base_lr} -> 0.0001 (previous 3e-4 run collapsed)")
        base_lr = 1e-4
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    optimizer = torch.optim.AdamW(_split_param_groups(model, base_lr, weight_decay))
    total_epochs = int(train_cfg["epochs"])
    warmup_epochs = int(train_cfg.get("warmup_epochs", 3))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: _cosine_warmup_lambda(epoch, warmup_epochs, total_epochs)
    )
    if device.type == "cuda" and n_gpus >= 2:
        model = nn.DataParallel(model)
        print(f"DATAPARALLEL n_gpus={n_gpus} batch_size={batch_size}")
    amp = bool(train_cfg.get("mixed_precision", True) and device.type in ("cuda", "xpu"))
    scaler_device = device.type if device.type in ("cuda", "xpu") else "cuda"
    scaler = torch.amp.GradScaler(scaler_device, enabled=amp)
    checkpoint = resolve_path(config, train_cfg["checkpoint_path"])
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"TRAIN_SETUP batch_size={batch_size} n_gpus={n_gpus} base_lr={base_lr} "
        f"encoder_lr={base_lr * 0.1} decoder_lr={base_lr} amp={amp} "
        f"epochs={total_epochs} warmup={warmup_epochs} "
        f"num_workers={io_workers} torch_threads={p_cores} "
        f"pixel_weights={pixel_weights} device={device} device_name={_device_name(device)}"
    )
    best_f1 = -1.0
    best_threshold = 0.5
    best_epoch = 0
    epochs_no_improve = 0
    patience = int(train_cfg.get("early_stopping_patience", 7))
    min_delta = float(train_cfg.get("early_stopping_min_delta", 0.0))
    for epoch in range(total_epochs):
        summary = {}
        model.train()
        loss_sum = count = 0
        tp_sum = fp_sum = fn_sum = 0
        skipped = 0
        for images, masks, weights in tqdm(loaders["train"], desc=f"train {epoch + 1}", leave=False):
            images, masks, weights = images.to(device), masks.to(device), weights.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(images)
                loss = trail_seg_loss(logits, masks, weights, pos_boost=pos_boost)
            if not torch.isfinite(loss):
                skipped += 1
                continue
            if amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            tp, fp, fn = confusion_counts(
                logits.detach(), masks, threshold=0.5, ignore=(weights == 0)
            )
            loss_sum += float(loss.detach().item())
            tp_sum += tp
            fp_sum += fp
            fn_sum += fn
            count += 1
        if count == 0:
            raise RuntimeError(f"Every training step was non-finite in epoch {epoch + 1}")
        train_iou, train_f1 = metrics_from_counts(tp_sum, fp_sum, fn_sum)
        summary["train"] = (loss_sum / count, train_iou, train_f1, tp_sum, fp_sum, fn_sum)

        model.eval()
        loss_sum = count = 0
        val_counts = {threshold: [0, 0, 0] for threshold in val_thresholds}
        legacy_counts = {threshold: [0, 0, 0] for threshold in val_thresholds}
        with torch.no_grad():
            for images, masks, weights in tqdm(loaders["val"], desc=f"val {epoch + 1}", leave=False):
                images, masks, weights = images.to(device), masks.to(device), weights.to(device)
                with torch.autocast(device_type=device.type, enabled=amp):
                    logits = model(images)
                    loss = trail_seg_loss(logits, masks, weights, pos_boost=pos_boost)
                if torch.isfinite(loss):
                    loss_sum += float(loss.detach().item())
                    count += 1
                probs = torch.sigmoid(logits.detach().float())
                truth = masks >= 0.5
                ignore = weights == 0
                for threshold in val_thresholds:
                    pred = probs >= threshold
                    tp, fp, fn = confusion_from_binary(pred, truth, ignore=ignore)
                    val_counts[threshold][0] += tp
                    val_counts[threshold][1] += fp
                    val_counts[threshold][2] += fn
                    ltp, lfp, lfn = confusion_from_binary(pred, truth, ignore=None)
                    legacy_counts[threshold][0] += ltp
                    legacy_counts[threshold][1] += lfp
                    legacy_counts[threshold][2] += lfn
        if count == 0:
            raise RuntimeError(f"Every validation step was non-finite in epoch {epoch + 1}")
        scored = []
        for threshold, (tp, fp, fn) in val_counts.items():
            iou, f1 = metrics_from_counts(tp, fp, fn)
            scored.append((f1, iou, threshold, tp, fp, fn))
        scored.sort(reverse=True)
        val_f1, val_iou, val_threshold, val_tp, val_fp, val_fn = scored[0]
        summary["val"] = (loss_sum / count, val_iou, val_f1, val_tp, val_fp, val_fn, val_threshold)
        current_lrs = [group["lr"] for group in optimizer.param_groups]
        print(f"epoch={epoch+1} train_loss={summary['train'][0]:.4f} "
              f"val_loss={summary['val'][0]:.4f} val_iou={summary['val'][1]:.4f} val_f1={summary['val'][2]:.4f}")
        print(
            f"epoch_counts={epoch+1} "
            f"train_tp={summary['train'][3]} train_fp={summary['train'][4]} train_fn={summary['train'][5]} "
            f"val_tp={summary['val'][3]} val_fp={summary['val'][4]} val_fn={summary['val'][5]} "
            f"threshold={val_threshold:.2f} skipped_steps={skipped}"
        )
        print(
            f"epoch_thresholds={epoch+1} "
            + " ".join(
                f"t{threshold:.2f}={metrics_from_counts(*val_counts[threshold])[1]:.4f}"
                for threshold in val_thresholds
            )
            + f" lrs={','.join(f'{lr:.2e}' for lr in current_lrs)}"
        )
        if any(sum(legacy_counts[t]) != sum(val_counts[t]) for t in val_thresholds):
            _, legacy_f1 = metrics_from_counts(*legacy_counts[val_threshold])
            print(
                f"epoch_legacy={epoch+1} threshold={val_threshold:.2f} "
                f"legacy_f1={legacy_f1:.4f} "
                f"legacy_tp={legacy_counts[val_threshold][0]} "
                f"legacy_fp={legacy_counts[val_threshold][1]} "
                f"legacy_fn={legacy_counts[val_threshold][2]}"
            )
        if device.type in ("cuda", "xpu") and epoch == 0:
            peak = _peak_memory_gb(device)
            peak_txt = "NA" if peak is None else f"{peak:.2f}"
            print(
                f"GPU_MEMORY_PEAK_GB={peak_txt} "
                f"device_count={n_gpus} device_name={_device_name(device)}"
            )
        if val_f1 > best_f1 + min_delta:
            best_f1 = val_f1
            best_threshold = float(val_threshold)
            best_epoch = epoch + 1
            epochs_no_improve = 0
            torch.save({
                "model": _unwrap(model).state_dict(),
                "config": model_cfg,
                "epoch": epoch + 1,
                "val_f1": best_f1,
                "val_iou": val_iou,
                "val_tp": val_tp,
                "val_fp": val_fp,
                "val_fn": val_fn,
                "best_threshold": best_threshold,
                "seed": seed,
            }, checkpoint)
            _maybe_write_threshold(
                config_path, best_threshold,
                enabled=bool(train_cfg.get("writeback_threshold", False)),
            )
            print(f"CHECKPOINT_SAVED epoch={epoch+1} val_f1={best_f1:.4f} threshold={best_threshold:.2f}")
        else:
            epochs_no_improve += 1
            if patience > 0 and epochs_no_improve >= patience:
                print(
                    f"EARLY_STOP epoch={epoch+1} best_epoch={best_epoch} "
                    f"best_val_f1={best_f1:.4f} best_threshold={best_threshold:.2f}"
                )
                break
        scheduler.step()
    print(
        f"Best validation F1: {best_f1:.4f}; checkpoint: {checkpoint} "
        f"best_epoch={best_epoch} best_threshold={best_threshold:.2f}"
    )
    print(f"Held-out test blocks reserved: {len(splits['test'])} tiles")
    return checkpoint
