"""Held-out geographic-block metrics and simple OSM novelty count."""
from pathlib import Path
import math

import geopandas as gpd
import torch
from torch.utils.data import DataLoader

from trailnet.config import load_config, resolve_path
from trailnet.data.dataset import TrailTileDataset, read_indexes, spatial_split
from trailnet.models.unet import build_model
from trailnet.train import confusion_from_binary, metrics_from_counts, select_device


def parse_threshold_list(text):
    if text is None or str(text).strip() == "":
        return None
    vals = tuple(float(x.strip()) for x in str(text).split(",") if x.strip())
    if not vals or min(vals) <= 0 or max(vals) >= 1:
        raise ValueError("thresholds must be in (0, 1)")
    return vals


def format_sweep_line(threshold, tp, fp, fn, iou, f1, legacy_f1=None, legacy_iou=None, area=None):
    area_bit = f" area={area}" if area else ""
    line = (
        f"eval_threshold_sweep{area_bit} t={threshold:.2f} tp={tp} fp={fp} fn={fn} "
        f"iou={iou:.4f} f1={f1:.4f}"
    )
    if legacy_f1 is not None:
        line += f" legacy_f1={legacy_f1:.4f}"
    if legacy_iou is not None:
        line += f" legacy_iou={legacy_iou:.4f}"
    return line


def _has_finite_bounds(frame) -> bool:
    try:
        bounds = frame.total_bounds
    except Exception:
        return False
    if bounds is None:
        return False
    try:
        values = [float(value) for value in bounds]
    except (TypeError, ValueError):
        return False
    return len(values) == 4 and all(math.isfinite(value) for value in values)


def _usable_geometries(frame, label, area_name):
    """Drop empty/invalid/NaN geometries; return a cleaned frame or None to skip."""
    if frame is None or frame.empty:
        print(f"skip novelty for {area_name}: {label} GeoDataFrame is empty")
        return None
    geometry = frame.geometry
    mask = geometry.notna()
    try:
        mask = mask & ~geometry.is_empty
    except Exception:
        pass
    try:
        mask = mask & geometry.is_valid
    except Exception:
        pass
    cleaned = frame.loc[mask].copy()
    dropped = len(frame) - len(cleaned)
    if cleaned.empty:
        print(
            f"skip novelty for {area_name}: no valid {label} geometries remain "
            f"(dropped {dropped} empty/invalid/NaN rows)"
        )
        return None
    if not _has_finite_bounds(cleaned):
        print(
            f"skip novelty for {area_name}: {label} total_bounds has NaN/None "
            f"({getattr(cleaned, 'total_bounds', None)})"
        )
        return None
    if dropped:
        print(f"novelty {area_name}: dropped {dropped} empty/invalid {label} geometries")
    return cleaned


def evaluate(config_path=None, checkpoint_path=None, thresholds=None):
    config = load_config(config_path); training = config["training"]
    indexes = [resolve_path(config, config["paths"]["tile_dir"]) / a["name"] / "index.csv"
               for a in config["areas"]]
    rows = read_indexes(indexes)
    split = spatial_split(rows, training["block_size_m"], training["val_fraction"],
                          training["test_fraction"], config.get("seed", 42))["test"]
    if not split:
        raise ValueError("No held-out test tiles; adjust spatial split settings or prepare more areas")
    model_cfg = config["model"]; device = select_device()
    model = build_model(model_cfg, encoder_weights=None).to(device)
    checkpoint_path = resolve_path(config, checkpoint_path or training["checkpoint_path"])
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"]); model.eval()
    threshold = float(checkpoint.get("best_threshold", config.get("inference", {}).get("threshold", 0.5)))
    print(f"eval_threshold={threshold:.2f} source={'checkpoint' if 'best_threshold' in checkpoint else 'config'}")
    if thresholds is None:
        sweep = (threshold,)
    else:
        sweep = tuple(float(t) for t in thresholds)
    dataset = TrailTileDataset(split, pixel_weights=config.get("loss", {}).get("pixel_weights", {}))
    batch_size = max(1, int(config.get("inference", {}).get("batch_size", training.get("batch_size", 4))))
    global_skip = {t: [0, 0, 0] for t in sweep}
    global_legacy = {t: [0, 0, 0] for t in sweep}
    area_skip = {}
    area_legacy = {}
    with torch.inference_mode():
        for start in range(0, len(split), batch_size):
            chunk = split[start:start + batch_size]
            items = [dataset[start + i] for i in range(len(chunk))]
            images = torch.stack([item[0] for item in items]).to(device)
            masks = torch.stack([item[1] for item in items])
            weights = torch.stack([item[2] for item in items])
            probs = torch.sigmoid(model(images)).cpu()
            truth = masks >= 0.5
            ignore = weights == 0
            for j, row in enumerate(chunk):
                area = row.get("area", "unknown")
                area_skip.setdefault(area, {t: [0, 0, 0] for t in sweep})
                area_legacy.setdefault(area, {t: [0, 0, 0] for t in sweep})
                p = probs[j:j + 1]
                m = truth[j:j + 1]
                ign = ignore[j:j + 1]
                for t in sweep:
                    pred = p >= t
                    tp, fp, fn = confusion_from_binary(pred, m, ignore=ign)
                    global_skip[t][0] += tp; global_skip[t][1] += fp; global_skip[t][2] += fn
                    area_skip[area][t][0] += tp; area_skip[area][t][1] += fp; area_skip[area][t][2] += fn
                    ltp, lfp, lfn = confusion_from_binary(pred, m, ignore=None)
                    global_legacy[t][0] += ltp; global_legacy[t][1] += lfp; global_legacy[t][2] += lfn
                    area_legacy[area][t][0] += ltp; area_legacy[area][t][1] += lfp; area_legacy[area][t][2] += lfn
    primary = threshold if threshold in global_skip else sweep[0]
    tp, fp, fn = global_skip[primary]
    iou, f1 = metrics_from_counts(tp, fp, fn)
    print(f"held-out tile count={len(split)} IoU={iou:.4f} F1={f1:.4f} threshold={primary:.2f} tp={tp} fp={fp} fn={fn}")
    if thresholds is not None or any(sum(global_legacy[t]) != sum(global_skip[t]) for t in sweep):
        for t in sweep:
            tp, fp, fn = global_skip[t]
            iou, f1 = metrics_from_counts(tp, fp, fn)
            ltp, lfp, lfn = global_legacy[t]
            liou, lf1 = metrics_from_counts(ltp, lfp, lfn)
            print(format_sweep_line(t, tp, fp, fn, iou, f1, legacy_f1=lf1, legacy_iou=liou))
            for area in sorted(area_skip):
                atp, afp, afn = area_skip[area][t]
                aiou, af1 = metrics_from_counts(atp, afp, afn)
                print(format_sweep_line(t, atp, afp, afn, aiou, af1, area=area))
    predicted_dir = resolve_path(config, config["paths"]["vector_dir"])
    novelty = {}
    for area in config["areas"]:
        # Prefer dedicated novelty product; else count unmatched from unfiltered trails.
        novelty_path = predicted_dir / f"{area['name']}_novelty.geojson"
        pred_path = novelty_path if novelty_path.exists() else predicted_dir / f"{area['name']}_trails.geojson"
        osm_path = resolve_path(config, area["trails"])
        if not pred_path.exists() or not osm_path.exists():
            print(f"skip novelty for {area['name']}: missing pred or OSM file")
            continue
        try:
            predicted, existing = gpd.read_file(pred_path), gpd.read_file(osm_path)
            if predicted.crs is None or existing.crs is None:
                print(f"skip novelty for {area['name']}: missing CRS on predicted or OSM layer")
                continue
            predicted = _usable_geometries(predicted, "predicted", area["name"])
            existing = _usable_geometries(existing, "OSM", area["name"])
            if predicted is None:
                continue
            if novelty_path.exists():
                # File already novelty-filtered at vectorize time.
                novelty[area["name"]] = int(len(predicted))
                continue
            if existing is None:
                novelty[area["name"]] = int(len(predicted))
                print(
                    f"novelty {area['name']}: OSM layer unusable; "
                    f"counting all {novelty[area['name']]} predicted features as unmatched"
                )
                continue
            metric_crs = predicted.estimate_utm_crs()
            if metric_crs is None:
                print(f"skip novelty for {area['name']}: estimate_utm_crs returned None")
                continue
            predicted = predicted.to_crs(metric_crs)
            existing = existing.to_crs(metric_crs)
            tree = existing.sindex
            tolerance = float(config["postprocess"]["dedup_tolerance_m"])
            unmatched = 0
            for line in predicted.geometry:
                if line is None or line.is_empty:
                    continue
                ids = list(tree.query(line.buffer(tolerance)))
                if not any(line.distance(existing.geometry.iloc[int(i)]) <= tolerance for i in ids):
                    unmatched += 1
            novelty[area["name"]] = unmatched
        except Exception as exc:
            print(f"skip novelty for {area['name']}: {type(exc).__name__}: {exc}")
            continue
    print("predicted line features not matching OSM within tolerance:", novelty)
    return {"iou": iou, "f1": f1, "unmatched_by_area": novelty}


if __name__ == "__main__":
    evaluate()
