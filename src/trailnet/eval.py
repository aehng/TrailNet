"""Held-out geographic-block metrics and simple OSM novelty count."""
from pathlib import Path

import geopandas as gpd
import torch
from torch.utils.data import DataLoader

from trailnet.config import load_config, resolve_path
from trailnet.data.dataset import TrailTileDataset, read_indexes, spatial_split
from trailnet.models.unet import TrailUNet


def evaluate(config_path=None, checkpoint_path=None):
    config = load_config(config_path); training = config["training"]
    indexes = [resolve_path(config, config["paths"]["tile_dir"]) / a["name"] / "index.csv"
               for a in config["areas"]]
    rows = read_indexes(indexes)
    split = spatial_split(rows, training["block_size_m"], training["val_fraction"],
                          training["test_fraction"], config.get("seed", 42))["test"]
    if not split:
        raise ValueError("No held-out test tiles; adjust spatial split settings or prepare more areas")
    model_cfg = config["model"]; device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TrailUNet(model_cfg["encoder_name"], None, model_cfg["in_channels"],
                      model_cfg["classes"], model_cfg.get("dilated_center", False)).to(device)
    checkpoint_path = resolve_path(config, checkpoint_path or training["checkpoint_path"])
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"]); model.eval()
    tp = fp = fn = 0
    with torch.inference_mode():
        for images, masks in DataLoader(TrailTileDataset(split), batch_size=training["batch_size"]):
            logits = model(images.to(device))
            pred = torch.sigmoid(logits).cpu() >= 0.5; truth = masks >= 0.5
            tp += int((pred & truth).sum()); fp += int((pred & ~truth).sum()); fn += int((~pred & truth).sum())
    iou = tp / (tp + fp + fn) if tp + fp + fn else 1.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
    print(f"held-out tile count={len(split)} IoU={iou:.4f} F1={f1:.4f}")
    predicted_dir = resolve_path(config, config["paths"]["vector_dir"])
    novelty = {}
    for area in config["areas"]:
        pred_path = predicted_dir / f"{area['name']}_trails.geojson"
        osm_path = resolve_path(config, area["trails"])
        if not pred_path.exists() or not osm_path.exists():
            continue
        predicted, existing = gpd.read_file(pred_path), gpd.read_file(osm_path)
        if predicted.crs is None or existing.crs is None:
            continue
        metric_crs = predicted.estimate_utm_crs()
        if metric_crs is None:
            continue
        predicted = predicted.to_crs(metric_crs)
        existing = existing.to_crs(metric_crs); tree = existing.sindex
        tolerance = float(config["postprocess"]["dedup_tolerance_m"])
        unmatched = 0
        for line in predicted.geometry:
            ids = list(tree.query(line.buffer(tolerance)))
            if not any(line.distance(existing.geometry.iloc[int(i)]) <= tolerance for i in ids):
                unmatched += 1
        novelty[area["name"]] = unmatched
    print("predicted line features not matching OSM within tolerance:", novelty)
    return {"iou": iou, "f1": f1, "unmatched_by_area": novelty}


if __name__ == "__main__":
    evaluate()
