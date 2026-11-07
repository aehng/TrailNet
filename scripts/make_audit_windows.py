"""Create blinded, no-overlay audit panels from the frozen v4 tile index."""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio


def frozen_spatial_split(rows: list[dict], seed: int = 42, block_size_m: int = 2500) -> dict[str, list[dict]]:
    """Mirror TrailNet's seed-42 split hash without importing torch/XPU runtime."""
    result = {"train": [], "val": [], "test": []}
    for row in rows:
        block_x = int(np.floor(row["x"] / block_size_m))
        block_y = int(np.floor(row["y"] / block_size_m))
        key = f"{seed}:{row['area']}:{block_x}:{block_y}".encode()
        value = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / 2**64
        split_name = "test" if value < 0.15 else ("val" if value < 0.30 else "train")
        result[split_name].append(row)
    return result


def read_indexes(index_paths: list[Path]) -> list[dict]:
    rows = []
    for index_path in index_paths:
        with index_path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                row["index_dir"] = str(index_path.parent)
                row["x"], row["y"] = float(row["x"]), float(row["y"])
                rows.append(row)
    return rows


def stable_order(rows: list[dict], seed: int) -> list[dict]:
    return sorted(rows, key=lambda row: hashlib.sha256(
        f"{seed}:{row['area']}:{row['image']}".encode()).hexdigest())


def _sample(row: dict, crop_px: int) -> tuple[dict, np.ndarray, np.ndarray]:
    base = Path(row["index_dir"])
    with rasterio.open(base / row["image"]) as image_ds:
        width, height = image_ds.width, image_ds.height
        left = max(0, (width - crop_px) // 2)
        top = max(0, (height - crop_px) // 2)
        window = rasterio.windows.Window(left, top, min(crop_px, width-left), min(crop_px, height-top))
        image = image_ds.read(window=window).astype(np.float32)
        valid = np.all(image_ds.read_masks(window=window) > 0, axis=0)
        transform = image_ds.window_transform(window)
        center_x, center_y = transform * (window.width / 2, window.height / 2)
        resolution_m = math.sqrt(abs(image_ds.transform.a * image_ds.transform.e))
    with rasterio.open(base / row["class_map"]) as class_ds:
        labels = class_ds.read(1, window=window)
    info = dict(row)
    info.update({"crop_left_px": left, "crop_top_px": top, "crop_width_px": int(window.width),
                 "crop_height_px": int(window.height), "center_x": center_x, "center_y": center_y,
                 "resolution_m": resolution_m, "valid_fraction": float(valid.mean()) if valid.size else 0.0})
    return info, image, labels


def _center_label_stratum(row: dict) -> str:
    base = Path(row["index_dir"])
    with rasterio.open(base / row["class_map"]) as class_ds:
        width, height = class_ds.width, class_ds.height
        crop_px = round(128 / math.sqrt(abs(class_ds.transform.a * class_ds.transform.e)))
        left, top = max(0, (width - crop_px) // 2), max(0, (height - crop_px) // 2)
        window = rasterio.windows.Window(left, top, min(crop_px, width-left), min(crop_px, height-top))
        labels = class_ds.read(1, window=window)
    return "reference_positive" if np.isin(labels, [1, 2, 3]).any() else "apparently_empty"


def build_audit(index_root: Path, output: Path, seed: int = 42,
                crop_m: float = 128.0, max_windows: int = 48) -> dict:
    index_paths = sorted(index_root.glob("*/index.csv"))
    if not index_paths:
        raise FileNotFoundError(f"No area index.csv files below {index_root}")
    rows = read_indexes(index_paths)
    split = frozen_spatial_split(rows, seed=seed, block_size_m=2500)
    output.mkdir(parents=True, exist_ok=True)
    half = max_windows // 2
    area_order = ["park_city", "corner_canyon", "kelly_canyon", "snowbasin", "teton_pass"]
    allocations = {"park_city": {"train": 12, "val": 12}}
    other_areas = [area for area in area_order if area != "park_city"]
    base, extra = divmod(max_windows - 24, len(other_areas))
    for area in other_areas:
        allocations[area] = {"train": (base + (1 if extra > 0 else 0)) // 2,
                             "val": (base + (1 if extra > 0 else 0)) -
                                    (base + (1 if extra > 0 else 0)) // 2}
        if extra > 0:
            extra -= 1

    selected: list[tuple[dict, np.ndarray, np.ndarray, str]] = []
    shortages = []
    for area in area_order:
        for split_name in ("train", "val"):
            wanted = allocations.get(area, {}).get(split_name, 0)
            candidates = [r for r in split[split_name] if r["area"] == area]
            positive, empty = [], []
            pool = stable_order(candidates, seed)[:max(wanted * 4, wanted)]
            for row in pool:
                if _center_label_stratum(row) == "reference_positive":
                    positive.append(row)
                else:
                    empty.append(row)
            # Preserve positive/empty balance where available, then fill shortages from remaining candidates.
            buckets = [positive, empty]
            chosen = []
            for bucket in buckets:
                quota = math.ceil(wanted / 2)
                chosen.extend(bucket[:quota])
            used = {row["image"] for row in chosen}
            if len(chosen) < wanted:
                chosen.extend(row for row in pool
                              if row["image"] not in used and len(chosen) < wanted)
            if len(chosen) < wanted:
                shortages.append({"area": area, "split": split_name,
                                  "requested": wanted, "available": len(chosen)})
            for row in chosen:
                info, image, labels = _sample(row, crop_px=round(crop_m / 0.6))
                selected.append((info, image, labels, split_name))

    # Recompute true crop supervision stats, stored separately from the blinded contact sheets.
    key_rows = []
    for number, (info, image, labels, split_name) in enumerate(selected, 1):
        key_rows.append({"sample_id": f"W{number:03d}", "area": info["area"], "split": split_name,
                         "image": info["image"], "center_x": info["center_x"], "center_y": info["center_y"],
                         "crop_width_px": info["crop_width_px"], "crop_height_px": info["crop_height_px"],
                         "resolution_m": info["resolution_m"], "image_mask_valid_fraction": round(info["valid_fraction"], 6),
                         "class_1_px": int((labels == 1).sum()),
                         "class_2_px": int((labels == 2).sum()), "class_3_px": int((labels == 3).sum()),
                         "ignore_px": int((labels == 4).sum()), "background_px": int((labels == 0).sum()),
                         "selection_stratum": "reference_positive" if np.isin(labels, [1, 2, 3]).any() else "apparently_empty"})

    with (output / "audit_samples.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(key_rows[0]) if key_rows else ["sample_id"])
        writer.writeheader(); writer.writerows(key_rows)

    # The key is intentionally separate: sheets contain RGB, CIR and relief only, no labels/predictions.
    cols = 4
    for sheet_index, start in enumerate(range(0, len(selected), 16), 1):
        batch = selected[start:start+16]
        fig, axes = plt.subplots(len(batch), 3, figsize=(11, max(3, len(batch)*2.1)), squeeze=False)
        for local_index, (info, image, _labels, _split) in enumerate(batch):
            row_index = start + local_index
            rgb = np.moveaxis(image[:3], 0, -1)
            rgb = np.clip(rgb / max(1.0, float(np.nanpercentile(rgb, 99))), 0, 1)
            cir = np.moveaxis(image[[3, 0, 1]], 0, -1)
            cir = np.clip(cir / max(1.0, float(np.nanpercentile(cir, 99))), 0, 1)
            relief = image[6]
            axes[local_index, 0].imshow(rgb); axes[local_index, 1].imshow(cir)
            axes[local_index, 2].imshow(relief, cmap="gray")
            for axis, label in zip(axes[local_index], ("RGB", "NIR/R/G", "3DEP relief")):
                axis.set_title(label, fontsize=8); axis.set_xticks([]); axis.set_yticks([])
            axes[local_index, 0].set_ylabel(f"W{row_index+1:03d}", rotation=0, labelpad=20, fontsize=8)
        fig.tight_layout()
        fig.savefig(output / f"contact_sheet_{sheet_index:02d}.png", dpi=130)
        plt.close(fig)

    return {"seed": seed, "block_size_m": 2500, "index_root": str(index_root),
            "grid": "seed-42 A-compatible index tile centers; center crop, fixed area/split allocation",
            "crop_target_m": crop_m, "requested_windows": max_windows,
            "sampled_windows": len(selected), "shortages": shortages,
            "tile_raster_pixel_identity_verified": False,
            "counts_by_area_split_stratum": {
                area: {split_name: {stratum: sum(1 for item in key_rows
                    if item["area"] == area and item["split"] == split_name and item["selection_stratum"] == stratum)
                    for stratum in ("reference_positive", "apparently_empty")}
                    for split_name in ("train", "val")}
                for area in area_order},
            "limitations": ["Selection is from existing tiled data and frozen seed-42 spatial split.",
                            "No independent human adjudication; class-map counts are exposed only in audit_samples.csv.",
                            "Snow/cloud/shadow/canopy visibility must be reviewed from the no-overlay contact sheets.",
                            "Tile rasters were sampled from the local tiles_v4 folder; only index CSV hashes match recovered A, not per-tile raster hashes.",
                            "Center-point selection is a first-pass sampler, not a substitute for precise block-footprint clipping."]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("experiments/overnight-20261107/audit"))
    args = parser.parse_args()
    summary = build_audit(args.index_root, args.output)
    (args.output / "audit_window_summary.json").write_text(__import__("json").dumps(summary, indent=2), encoding="utf-8")
    print(f"WINDOWS {summary['sampled_windows']}/{summary['requested_windows']}; shortages={len(summary['shortages'])}")


if __name__ == "__main__":
    main()
