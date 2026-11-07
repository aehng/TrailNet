"""Indexed seven-channel tile dataset with class-aware pixel weights."""
import csv
import hashlib
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset


def read_indexes(index_paths):
    rows = []
    for index_path in index_paths:
        path = Path(index_path)
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                row["index_dir"] = str(path.parent)
                row["x"], row["y"] = float(row["x"]), float(row["y"])
                rows.append(row)
    return rows


def spatial_split(rows, block_size_m=5000, val_fraction=0.15, test_fraction=0.15, seed=42):
    """Assign whole projected blocks to splits using a stable hash of area/block."""
    if block_size_m <= 0 or val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1:
        raise ValueError("Invalid spatial split fractions or block size")
    result = {"train": [], "val": [], "test": []}
    for row in rows:
        block_x = int(np.floor(row["x"] / block_size_m))
        block_y = int(np.floor(row["y"] / block_size_m))
        key = f"{seed}:{row['area']}:{block_x}:{block_y}".encode()
        value = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / 2**64
        split = "test" if value < test_fraction else ("val" if value < test_fraction + val_fraction else "train")
        result[split].append(row)
    return result


class TrailTileDataset(Dataset):
    def __init__(self, rows, augment=False, pixel_weights=None,
                 track_weight=0.30, uncertain_weight=0.65):
        self.rows = list(rows)
        self.augment = augment
        configured = pixel_weights or {}
        self.path_weight = float(configured.get("path", 1.0))
        self.track_weight = float(configured.get("track", track_weight))
        self.uncertain_weight = float(configured.get("uncertain", uncertain_weight))
        self.background_weight = float(configured.get("background", 1.0))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        base = Path(row["index_dir"])
        with rasterio.open(base / row["image"]) as ds:
            image = ds.read().astype(np.float32)
        with rasterio.open(base / row["mask"]) as ds:
            mask = ds.read(1).astype(np.float32)[None]
        if image.shape[0] != 7:
            raise ValueError(f"Expected 7 input channels, got {image.shape[0]}")
        class_map_path = row.get("class_map")
        if class_map_path:
            with rasterio.open(base / class_map_path) as ds:
                class_map = ds.read(1)
        else:
            # Older binary indexes have no class map; retain their positives as path labels.
            class_map = (mask[0] > 0).astype(np.uint8)
        weights = np.full(mask.shape, self.background_weight, dtype=np.float32)
        weights[0, class_map == 1] = self.path_weight
        weights[0, class_map == 2] = self.track_weight
        weights[0, class_map == 3] = self.uncertain_weight
        # Soft GPS band: fractional mask on background class_map cells.
        soft = (mask[0] > 0) & (mask[0] < 1) & (class_map == 0)
        weights[0, soft] = self.path_weight * mask[0, soft]
        # Explicit ignore (sidewalks / ignore annulus) never supervises.
        weights[0, class_map == 4] = 0.0
        mask[0, class_map == 4] = 0.0
        if self.augment:
            turns = int(torch.randint(0, 4, ()).item())
            image = np.rot90(image, turns, axes=(1, 2)).copy()
            mask = np.rot90(mask, turns, axes=(1, 2)).copy()
            weights = np.rot90(weights, turns, axes=(1, 2)).copy()
            if torch.rand(()) < 0.5:
                image, mask = image[:, :, ::-1].copy(), mask[:, :, ::-1].copy()
                weights = weights[:, :, ::-1].copy()
            if torch.rand(()) < 0.5:
                image, mask = image[:, ::-1, :].copy(), mask[:, ::-1, :].copy()
                weights = weights[:, ::-1, :].copy()
        return torch.from_numpy(image), torch.from_numpy(mask), torch.from_numpy(weights)
