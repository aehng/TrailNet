"""Sliding-window inference over one area's imagery/DTM pair."""
import math

import numpy as np
import rasterio
import torch
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.windows import Window, transform as window_transform
from tqdm import tqdm

from trailnet.config import load_config, resolve_path
from trailnet.data.terrain import derive_terrain
from trailnet.models.unet import TrailUNet


def _input_tile(image_ds, dtm_ds, row, col, size, relief_radius):
    h, w = min(size, image_ds.height-row), min(size, image_ds.width-col)
    window = Window(col, row, size, size)
    imagery = image_ds.read([1, 2, 3, 4], window=window, boundless=True,
                            out_shape=(4, size, size), fill_value=0)
    integer_input = np.issubdtype(imagery.dtype, np.integer)
    scale = (float(np.iinfo(imagery.dtype).max) if integer_input else
             (65535.0 if np.nanmax(imagery) > 255 else (255.0 if np.nanmax(imagery) > 1 else 1.0)))
    imagery = np.clip(imagery / scale, 0, 1)
    transform = window_transform(Window(col, row, size, size), image_ds.transform)
    dtm = np.full((size, size), np.nan, dtype=np.float32)
    reproject(source=rasterio.band(dtm_ds, 1), destination=dtm,
              src_transform=dtm_ds.transform, src_crs=dtm_ds.crs,
              src_nodata=dtm_ds.nodata,
              dst_transform=transform, dst_crs=image_ds.crs,
              resampling=Resampling.bilinear, dst_nodata=np.nan)
    hillshade, slope, _ = derive_terrain(dtm, image_ds.transform.a, image_ds.transform.e, relief_radius)
    return np.concatenate((imagery, hillshade[None], slope[None]), axis=0), transform


def predict_area(config, area, checkpoint_path=None):
    model_cfg = config["model"]
    checkpoint_path = resolve_path(config, checkpoint_path or config["training"]["checkpoint_path"])
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TrailUNet(model_cfg["encoder_name"], None, model_cfg["in_channels"],
                      model_cfg["classes"], model_cfg.get("dilated_center", False)).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"]); model.eval()
    image_path, dtm_path = resolve_path(config, area["imagery"]), resolve_path(config, area["dtm"])
    out_path = resolve_path(config, config["paths"]["prediction_dir"]) / f"{area['name']}_probability.tif"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    size = int(config["tiling"]["tile_size"])
    overlap = int(config["tiling"]["overlap"])
    stride = size - overlap
    radius = config.get("terrain", {}).get("local_relief_radius_px", 25)
    with rasterio.open(image_path) as image_ds, rasterio.open(dtm_path) as dtm_ds:
        if image_ds.count < 4:
            raise ValueError("Inference requires four NAIP bands (R,G,B,NIR)")
        profile = image_ds.profile.copy(); profile.update(count=1, dtype="float32", nodata=0, compress="deflate")
        with rasterio.open(out_path, "w", **profile) as output:
            for row in tqdm(range(0, image_ds.height, stride), desc=area["name"]):
                for col in range(0, image_ds.width, stride):
                    inputs, transform = _input_tile(image_ds, dtm_ds, row, col, size, radius)
                    with torch.inference_mode():
                        batch = torch.from_numpy(inputs[None]).to(device)
                        prob = torch.sigmoid(model(batch))[0, 0].cpu().numpy()
                    h, w = min(size, image_ds.height-row), min(size, image_ds.width-col)
                    top = 0 if row == 0 else overlap // 2
                    left = 0 if col == 0 else overlap // 2
                    bottom = h if row + stride >= image_ds.height else min(h, size - (overlap - overlap // 2))
                    right = w if col + stride >= image_ds.width else min(w, size - (overlap - overlap // 2))
                    if bottom > top and right > left:
                        window = Window(col + left, row + top, right-left, bottom-top)
                        output.write(prob[top:bottom, left:right].astype(np.float32), 1, window=window)
    return out_path
