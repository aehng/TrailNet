"""Download bounded paired Park City 2021 NAIP / 2018 municipal image chips."""
from __future__ import annotations

import csv
import base64
import hashlib
import io
import json
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import requests
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio.transform import array_bounds
from rasterio.windows import Window


ROOT = Path(__file__).resolve().parent
OUTPUT = Path("/kaggle/working/park-city-imagery")
SERVICE = "https://gis.parkcity.org/server/rest/services/PC2018Imagery/ImageServer/exportImage"
NAIP = Path("/kaggle/input/datasets/elikearl/trailnet-multiaoi-v4/park_city_naip_rgbn.tif")
TARGET_SIZE = 256
MAX_CHIPS = 24
SAMPLES_CSV_B64 = 'c2FtcGxlX2lkLGNlbnRlcl94LGNlbnRlcl95LGNyb3Bfd2lkdGhfcHgsY3JvcF9oZWlnaHRfcHgscmVzb2x1dGlvbl9tClcwMDEsNDQ5Njk0LjMwMDAwMDAwMDA1LDQ0OTY5MzUuNDk5OTk5OTk5LDIxMywyMTMsMC42ClcwMDIsNDUyNDU5LjEwMDAwMDAwMDAzLDQ0OTY5MzUuNDk5OTk5OTk5LDIxMywyMTMsMC42ClcwMDMsNDQ4NTQyLjMwMDAwMDAwMDA1LDQ0OTczOTYuMjk5OTk5OTk5LDIxMywyMTMsMC42ClcwMDQsNDQ5MDAzLjEwMDAwMDAwMDAzLDQ0OTYwMTMuODk5OTk5OTk5LDIxMywyMTMsMC42ClcwMDUsNDUyNjg5LjUwMDAwMDAwMDA2LDQ0OTcxNjUuODk5OTk5OTk5LDIxMywyMTMsMC42ClcwMDYsNDUxMzA3LjEwMDAwMDAwMDAzLDQ0OTYwMTMuODk5OTk5OTk5LDIxMywyMTMsMC42ClcwMDcsNDQ4NTQyLjMwMDAwMDAwMDA1LDQ0OTU1NTMuMSwyMTMsMjEzLDAuNgpXMDA4LDQ0OTY5NC4zMDAwMDAwMDAwNSw0NDk3MTY1Ljg5OTk5OTk5OSwyMTMsMjEzLDAuNgpXMDA5LDQ1MTA3Ni43MDAwMDAwMDAwNyw0NDk1MzIyLjY5OTk5OTk5OSwyMTMsMjEzLDAuNgpXMDEwLDQ1MDg0Ni4zMDAwMDAwMDAwNSw0NDk2OTM1LjQ5OTk5OTk5OSwyMTMsMjEzLDAuNgpXMDExLDQ1MDE1NS4xMDAwMDAwMDAwMyw0NDk0NjMxLjQ5OTk5OTk5OSwyMTMsMjEzLDAuNgpXMDEyLDQ0OTY5NC4zMDAwMDAwMDAwNSw0NDk1MzIyLjY5OTk5OTk5OSwyMTMsMjEzLDAuNgpXMDEzLDQ1MDE1NS4xMDAwMDAwMDAwMyw0NDk4Nzc4LjY5OTk5OTk5OSwyMTMsMjEzLDAuNgpXMDE0LDQ1MTc2Ny45LDQ0OTc4NTcuMSwyMTMsMjEzLDAuNgpXMDE1LDQ1MTA3Ni43MDAwMDAwMDAwNyw0NDk5MjM5LjQ5OTk5OTk5OSwyMTMsMjEzLDAuNgpXMDE2LDQ1MDYxNS45LDQ0OTk3MDAuMjk5OTk5OTk5LDIxMywyMTMsMC42ClcwMTcsNDUwMTU1LjEwMDAwMDAwMDAzLDQ0OTkwMDkuMSwyMTMsMjEzLDAuNgpXMDE4LDQ1MDg0Ni4zMDAwMDAwMDAwNSw0NDk5NDY5Ljg5OTk5OTk5OSwyMTMsMjEzLDAuNgpXMDE5LDQ1MTA3Ni43MDAwMDAwMDAwNyw0NDk3NjI2LjY5OTk5OTk5OSwyMTMsMjEzLDAuNgpXMDIwLDQ1MTk5OC4zMDAwMDAwMDAwNSw0NDk5OTMwLjY5OTk5OTk5OSwyMTMsMjEzLDAuNgpXMDIxLDQ1MjQ1OS4xMDAwMDAwMDAwMyw0NDk4MDg3LjQ5OTk5OTk5OSwyMTMsMjEzLDAuNgpXMDIyLDQ1MTk5OC4zMDAwMDAwMDAwNSw0NDk5NzAwLjI5OTk5OTk5OSwyMTMsMjEzLDAuNgpXMDIzLDQ1MjY4OS41MDAwMDAwMDAwNiw0NDk0ODYxLjg5OTk5OTk5OSwyMTMsMjEzLDAuNgpXMDI0LDQ1MDE1NS4xMDAwMDAwMDAwMyw0NDk5NzAwLjI5OTk5OTk5OSwyMTMsMjEzLDAuNgo='


def callback(status: str, **fields: object) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "callback.json").write_text(json.dumps({"status": status, **fields}, indent=2), encoding="utf-8")


def render_rgb(array: np.ndarray) -> np.ndarray:
    rgb = np.moveaxis(array[:3], 0, -1).astype(np.float32)
    finite = rgb[np.isfinite(rgb)]
    if finite.size:
        lo, hi = np.percentile(finite, (1, 99))
        if hi > lo:
            rgb = np.clip((rgb - lo) / (hi - lo), 0, 1)
    return rgb


def read_samples() -> list[dict[str, str]]:
    data = base64.b64decode(SAMPLES_CSV_B64).decode("utf-8-sig")
    samples = list(csv.DictReader(io.StringIO(data, newline="")))
    samples = samples[:MAX_CHIPS]
    if not samples:
        raise RuntimeError(f"No comparison windows found in {SAMPLES}")
    return samples


def export_candidate(session: requests.Session, bbox: tuple[float, float, float, float]) -> tuple[np.ndarray, np.ndarray, str]:
    params = {
        "bbox": ",".join(f"{value:.6f}" for value in bbox),
        "bboxSR": "32612",
        "imageSR": "32612",
        "size": f"{TARGET_SIZE},{TARGET_SIZE}",
        "format": "tiff",
        "pixelType": "U8",
        "f": "image",
    }
    response = session.get(SERVICE, params=params, timeout=(15, 60))
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "")
    if "json" in content_type.lower() or response.content[:1] == b"{":
        raise RuntimeError(f"ImageServer returned an error payload: {response.text[:400]}")
    with MemoryFile(response.content) as memory:
        with memory.open() as dataset:
            if dataset.count < 3:
                raise RuntimeError(f"Expected at least three imagery bands, got {dataset.count}")
            values = dataset.read()
            masks = dataset.read_masks(indexes=list(range(1, dataset.count + 1))) > 0
            valid = np.logical_and.reduce(masks)
    return values, valid, content_type


def main() -> None:
    started = time.time()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "chips").mkdir(exist_ok=True)
    samples = read_samples()
    records: list[dict[str, object]] = []
    panels: list[tuple[str, np.ndarray, np.ndarray, float, float]] = []

    with rasterio.open(NAIP) as source, requests.Session() as session:
        if source.crs is None or source.crs.to_epsg() != 32612:
            raise RuntimeError(f"Expected Park City NAIP source in EPSG:32612, got {source.crs}")
        current_source_metadata = {
            "count": source.count,
            "dtype": list(source.dtypes),
            "color_interpretation": [value.name for value in source.colorinterp],
            "band_descriptions": list(source.descriptions),
            "nodata": source.nodata,
            "resolution_m": list(source.res),
            "crs": str(source.crs),
            "bounds": list(source.bounds),
            "tags": source.tags(),
            "mask_warning_seen": "Rasterio reported NodataShadowWarning; current-mask fractions need validation against source masks, not numeric zero alone.",
        }
        session.headers.update({"User-Agent": "TrailNet paired image visibility screen; bounded sample; contact project maintainer"})
        for position, row in enumerate(samples, start=1):
            sample_id = row["sample_id"]
            cx, cy = float(row["center_x"]), float(row["center_y"])
            width_px, height_px = int(row["crop_width_px"]), int(row["crop_height_px"])
            resolution = float(row.get("resolution_m") or abs(source.transform.a))
            width_m, height_m = width_px * resolution, height_px * resolution
            left, right = cx - width_m / 2, cx + width_m / 2
            bottom, top = cy - height_m / 2, cy + height_m / 2
            src_window = Window(
                source.index(left, top)[1],
                source.index(left, top)[0],
                width_px,
                height_px,
            )
            window_transform = rasterio.windows.transform(src_window, source.transform)
            bbox = tuple(array_bounds(height_px, width_px, window_transform))
            current = source.read(indexes=[1, 2, 3, 4], window=src_window, boundless=True, fill_value=0)
            current_mask = source.read_masks(indexes=[1, 2, 3, 4], window=src_window, boundless=True) > 0
            current_valid = np.logical_and.reduce(current_mask)
            with MemoryFile() as memory:
                profile = {
                    "driver": "GTiff", "width": width_px, "height": height_px,
                    "count": 4, "dtype": current.dtype, "crs": source.crs,
                    "transform": window_transform,
                }
                with memory.open(**profile) as chip:
                    chip.write(current)
                    chip.write_mask((current_valid.astype(np.uint8) * 255))
                    current_rgb = chip.read(indexes=[1, 2, 3], out_shape=(3, TARGET_SIZE, TARGET_SIZE), resampling=Resampling.bilinear)
            candidate, candidate_valid, content_type = export_candidate(session, bbox)
            np.savez_compressed(OUTPUT / "chips" / f"{sample_id}.npz", current_rgb=current, current_valid=current_valid,
                                candidate=candidate, candidate_valid=candidate_valid, bbox=np.asarray(bbox))
            current_fraction = float(current_valid.mean())
            candidate_mask_fraction = float(candidate_valid.mean())
            candidate_zero_fraction = float(np.all(candidate[:3] == 0, axis=0).mean())
            candidate_nonzero_fraction = 1.0 - candidate_zero_fraction
            panels.append((sample_id, render_rgb(current_rgb), render_rgb(candidate), current_fraction, candidate_zero_fraction))
            records.append({
                "sample_id": sample_id,
                "center_epsg32612_m": [cx, cy],
                "bbox_epsg32612_m": list(bbox),
                "ground_window_m": [width_m, height_m],
                "current_source": "park_city_naip_rgbn.tif (2021 September/November mosaic)",
                "candidate_source": SERVICE,
                "candidate_response_content_type": content_type,
                "current_mask_valid_fraction": round(current_fraction, 6),
                "candidate_mask_valid_fraction": round(candidate_mask_fraction, 6),
                "candidate_rgb_zero_fraction": round(candidate_zero_fraction, 6),
                "candidate_rgb_nonzero_fraction": round(candidate_nonzero_fraction, 6),
                "candidate_zero_return_suspected": bool(candidate_zero_fraction > 0.99),
                "candidate_band_count": int(candidate.shape[0]),
                "candidate_band_order": "not verified in service metadata; first three bands shown as served",
            })
            print(f"CHIP_COMPLETE {position}/{len(samples)} {sample_id}", flush=True)

    for sheet_start in range(0, len(panels), 6):
        sheet = panels[sheet_start:sheet_start + 6]
        fig, axes = plt.subplots(len(sheet), 2, figsize=(12, max(4, len(sheet) * 2.7)), squeeze=False)
        for index, (sample_id, current, candidate, current_valid, candidate_zero) in enumerate(sheet):
            axes[index, 0].imshow(current)
            axes[index, 0].set_title(f"{sample_id} current NAIP 2021 | mask {current_valid:.1%}")
            axes[index, 1].imshow(candidate)
            axes[index, 1].set_title(f"{sample_id} PC2018 bands 1–3 | zero pixels {candidate_zero:.1%}")
            for axis in axes[index]:
                axis.set_xticks([]); axis.set_yticks([])
        fig.suptitle("Same 128 m split-exclusive Park City windows; images only, no labels or predictions", fontsize=12)
        fig.tight_layout()
        sheet_number = sheet_start // 6 + 1
        fig.savefig(OUTPUT / f"paired-contact-sheet-{sheet_number:02d}.png", dpi=130)
        plt.close(fig)

    summary = {
        "status": "complete",
        "started_epoch": started,
        "elapsed_seconds": round(time.time() - started, 1),
        "sample_count": len(records),
        "sample_manifest_source": "24 Park City rows selected from the completed A replay audit_samples.csv; exact recovered-A window centers and dimensions",
        "sample_manifest_sha256": hashlib.sha256(base64.b64decode(SAMPLES_CSV_B64)).hexdigest(),
        "window_size_target_m": 128,
        "candidate_flight_dates": ["2018-06-24", "2018-07-01"],
        "candidate_gsd_m": 0.5,
        "current_gsd_m": 0.6,
        "current_source_metadata": current_source_metadata,
        "review_protocol": "Contact sheets omit labels and predictions; give sheets to a reviewer without the sample key.",
        "human_visibility_review": "pending; contact sheets contain imagery only",
        "snow_shadow_cloud_canopy_scored": False,
        "candidate_rgb_allzero_windows": sum(row["candidate_zero_return_suspected"] for row in records),
        "candidate_zero_filled_pixel_fraction_mean": round(float(np.mean([row["candidate_rgb_zero_fraction"] for row in records])), 6),
        "band_order_verified": False,
        "ml_reuse_permission": "unknown; pixels retrieved only for bounded internal visibility screening",
        "source_item_identity": "current 2021 item IDs reconstructed from STAC; PC2018 service metadata lists flight dates but not sample-specific source IDs",
        "windows": records,
    }
    (OUTPUT / "paired-imagery-screen.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    callback("complete", sample_count=len(records), elapsed_seconds=round(time.time() - started, 1))
    print(f"SCREEN_COMPLETE samples={len(records)} output={OUTPUT}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        callback("failed", error=repr(error))
        raise
