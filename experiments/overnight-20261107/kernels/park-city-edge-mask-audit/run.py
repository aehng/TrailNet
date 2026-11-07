"""Read-only edge-mask audit on a Kaggle regeneration of the A Park City inputs."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

WORK = Path("/kaggle/working")
INPUT_ROOT = Path("/kaggle/input")
PROJECT = Path("/kaggle/temp/TrailNet-edge-audit")
CONFIG_NAME = "config_exp_t4e2_a_ignore5.yaml"
RASTER_SLUG = "trailnet-multiaoi-v4"
CODE_SLUG = "trailnet-t4e-code"
TOP_TIMEOUT_SECONDS = 45 * 60
COMMAND_TIMEOUT_SECONDS = 30 * 60
EXPECTED_INDEX_SHA256 = "b288e2b19654911aee4b81b5d51c1679affc3227a75b9964c0eb359a834a8ba2"
AREA_NAMES = ["corner_canyon", "kelly_canyon", "park_city", "snowbasin", "teton_pass"]
EXPECTED_INDEX_HASHES = {
    "corner_canyon": "277eac2dad6bdc33386eba744838a82fccea0eea0cf9e90f69a83faee7ed2411",
    "kelly_canyon": "60bb9f2c139cfaa47c6de7fda1121d325bfb4807ff1c056ad2dbb2d295a3f63d",
    "park_city": "b288e2b19654911aee4b81b5d51c1679affc3227a75b9964c0eb359a834a8ba2",
    "snowbasin": "250e16679bacb0a0d20d1160ad85e1724718a20576e3b8a19535eeb126732500",
    "teton_pass": "f04eff6be5a833ad08d2a882c3e7a88eaf16b53cef4a948eea69ab735b50984f",
}


def log(message: str) -> None:
    print(message, flush=True)
    with (WORK / "edge-audit.log").open("a", encoding="utf-8") as stream:
        stream.write(message + "\n")


def callback(stage: str, status: str, **extra) -> None:
    directory = WORK / "callbacks"
    directory.mkdir(parents=True, exist_ok=True)
    record = {"stage": stage, "status": status,
              "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **extra}
    final = directory / f"edge-audit-{stage}.callback.json"
    temporary = final.with_suffix(final.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2), encoding="utf-8")
    temporary.replace(final)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_mount(slug: str) -> Path:
    direct = INPUT_ROOT / slug
    if direct.is_dir():
        return direct
    matches = [path for path in INPUT_ROOT.rglob(slug) if path.is_dir()]
    if matches:
        return matches[0]
    visible = sorted(path.name for path in INPUT_ROOT.iterdir()) if INPUT_ROOT.is_dir() else []
    raise FileNotFoundError(f"dataset {slug} not mounted; visible={visible}")


def find_file(root: Path, name: str) -> Path | None:
    direct = root / name
    if direct.is_file():
        return direct
    return next(iter(root.rglob(name)), None)


def stage_inputs() -> None:
    rasters, code_mount = find_mount(RASTER_SLUG), find_mount(CODE_SLUG)
    source_roots = [path for path in (code_mount, code_mount / "trailnet_code")
                    if (path / "src").is_dir()]
    if source_roots:
        source_root = source_roots[0]
        for name in ("src", "scripts", "requirements.txt", "label_overrides"):
            source = source_root / name
            if source.is_dir():
                shutil.copytree(source, PROJECT / name, dirs_exist_ok=True)
            elif source.is_file():
                shutil.copy2(source, PROJECT / name)
        yaml_root = code_mount if any(code_mount.glob("config*.yaml")) else source_root
        for path in yaml_root.glob("config*.yaml"):
            shutil.copy2(path, PROJECT / path.name)
    else:
        raise FileNotFoundError("TrailNet source tree missing from code dataset")

    if not (PROJECT / "label_overrides").is_dir():
        archive = find_file(code_mount, "label_overrides.zip")
        if archive is not None:
            with zipfile.ZipFile(archive) as bundle:
                bundle.extractall(PROJECT)

    if not (PROJECT / CONFIG_NAME).is_file():
        raise FileNotFoundError(f"missing {CONFIG_NAME} on code dataset")
    for area in AREA_NAMES:
        if not (PROJECT / "label_overrides" / area / "trails.gpkg").is_file():
            raise FileNotFoundError(f"{area} A label override missing")
        data_dir = PROJECT / "data" / area
        data_dir.mkdir(parents=True, exist_ok=True)
        for stem, suffix in (("naip_rgbn", ".tif"), ("3dep_dtm", ".tif"), ("trails", ".gpkg")):
            source = find_file(rasters, f"{area}_{stem}{suffix}")
            if source is None:
                raise FileNotFoundError(f"missing raw A input {area}_{stem}{suffix}")
            shutil.copy2(source, data_dir / f"{stem}{suffix}")
        shutil.copy2(PROJECT / "label_overrides" / area / "trails.gpkg",
                     data_dir / "trails.gpkg")
    module = (PROJECT / "src" / "trailnet" / "data" / "tiles.py").read_text(encoding="utf-8")
    if "def _write_tile" not in module:
        raise RuntimeError("unexpected tile generator source")
    log("INPUTS_STAGED from A source datasets; no model/checkpoint mounted")


def run_prepare_data() -> None:
    missing = [name for name, module in (("rasterio", "rasterio"), ("geopandas", "geopandas"),
                                         ("pyyaml", "yaml"))
               if importlib.util.find_spec(module) is None]
    if missing:
        command = [sys.executable, "-m", "pip", "install", "-q", *missing]
        install = subprocess.run(command, cwd=PROJECT, text=True, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, timeout=COMMAND_TIMEOUT_SECONDS)
        if install.returncode:
            raise RuntimeError(f"dependency install failed: {install.stdout[-3000:]}")
    command = [sys.executable, "scripts/prepare_data.py", "--config", CONFIG_NAME,
               "--all-areas"]
    started = time.monotonic()
    process = subprocess.Popen(command, cwd=PROJECT, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, errors="replace")
    try:
        output, _ = process.communicate(timeout=COMMAND_TIMEOUT_SECONDS)
        timed_out = False
    except subprocess.TimeoutExpired:
        process.kill()
        output, _ = process.communicate()
        timed_out = True
    for line in output.splitlines():
        log(line)
    callback("prepare-data", "timed_out" if timed_out else
             ("complete" if process.returncode == 0 else "failed"),
             elapsed_seconds=round(time.monotonic() - started, 3),
             timeout_seconds=COMMAND_TIMEOUT_SECONDS, exit_code=process.returncode,
             timed_fallback_triggered=timed_out, output_tail=output[-4000:])
    if timed_out:
        raise TimeoutError("prepare_data timed out after 30 minutes")
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command, output=output)


def audit() -> dict:
    import numpy as np
    import rasterio
    from trailnet.config import load_config, resolve_path

    config = load_config(CONFIG_NAME)
    seed = int(config.get("seed", 42))
    training = config["training"]
    block_size = float(training["block_size_m"])
    val_fraction = float(training["val_fraction"])
    test_fraction = float(training["test_fraction"])
    background_weight = float(config.get("loss", {}).get("pixel_weights", {}).get("background", 1.0))
    area_results = {}
    total_tiles = {name: 0 for name in ("train", "val", "test")}

    for area_name in AREA_NAMES:
        area = next(row for row in config["areas"] if row["name"] == area_name)
        source_path = resolve_path(config, area["imagery"])
        tile_dir = resolve_path(config, config["paths"]["tile_dir"]) / area_name
        index_path = tile_dir / "index.csv"
        actual_index_hash = sha256(index_path)
        if actual_index_hash != EXPECTED_INDEX_HASHES[area_name]:
            raise RuntimeError(f"{area_name} index mismatch: {actual_index_hash}")
        with rasterio.open(source_path) as source:
            bounds = source.bounds
            source_transform = source.transform
            source_meta = {"sha256": sha256(source_path), "width": source.width,
                           "height": source.height, "crs": str(source.crs),
                           "bounds": [bounds.left, bounds.bottom, bounds.right, bounds.top],
                           "resolution": list(source.res), "nodata": source.nodata,
                           "colorinterp": [item.name for item in source.colorinterp],
                           "mask_flags": [[item.name for item in flags]
                                          for flags in source.mask_flag_enums]}
        with index_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        tile_meta = []
        for row in rows:
            with rasterio.open(tile_dir / row["image"]) as image_ds:
                height, width = image_ds.height, image_ds.width
                transform = image_ds.transform
                row_off = int(round((source_transform.f - transform.f) / -source_transform.e))
                col_off = int(round((transform.c - source_transform.c) / source_transform.a))
            x, y = float(row["x"]), float(row["y"])
            bx, by = math.floor(x / block_size), math.floor(y / block_size)
            score = int.from_bytes(hashlib.sha256(
                f"{seed}:{area_name}:{bx}:{by}".encode()).digest()[:8], "big") / 2**64
            split = "test" if score < test_fraction else (
                "val" if score < test_fraction + val_fraction else "train")
            total_tiles[split] += 1
            tile_meta.append({"row": row, "height": height, "width": width,
                              "row_off": row_off, "col_off": col_off,
                              "split": split, "block": f"{bx},{by}"})

        max_rows = max(source_meta["height"], max(item["row_off"] + item["height"] for item in tile_meta))
        max_cols = max(source_meta["width"], max(item["col_off"] + item["width"] for item in tile_meta))
        positive_union = {name: np.zeros((max_rows, max_cols), dtype=bool)
                          for name in ("train", "val", "test")}
        area_result = {"tile_index_sha256": actual_index_hash,
                       "tile_index_matches_recovered_a": True,
                       "source": source_meta,
                       "tile_count": len(rows),
                       "split_tile_counts": {name: sum(1 for item in tile_meta if item["split"] == name)
                                             for name in ("train", "val", "test")},
                       "edge_tile_counts": {name: 0 for name in ("train", "val", "test")},
                       "positive_edge_tile_counts": {name: 0 for name in ("train", "val", "test")},
                       "positive_edge_blocks": {name: [] for name in ("train", "val", "test")},
                       "repeated_tile_pixel_counts": {name: {"outside": 0, "positive_outside": 0,
                                                              "class0_background_outside": 0,
                                                              "class4_ignore_outside": 0}
                                                     for name in ("train", "val", "test")},
                       "positive_edge_tiles": []}
        block_sets = {name: set() for name in ("train", "val", "test")}
        for item in tile_meta:
            row, height, width = item["row"], item["height"], item["width"]
            row_off, col_off, split = item["row_off"], item["col_off"], item["split"]
            with rasterio.open(tile_dir / row["image"]) as image_ds:
                rr, cc = np.indices((height, width))
                x = image_ds.transform.c + (cc + 0.5) * image_ds.transform.a
                y = image_ds.transform.f + (rr + 0.5) * image_ds.transform.e
                outside = ((x < bounds.left) | (x >= bounds.right) |
                           (y < bounds.bottom) | (y >= bounds.top))
                if outside.any():
                    optical = image_ds.read([1, 2, 3, 4])
                else:
                    optical = None
            with rasterio.open(tile_dir / row["class_map"]) as class_ds:
                classes = class_ds.read(1)
            with rasterio.open(tile_dir / row["mask"]) as mask_ds:
                targets = mask_ds.read(1)
            positive = np.isin(classes, [1, 2, 3]) | (targets > 0)
            target_map = positive_union[split]
            target_map[row_off:row_off + height, col_off:col_off + width] |= positive
            if not outside.any():
                continue
            outside_positive = int((positive & outside).sum())
            counts = area_result["repeated_tile_pixel_counts"][split]
            counts["outside"] += int(outside.sum())
            counts["positive_outside"] += outside_positive
            counts["class0_background_outside"] += int(((classes == 0) & outside).sum())
            counts["class4_ignore_outside"] += int(((classes == 4) & outside).sum())
            area_result["edge_tile_counts"][split] += 1
            if outside_positive:
                area_result["positive_edge_tile_counts"][split] += 1
                block_sets[split].add(item["block"])
                area_result["positive_edge_tiles"].append({"image": row["image"],
                                                             "split": split,
                                                             "block": item["block"],
                                                             "outside_pixels": int(outside.sum()),
                                                             "positive_outside": outside_positive})
            if optical is not None:
                counts["optical_allzero_outside"] = counts.get("optical_allzero_outside", 0) + int(
                    ((np.all(optical == 0, axis=0)) & outside).sum())

        for name in ("train", "val", "test"):
            area_result["positive_edge_blocks"][name] = sorted(block_sets[name])
        source_inside = np.zeros((max_rows, max_cols), dtype=bool)
        source_inside[:source_meta["height"], :source_meta["width"]] = True
        area_result["unique_target_pixels_by_split"] = {}
        for name, pixels in positive_union.items():
            total = int(pixels.sum())
            inside = int((pixels & source_inside).sum())
            outside = total - inside
            area_result["unique_target_pixels_by_split"][name] = {
                "all": total, "inside_source_footprint": inside,
                "outside_source_footprint": outside,
                "outside_fraction_of_unique_targets": round(outside / total, 8) if total else 0.0,
                "unique_outside_background_weight_equivalent": outside * background_weight,
            }
        any_split_positive = positive_union["train"] | positive_union["val"] | positive_union["test"]
        area_result["unique_target_pixels_by_split"]["all_splits_union"] = {
            "all": int(any_split_positive.sum()),
            "outside_source_footprint": int((any_split_positive & ~source_inside).sum())}
        area_results[area_name] = area_result

    if total_tiles["val"] != 172 or total_tiles["test"] != 358 or total_tiles["train"] != 1745:
        raise RuntimeError(f"frozen split count mismatch: {total_tiles}")
    return {"status": "complete", "experiment": "A-source all-AOI edge-mask audit",
            "no_train": True, "heldout_scored": False,
            "project_config": CONFIG_NAME, "split_seed": seed,
            "block_size_m": block_size, "val_fraction": val_fraction,
            "test_fraction": test_fraction, "background_weight": background_weight,
            "total_tiles_by_split": total_tiles, "areas": area_results,
            "interpretation": "Fresh A-source tile regeneration, not a pixel-hash comparison with archived tensors. All five generated index hashes match recovered A. Counts are stratified by the frozen split and include unique positive-target pixel unions within each split plus repeated tile-pixel counts."
            }


def top_timeout() -> None:
    callback("final", "timed_out", timeout_seconds=TOP_TIMEOUT_SECONDS,
             timed_fallback_triggered=True)
    os._exit(124)


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    timer = threading.Timer(TOP_TIMEOUT_SECONDS, top_timeout)
    timer.daemon = True
    timer.start()
    started = time.monotonic()
    callback("start", "running", timeout_seconds=TOP_TIMEOUT_SECONDS,
             training_launched=False, gpu_enabled=False)
    try:
        stage_inputs()
        run_prepare_data()
        sys.path.insert(0, str(PROJECT / "src"))
        result = audit()
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        output = WORK / "park-city-edge-mask-audit.json"
        temporary = output.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
        temporary.replace(output)
        callback("final", "complete", elapsed_seconds=result["elapsed_seconds"],
                 timed_fallback_triggered=False, total_tiles_by_split=result["total_tiles_by_split"])
        log(f"EDGE_AUDIT_COMPLETE areas={len(result['areas'])} splits={result['total_tiles_by_split']}")
    except Exception as exc:
        callback("final", "failed", error=repr(exc), elapsed_seconds=round(time.monotonic() - started, 3),
                 timed_fallback_triggered=False)
        raise
    finally:
        timer.cancel()


if __name__ == "__main__":
    main()
