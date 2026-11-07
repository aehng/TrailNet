"""Read-only edge-mask audit on a Kaggle regeneration of the A Park City inputs."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
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
    if not (PROJECT / "label_overrides" / "park_city" / "trails.gpkg").is_file():
        raise FileNotFoundError("Park City A label override missing")
    data_dir = PROJECT / "data" / "park_city"
    data_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("park_city_naip_rgbn.tif", "park_city_3dep_dtm.tif", "park_city_trails.gpkg"):
        source = find_file(rasters, filename)
        if source is None:
            raise FileNotFoundError(f"missing raw A input {filename}")
        shutil.copy2(source, data_dir / filename.removeprefix("park_city_"))
    shutil.copy2(PROJECT / "label_overrides" / "park_city" / "trails.gpkg",
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
               "--area", "park_city"]
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
    area = next(row for row in config["areas"] if row["name"] == "park_city")
    source_path = resolve_path(config, area["imagery"])
    tile_dir = resolve_path(config, config["paths"]["tile_dir"]) / "park_city"
    index_path = tile_dir / "index.csv"
    actual_index_hash = sha256(index_path)
    if actual_index_hash != EXPECTED_INDEX_SHA256:
        raise RuntimeError(f"Park City index mismatch: {actual_index_hash}")

    with rasterio.open(source_path) as source:
        bounds = source.bounds
        source_meta = {"sha256": sha256(source_path), "width": source.width,
                       "height": source.height, "crs": str(source.crs),
                       "bounds": [bounds.left, bounds.bottom, bounds.right, bounds.top],
                       "resolution": list(source.res), "nodata": source.nodata,
                       "colorinterp": [item.name for item in source.colorinterp],
                       "mask_flags": [[item.name for item in flags]
                                      for flags in source.mask_flag_enums]}

    with index_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    result = {"status": "complete", "experiment": "A-source Park City edge-mask audit",
              "no_train": True, "heldout_scored": False,
              "project_config": CONFIG_NAME, "tile_index_sha256": actual_index_hash,
              "tile_index_matches_recovered_a": True, "tile_count": len(rows),
              "source": source_meta,
              "background_weight": float(config.get("loss", {}).get("pixel_weights", {}).get("background", 1.0)),
              "counts": {"edge_tiles": 0, "out_of_footprint_tile_pixels": 0,
                         "optical_allzero_outside": 0, "optical_nonzero_outside": 0,
                         "class0_background_outside": 0, "class4_ignore_outside": 0,
                         "positive_class_outside": 0, "target_nonzero_outside": 0},
              "edge_tiles": []}
    for row in rows:
        with rasterio.open(tile_dir / row["image"]) as image_ds:
            height, width = image_ds.height, image_ds.width
            transform = image_ds.transform
            rr, cc = np.indices((height, width))
            x = transform.c + (cc + 0.5) * transform.a
            y = transform.f + (rr + 0.5) * transform.e
            outside = ((x < bounds.left) | (x >= bounds.right) |
                       (y < bounds.bottom) | (y >= bounds.top))
            if not outside.any():
                continue
            optical = image_ds.read([1, 2, 3, 4])
        with rasterio.open(tile_dir / row["class_map"]) as class_ds:
            classes = class_ds.read(1)
        with rasterio.open(tile_dir / row["mask"]) as mask_ds:
            targets = mask_ds.read(1)
        zero = np.all(optical == 0, axis=0) & outside
        counts = result["counts"]
        counts["edge_tiles"] += 1
        counts["out_of_footprint_tile_pixels"] += int(outside.sum())
        counts["optical_allzero_outside"] += int(zero.sum())
        counts["optical_nonzero_outside"] += int((outside & ~np.all(optical == 0, axis=0)).sum())
        counts["class0_background_outside"] += int(((classes == 0) & outside).sum())
        counts["class4_ignore_outside"] += int(((classes == 4) & outside).sum())
        counts["positive_class_outside"] += int((np.isin(classes, [1, 2, 3]) & outside).sum())
        counts["target_nonzero_outside"] += int(((targets != 0) & outside).sum())
        result["edge_tiles"].append({"image": row["image"],
                                     "outside_pixels": int(outside.sum()),
                                     "outside_fraction": round(float(outside.mean()), 6),
                                     "positive_class_pixels": int((np.isin(classes, [1, 2, 3]) & outside).sum())})

    result["interpretation"] = (
        "This is a new tile build from the A source datasets and code, not a pixel-hash comparison "
        "with archived training tensors. Index membership matches the recovered A Park City index. "
        "Outside-footprint class-0 pixels are ordinary background in the staged config; the separate "
        "opt-in ignore setting was not enabled."
    )
    return result


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
                 timed_fallback_triggered=False, tile_count=result["tile_count"],
                 edge_tiles=result["counts"]["edge_tiles"])
        log(f"EDGE_AUDIT_COMPLETE tiles={result['tile_count']} edge_tiles={result['counts']['edge_tiles']}")
    except Exception as exc:
        callback("final", "failed", error=repr(exc), elapsed_seconds=round(time.monotonic() - started, 3),
                 timed_fallback_triggered=False)
        raise
    finally:
        timer.cancel()


if __name__ == "__main__":
    main()
