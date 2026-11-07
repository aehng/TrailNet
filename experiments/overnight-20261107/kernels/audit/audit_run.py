"""Regenerate frozen A tiles on Kaggle and emit bounded, hashed audit windows."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import base_runner as base


EXPECTED_INDEX_SHA256 = {
    "corner_canyon": "277eac2dad6bdc33386eba744838a82fccea0eea0cf9e90f69a83faee7ed2411",
    "kelly_canyon": "60bb9f2c139cfaa47c6de7fda1121d325bfb4807ff1c056ad2dbb2d295a3f63d",
    "park_city": "b288e2b19654911aee4b81b5d51c1679affc3227a75b9964c0eb359a834a8ba2",
    "snowbasin": "250e16679bacb0a0d20d1160ad85e1724718a20576e3b8a19535eeb126732500",
    "teton_pass": "f04eff6be5a833ad08d2a882e7c3a88eaf16b53cef4a948eea69ab735b50984f",
}
AREAS = tuple(EXPECTED_INDEX_SHA256)
OUTPUT = Path("/kaggle/working")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    started = time.time()
    base.PROJECT = Path("/kaggle/temp/TrailNet")
    base.LOG_PATH = OUTPUT / "audit-run.log"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    base.log("TrailNet T4Q exact-A tile audit starting; no training or prediction")
    base.stage_inputs()

    missing = [name for name in ("geopandas", "rasterio", "numpy", "matplotlib", "yaml")
               if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(f"Required preinstalled packages are missing; refusing pip install: {missing}")

    audit_script = Path(__file__).resolve().parent / "make_audit_windows.py"
    target_script = base.PROJECT / "scripts" / "make_audit_windows.py"
    target_script.parent.mkdir(parents=True, exist_ok=True)
    target_script.write_bytes(audit_script.read_bytes())

    base.run_command([sys.executable, "scripts/prepare_data.py", "--config",
                      base.CONFIG_NAME, "--all-areas"])
    tiles = base.PROJECT / "data" / "tiles_t4e_a"
    actual_indexes = {}
    for area, expected in EXPECTED_INDEX_SHA256.items():
        path = tiles / area / "index.csv"
        actual = sha256(path)
        actual_indexes[area] = {"sha256": actual, "expected_recovered_a_sha256": expected,
                                "matches_recovered_a": actual == expected}
    if not all(row["matches_recovered_a"] for row in actual_indexes.values()):
        raise RuntimeError(f"Regenerated A tile index mismatch: {actual_indexes}")

    raster_mount = base.find_mount(base.RASTER_SLUG)
    checksum_path = base.find_file(raster_mount, "SHA256SUMS.txt")
    if checksum_path is None:
        raise FileNotFoundError("raster dataset SHA256SUMS.txt missing")
    expected_files = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        fields = line.strip().split(maxsplit=1)
        if len(fields) == 2:
            expected_files[fields[1].lstrip("* ")] = fields[0].lower()
    raw_hashes = []
    for area in AREAS:
        for suffix in ("naip_rgbn.tif", "3dep_dtm.tif", "trails.gpkg"):
            name = f"{area}_{suffix}"
            path = base.find_file(raster_mount, name)
            if path is None:
                raise FileNotFoundError(f"missing raster dataset source file {name}")
            actual = sha256(path)
            raw_hashes.append({"file": name, "sha256": actual,
                               "expected_sha256": expected_files.get(name),
                               "manifest_match": expected_files.get(name) == actual})
    if not all(row["manifest_match"] for row in raw_hashes):
        raise RuntimeError("Mounted raw raster/vector inputs do not match their published SHA256SUMS.txt")

    base.run_command([sys.executable, "scripts/make_audit_windows.py",
                      "--index-root", str(tiles), "--output", str(OUTPUT / "audit")])
    runtime = {"python": sys.version, "elapsed_seconds": round(time.time() - started, 1),
               "device": "T4 allocated by Kaggle; tile generation and audit sampling are CPU geospatial operations"}
    manifest = {
        "status": "complete",
        "source_datasets": ["elikearl/trailnet-multiaoi-v4", "elikearl/trailnet-t4e-code"],
        "raw_source_files": raw_hashes,
        "raw_source_files_match_published_manifest": all(row["manifest_match"] for row in raw_hashes),
        "regenerated_a_index_hashes": actual_indexes,
        "sample_tile_image_and_class_map_hashes": "audit/audit_samples.csv",
        "config": base.CONFIG_NAME,
        "tile_pixel_identity_check": "compare audit sample hashes with local provisional audit_samples.csv; this checks sampled tiles only",
        "runtime": runtime,
        "no_training": True,
        "no_prediction": True,
    }
    (OUTPUT / "preflight.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (OUTPUT / "audit-result.json").write_text(json.dumps({
        "status": "complete", "windows": "audit/audit_samples.csv",
        "summary": "audit/audit_window_summary.json", "contact_sheets": 3,
        "training": False, "elapsed_seconds": runtime["elapsed_seconds"]
    }, indent=2), encoding="utf-8")
    base.log("AUDIT_COMPLETE " + json.dumps(manifest, default=str))


if __name__ == "__main__":
    main()
