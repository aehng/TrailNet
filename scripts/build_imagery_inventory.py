"""Build source-level imagery/terrain provenance and sampled raster diagnostics."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling


AREAS = ("corner_canyon", "kelly_canyon", "park_city", "snowbasin", "teton_pass")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_hashes(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.strip().split(maxsplit=1)
        if len(fields) == 2:
            result[fields[1].lstrip("* ")] = fields[0].lower()
    return result


def raster_summary(path: Path, *, sample_factor: int = 32) -> dict:
    with rasterio.open(path) as dataset:
        out_h = max(1, dataset.height // sample_factor)
        out_w = max(1, dataset.width // sample_factor)
        values = dataset.read(
            out_shape=(dataset.count, out_h, out_w),
            resampling=Resampling.nearest,
        )
        masks = dataset.read_masks(
            out_shape=(dataset.count, out_h, out_w),
            resampling=Resampling.nearest,
        ) > 0
        per_band = []
        for index, band in enumerate(values, 1):
            valid = masks[index - 1]
            selected = band[valid]
            if selected.size:
                dtype_max = np.iinfo(band.dtype).max if np.issubdtype(band.dtype, np.integer) else None
                per_band.append({
                    "band": index,
                    "dtype": str(band.dtype),
                    "minimum": float(np.nanmin(selected)),
                    "maximum": float(np.nanmax(selected)),
                    "zero_fraction": round(float(np.mean(selected == 0)), 6),
                    "max_value_fraction": round(float(np.mean(selected == dtype_max)), 6) if dtype_max is not None else None,
                    "constant": bool(np.nanmin(selected) == np.nanmax(selected)),
                })
            else:
                per_band.append({"band": index, "empty_sample": True})
        all_valid = np.logical_and.reduce(masks)
        return {
            "path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size,
            "width": dataset.width, "height": dataset.height, "count": dataset.count,
            "dtype": list(dataset.dtypes), "crs": str(dataset.crs),
            "transform": list(dataset.transform)[:6], "resolution": list(dataset.res),
            "bounds": list(dataset.bounds), "nodata": dataset.nodata,
            "tags": dataset.tags(), "sample_factor": sample_factor,
            "sample_mask_valid_fraction": round(float(np.mean(all_valid)), 6),
            "sample_by_band": per_band,
        }


def build_inventory(data_root: Path, index_root: Path, manifest_path: Path,
                    summary_path: Path, naip_trace_path: Path, checkpoint: Path,
                    config: Path, output: Path) -> dict:
    expected = expected_hashes(manifest_path)
    source_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary_by_area = {row["name"]: row for row in source_summary["areas"]}
    trace_by_area = {row["area"]: row for row in json.loads(naip_trace_path.read_text(encoding="utf-8"))["areas"]}
    rows, input_hashes, mismatch = [], [], []
    for area in AREAS:
        area_summary = summary_by_area[area]
        components = {}
        for key, filename in (("naip", "naip_rgbn.tif"), ("dtm", "3dep_dtm.tif"),
                              ("labels", "trails.gpkg")):
            path = data_root / area / filename
            actual = sha256(path)
            manifest_key = f"{area}_{filename}"
            expected_sha = expected.get(manifest_key)
            matches = expected_sha == actual
            input_hashes.append({"file": manifest_key, "sha256": actual,
                                 "expected_sha256": expected_sha, "manifest_match": matches})
            if not matches:
                mismatch.append(manifest_key)
            if key in {"naip", "dtm"}:
                components[key] = raster_summary(path)
            else:
                components[key] = {"path": str(path), "sha256": actual,
                                   "bytes": path.stat().st_size, "feature_source_counts": area_summary.get("sources")}
        imagery, dtm = components["naip"], components["dtm"]
        trace = trace_by_area[area]
        reconstructed_items = trace.get("items_selected_by_latest_year_rule", [])
        fetched_from_stac = "fetched" in str(area_summary.get("raster_acquisition", "")).lower()
        rows.append({
            "area": area,
            "imagery_source": area_summary.get("raster_acquisition"),
            "imagery_acquisition_dates_reconstructed": ";".join(sorted({item.get("datetime", "")[:10] for item in reconstructed_items})) if fetched_from_stac else "unknown: raster was reused and source-item ID/date was not retained",
            "imagery_source_item_ids_reconstructed": json.dumps([item.get("id") for item in reconstructed_items]) if fetched_from_stac else "unknown",
            "imagery_item_identity_caveat": trace.get("source_identity_caveat", "source ID/date unavailable"),
            "imagery_gsd_m": "x".join(str(v) for v in imagery["resolution"]),
            "imagery_crs": imagery["crs"], "imagery_bounds": json.dumps(imagery["bounds"]),
            "imagery_transform": json.dumps(imagery["transform"]), "imagery_bands": "R,G,B,NIR (inferred from producer convention)",
            "imagery_sample_valid_fraction": imagery["sample_mask_valid_fraction"],
            "imagery_nodata": imagery["nodata"], "imagery_sample_band_stats": json.dumps(imagery["sample_by_band"]),
            "dtm_source": "USGS TNMAccess 3DEP (source dataset summary)",
            "dtm_product_or_acquisition_date": "unknown in retained source metadata",
            "dtm_gsd_m": "x".join(str(v) for v in dtm["resolution"]), "dtm_crs": dtm["crs"],
            "dtm_bounds": json.dumps(dtm["bounds"]), "dtm_nodata": dtm["nodata"],
            "label_sources": json.dumps(area_summary.get("sources")),
            "input_hashes_match_published_manifest": all(item["manifest_match"] for item in input_hashes[-3:]),
            "limitations": "Date/scene identity untracked; sampled mask validity is not a pixel-accurate scene-footprint audit; shadows/snow/cloud require visual review.",
        })
    output.mkdir(parents=True, exist_ok=True)
    with (output / "imagery_inventory.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    index_hashes = {}
    for area in AREAS:
        a_path = index_root / area / "index.csv"
        b_path = data_root / "tiles_v4" / area / "index.csv"
        index_hashes[area] = {"recovered_a_index_sha256": sha256(a_path),
                              "local_tiles_v4_index_sha256": sha256(b_path),
                              "index_match": sha256(a_path) == sha256(b_path),
                              "tile_raster_pixel_identity_verified": False}
    checkpoint_sha = sha256(checkpoint)
    preflight = {
        "source_dataset_checksum_manifest": str(manifest_path),
        "raw_source_inputs": input_hashes,
        "raw_source_mismatches": mismatch,
        "all_raw_source_inputs_match_manifest": not mismatch,
        "recovered_a_index_comparison": index_hashes,
        "all_recovered_a_indexes_match_local_v4_indexes": all(item["index_match"] for item in index_hashes.values()),
        "tile_raster_pixel_identity_verified": False,
        "checkpoint": {"path": str(checkpoint), "sha256": checkpoint_sha},
        "expected_checkpoint_sha256": "73b9f79ce0d20b9056a36c37cea8645f8a128eaea313672dae13c2f96ff16e4a",
        "checkpoint_matches_handoff": checkpoint_sha == "73b9f79ce0d20b9056a36c37cea8645f8a128eaea313672dae13c2f96ff16e4a",
        "config": {"path": str(config), "sha256": sha256(config)},
        "split": {"block_size_m": 2500, "seed": 42, "val_fraction": 0.15, "test_fraction": 0.15},
        "data_entry_gate": "Raw input files and tile indexes match source evidence; local tile pixel files have not yet been compared with regenerated A tiles.",
    }
    (output / "preflight.json").write_text(json.dumps(preflight, indent=2), encoding="utf-8")
    return preflight


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--index-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--naip-trace", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("experiments/overnight-20261107"))
    args = parser.parse_args()
    result = build_inventory(args.data_root, args.index_root, args.manifest,
                             args.summary, args.naip_trace, args.checkpoint, args.config, args.output)
    print(json.dumps({"raw_inputs_match": result["all_raw_source_inputs_match_manifest"],
                      "indexes_match": result["all_recovered_a_indexes_match_local_v4_indexes"],
                      "checkpoint_matches": result["checkpoint_matches_handoff"]}))


if __name__ == "__main__":
    main()
