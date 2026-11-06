"""Vector-level length-weighted precision/recall vs labels (metric CRS only)."""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
from shapely.ops import unary_union


DEFAULT_TOLERANCE_SWEEP_M = (1.0, 3.0, 5.0, 8.0, 10.0)


def _length_m(geom):
    if geom is None or geom.is_empty:
        return 0.0
    return float(geom.length)


def _linear_units_are_metres(crs):
    if crs is None or not getattr(crs, "is_projected", False):
        return False
    units = ""
    if getattr(crs, "axis_info", None):
        units = (crs.axis_info[0].unit_name or "").lower()
    if not units and getattr(crs, "linear_units", None):
        units = str(crs.linear_units).lower()
    return units in {"meter", "metre", "meters", "metres"}


def _coords_look_geographic(gdf):
    """True when XY look like lon/lat degrees (catches UTM metres tagged as 4326)."""
    if gdf is None or gdf.empty:
        return False
    minx, miny, maxx, maxy = gdf.total_bounds
    return abs(minx) <= 180 and abs(maxx) <= 180 and abs(miny) <= 90 and abs(maxy) <= 90


def _require_metric_frame(gdf, name):
    if gdf is None:
        return None
    if gdf.crs is None:
        raise ValueError(f"{name} trails have no CRS; refuse silent degree/metre matching")
    if not gdf.crs.is_projected or not _linear_units_are_metres(gdf.crs):
        # Geographic CRS with degree-like bounds → project to UTM.
        if _coords_look_geographic(gdf):
            metric = gdf.estimate_utm_crs()
            return gdf.to_crs(metric)
        raise ValueError(
            f"{name} trails CRS {gdf.crs} is not a projected metre system and "
            "coordinates are not geographic; check for metre values tagged as EPSG:4326"
        )
    # Projected CRS but coordinates look like lon/lat → almost certainly mis-tagged.
    if _coords_look_geographic(gdf):
        raise ValueError(
            f"{name} trails claim projected CRS {gdf.crs} but bounds look geographic; "
            "refusing match (likely CRS mis-tag)"
        )
    return gdf


def _aligned_metric_pair(pred_gdf, label_gdf):
    pred = _require_metric_frame(pred_gdf.copy(), "predicted")
    if label_gdf is None or label_gdf.empty:
        return pred, None
    labels = label_gdf.copy()
    if labels.crs is None:
        raise ValueError("label trails have no CRS; refuse silent degree/metre matching")
    if not labels.crs.is_projected or not _linear_units_are_metres(labels.crs):
        if _coords_look_geographic(labels):
            labels = labels.to_crs(pred.crs)
        else:
            raise ValueError(
                f"label trails CRS {labels.crs} is not metric and bounds are not geographic"
            )
    else:
        labels = labels.to_crs(pred.crs)
    if _coords_look_geographic(labels) and _linear_units_are_metres(labels.crs):
        raise ValueError("label trails look geographic after projection; CRS mismatch")
    return pred, labels


def _covered_length(geom, others, tolerance_m):
    """Length of geom that lies within tolerance_m of any geometry in others."""
    length = _length_m(geom)
    if length <= 0 or others is None or len(others) == 0:
        return 0.0
    hits = list(others.sindex.query(geom.buffer(tolerance_m)))
    if not hits:
        return 0.0
    buf = unary_union([g.buffer(tolerance_m) for g in others.iloc[hits]])
    inter = geom.intersection(buf)
    if inter.is_empty:
        return 0.0
    return min(length, float(inter.length))


def vector_noise_metrics(pred_gdf, label_gdf, match_tolerance_m=5.0, noise_length_m=20.0):
    """Symmetric length-weighted precision/recall in a shared projected metre CRS.

    precision = matched predicted length / predicted length
    recall = matched label length / label length

    Does not choose a "best" tolerance; callers that need multiple tolerances should
    use ``vector_metrics_tolerance_sweep`` and report each value separately.
    """
    match_tolerance_m = float(match_tolerance_m)
    noise_length_m = float(noise_length_m)
    if match_tolerance_m < 0:
        raise ValueError("match_tolerance_m must be >= 0")

    if pred_gdf is None or pred_gdf.empty:
        if label_gdf is None or label_gdf.empty:
            label_km = 0.0
        else:
            labels = _require_metric_frame(label_gdf.copy(), "label")
            label_km = float(labels.geometry.length.sum() / 1000.0)
        return {
            "pred_km": 0.0,
            "label_km": label_km,
            "matched_pred_km": 0.0,
            "matched_label_km": 0.0,
            "precision": 1.0 if label_km == 0 else 0.0,
            "recall": 1.0 if label_km == 0 else 0.0,
            "n_pred": 0,
            "n_label": 0 if label_gdf is None else int(len(label_gdf)),
            "noise_fragments": 0,
            "noise_km": 0.0,
            "match_tolerance_m": match_tolerance_m,
            "noise_length_m": noise_length_m,
            "crs": None if label_gdf is None or label_gdf.empty else str(label_gdf.crs),
        }

    pred, labels = _aligned_metric_pair(pred_gdf, label_gdf)
    pred_len = float(pred.geometry.length.sum())
    label_len = 0.0 if labels is None else float(labels.geometry.length.sum())
    matched_pred = 0.0
    noise_n = 0
    noise_m = 0.0

    if labels is not None and not labels.empty:
        label_geoms = labels.geometry
        for geom in pred.geometry:
            length = _length_m(geom)
            if length <= 0:
                continue
            covered = _covered_length(geom, label_geoms, match_tolerance_m)
            matched_pred += covered
            leftover = max(0.0, length - covered)
            if leftover >= 0.8 * length and length < noise_length_m:
                noise_n += 1
                noise_m += leftover
        pred_geoms = pred.geometry
        matched_label = 0.0
        for geom in labels.geometry:
            length = _length_m(geom)
            if length <= 0:
                continue
            matched_label += _covered_length(geom, pred_geoms, match_tolerance_m)
    else:
        matched_label = 0.0
        short = pred.geometry.length < noise_length_m
        noise_n = int(short.sum())
        noise_m = float(pred.loc[short].geometry.length.sum())

    precision = matched_pred / pred_len if pred_len else 1.0
    recall = matched_label / label_len if label_len else 1.0
    return {
        "pred_km": pred_len / 1000.0,
        "label_km": label_len / 1000.0,
        "matched_pred_km": matched_pred / 1000.0,
        "matched_label_km": matched_label / 1000.0,
        "precision": precision,
        "recall": recall,
        "n_pred": int(len(pred)),
        "n_label": 0 if labels is None else int(len(labels)),
        "noise_fragments": int(noise_n),
        "noise_km": noise_m / 1000.0,
        "match_tolerance_m": match_tolerance_m,
        "noise_length_m": noise_length_m,
        "crs": str(pred.crs),
    }


def vector_metrics_tolerance_sweep(
    pred_gdf,
    label_gdf,
    tolerances_m=DEFAULT_TOLERANCE_SWEEP_M,
    noise_length_m=20.0,
):
    """Evaluate length-weighted P/R at each tolerance. Does not pick a winner."""
    rows = []
    for tol in tolerances_m:
        metrics = vector_noise_metrics(
            pred_gdf, label_gdf, match_tolerance_m=float(tol), noise_length_m=noise_length_m
        )
        rows.append(metrics)
    return rows



def _clip_pair_to_raster(pred_gdf, label_gdf, raster_path):
    """Clip pred/label geometries to the probability raster footprint (metric CRS)."""
    import rasterio
    from shapely.geometry import box

    with rasterio.open(raster_path) as ds:
        bounds = ds.bounds
        raster_crs = ds.crs
    if raster_crs is None:
        return pred_gdf, label_gdf
    footprint = gpd.GeoDataFrame({"geometry": [box(*bounds)]}, crs=raster_crs)
    if pred_gdf is not None and not pred_gdf.empty:
        if pred_gdf.crs != footprint.crs:
            footprint_pred = footprint.to_crs(pred_gdf.crs)
        else:
            footprint_pred = footprint
        pred_gdf = gpd.clip(pred_gdf, footprint_pred)
    if label_gdf is not None and not label_gdf.empty:
        if label_gdf.crs != footprint.crs:
            footprint_lab = footprint.to_crs(label_gdf.crs)
        else:
            footprint_lab = footprint
        label_gdf = gpd.clip(label_gdf, footprint_lab)
    return pred_gdf, label_gdf




def assign_block_split(area, block_x, block_y, seed=42, val_fraction=0.15, test_fraction=0.15):
    """Return train/val/test for one geographic block — same hash as ``spatial_split``."""
    import hashlib

    if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1:
        raise ValueError("Invalid spatial split fractions")
    key = f"{seed}:{area}:{int(block_x)}:{int(block_y)}".encode()
    value = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / 2**64
    if value < test_fraction:
        return "test"
    if value < test_fraction + val_fraction:
        return "val"
    return "train"


def unique_blocks_from_rows(rows, block_size_m=2500):
    """Unique (area, block_x, block_y) from tile-index rows with projected centres."""
    import numpy as np

    if block_size_m <= 0:
        raise ValueError("block_size_m must be > 0")
    seen = set()
    out = []
    for row in rows:
        area = row["area"]
        bx = int(np.floor(float(row["x"]) / block_size_m))
        by = int(np.floor(float(row["y"]) / block_size_m))
        key = (area, bx, by)
        if key in seen:
            continue
        seen.add(key)
        out.append({"area": area, "block_x": bx, "block_y": by, "crs": row.get("crs")})
    return out


def split_block_polygons(
    rows,
    block_size_m=2500,
    val_fraction=0.15,
    test_fraction=0.15,
    seed=42,
    area_name=None,
    crs=None,
):
    """Build per-split unions of block boxes from the same assignment as ``spatial_split``.

    Returns dict mapping ``train``/``val``/``test`` to a GeoDataFrame of block boxes
    (may be empty). Coordinates are projected metres matching tile ``x``/``y``.
    """
    from shapely.geometry import box

    blocks = unique_blocks_from_rows(rows, block_size_m=block_size_m)
    if area_name is not None:
        blocks = [b for b in blocks if b["area"] == area_name]
    if crs is None:
        for b in blocks:
            if b.get("crs"):
                crs = b["crs"]
                break
        if crs is None and rows:
            filtered = [r for r in rows if area_name is None or r["area"] == area_name]
            if filtered and filtered[0].get("crs"):
                crs = filtered[0]["crs"]
    if crs is None:
        crs = "EPSG:32612"

    buckets = {"train": [], "val": [], "test": []}
    for b in blocks:
        split = assign_block_split(
            b["area"], b["block_x"], b["block_y"],
            seed=seed, val_fraction=val_fraction, test_fraction=test_fraction,
        )
        bx, by = b["block_x"], b["block_y"]
        buckets[split].append(
            box(bx * block_size_m, by * block_size_m, (bx + 1) * block_size_m, (by + 1) * block_size_m)
        )
    out = {}
    for split, geoms in buckets.items():
        out[split] = gpd.GeoDataFrame({"geometry": geoms}, crs=crs) if geoms else gpd.GeoDataFrame(
            {"geometry": []}, crs=crs
        )
    return out


def _clip_pair_to_polygon(pred_gdf, label_gdf, clip_gdf):
    """Clip pred/label to the union of clip_gdf polygons (metric CRS)."""
    if clip_gdf is None or clip_gdf.empty:
        empty_pred = pred_gdf.iloc[0:0].copy() if pred_gdf is not None else None
        empty_lab = label_gdf.iloc[0:0].copy() if label_gdf is not None else None
        return empty_pred, empty_lab
    clip_union = clip_gdf
    if pred_gdf is not None and not pred_gdf.empty:
        if clip_union.crs != pred_gdf.crs:
            clip_pred = clip_union.to_crs(pred_gdf.crs)
        else:
            clip_pred = clip_union
        pred_gdf = gpd.clip(pred_gdf, clip_pred)
    if label_gdf is not None and not label_gdf.empty:
        if clip_union.crs != label_gdf.crs:
            clip_lab = clip_union.to_crs(label_gdf.crs)
        else:
            clip_lab = clip_union
        label_gdf = gpd.clip(label_gdf, clip_lab)
    return pred_gdf, label_gdf


def length_weighted_aggregate(per_area_metrics, report_tolerance_m=5.0):
    """Aggregate length-weighted P/R across areas for a single tolerance headline."""
    pred_km = matched_pred = label_km = matched_label = 0.0
    areas_used = []
    for area_name, metrics in per_area_metrics.items():
        if not metrics or metrics.get("skipped") or metrics.get("empty_scope"):
            continue
        # Prefer explicit primary fields; fall back to tolerance_sweep
        row = metrics
        if "precision" not in row and "tolerance_sweep" in row:
            report_tol = float(metrics.get("report_tolerance_m", report_tolerance_m))
            sweep = metrics["tolerance_sweep"]
            row = next(
                (r for r in sweep if abs(r["match_tolerance_m"] - report_tol) < 1e-9),
                sweep[0] if sweep else None,
            )
            if row is None:
                continue
        pred_km += float(row.get("pred_km", 0.0))
        matched_pred += float(row.get("matched_pred_km", 0.0))
        label_km += float(row.get("label_km", 0.0))
        matched_label += float(row.get("matched_label_km", 0.0))
        areas_used.append(area_name)
    precision = (matched_pred / pred_km) if pred_km else 1.0
    recall = (matched_label / label_km) if label_km else 1.0
    return {
        "precision": precision,
        "recall": recall,
        "pred_km": pred_km,
        "label_km": label_km,
        "matched_pred_km": matched_pred,
        "matched_label_km": matched_label,
        "report_tolerance_m": float(report_tolerance_m),
        "areas": areas_used,
        "n_areas": len(areas_used),
    }


def _empty_scope_result(area_name, scope, report_tol, reason="empty_scope"):
    return {
        "area": area_name,
        "scope": scope,
        "empty_scope": True,
        "skipped": reason,
        "pred_km": 0.0,
        "label_km": 0.0,
        "matched_pred_km": 0.0,
        "matched_label_km": 0.0,
        "precision": None,
        "recall": None,
        "tolerance_sweep": [],
        "report_tolerance_m": report_tol,
        "clipped_to_prediction": True,
        "clipped_to_blocks": scope != "full_footprint",
    }


def _metrics_bundle(pred, labels, tolerances_m, report_tol, area_name, scope, clipped_pred, clipped_blocks):
    if (pred is None or pred.empty) and (labels is None or (hasattr(labels, "empty") and labels.empty)):
        return _empty_scope_result(area_name, scope, report_tol, reason="empty_after_clip")
    sweep = vector_metrics_tolerance_sweep(pred, labels, tolerances_m=tolerances_m)
    if not sweep:
        return _empty_scope_result(area_name, scope, report_tol, reason="empty_sweep")
    primary = next(
        (row for row in sweep if abs(row["match_tolerance_m"] - report_tol) < 1e-9),
        sweep[0],
    )
    out = dict(primary)
    out["area"] = area_name
    out["scope"] = scope
    out["tolerance_sweep"] = sweep
    out["report_tolerance_m"] = report_tol
    out["clipped_to_prediction"] = clipped_pred
    out["clipped_to_blocks"] = clipped_blocks
    out["empty_scope"] = False
    return out


def vector_metrics_for_area(
    config,
    area,
    predicted_path=None,
    label_path=None,
    tolerances_m=None,
    index_rows=None,
    scopes=("full_footprint", "val_blocks", "test_blocks"),
):
    """Length-weighted vector P/R for one AOI, optionally scoped to val/test blocks.

    ``full_footprint`` clips pred+label to the probability raster footprint (existing).
    ``val_blocks`` / ``test_blocks`` further clip to the geographic block unions from the
    same ``spatial_split`` hash used for pixel eval (block_size_m / fractions / seed).
    """
    from trailnet.config import resolve_path

    area_name = area["name"] if isinstance(area, dict) else str(area)
    pred_path = Path(predicted_path) if predicted_path else (
        resolve_path(config, config["paths"]["vector_dir"]) / f"{area_name}_trails.geojson"
    )
    lab_path = Path(label_path) if label_path else resolve_path(config, area["trails"])
    if not pred_path.exists():
        return {"area": area_name, "skipped": "missing predicted vectors"}
    pred = gpd.read_file(pred_path)
    labels = gpd.read_file(lab_path) if lab_path.exists() else None
    if tolerances_m is None:
        tolerances_m = config.get("evaluation", {}).get(
            "vector_match_tolerances_m", list(DEFAULT_TOLERANCE_SWEEP_M)
        )
    report_tol = float(config.get("evaluation", {}).get("vector_report_tolerance_m", 5.0))

    # Optional clip both layers to probability-raster footprint so label_km is not full-AOI
    # while predictions only cover the inferred mosaic.
    clip_path = None
    clipped_pred = False
    if config.get("evaluation", {}).get("clip_vector_metrics_to_prediction", True):
        pred_dir = resolve_path(config, config["paths"]["prediction_dir"])
        clip_path = pred_dir / f"{area_name}_probability.tif"
    if clip_path is not None and clip_path.exists():
        pred, labels = _clip_pair_to_raster(pred, labels, clip_path)
        clipped_pred = True

    # Load tile index rows for block assignment when scopes need them.
    need_blocks = any(s in ("val_blocks", "test_blocks") for s in scopes)
    training = config.get("training", {})
    block_size_m = float(training.get("block_size_m", 2500))
    val_fraction = float(training.get("val_fraction", 0.15))
    test_fraction = float(training.get("test_fraction", 0.15))
    seed = int(config.get("seed", 42))
    block_polys = None
    if need_blocks:
        rows = index_rows
        if rows is None:
            from trailnet.data.dataset import read_indexes
            tile_dir = resolve_path(config, config["paths"]["tile_dir"])
            index_path = tile_dir / area_name / "index.csv"
            if index_path.exists():
                rows = read_indexes([index_path])
            else:
                rows = []
        if rows:
            block_polys = split_block_polygons(
                rows,
                block_size_m=block_size_m,
                val_fraction=val_fraction,
                test_fraction=test_fraction,
                seed=seed,
                area_name=area_name,
            )

    scoped = {}
    for scope in scopes:
        if scope == "full_footprint":
            scoped[scope] = _metrics_bundle(
                pred, labels, tolerances_m, report_tol, area_name, scope, clipped_pred, False
            )
            continue
        split_name = "val" if scope == "val_blocks" else ("test" if scope == "test_blocks" else None)
        if split_name is None:
            continue
        if block_polys is None:
            scoped[scope] = _empty_scope_result(
                area_name, scope, report_tol, reason="missing_tile_index"
            )
            continue
        clip_gdf = block_polys.get(split_name)
        if clip_gdf is None or clip_gdf.empty:
            scoped[scope] = _empty_scope_result(
                area_name, scope, report_tol, reason="no_blocks_for_split"
            )
            continue
        pred_s, lab_s = _clip_pair_to_polygon(pred, labels, clip_gdf)
        scoped[scope] = _metrics_bundle(
            pred_s, lab_s, tolerances_m, report_tol, area_name, scope, clipped_pred, True
        )

    # Backward-compatible top-level fields = full_footprint headline.
    primary_scope = scoped.get("full_footprint") or next(iter(scoped.values()), None)
    out = dict(primary_scope) if primary_scope else {"area": area_name, "skipped": "no_scopes"}
    out["area"] = area_name
    out["scopes"] = scoped
    out["block_split"] = {
        "block_size_m": block_size_m,
        "val_fraction": val_fraction,
        "test_fraction": test_fraction,
        "seed": seed,
    }
    return out


