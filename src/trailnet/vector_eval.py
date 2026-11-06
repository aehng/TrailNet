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


def vector_metrics_for_area(config, area, predicted_path=None, label_path=None, tolerances_m=None):
    from trailnet.config import resolve_path

    pred_path = Path(predicted_path) if predicted_path else (
        resolve_path(config, config["paths"]["vector_dir"]) / f"{area['name']}_trails.geojson"
    )
    lab_path = Path(label_path) if label_path else resolve_path(config, area["trails"])
    if not pred_path.exists():
        return {"area": area["name"], "skipped": "missing predicted vectors"}
    pred = gpd.read_file(pred_path)
    labels = gpd.read_file(lab_path) if lab_path.exists() else None
    if tolerances_m is None:
        tolerances_m = config.get("evaluation", {}).get(
            "vector_match_tolerances_m", list(DEFAULT_TOLERANCE_SWEEP_M)
        )
    sweep = vector_metrics_tolerance_sweep(pred, labels, tolerances_m=tolerances_m)
    # Keep backward-compatible single-tolerance fields from config default, not "best".
    default_tol = float(config.get("postprocess", {}).get("dedup_tolerance_m", 5.0))
    primary = next((row for row in sweep if abs(row["match_tolerance_m"] - default_tol) < 1e-9), sweep[0])
    out = dict(primary)
    out["area"] = area["name"]
    out["tolerance_sweep"] = sweep
    return out
