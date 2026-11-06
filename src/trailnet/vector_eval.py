"""Vector-level precision/recall vs labels and short unmatched fragment counts."""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd


def _length_m(geom):
    if geom is None or geom.is_empty:
        return 0.0
    return float(geom.length)


def vector_noise_metrics(pred_gdf, label_gdf, match_tolerance_m=5.0, noise_length_m=20.0):
    """Match predicted lines to labels in a projected metric CRS.

    precision = matched predicted length / predicted length
    recall = matched label length / label length
    noise_fragments = unmatched predicted parts shorter than noise_length_m
    """
    if pred_gdf is None or pred_gdf.empty:
        label_km = 0.0 if label_gdf is None or label_gdf.empty else float(label_gdf.geometry.length.sum() / 1000.0)
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
        }
    pred = pred_gdf.copy()
    labels = label_gdf.copy() if label_gdf is not None and not label_gdf.empty else None
    if pred.crs is None:
        raise ValueError("predicted trails have no CRS")
    if not pred.crs.is_projected:
        metric = pred.estimate_utm_crs()
        pred = pred.to_crs(metric)
        if labels is not None and labels.crs is not None:
            labels = labels.to_crs(metric)
    elif labels is not None and labels.crs is not None:
        labels = labels.to_crs(pred.crs)

    pred_len = float(pred.geometry.length.sum())
    label_len = 0.0 if labels is None else float(labels.geometry.length.sum())
    matched_pred = 0.0
    noise_n = 0
    noise_m = 0.0
    if labels is not None and not labels.empty:
        tree = labels.sindex
        for geom in pred.geometry:
            length = _length_m(geom)
            if length <= 0:
                continue
            hits = list(tree.query(geom.buffer(match_tolerance_m)))
            covered = 0.0
            if hits:
                others = labels.geometry.iloc[hits]
                inter = geom.intersection(others.buffer(match_tolerance_m).unary_union)
                covered = 0.0 if inter.is_empty else float(inter.length)
            matched_pred += min(length, covered)
            leftover = max(0.0, length - covered)
            if leftover >= 0.8 * length and length < noise_length_m:
                noise_n += 1
                noise_m += leftover
        tree_p = pred.sindex
        matched_label = 0.0
        for geom in labels.geometry:
            length = _length_m(geom)
            if length <= 0:
                continue
            hits = list(tree_p.query(geom.buffer(match_tolerance_m)))
            if not hits:
                continue
            others = pred.geometry.iloc[hits]
            inter = geom.intersection(others.buffer(match_tolerance_m).unary_union)
            if not inter.is_empty:
                matched_label += min(length, float(inter.length))
    else:
        matched_label = 0.0
        noise_n = int((pred.geometry.length < noise_length_m).sum())
        noise_m = float(pred.loc[pred.geometry.length < noise_length_m].geometry.length.sum())

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
        "match_tolerance_m": float(match_tolerance_m),
        "noise_length_m": float(noise_length_m),
    }


def vector_metrics_for_area(config, area, predicted_path=None, label_path=None):
    from trailnet.config import resolve_path
    pred_path = Path(predicted_path) if predicted_path else (
        resolve_path(config, config["paths"]["vector_dir"]) / f"{area['name']}_trails.geojson"
    )
    lab_path = Path(label_path) if label_path else resolve_path(config, area["trails"])
    if not pred_path.exists():
        return {"area": area["name"], "skipped": "missing predicted vectors"}
    pred = gpd.read_file(pred_path)
    labels = gpd.read_file(lab_path) if lab_path.exists() else None
    metrics = vector_noise_metrics(
        pred, labels,
        match_tolerance_m=float(config.get("postprocess", {}).get("dedup_tolerance_m", 5.0)),
    )
    metrics["area"] = area["name"]
    return metrics
