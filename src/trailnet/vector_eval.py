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


SCOPED_TOLERANCES_M = (3.0, 5.0, 8.0, 10.0)


def _empty_lines(crs):
    return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=crs)


def _linear_pieces(gdf):
    """Explode to LineStrings so each block-border piece can be kept or dropped."""
    if gdf is None or gdf.empty:
        return gdf
    out = gdf.explode(index_parts=False, ignore_index=True)
    out = out[out.geometry.map(lambda geom: geom is not None and not geom.is_empty)].copy()
    if out.empty:
        return out
    out = out.explode(index_parts=False, ignore_index=True)
    out = out[out.geometry.geom_type.isin(["LineString", "LinearRing"])].copy()
    if out.empty:
        return out
    return out[out.geometry.length > 0].copy()


def _length_km(gdf):
    if gdf is None or gdf.empty:
        return 0.0
    return float(gdf.geometry.length.sum() / 1000.0)


def block_index(x, y, block_size_m):
    """Floor block index. Same operation as ``spatial_split`` on a tile centre."""
    import numpy as np

    size = float(block_size_m)
    return int(np.floor(float(x) / size)), int(np.floor(float(y) / size))


def resolve_sidewalk_uncertain_buffer_m(configured, ignore_buffer_m=0.0):
    """Match ``tiles._trail_geometries`` / ``build_area_tiles``.

    YAML key is ``tiling.sidewalk_uncertain_buffer_m`` (A recipe sets 2.0).
    When the key is absent, the rasterizer uses ``tiling.ignore_buffer_m`` if
    that is positive, otherwise 2 m. Explicit 0 stays 0. This is not the path
    ignore annulus and it is not added on top of ``ignore_buffer_m`` when set.
    """
    if configured is None:
        ignore_buffer_m = float(ignore_buffer_m or 0.0)
        return ignore_buffer_m if ignore_buffer_m > 0 else 2.0
    return float(configured or 0.0)


def sidewalk_core_width_m(trail_buffers):
    """Width chain used when rasterizing explicit sidewalks to class 4."""
    buffers = trail_buffers or {}
    width = buffers.get(
        "sidewalk",
        buffers.get("footway", buffers.get("path", buffers.get("default", 1.0))),
    )
    try:
        width = float(1.0 if width is None else width)
    except (TypeError, ValueError):
        width = 1.0
    if width <= 0:
        width = 1.0
    return width


def _sidewalk_exclusion_polygon(label_gdf, trail_buffers, sidewalk_uncertain_buffer_m):
    """Ignore disk = sidewalk core width + configured uncertain buffer.

    Predictions and reference length inside this disk are removed from both
    denominators. They are not background false positives.
    """
    from trailnet.data.tiles import _is_explicit_sidewalk

    if label_gdf is None or label_gdf.empty:
        return None
    radius = sidewalk_core_width_m(trail_buffers) + float(sidewalk_uncertain_buffer_m or 0.0)
    parts = []
    for _, row in label_gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if not _is_explicit_sidewalk(row):
            continue
        parts.append(geom.buffer(radius) if radius > 0 else geom)
    if not parts:
        return None
    return unary_union(parts)


def _drop_explicit_sidewalks(gdf):
    from trailnet.data.tiles import _is_explicit_sidewalk

    if gdf is None or gdf.empty:
        return gdf
    keep = [not _is_explicit_sidewalk(row) for _, row in gdf.iterrows()]
    return gdf.loc[keep].copy()


def _subtract_polygon(gdf, polygon):
    if gdf is None or gdf.empty or polygon is None or getattr(polygon, "is_empty", True):
        return gdf
    out = gdf.copy()
    out["geometry"] = out.geometry.difference(polygon)
    return _linear_pieces(out)


def _block_region(block_keys, block_size_m):
    from shapely.geometry import box

    if not block_keys:
        return None
    size = float(block_size_m)
    return unary_union([
        box(bx * size, by * size, (bx + 1) * size, (by + 1) * size)
        for _area, bx, by in block_keys
    ])


def _clip_lines_to_region(gdf, region):
    if gdf is None or gdf.empty or region is None or region.is_empty:
        crs = None if gdf is None else gdf.crs
        return _empty_lines(crs)
    out = gdf.copy()
    out["geometry"] = out.geometry.intersection(region)
    return _linear_pieces(out)


def _filter_pieces_to_blocks(gdf, block_keys, block_size_m):
    """Keep a piece only when its midpoint's block is in this split.

    Closed squares share edges. ``floor`` assigns an exact boundary coordinate
    to the next block, matching tile-centre assignment, so a boundary line is
    counted in one split only.
    """
    if gdf is None or gdf.empty:
        return gdf
    allowed = {(bx, by) for _area, bx, by in block_keys}
    size = float(block_size_m)
    keep = []
    for geom in gdf.geometry:
        mid = geom.interpolate(0.5, normalized=True)
        bx, by = block_index(mid.x, mid.y, size)
        keep.append((bx, by) in allowed)
    return gdf.loc[keep].copy()


def split_lines_to_blocks(gdf, block_keys, block_size_m, footprint=None):
    """Split lines on block borders *before* they become matcher inputs.

    Geometry outside the split is discarded here. A later tolerance buffer cannot
    borrow length from the other side of the border.
    """
    region = _block_region(block_keys, block_size_m)
    if region is not None and footprint is not None and not footprint.is_empty:
        region = region.intersection(footprint)
    clipped = _clip_lines_to_region(gdf, region)
    return _filter_pieces_to_blocks(clipped, block_keys, block_size_m)


def _tile_union(rows, tile_size_px, pixel_w, pixel_h, footprint=None):
    from shapely.geometry import box

    if not rows or not tile_size_px or not pixel_w or not pixel_h:
        return None
    half_w = (float(tile_size_px) * float(pixel_w)) / 2.0
    half_h = (float(tile_size_px) * float(pixel_h)) / 2.0
    region = unary_union([
        box(float(row["x"]) - half_w, float(row["y"]) - half_h,
            float(row["x"]) + half_w, float(row["y"]) + half_h)
        for row in rows
    ])
    if footprint is not None and not footprint.is_empty:
        region = region.intersection(footprint)
    return region


def _project_to_crs(gdf, crs):
    if gdf is None or gdf.empty or crs is None or gdf.crs is None or gdf.crs == crs:
        return gdf
    return gdf.to_crs(crs)


def _null_scope(split_name, reason, n_blocks=0, crs=None):
    return {
        "split": split_name,
        "empty_scope": True,
        "no_in_scope_length": True,
        "reason": reason,
        "precision": None,
        "recall": None,
        "pred_km": 0.0,
        "label_km": 0.0,
        "matched_pred_km": 0.0,
        "matched_label_km": 0.0,
        "n_pred": 0,
        "n_label": 0,
        "n_blocks": int(n_blocks),
        "n_tiles": 0,
        "tolerance_sweep": [],
        "crs": None if crs is None else str(crs),
        "fabricated_perfect_score": False,
        "novelty_filter_applied": False,
        "unfiltered_predictions": True,
    }


def _scrub_empty_recall(row, label_km):
    """``vector_noise_metrics`` reports recall 1 when label length is 0. Do not."""
    if label_km <= 0:
        row["recall"] = None
        row["matched_label_km"] = 0.0
    if row.get("pred_km", 0) <= 0 and label_km > 0:
        row["precision"] = 0.0
        row["matched_pred_km"] = 0.0
        row["recall"] = 0.0
    return row


def scoped_vector_metrics(
    pred_gdf,
    label_gdf,
    tile_rows,
    split,
    block_size_m,
    val_fraction,
    test_fraction,
    seed,
    tolerances_m=SCOPED_TOLERANCES_M,
    report_tolerance_m=5.0,
    footprint_geom=None,
    footprint_crs=None,
    pixel_size_xy=None,
    tile_size_px=None,
    sidewalk_uncertain_buffer_m=None,
    ignore_buffer_m=0.0,
    trail_buffers=None,
    area_name=None,
    noise_length_m=20.0,
):
    """Length-weighted P/R on one disjoint geographic split.

    Block membership uses ``assign_block_split``, the same hash as
    ``spatial_split`` (seed, block size, val/test fractions, tile-centre block).
    The match scope is those block squares intersected with the raster footprint.
    Overlapping tile windows are not the match scope; their length difference is
    reported separately. Explicit sidewalks/crossings and the configured uncertain
    buffer are removed from predictions and from the reference before lengths are
    summed. Unnamed paths stay. Empty scopes return null P/R, not 1.
    """
    split = str(split)
    if split not in {"train", "val", "test"}:
        raise ValueError(f"split must be train, val, or test; got {split!r}")
    rows = []
    for row in tile_rows or []:
        if area_name is not None and str(row.get("area", area_name)) != str(area_name):
            continue
        item = dict(row)
        item["area"] = str(item.get("area") or area_name or "area")
        rows.append(item)

    sidewalk_m = resolve_sidewalk_uncertain_buffer_m(
        sidewalk_uncertain_buffer_m, ignore_buffer_m
    )
    block_keys = []
    seen = set()
    for row in rows:
        bx, by = block_index(row["x"], row["y"], block_size_m)
        key = (row["area"], bx, by)
        if key in seen:
            continue
        seen.add(key)
        if assign_block_split(
            row["area"], bx, by, seed=seed,
            val_fraction=val_fraction, test_fraction=test_fraction,
        ) == split:
            block_keys.append(key)
    allowed = {(bx, by) for _area, bx, by in block_keys}
    split_rows = []
    for row in rows:
        bx, by = block_index(row["x"], row["y"], block_size_m)
        if (bx, by) in allowed:
            split_rows.append(row)

    exclusion_note = (
        "explicit sidewalk/crossing core plus tiling.sidewalk_uncertain_buffer_m "
        "removed from prediction and reference denominators; not scored as "
        "background false positives; unnamed paths kept"
    )
    if not block_keys:
        out = _null_scope(split, "no tile centers assigned to this split")
        out["sidewalk_uncertain_buffer_m"] = sidewalk_m
        out["sidewalk_exclusion"] = exclusion_note
        out["boundary_vs_tile_assignment"] = None
        return out

    pred_in = pred_gdf if pred_gdf is not None else _empty_lines(None)
    if pred_in is None or pred_in.empty:
        # Still need a metric CRS from labels or the footprint.
        if label_gdf is not None and not label_gdf.empty:
            labels_metric = _require_metric_frame(label_gdf.copy(), "label")
            pred_metric = _empty_lines(labels_metric.crs)
        elif footprint_crs is not None:
            pred_metric = _empty_lines(footprint_crs)
            labels_metric = None if label_gdf is None else _empty_lines(footprint_crs)
        else:
            out = _null_scope(split, "no geometries and no metric CRS", n_blocks=len(block_keys))
            out["sidewalk_uncertain_buffer_m"] = sidewalk_m
            return out
    else:
        pred_metric, labels_metric = _aligned_metric_pair(pred_in, label_gdf)

    working_crs = None
    if footprint_crs is not None:
        working_crs = footprint_crs
    elif pred_metric is not None and pred_metric.crs is not None:
        working_crs = pred_metric.crs
    elif labels_metric is not None and labels_metric.crs is not None:
        working_crs = labels_metric.crs
    pred_metric = _project_to_crs(pred_metric, working_crs)
    labels_metric = _project_to_crs(labels_metric, working_crs)
    if pred_metric is None:
        pred_metric = _empty_lines(working_crs)
    if labels_metric is None:
        labels_metric = _empty_lines(working_crs)

    footprint = None
    if footprint_geom is not None and not getattr(footprint_geom, "is_empty", False):
        footprint = footprint_geom
        if footprint_crs is not None and working_crs is not None and footprint_crs != working_crs:
            footprint = gpd.GeoSeries([footprint_geom], crs=footprint_crs).to_crs(working_crs).iloc[0]

    exclusion = _sidewalk_exclusion_polygon(labels_metric, trail_buffers, sidewalk_m)
    pred_base = _subtract_polygon(pred_metric, exclusion)
    label_base = _subtract_polygon(_drop_explicit_sidewalks(labels_metric), exclusion)
    pred_scoped = split_lines_to_blocks(pred_base, block_keys, block_size_m, footprint=footprint)
    label_scoped = split_lines_to_blocks(label_base, block_keys, block_size_m, footprint=footprint)

    pixel_w = pixel_h = None
    if pixel_size_xy:
        pixel_w, pixel_h = float(pixel_size_xy[0]), float(pixel_size_xy[1])
    tile_region = _tile_union(
        split_rows, tile_size_px, pixel_w, pixel_h, footprint=footprint,
    )
    block_region = _block_region(block_keys, block_size_m)
    if block_region is not None and footprint is not None and not footprint.is_empty:
        block_region = block_region.intersection(footprint)
    pred_tiles = _clip_lines_to_region(pred_base, tile_region)
    label_tiles = _clip_lines_to_region(label_base, tile_region)
    boundary = {
        "block_pred_km": _length_km(pred_scoped),
        "block_label_km": _length_km(label_scoped),
        "tile_union_pred_km": _length_km(pred_tiles),
        "tile_union_label_km": _length_km(label_tiles),
        "pred_km_in_tiles_not_blocks": _length_km(_subtract_polygon(pred_tiles, block_region)),
        "label_km_in_tiles_not_blocks": _length_km(_subtract_polygon(label_tiles, block_region)),
        "pred_km_in_blocks_not_tiles": _length_km(_subtract_polygon(pred_scoped, tile_region)),
        "label_km_in_blocks_not_tiles": _length_km(_subtract_polygon(label_scoped, tile_region)),
        "note": (
            "Match scope is the disjoint block grid plus raster footprint. "
            "Pixel metrics use full tiles whose centers fall in the split; overlap "
            "can enter neighboring splits. Lengths are after sidewalk exclusion. "
            "This is a boundary difference, not a trail-truth score."
        ),
    }

    block_pred_km = boundary["block_pred_km"]
    block_label_km = boundary["block_label_km"]
    if block_pred_km <= 0 and block_label_km <= 0:
        out = _null_scope(
            split, "split blocks contain no prediction or reference length",
            n_blocks=len(block_keys), crs=working_crs,
        )
        out["empty_scope"] = False
        out["no_in_scope_length"] = True
        out["n_tiles"] = len(split_rows)
        out["boundary_vs_tile_assignment"] = boundary
        out["sidewalk_uncertain_buffer_m"] = sidewalk_m
        out["sidewalk_exclusion"] = exclusion_note
        return out

    sweep = vector_metrics_tolerance_sweep(
        pred_scoped if not pred_scoped.empty else _empty_lines(working_crs),
        label_scoped if not label_scoped.empty else _empty_lines(working_crs),
        tolerances_m=tolerances_m,
        noise_length_m=noise_length_m,
    )
    for row in sweep:
        _scrub_empty_recall(row, block_label_km)
    report_tolerance_m = float(report_tolerance_m)
    primary = next(
        (row for row in sweep if abs(float(row["match_tolerance_m"]) - report_tolerance_m) < 1e-9),
        sweep[0],
    )
    out = dict(primary)
    out.update({
        "split": split,
        "empty_scope": False,
        "no_in_scope_length": False,
        "reason": None,
        "n_blocks": len(block_keys),
        "n_tiles": len(split_rows),
        "tolerance_sweep": sweep,
        "report_tolerance_m": report_tolerance_m,
        "boundary_vs_tile_assignment": boundary,
        "sidewalk_uncertain_buffer_m": sidewalk_m,
        "sidewalk_core_width_m": sidewalk_core_width_m(trail_buffers),
        "sidewalk_exclusion": exclusion_note,
        "unfiltered_predictions": True,
        "novelty_filter_applied": False,
        "scope": "disjoint_geographic_blocks_and_raster_footprint",
        "split_geometry_before_match": True,
        "fabricated_perfect_score": False,
    })
    return out


def length_weighted_aggregate(per_area_metrics, report_tolerance_m=5.0):
    """Aggregate length-weighted P/R. Empty scopes do not become precision=recall=1."""
    pred_km = matched_pred = label_km = matched_label = 0.0
    areas_used = []
    areas_empty = []
    for area_name, metrics in per_area_metrics.items():
        if not metrics or metrics.get("skipped") or metrics.get("empty_scope") or metrics.get("no_in_scope_length"):
            areas_empty.append(area_name)
            continue
        row = metrics
        report_tol = float(metrics.get("report_tolerance_m", report_tolerance_m))
        if metrics.get("tolerance_sweep"):
            picked = next(
                (item for item in metrics["tolerance_sweep"]
                 if abs(float(item["match_tolerance_m"]) - float(report_tolerance_m)) < 1e-9),
                None,
            )
            if picked is not None:
                row = picked
        if "precision" not in row and "pred_km" not in row:
            areas_empty.append(area_name)
            continue
        areas_used.append(area_name)
        pred_km += float(row.get("pred_km", 0.0))
        matched_pred += float(row.get("matched_pred_km", 0.0))
        label_km += float(row.get("label_km", 0.0))
        matched_label += float(row.get("matched_label_km", 0.0))
    if not areas_used or (pred_km <= 0 and label_km <= 0):
        return {
            "precision": None,
            "recall": None,
            "pred_km": pred_km,
            "label_km": label_km,
            "matched_pred_km": matched_pred,
            "matched_label_km": matched_label,
            "report_tolerance_m": float(report_tolerance_m),
            "areas": areas_used,
            "areas_empty": areas_empty,
            "n_areas": len(areas_used),
            "empty_scope": True,
            "fabricated_perfect_score": False,
        }
    return {
        "precision": (matched_pred / pred_km) if pred_km else 0.0,
        "recall": (matched_label / label_km) if label_km else None,
        "pred_km": pred_km,
        "label_km": label_km,
        "matched_pred_km": matched_pred,
        "matched_label_km": matched_label,
        "report_tolerance_m": float(report_tolerance_m),
        "areas": areas_used,
        "areas_empty": areas_empty,
        "n_areas": len(areas_used),
        "empty_scope": False,
        "fabricated_perfect_score": False,
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
    ``val_blocks`` / ``test_blocks`` use ``scoped_vector_metrics``: disjoint block squares
    (same hash as pixel ``spatial_split``), raster footprint, sidewalk/uncertain-buffer
    exclusion, and lines split on block borders before matching.
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
    pred_raw, labels_raw = pred, labels
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
    footprint_geom = None
    footprint_crs = None
    pixel_size = None
    raster_for_scope = None
    if clip_path is not None and clip_path.exists():
        pred, labels = _clip_pair_to_raster(pred, labels, clip_path)
        clipped_pred = True
        raster_for_scope = clip_path
    elif area.get("imagery"):
        imagery = resolve_path(config, area["imagery"])
        if imagery.exists():
            raster_for_scope = imagery
    if raster_for_scope is not None:
        footprint_geom, footprint_crs, pixel_size, _grid_transform = _raster_footprint(raster_for_scope)

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
        if not rows:
            scoped[scope] = _empty_scope_result(
                area_name, scope, report_tol, reason="missing_tile_index"
            )
            continue
        tiling = config.get("tiling", {})
        sidewalk_cfg = (
            tiling["sidewalk_uncertain_buffer_m"]
            if "sidewalk_uncertain_buffer_m" in tiling else None
        )
        metrics = scoped_vector_metrics(
            pred_raw,
            labels_raw,
            rows,
            split_name,
            block_size_m=block_size_m,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            seed=seed,
            tolerances_m=tolerances_m,
            report_tolerance_m=report_tol,
            footprint_geom=footprint_geom,
            footprint_crs=footprint_crs,
            pixel_size_xy=pixel_size,
            tile_size_px=int(tiling.get("tile_size") or 0) or None,
            sidewalk_uncertain_buffer_m=sidewalk_cfg,
            ignore_buffer_m=float(tiling.get("ignore_buffer_m") or 0.0),
            trail_buffers=tiling.get("trail_buffer_m") or {},
            area_name=area_name,
        )
        metrics["area"] = area_name
        metrics["scope"] = scope
        metrics["clipped_to_prediction"] = footprint_geom is not None or clipped_pred
        metrics["clipped_to_blocks"] = True
        if metrics.get("empty_scope"):
            metrics["skipped"] = metrics.get("reason") or "empty_scope"
        scoped[scope] = metrics

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


def _raster_footprint(raster_path):
    """Bounds, CRS, and absolute pixel size of a metric raster. Not trail truth."""
    import rasterio
    from shapely.geometry import box

    with rasterio.open(raster_path) as ds:
        if ds.crs is None:
            raise ValueError(f"Raster {raster_path} has no CRS")
        geom = box(*ds.bounds)
        pixel = (abs(float(ds.transform.a)), abs(float(ds.transform.e)))
        return geom, ds.crs, pixel, ds.transform


def _finite(value):
    import math

    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def assert_metric_raster_grid(crs, transform, width, height):
    """Reject a missing, geographic, or degenerate affine before a round-trip check.

    A passing grid only means the raster can host a 1 px centerline comparison.
    It is not evidence that the lines are true trails.
    """
    if crs is None or not getattr(crs, "is_projected", False):
        raise ValueError(f"Round-trip raster CRS {crs} is not projected")
    if not _linear_units_are_metres(crs):
        raise ValueError(f"Round-trip raster CRS {crs} is not metre-based")
    pixel_w = float(transform.a)
    pixel_h = float(transform.e)
    if pixel_w == 0 or pixel_h == 0 or not all(_finite(v) for v in (pixel_w, pixel_h, transform.c, transform.f)):
        raise ValueError("Round-trip affine is degenerate or non-finite")
    if abs(pixel_w) > 50 or abs(pixel_h) > 50:
        raise ValueError(f"Round-trip pixel size {pixel_w}, {pixel_h} is not a trail-scale metre grid")
    if abs(float(transform.c)) <= 180 and abs(float(transform.f)) <= 90 and abs(pixel_w) < 0.05:
        raise ValueError("Round-trip affine origin looks geographic under a projected CRS")
    if int(width) <= 0 or int(height) <= 0:
        raise ValueError("Round-trip raster has no pixels")
    return {
        "crs": str(crs),
        "pixel_size_x_m": abs(pixel_w),
        "pixel_size_y_m": abs(pixel_h),
        "width": int(width),
        "height": int(height),
        "affine_ok": True,
        "roundtrip_is_trail_truth": False,
    }


def directional_reraster_scores(skeleton, vector_burn, dilate_px=1):
    """Directional centerline round-trip at ``dilate_px``.

    Recall = skeleton pixels covered by the dilated vector burn / skeleton pixels.
    Precision = vector-burn pixels covered by the dilated skeleton / vector-burn pixels.
    The previous check divided the skeleton-side intersection by the vector pixel
    count and could exceed 1. These ratios stay in [0, 1].

    Agreement is vectorization consistency only, not trail-label truth.
    """
    import numpy as np
    from skimage.morphology import binary_dilation, disk

    dilate_px = int(dilate_px)
    if dilate_px < 0:
        raise ValueError("dilate_px must be >= 0")
    skel = np.asarray(skeleton).astype(bool)
    burn = np.asarray(vector_burn).astype(bool)
    if skel.shape != burn.shape:
        raise ValueError(f"skeleton shape {skel.shape} != vector burn shape {burn.shape}")
    if dilate_px == 0:
        skel_cover = burn
        burn_cover = skel
    else:
        skel_cover = binary_dilation(burn, disk(dilate_px))
        burn_cover = binary_dilation(skel, disk(dilate_px))
    recall_num = int((skel & skel_cover).sum())
    precision_num = int((burn & burn_cover).sum())
    skel_n = int(skel.sum())
    burn_n = int(burn.sum())
    precision = (precision_num / burn_n) if burn_n else None
    recall = (recall_num / skel_n) if skel_n else None
    if precision is not None and precision > 1.0 + 1e-9:
        raise RuntimeError("directional precision exceeded 1")
    if recall is not None and recall > 1.0 + 1e-9:
        raise RuntimeError("directional recall exceeded 1")
    legacy = None if burn_n == 0 else recall_num / burn_n
    return {
        "dilate_px": dilate_px,
        "skel_px": skel_n,
        "vector_burn_px": burn_n,
        "precision_numerator_px": precision_num,
        "precision_denominator_px": burn_n,
        "recall_numerator_px": recall_num,
        "recall_denominator_px": skel_n,
        "roundtrip_precision": None if precision is None else round(precision, 4),
        "roundtrip_recall": None if recall is None else round(recall, 4),
        "legacy_precision_skel_inter_over_burn": legacy,
        "roundtrip_is_trail_truth": False,
        "meaning": (
            f"centerline round-trip of skeleton vs re-rasterized vectors at {dilate_px}px; "
            "not agreement with trail reference labels"
        ),
    }


