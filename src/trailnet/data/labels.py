"""Cached, bbox-scoped free trail label retrieval and alignment."""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiLineString, box, shape

from trailnet.data.download import USER_AGENT, fetch_osm_trails


UGRC_TRAILS_URL = (
    "https://services1.arcgis.com/99lidPhWCzftIe9K/ArcGIS/rest/services/"
    "TrailsAndPathways/FeatureServer/0/query"
)
USFS_TRAILS_URL = (
    "https://apps.fs.usda.gov/arcx/rest/services/EDW/EDW_TrailNFSPublish_01/"
    "MapServer/0/query"
)
CANONICAL_FIELDS = [
    "highway", "footway", "trail_type", "name", "source", "cartocode", "class_group",
    "label_weight", "priority", "geometry",
]
SIDEWALK_TRAIL_TYPES = frozenset({"sidewalk", "crossing"})


def _bbox_key(source, endpoint, bbox):
    payload = json.dumps([source, endpoint.rstrip("/"), list(map(float, bbox))],
                         separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _arcgis_page(endpoint, bbox, offset, page_size, timeout):
    west, south, east, north = map(float, bbox)
    params = {
        "where": "1=1",
        "geometry": f"{west:.8f},{south:.8f},{east:.8f},{north:.8f}",
        "geometryType": "esriGeometryEnvelope",
        "inSR": 4326,
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "*",
        "outSR": 4326,
        "resultOffset": int(offset),
        "resultRecordCount": min(int(page_size), 2000),
        "f": "geojson",
    }
    request = Request(endpoint + "?" + urlencode(params),
                      headers={"User-Agent": USER_AGENT, "Accept": "application/geo+json, application/json"})
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read(1000).decode("utf-8", errors="replace")
        raise RuntimeError(f"ArcGIS query returned HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach ArcGIS endpoint {endpoint}: {exc.reason}") from exc


def fetch_arcgis_geojson(endpoint, bbox, cache_dir, *, source, page_size=2000,
                          timeout=90, refresh=False):
    """Fetch all pages for one bbox and cache the combined GeoJSON response."""
    if not (1 <= int(page_size) <= 2000):
        raise ValueError("ArcGIS page size must be between 1 and 2000")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{source.lower()}_{_bbox_key(source, endpoint, bbox)}.geojson"
    if cache_path.exists() and not refresh:
        with cache_path.open(encoding="utf-8") as stream:
            return json.load(stream)

    features = []
    seen_ids = set()
    offset = 0
    for _page_number in range(1000):
        payload = _arcgis_page(endpoint, bbox, offset, page_size, timeout)
        if payload.get("error"):
            raise RuntimeError(f"ArcGIS query error: {payload['error']}")
        page = payload.get("features") or []
        if not page:
            break
        ids = []
        for feature in page:
            properties = feature.get("properties") or {}
            oid = next((properties[key] for key in properties
                        if key.lower() in {"objectid", "oid", "fid"}), None)
            ids.append(str(oid) if oid is not None else hashlib.sha1(
                json.dumps(feature, sort_keys=True).encode("utf-8")).hexdigest())
        new_ids = [oid for oid in ids if oid not in seen_ids]
        if not new_ids:
            raise RuntimeError("ArcGIS pagination repeated a page; refusing incomplete labels")
        features.extend(feature for feature, oid in zip(page, ids) if oid not in seen_ids)
        seen_ids.update(ids)
        offset += len(page)
        if len(page) < min(int(page_size), 2000) and not payload.get("exceededTransferLimit", False):
            break
    else:
        raise RuntimeError("ArcGIS query exceeded the 1000-page safety limit")

    combined = {"type": "FeatureCollection", "features": features}
    stage_path = cache_path.with_suffix(".tmp")
    with stage_path.open("w", encoding="utf-8") as stream:
        json.dump(combined, stream, separators=(",", ":"))
    os.replace(stage_path, cache_path)
    return combined


def _line_geometry(feature):
    raw = feature.get("geometry")
    if not raw:
        return None
    try:
        geom = shape(raw)
    except (TypeError, ValueError):
        return None
    if geom.geom_type in {"LineString", "MultiLineString"}:
        return geom if not geom.is_empty else None
    if geom.geom_type == "GeometryCollection":
        lines = [part for part in geom.geoms if part.geom_type == "LineString" and not part.is_empty]
        if len(lines) == 1:
            return lines[0]
        if lines:
            return MultiLineString(lines)
    return None


def _properties(feature):
    return {str(key).lower(): value for key, value in (feature.get("properties") or {}).items()}


def parse_arcgis_geojson(payload, *, source="UGRC"):
    """Parse Utah UGRC CartoCode records into path/track labels."""
    features = []
    for feature in payload.get("features", []):
        props = _properties(feature)
        code_value = props.get("cartocode")
        match = re.match(r"^\s*([1-9])(?:\s|$|-)", str(code_value or ""))
        code = match.group(1) if match else str(code_value or "").strip()
        if code not in {"1", "2", "4", "5"}:
            continue
        class_group = "track" if code == "4" else "path"
        geom = _line_geometry(feature)
        if geom is None:
            continue
        features.append({
            "highway": "track" if class_group == "track" else "path",
            "trail_type": "ugrc_doubletrack" if class_group == "track" else "ugrc_singletrack",
            "name": props.get("primaryname") or props.get("name"),
            "source": "Utah UGRC TrailsAndPathways",
            "cartocode": code,
            "class_group": class_group,
            "label_weight": 1.0,
            "priority": 3,
            "geometry": geom,
        })
    return gpd.GeoDataFrame(features, columns=CANONICAL_FIELDS, geometry="geometry", crs="EPSG:4326")


def _yes(value):
    return str(value).strip().lower() in {"yes", "y", "true", "1", "allowed", "open", "managed"}


def _no(value):
    return str(value).strip().lower() in {"no", "n", "false", "0", "not allowed", "prohibited", "closed"}


def parse_usfs_geojson(payload, *, source="USFS"):
    """Parse TrailNFS lines, using motor-use fields when a forest publishes them."""
    features = []
    motor_fields = ("terra_motorized", "atv_accpt", "motorcycle_accpt", "fourwd_accpt")
    for feature in payload.get("features", []):
        props = _properties(feature)
        geom = _line_geometry(feature)
        if geom is None:
            continue
        values = [props.get(key) for key in motor_fields if props.get(key) not in (None, "")]
        if any(_yes(value) for value in values):
            class_group, label_weight = "track", 1.0
        elif values and all(_no(value) for value in values):
            class_group, label_weight = "path", 1.0
        else:
            class_group, label_weight = "path", 0.65
        features.append({
            "highway": "track" if class_group == "track" else "path",
            "trail_type": props.get("trail_type"),
            "name": props.get("trail_name") or props.get("name"),
            "source": "USFS TrailNFS",
            "cartocode": None,
            "class_group": class_group,
            "label_weight": label_weight,
            "priority": 2,
            "geometry": geom,
        })
    return gpd.GeoDataFrame(features, columns=CANONICAL_FIELDS, geometry="geometry", crs="EPSG:4326")


def _source_priority(source):
    source = str(source or "").lower()
    if "ugrc" in source:
        return 3
    if "usfs" in source or "forest service" in source:
        return 2
    if "openstreetmap" in source or source == "osm":
        return 1
    return 0


def deduplicate_trails(frame, *, target_crs, tolerance_m=4.0, overlap_fraction=0.8):
    """Keep one preferred record when projected linework materially overlaps."""
    if frame.empty:
        return gpd.GeoDataFrame(columns=CANONICAL_FIELDS, geometry="geometry", crs=target_crs)
    if frame.crs is None:
        raise ValueError("Trail labels have no CRS")
    projected = frame.to_crs(target_crs).copy()
    crs = projected.crs
    units = crs.axis_info[0].unit_name.lower() if crs.axis_info else ""
    if not crs.is_projected or units not in {"meter", "metre", "meters", "metres"}:
        raise ValueError("Trail deduplication requires a projected metric CRS")
    if "priority" not in projected:
        projected["priority"] = projected["source"].map(_source_priority)
    projected["priority"] = projected["priority"].fillna(0)
    projected = projected.sort_values("priority", ascending=False, kind="stable").reset_index(drop=True)
    spatial_index = projected.sindex
    kept = []
    kept_set = set()
    for i, row in projected.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty or geom.length <= 0:
            continue
        duplicated = False
        candidate_ids = spatial_index.query(geom.buffer(tolerance_m))
        for j in candidate_ids:
            j = int(j)
            if j not in kept_set:
                continue
            other = projected.geometry.iloc[j]
            if geom.distance(other) > tolerance_m:
                continue
            min_length = max(min(geom.length, other.length), 1e-6)
            close_forward = geom.intersection(other.buffer(tolerance_m)).length / geom.length
            close_reverse = other.intersection(geom.buffer(tolerance_m)).length / other.length
            if min(close_forward, close_reverse) >= overlap_fraction and min_length > 0:
                duplicated = True
                break
        if not duplicated:
            kept.append(i)
            kept_set.add(i)
    result = projected.loc[kept].copy()
    for field in CANONICAL_FIELDS:
        if field not in result:
            result[field] = None
    return result[CANONICAL_FIELDS].reset_index(drop=True)


def _is_unnamed_value(value):
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower() in {"", "nan", "none", "null"}


def _drop_unnamed_osm_footway(frame, enabled: bool):
    if not enabled or frame is None or frame.empty:
        return frame
    result = frame.copy()
    if "trail_type" in result:
        trail = result["trail_type"]
    elif "highway" in result:
        trail = result["highway"]
    else:
        return result
    if "source" in result:
        is_osm = result["source"].map(
            lambda s: "openstreetmap" in str(s).lower() or str(s).lower() == "osm"
        )
    else:
        return result
    is_fw = trail.astype(str).str.lower().eq("footway")
    unnamed = result["name"].map(_is_unnamed_value) if "name" in result else True
    drop = is_osm & is_fw & unnamed
    return result.loc[~drop].copy()


def filter_unnamed_osm_footway_gpkg(path, *, enabled=True):
    """In-place GPKG row drop used by T4-B (no --fetch-labels)."""
    path = Path(path)
    frame = gpd.read_file(path)
    before = len(frame)
    kept = _drop_unnamed_osm_footway(frame, enabled=enabled)
    if enabled:
        kept.to_file(path, layer="trails", driver="GPKG")
    return {"path": str(path), "before": before, "after": int(len(kept))}


def _z01(arr):
    import numpy as np
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=np.float32)
    lo, hi = np.percentile(arr[finite], [2, 98])
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0, 1).astype(np.float32)


def _heading_at(geom, distance):
    import math
    d0 = max(0.0, min(geom.length, distance - 8.0))
    d1 = max(0.0, min(geom.length, distance + 8.0))
    p0 = geom.interpolate(d0)
    p1 = geom.interpolate(d1)
    dx, dy = p1.x - p0.x, p1.y - p0.y
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        return None
    return dx / norm, dy / norm


def _sample_grid(array, transform, xs, ys, order=1):
    import numpy as np
    from scipy.ndimage import map_coordinates
    inv = ~transform
    cols = np.empty(len(xs), dtype=np.float64)
    rows = np.empty(len(ys), dtype=np.float64)
    for i, (x, y) in enumerate(zip(xs, ys)):
        c, r = inv * (x, y)
        cols[i] = c
        rows[i] = r
    coords = np.vstack([rows - 0.5, cols - 0.5])
    return map_coordinates(array, coords, order=order, mode="nearest")


def _line_parts(geom):
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "MultiLineString":
        return list(geom.geoms)
    if geom.geom_type == "LineString":
        return [geom]
    return []


def _shift_linestring(part, offset_m, heading_fn=None):
    """Shift every vertex by offset_m along the local left-normal."""
    coords = list(part.coords)
    if len(coords) < 2:
        return part
    new_coords = []
    length = part.length
    traveled = 0.0
    for i, (x, y) in enumerate(coords):
        if i > 0:
            px, py = coords[i - 1]
            traveled += ((x - px) ** 2 + (y - py) ** 2) ** 0.5
        heading = _heading_at(part, traveled if heading_fn is None else traveled)
        if heading is None:
            new_coords.append((x, y))
            continue
        dx, dy = heading
        nx, ny = -dy, dx
        new_coords.append((x + offset_m * nx, y + offset_m * ny))
    return LineString(new_coords)


def _reconstruct_from_samples(part, samples, reconstruct="smooth_line"):
    """samples: list of (distance, x, y, nx, ny, offset)."""
    accepted = [s for s in samples if abs(s[5]) > 1e-9]
    if len(accepted) < 3:
        if not accepted:
            return part
        median = float(pd.Series([s[5] for s in accepted]).median())
        return _shift_linestring(part, median)
    distances = [s[0] for s in samples]
    offsets = [s[5] for s in samples]
    if reconstruct == "smooth_line" and len(offsets) >= 3:
        smoothed = list(offsets)
        for i in range(len(offsets)):
            lo = max(0, i - 1)
            hi = min(len(offsets), i + 2)
            smoothed[i] = sum(offsets[lo:hi]) / (hi - lo)
        offsets = smoothed
    coords = list(part.coords)
    if len(coords) < 2:
        return part
    new_coords = []
    traveled = 0.0
    for i, (x, y) in enumerate(coords):
        if i > 0:
            px, py = coords[i - 1]
            traveled += ((x - px) ** 2 + (y - py) ** 2) ** 0.5
        heading = _heading_at(part, traveled)
        if heading is None:
            new_coords.append((x, y))
            continue
        dx, dy = heading
        nx, ny = -dy, dx
        offset = float(np.interp(traveled, distances, offsets)) if distances else 0.0
        new_coords.append((x + offset * nx, y + offset * ny))
    return LineString(new_coords)


def snap_lines_to_relief(
    frame: gpd.GeoDataFrame,
    dtm_path: Path,
    *,
    max_offset_m: float = 15.0,
    step_m: float = 0.6,
    sample_spacing_m: float = 25.0,
    score_margin: float = 0.08,
    sharpness_ratio: float = 1.20,
    reconstruct: str = "smooth_line",
) -> gpd.GeoDataFrame:
    """Sample every sample_spacing_m and snap accepted points to a relief-edge trough."""
    import numpy as np
    import rasterio
    from scipy.ndimage import sobel
    from shapely.geometry import MultiLineString
    from trailnet.data.terrain import derive_terrain

    if frame is None or frame.empty:
        return frame
    dtm_path = Path(dtm_path)
    with rasterio.open(dtm_path) as ds:
        z = ds.read(1).astype(np.float32)
        transform = ds.transform
        crs = ds.crs
        nodata = ds.nodata
    if nodata is not None:
        z = np.where(z == nodata, np.nan, z)
    projected = frame.to_crs(crs) if frame.crs is not None and frame.crs != crs else frame.copy()
    hillshade, _slope, relief = derive_terrain(z, transform.a, transform.e, 15)
    edge = np.hypot(sobel(hillshade, axis=1), sobel(hillshade, axis=0))
    trough = 1.0 - relief
    response = 0.55 * _z01(edge) + 0.45 * _z01(trough)
    offsets = np.arange(-float(max_offset_m), float(max_offset_m) + 1e-9, float(step_m))
    center = int(np.argmin(np.abs(offsets)))

    def snap_part(part):
        if part is None or part.is_empty or part.length < 2.0:
            return part
        spacing = float(sample_spacing_m)
        n = max(2, int(part.length // spacing))
        samples = []
        for i in range(1, n):
            dist = min(part.length, i * spacing)
            heading = _heading_at(part, dist)
            if heading is None:
                continue
            dx, dy = heading
            nx, ny = -dy, dx
            pt = part.interpolate(dist)
            xs = np.array([pt.x + o * nx for o in offsets])
            ys = np.array([pt.y + o * ny for o in offsets])
            scores = _sample_grid(response, transform, xs, ys, order=1)
            if not np.isfinite(scores).any():
                samples.append((dist, pt.x, pt.y, nx, ny, 0.0))
                continue
            best = int(np.nanargmax(scores))
            score_best = float(scores[best])
            score_zero = float(scores[center])
            adj = []
            if best > 0 and np.isfinite(scores[best - 1]):
                adj.append(float(scores[best - 1]))
            if best < len(scores) - 1 and np.isfinite(scores[best + 1]):
                adj.append(float(scores[best + 1]))
            sharp = bool(adj) and score_best >= float(sharpness_ratio) * (sum(adj) / len(adj))
            accept = score_best >= score_zero + float(score_margin) and sharp
            offset = float(offsets[best]) if accept else 0.0
            samples.append((dist, pt.x, pt.y, nx, ny, offset))
        if not samples:
            return part
        return _reconstruct_from_samples(part, samples, reconstruct=reconstruct)

    new_geoms = []
    for geom in projected.geometry:
        parts = [snap_part(part) for part in _line_parts(geom)]
        parts = [p for p in parts if p is not None and not p.is_empty]
        if not parts:
            new_geoms.append(geom)
        elif len(parts) == 1:
            new_geoms.append(parts[0])
        else:
            new_geoms.append(MultiLineString(parts))
    out = projected.copy()
    out.geometry = new_geoms
    return out


def _is_sidewalk_row(row):
    """True only for explicit OSM footway=sidewalk/crossing (not unnamed footway)."""
    trail = str(row.get("trail_type") or "").strip().lower()
    if trail in SIDEWALK_TRAIL_TYPES:
        return True
    footway = str(row.get("footway") or "").strip().lower()
    highway = str(row.get("highway") or "").strip().lower()
    return highway == "footway" and footway in SIDEWALK_TRAIL_TYPES


def _canonicalize_existing(frame):
    if frame.empty:
        return gpd.GeoDataFrame(columns=CANONICAL_FIELDS, geometry="geometry", crs=frame.crs)
    result = frame.copy()
    for field in CANONICAL_FIELDS:
        if field not in result:
            result[field] = None
    for index, row in result.iterrows():
        trail = str(row.get("trail_type") or "").strip().lower()
        footway = row.get("footway")
        if isinstance(footway, str):
            footway = footway.strip().lower() or None
        else:
            footway = None
        if footway is None and trail in SIDEWALK_TRAIL_TYPES:
            footway = trail
        result.at[index, "footway"] = footway
        if _is_sidewalk_row({**row.to_dict(), "footway": footway, "trail_type": trail or row.get("trail_type")}):
            result.at[index, "class_group"] = "sidewalk"
            result.at[index, "trail_type"] = footway or trail
            # Keep highway=footway so source tags remain auditable; not a path positive.
            result.at[index, "highway"] = "footway"
        else:
            highway = str(row.get("highway") or row.get("trail_type") or "").lower()
            group = row.get("class_group")
            if group not in {"path", "track"}:
                group = "track" if highway == "track" else "path"
            result.at[index, "class_group"] = group
            result.at[index, "highway"] = "track" if group == "track" else "path"
            if not trail:
                result.at[index, "trail_type"] = result.at[index, "highway"]
        label_weight = row.get("label_weight")
        result.at[index, "label_weight"] = (1.0 if pd.isna(label_weight) else float(label_weight))
        result.at[index, "priority"] = _source_priority(row.get("source"))
    return result[CANONICAL_FIELDS]


def merge_area_labels(bbox, output_path, *, area_name, target_crs, cache_dir,
                      include_osm=True, include_ugrc=False, include_usfs=False,
                      osm_endpoint="https://overpass-api.de/api/interpreter",
                      ugrc_endpoint=UGRC_TRAILS_URL, usfs_endpoint=USFS_TRAILS_URL,
                      existing_path=None, refresh=False, token_env=None,
                      drop_unnamed_osm_footway=False, snap_to_relief=None, dtm_path=None):
    """Fetch available free labels, clip, project, deduplicate, and stage a GPKG."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    errors = {}

    if existing_path and Path(existing_path).exists():
        existing = gpd.read_file(existing_path)
        if existing.crs is None:
            raise ValueError(f"Existing label file has no CRS: {existing_path}")
        frames.append(_drop_unnamed_osm_footway(
            _canonicalize_existing(existing), drop_unnamed_osm_footway
        ))

    if include_osm:
        try:
            with tempfile.TemporaryDirectory(prefix="trailnet-osm-labels-") as scratch:
                temp_gpkg = Path(scratch) / "osm.gpkg"
                fetch_osm_trails(bbox, temp_gpkg, area_name=area_name, endpoint=osm_endpoint,
                                 token_env=token_env, overwrite=True, cache_dir=cache_dir / "osm",
                                 refresh=refresh)
                frames.append(_drop_unnamed_osm_footway(
                    _canonicalize_existing(gpd.read_file(temp_gpkg)), drop_unnamed_osm_footway
                ))
        except Exception as exc:
            errors["osm"] = str(exc)

    for include, source, endpoint, parser in (
        (include_ugrc, "ugrc", ugrc_endpoint, parse_arcgis_geojson),
        (include_usfs, "usfs", usfs_endpoint, parse_usfs_geojson),
    ):
        if not include:
            continue
        try:
            payload = fetch_arcgis_geojson(endpoint, bbox, cache_dir / "arcgis",
                                           source=source, refresh=refresh)
            frames.append(_drop_unnamed_osm_footway(parser(payload), drop_unnamed_osm_footway))
        except Exception as exc:
            errors[source] = str(exc)

    if not frames or not any(not frame.empty for frame in frames):
        raise RuntimeError(f"No usable free trail labels for {area_name}: {errors}")
    clip = box(*map(float, bbox))
    clipped = []
    for frame in frames:
        if frame.empty:
            continue
        wgs84 = frame.to_crs("EPSG:4326")
        wgs84 = wgs84.copy()
        wgs84.geometry = wgs84.geometry.intersection(clip)
        wgs84 = wgs84[~wgs84.geometry.is_empty]
        wgs84["priority"] = wgs84["source"].map(_source_priority)
        clipped.append(wgs84)
    merged = gpd.GeoDataFrame(pd.concat(clipped, ignore_index=True),
                              geometry="geometry", crs="EPSG:4326")
    merged = deduplicate_trails(merged, target_crs=target_crs, tolerance_m=4.0)
    snap_cfg = snap_to_relief or {}
    if snap_cfg.get("enabled") and dtm_path:
        merged = snap_lines_to_relief(
            merged, dtm_path,
            max_offset_m=float(snap_cfg.get("max_offset_m", 15.0)),
            step_m=float(snap_cfg.get("step_m", 0.6)),
            sample_spacing_m=float(snap_cfg.get("sample_spacing_m", 25.0)),
            score_margin=float(snap_cfg.get("score_margin", 0.08)),
            sharpness_ratio=float(snap_cfg.get("sharpness_ratio", 1.20)),
            reconstruct=str(snap_cfg.get("reconstruct", "smooth_line")),
        )
    merged["area"] = area_name

    with tempfile.TemporaryDirectory(prefix="trailnet-labels-", dir=output_path.parent) as scratch:
        staged = Path(scratch) / output_path.name
        merged.to_file(staged, layer="trails", driver="GPKG")
        os.replace(staged, output_path)
    source_counts = {str(source): int(count) for source, count in merged["source"].value_counts().items()}
    return {"path": str(output_path), "feature_count": len(merged),
            "source_counts": source_counts, "source_errors": errors,
            "crs": str(merged.crs)}
