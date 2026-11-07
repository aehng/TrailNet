"""Raster alignment, trail-mask rasterization and overlapping tile export."""
import csv
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.warp import reproject
from rasterio.windows import Window, bounds as window_bounds, transform as window_transform
import geopandas as gpd
from shapely.geometry import box

from trailnet.config import resolve_path
from trailnet.data.terrain import derive_terrain

_WORKER = {}


def _normalize_imagery(arr):
    source = np.asarray(arr)
    arr = source.astype(np.float32)
    if np.issubdtype(source.dtype, np.integer):
        scale = float(np.iinfo(source.dtype).max)
    else:
        peak = float(np.nanmax(arr))
        scale = 65535.0 if peak > 255 else (255.0 if peak > 1.0 else 1.0)
    return np.clip(arr / scale, 0.0, 1.0)


def _is_explicit_sidewalk(row):
    """Exclude only OSM footway=sidewalk/crossing; unnamed footway stays eligible."""
    trail = str(row.get("trail_type") or "").strip().lower()
    if trail in {"sidewalk", "crossing"}:
        return True
    group = str(row.get("class_group") or "").strip().lower()
    if group == "sidewalk":
        return True
    footway = str(row.get("footway") or "").strip().lower()
    highway = str(row.get("highway") or "").strip().lower()
    return highway == "footway" and footway in {"sidewalk", "crossing"}


def _trail_geometries(gdf, buffers, ignore_buffer_m=0.0, sidewalk_uncertain_buffer_m=None):
    if gdf.crs is None:
        raise ValueError("Trail vector file has no CRS")
    if not gdf.crs.is_projected:
        raise ValueError("Trail vectors must be projected to the imagery CRS before buffering")
    ignore_buffer_m = float(ignore_buffer_m or 0.0)
    # Sidewalk GPS/edge uncertainty: default to the same ignore ring used for trails.
    if sidewalk_uncertain_buffer_m is None:
        sidewalk_uncertain_buffer_m = ignore_buffer_m if ignore_buffer_m > 0 else 2.0
    sidewalk_uncertain_buffer_m = float(sidewalk_uncertain_buffer_m or 0.0)
    result = {1: [], 2: [], 3: [], 4: []}
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        # Explicit sidewalks/crossings are never path/track positives. Rasterize as
        # ignore (class 4) so supervised loss does not treat them as background.
        if _is_explicit_sidewalk(row):
            width = float(buffers.get("sidewalk", buffers.get("footway", buffers.get("path",
                             buffers.get("default", 1.0)))))
            if width <= 0:
                width = 1.0
            core = geom.buffer(width)
            result[4].append((core, 4))
            if sidewalk_uncertain_buffer_m > 0:
                ring = geom.buffer(width + sidewalk_uncertain_buffer_m).difference(core)
                if ring is not None and not ring.is_empty:
                    result[4].append((ring, 4))
            continue
        raw_kind = ""
        for field in ("class_group", "highway", "fclass", "trail_type"):
            value = row.get(field)
            if value is not None and str(value).strip().lower() not in {"", "nan", "none"}:
                raw_kind = str(value).strip().lower()
                break
        if raw_kind == "track":
            class_code, buffer_key = 2, "track"
        elif raw_kind in {"path", "path_singletrack", "footway", "cycleway"}:
            class_code, buffer_key = 1, "path"
        elif raw_kind in {"track_atv", "track_doubletrack"}:
            class_code, buffer_key = 2, "track"
        else:
            # Unknown USFS use is retained as a medium-confidence path label.
            class_code, buffer_key = 3, "path"
        if class_code == 1 and "label_weight" in row and row["label_weight"] is not None:
            try:
                if float(row["label_weight"]) < 0.99:
                    class_code = 3
            except (TypeError, ValueError):
                pass
        width = float(buffers.get(buffer_key, buffers.get(raw_kind,
                                                         buffers.get("default", 1.0))))
        if width > 0:
            core = geom.buffer(width)
            result[class_code].append((core, class_code))
            if ignore_buffer_m > 0:
                ring = geom.buffer(width + ignore_buffer_m).difference(core)
                if ring is not None and not ring.is_empty:
                    result[4].append((ring, 4))
    return result


def _prepare_worker_count(cfg, tile_count):
    configured = cfg.get("num_workers")
    cpu = os.cpu_count() or 1
    # Lunar Lake-class 4P+4E: four workers keep the P-cores busy; extra
    # processes mostly fight GDAL for the same files.
    default = min(4, cpu)
    workers = default if configured is None else max(1, int(configured))
    if tile_count < workers * 2:
        return 1
    return workers


def _load_label_index(trails_path, dst_crs, buffers, ignore_buffer_m=0.0,
                      sidewalk_uncertain_buffer_m=None):
    vectors = gpd.read_file(trails_path).to_crs(dst_crs)
    buffered = _trail_geometries(
        vectors, buffers, ignore_buffer_m=ignore_buffer_m,
        sidewalk_uncertain_buffer_m=sidewalk_uncertain_buffer_m,
    )
    indexed_shapes = [(geom, code) for code, shapes in buffered.items()
                      for geom, _value in shapes]
    geometries = [geom for geom, _code in indexed_shapes]
    codes = [code for _geom, code in indexed_shapes]
    index = gpd.GeoSeries(geometries, crs=dst_crs).sindex if geometries else None
    return geometries, codes, index


def _write_tile(image_ds, dtm_ds, row, col, spec, shape_geometries, shape_codes, shape_index):
    size = spec["size"]
    window = Window(col, row, size, size)
    raw = image_ds.read([1, 2, 3, 4], window=window, boundless=True, fill_value=0)
    imagery = _normalize_imagery(raw)
    transform = window_transform(window, image_ds.transform)
    dtm = np.full((size, size), np.nan, dtype=np.float32)
    reproject(source=rasterio.band(dtm_ds, 1), destination=dtm,
              src_transform=dtm_ds.transform, src_crs=dtm_ds.crs,
              src_nodata=dtm_ds.nodata, dst_transform=transform,
              dst_crs=image_ds.crs, dst_nodata=np.nan,
              resampling=Resampling.bilinear)
    hillshade, slope, local_relief = derive_terrain(
        dtm, image_ds.transform.a, image_ds.transform.e, spec["relief_radius"])
    terrain = np.stack((hillshade, slope, local_relief), axis=0) * spec["channel_gain"]
    tile = np.concatenate((imagery, terrain), axis=0)
    bounds = window_bounds(window, image_ds.transform)
    local_classes = {code: [] for code in (1, 2, 3, 4)}
    if shape_index is not None:
        tile_bounds = box(*bounds)
        for i in shape_index.query(tile_bounds):
            index = int(i)
            geom = shape_geometries[index]
            if geom.intersects(tile_bounds):
                local_classes[shape_codes[index]].append((geom, shape_codes[index]))
    class_shapes = [shape for code in (4, 2, 3, 1) for shape in local_classes[code]]
    class_map = rasterize(class_shapes, out_shape=(size, size), transform=transform,
                          fill=0, dtype="uint8", all_touched=True) if class_shapes else np.zeros(
        (size, size), dtype="uint8"
    )
    if spec.get("ignore_outside_imagery", False):
        # Boundless optical reads are zero-filled outside the source footprint.
        # Those pixels are unknown coverage, not observed background. Use the
        # source grid indices, never pixel values, to preserve legitimate zeros.
        rows = row + np.arange(size)
        cols = col + np.arange(size)
        inside_source = ((rows[:, None] >= 0) & (rows[:, None] < image_ds.height) &
                         (cols[None, :] >= 0) & (cols[None, :] < image_ds.width))
        class_map[~inside_source] = 4
    soft_band = float(spec.get("soft_label_band_m") or 0.0)
    if soft_band > 0:
        # Soft confidence in the GPS uncertainty band beyond the hard core.
        # Hard core (classes 1/2/3) stays 1.0; ignore (4) stays unsupervised;
        # pixels in the soft annulus get a decaying target in (0,1).
        from rasterio.transform import xy as transform_xy
        soft = np.zeros((size, size), dtype=np.float32)
        # Approximate soft field by rasterizing successive buffers of positive cores.
        pos_shapes = [shape for code in (1, 2, 3) for shape in local_classes[code]]
        if pos_shapes:
            soft = np.isin(class_map, (1, 2, 3)).astype(np.float32)
            # Dilate rings at ~1px steps out to soft_band using buffered geometries.
            # local_classes cores are already width-buffered polygons; expand further.
            steps = max(1, int(round(soft_band / max(abs(transform.a), 1e-6))))
            for step in range(1, steps + 1):
                frac = 1.0 - (step / float(steps))
                ring_shapes = []
                for geom, code in pos_shapes:
                    # Expand core polygon by step pixels in map units.
                    expand = step * abs(transform.a)
                    grown = geom.buffer(expand)
                    if grown is not None and not grown.is_empty:
                        ring_shapes.append((grown, 1))
                if not ring_shapes:
                    continue
                layer = rasterize(ring_shapes, out_shape=(size, size), transform=transform,
                                  fill=0, dtype="uint8", all_touched=True)
                # Only fill background (not ignore, not already harder core)
                fill = (layer > 0) & (class_map == 0) & (soft < frac)
                soft[fill] = np.maximum(soft[fill], frac)
        # Sidewalk/ignore must never be supervised positives.
        soft[class_map == 4] = 0.0
        mask = soft
        mask_dtype = "float32"
    else:
        mask = np.isin(class_map, (1, 2, 3)).astype("uint8")
        mask_dtype = "uint8"
    stem = f"{spec['area_name']}_r{row:07d}_c{col:07d}"
    out_dir = Path(spec["out_dir"])
    profile = dict(spec["profile"])
    profile.update(width=size, height=size, count=7, dtype="float32", nodata=0,
                   transform=transform, compress="deflate")
    mask_profile = profile.copy()
    mask_profile.update(count=1, dtype=mask_dtype)
    with rasterio.open(out_dir / f"{stem}.tif", "w", **profile) as ds:
        ds.write(tile)
    with rasterio.open(out_dir / f"{stem}_mask.tif", "w", **mask_profile) as ds:
        ds.write(mask.astype(mask_dtype), 1)
    class_profile = profile.copy()
    class_profile.update(count=1, dtype="uint8")
    with rasterio.open(out_dir / f"{stem}_class.tif", "w", **class_profile) as ds:
        ds.write(class_map, 1)
    center = transform * (size / 2, size / 2)
    return {"area": spec["area_name"], "image": f"{stem}.tif",
            "mask": f"{stem}_mask.tif", "class_map": f"{stem}_class.tif",
            "x": center[0], "y": center[1], "crs": spec["crs"]}


def _init_tile_worker(spec):
    os.environ["GDAL_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    image_ds = rasterio.open(spec["imagery_path"])
    dtm_ds = rasterio.open(spec["dtm_path"])
    geometries, codes, index = _load_label_index(
        spec["trails_path"], image_ds.crs, spec["buffers"],
        ignore_buffer_m=spec.get("ignore_buffer_m", 0.0),
        sidewalk_uncertain_buffer_m=spec.get("sidewalk_uncertain_buffer_m"),
    )
    _WORKER.clear()
    _WORKER.update(spec=spec, image_ds=image_ds, dtm_ds=dtm_ds,
                   geometries=geometries, codes=codes, index=index)


def _write_one_tile(row_col):
    row, col = row_col
    return _write_tile(
        _WORKER["image_ds"], _WORKER["dtm_ds"], row, col, _WORKER["spec"],
        _WORKER["geometries"], _WORKER["codes"], _WORKER["index"],
    )


def build_area_tiles(config, area):
    """Stream an area's aligned output grid into tiles without loading its mosaic.

    The source NAIP, DTM, and trail vectors remain external inputs. Raster
    working memory is bounded to one configured tile plus terrain derivatives.
    """
    cfg = config["tiling"]
    size, overlap = int(cfg["tile_size"]), int(cfg["overlap"])
    if overlap < 0 or overlap >= size:
        raise ValueError("tiling.overlap must be in [0, tile_size)")
    stride = size - overlap
    imagery_path, dtm_path, trails_path = (resolve_path(config, area[k]) for k in ("imagery", "dtm", "trails"))
    for path in (imagery_path, dtm_path, trails_path):
        if not path.exists():
            raise FileNotFoundError(f"Required input is missing: {path}")
    out_dir = resolve_path(config, config["paths"]["tile_dir"]) / area["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "index.csv"
    terrain_cfg = config.get("terrain", {})
    relief_radius = terrain_cfg.get("local_relief_radius_px", 25)
    channel_gain = float(terrain_cfg.get("channel_gain", 1.0))
    if channel_gain <= 0:
        raise ValueError("terrain.channel_gain must be positive")
    with rasterio.open(imagery_path) as image_ds:
        if image_ds.count < 4:
            raise ValueError(f"Expected four NAIP bands (R,G,B,NIR), found {image_ds.count}")
        if not image_ds.crs or not image_ds.crs.is_projected:
            raise ValueError("Imagery must have a projected metric CRS")
        units = image_ds.crs.linear_units
        if units and units.lower() not in ("metre", "meter", "metres", "meters"):
            raise ValueError(f"Imagery CRS units must be metres, got {units!r}")
        height, width = image_ds.height, image_ds.width
        crs = image_ds.crs.to_string()
        profile = image_ds.profile.copy()
        for drop in ("blockxsize", "blockysize", "tiled", "interleave"):
            profile.pop(drop, None)
    coords = [(row, col) for row in range(0, height, stride) for col in range(0, width, stride)]
    workers = _prepare_worker_count(cfg, len(coords))
    spec = {
        "size": size,
        "imagery_path": str(imagery_path),
        "dtm_path": str(dtm_path),
        "trails_path": str(trails_path),
        "out_dir": str(out_dir),
        "area_name": area["name"],
        "buffers": cfg.get("trail_buffer_m", {}),
        "ignore_buffer_m": float(cfg.get("ignore_buffer_m", 0.0) or 0.0),
        "sidewalk_uncertain_buffer_m": (
            None if cfg.get("sidewalk_uncertain_buffer_m") is None
            else float(cfg.get("sidewalk_uncertain_buffer_m"))
        ),
        "relief_radius": relief_radius,
        "channel_gain": channel_gain,
        "ignore_outside_imagery": bool(cfg.get("ignore_outside_imagery", False)),
        "profile": {k: v for k, v in profile.items() if k != "crs"},
        "crs": crs,
    }
    # rasterio Profile needs the CRS string; Affine/CRS objects pickle poorly.
    spec["profile"]["crs"] = crs
    print(f"PREPARE_SETUP area={area['name']} tiles={len(coords)} workers={workers}")
    if workers <= 1:
        geometries, codes, index = _load_label_index(
            trails_path, crs, spec["buffers"],
            ignore_buffer_m=spec.get("ignore_buffer_m", 0.0),
            sidewalk_uncertain_buffer_m=spec.get("sidewalk_uncertain_buffer_m"),
        )
        rows = []
        with rasterio.open(imagery_path) as image_ds, rasterio.open(dtm_path) as dtm_ds:
            for row, col in coords:
                rows.append(_write_tile(image_ds, dtm_ds, row, col, spec,
                                        geometries, codes, index))
    else:
        chunksize = max(1, len(coords) // (workers * 4))
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_tile_worker,
                                 initargs=(spec,)) as pool:
            rows = list(pool.map(_write_one_tile, coords, chunksize=chunksize))
    with index_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["area", "image", "mask", "class_map",
                                                    "x", "y", "crs"])
        writer.writeheader(); writer.writerows(rows)
    return index_path
