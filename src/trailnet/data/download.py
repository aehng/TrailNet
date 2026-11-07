"""Metadata-first adapters for public NAIP, 3DEP, and OSM sources.

Nothing downloads at import time. Raster downloads are bounded by area and
byte limits and are written to a staging file before replacing the destination.
"""
from __future__ import annotations

import json
import hashlib
import math
import os
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin, urlsplit
from urllib.request import Request, urlopen

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import ColorInterp, Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds
from shapely.geometry import LineString


DEFAULT_NAIP_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
DEFAULT_PC_SIGN_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/sign"
DEFAULT_TNM_PRODUCTS = "https://tnmaccess.nationalmap.gov/api/v1/products"
DEFAULT_OVERPASS_URL = "https://overpass-api.de/api/interpreter"
TNM_1M_DATASET = "Digital Elevation Model (DEM) 1 meter"
USER_AGENT = "TrailNet/0.1 (public geospatial data preparation)"


def _validate_bbox(bbox):
    if len(bbox) != 4:
        raise ValueError("bbox must be (west, south, east, north) in EPSG:4326")
    west, south, east, north = map(float, bbox)
    if not all(math.isfinite(v) for v in (west, south, east, north)):
        raise ValueError("bbox coordinates must be finite")
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise ValueError("bbox must be ordered west,south,east,north in EPSG:4326")
    return west, south, east, north


def _area_km2(bbox):
    west, south, east, north = _validate_bbox(bbox)
    mean_lat = math.radians((south + north) / 2)
    width_km = (east - west) * 111.32 * math.cos(mean_lat)
    height_km = (north - south) * 110.57
    return width_km * height_km


def _request_json(url, *, method="GET", payload=None, headers=None, timeout=60):
    request_headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if headers:
        request_headers.update(headers)
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        request_headers["Content-Type"] = "application/json"
    request = Request(url, data=data, headers=request_headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read(1000).decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach {url}: {exc.reason}") from exc


def _stac_search(bbox, *, catalog_url=DEFAULT_NAIP_STAC, collection="naip",
                 max_items=1000, page_size=100, timeout=60):
    """Return bounded STAC items and whether more results were available."""
    bbox = _validate_bbox(bbox)
    if max_items < 1 or page_size < 1:
        raise ValueError("max_items and page_size must be positive")
    url = catalog_url.rstrip("/") + "/search"
    payload = {"collections": [collection], "bbox": list(bbox),
               "limit": min(page_size, max_items)}
    method = "POST"
    items = []
    while url and len(items) < max_items:
        result = _request_json(url, method=method, payload=payload, timeout=timeout)
        items.extend(result.get("features", []))
        next_link = next((link for link in result.get("links", []) if link.get("rel") == "next"), None)
        if not next_link:
            url = None
            break
        url = urljoin(catalog_url.rstrip("/") + "/", next_link["href"])
        method = next_link.get("method", "GET").upper()
        if method == "POST" and isinstance(next_link.get("body"), dict):
            payload = next_link["body"]
        else:
            method = "GET"
            payload = None
    return items[:max_items], bool(url)


def _item_datetime(item):
    properties = item.get("properties", {})
    return properties.get("datetime") or properties.get("start_datetime") or ""


def probe_naip(bbox, *, catalog_url=DEFAULT_NAIP_STAC, collection="naip",
               max_items=100, timeout=60):
    """Query NAIP STAC metadata only; never open or download imagery assets."""
    items, truncated = _stac_search(bbox, catalog_url=catalog_url, collection=collection,
                                    max_items=max_items, timeout=timeout)
    summaries = []
    for item in items:
        summaries.append({"id": item.get("id"), "datetime": _item_datetime(item),
                          "assets": sorted(item.get("assets", {}).keys()),
                          "bbox": item.get("bbox")})
    return {"catalog": catalog_url, "collection": collection, "item_count": len(items),
            "truncated": truncated, "items": summaries}


def _pc_signed_href(href, sign_url=DEFAULT_PC_SIGN_URL, timeout=60):
    """Ask Planetary Computer's public SAS endpoint to sign a public asset URL."""
    url = (sign_url or DEFAULT_PC_SIGN_URL) + "?" + urlencode({"href": href})
    result = _request_json(url, timeout=timeout)
    signed = result.get("href")
    if not signed:
        raise RuntimeError("Planetary Computer sign response did not include an href")
    return signed


def _utm_epsg_for_bbox(bbox):
    west, south, east, north = bbox
    lon = (west + east) / 2
    lat = (south + north) / 2
    zone = min(60, max(1, int(math.floor((lon + 180) / 6)) + 1))
    return (32600 if lat >= 0 else 32700) + zone


def _target_grid(bbox, resolution_m):
    bbox = _validate_bbox(bbox)
    if resolution_m <= 0:
        raise ValueError("resolution_m must be greater than zero")
    dst_crs = f"EPSG:{_utm_epsg_for_bbox(bbox)}"
    left, bottom, right, top = transform_bounds("EPSG:4326", dst_crs, *bbox, densify_pts=21)
    x0 = math.floor(left / resolution_m) * resolution_m
    y0 = math.ceil(top / resolution_m) * resolution_m
    width = int(math.ceil((right - x0) / resolution_m))
    height = int(math.ceil((y0 - bottom) / resolution_m))
    return dst_crs, from_origin(x0, y0, resolution_m, resolution_m), width, height


def _check_grid_size(width, height, max_pixels):
    pixels = int(width) * int(height)
    if max_pixels <= 0:
        raise ValueError("max_raster_pixels must be positive")
    if pixels > max_pixels:
        raise ValueError(f"Projected raster grid has {pixels:,} pixels, above the configured "
                         f"limit of {max_pixels:,}; use a smaller bbox or coarser resolution")
    return pixels


def _bounded_window(bounds, transform, width, height):
    raw = from_bounds(*bounds, transform=transform)
    col0 = max(0, int(math.floor(raw.col_off)))
    row0 = max(0, int(math.floor(raw.row_off)))
    col1 = min(width, int(math.ceil(raw.col_off + raw.width)))
    row1 = min(height, int(math.ceil(raw.row_off + raw.height)))
    if col1 <= col0 or row1 <= row0:
        return None
    return Window(col0, row0, col1-col0, row1-row0)


def fetch_naip(bbox, output_path, *, catalog_url=DEFAULT_NAIP_STAC,
               collection="naip", asset_key="image", resolution_m=0.6,
               max_raster_km2=25.0, max_raster_pixels=70_000_000,
               max_items=1000, timeout=120, overwrite=False,
               sign_url=DEFAULT_PC_SIGN_URL):
    """Build a four-band RGBN GeoTIFF from public NAIP STAC assets.

    Downloads are opt-in through the CLI. This function rejects bboxes larger
    than ``max_raster_km2`` before contacting the STAC or asset services.
    For multi-tile extents, the newest available acquisition year is mosaicked;
    data are read in windows and the destination is published atomically.
    """
    bbox = _validate_bbox(bbox)
    area = _area_km2(bbox)
    if area > max_raster_km2:
        raise ValueError(f"Raster bbox is about {area:.1f} km2, above the configured "
                         f"limit of {max_raster_km2:.1f} km2; use a smaller bbox or raise the limit explicitly")
    dst_crs, dst_transform, width, height = _target_grid(bbox, resolution_m)
    _check_grid_size(width, height, max_raster_pixels)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}; pass overwrite=True to replace it")
    items, truncated = _stac_search(bbox, catalog_url=catalog_url, collection=collection,
                                    max_items=max_items, timeout=min(timeout, 60))
    if truncated:
        raise ValueError(f"STAC returned more than {max_items} items; reduce bbox or increase max_items")
    if not items:
        raise FileNotFoundError("No NAIP STAC items intersect the configured bbox")
    latest_year = max((_item_datetime(item)[:4] for item in items if _item_datetime(item)), default="")
    selected = [item for item in items if latest_year and _item_datetime(item).startswith(latest_year)]
    if not selected:
        raise ValueError("NAIP STAC items have no acquisition datetime")
    for item in selected:
        if asset_key not in item.get("assets", {}):
            raise KeyError(f"NAIP item {item.get('id')} has no {asset_key!r} asset")

    with tempfile.TemporaryDirectory(prefix="trailnet-naip-", dir=output_path.parent) as temp_dir:
        staged_path = Path(temp_dir) / output_path.name
        signed_href = _pc_signed_href(selected[0]["assets"][asset_key]["href"], sign_url, timeout)
        with rasterio.open(signed_href) as first_source:
            if first_source.count < 4:
                raise ValueError(f"NAIP asset {selected[0].get('id')} has {first_source.count} bands; four required")
            dtype = first_source.dtypes[0]
            profile = {"driver": "GTiff", "height": height, "width": width,
                       "count": 4, "dtype": dtype, "crs": dst_crs,
                       "transform": dst_transform, "nodata": 0, "tiled": True,
                       "blockxsize": 512, "blockysize": 512, "compress": "deflate"}
            with rasterio.open(staged_path, "w+", **profile) as destination:
                # A four-band NAIP asset is RGB + near infrared. GeoTIFF's
                # default interpretation for band 4 is alpha, which is wrong
                # here and can mislead downstream GIS readers. Keep the band
                # values unchanged and mark NIR as an undefined color channel.
                destination.colorinterp = (ColorInterp.red, ColorInterp.green,
                                           ColorInterp.blue, ColorInterp.undefined)
                for item_number, item in enumerate(selected):
                    href = signed_href if item_number == 0 else _pc_signed_href(
                        item["assets"][asset_key]["href"], sign_url, timeout)
                    with rasterio.open(href) as source:
                        if source.count < 4 or source.dtypes[0] != dtype:
                            raise ValueError(f"NAIP item {item.get('id')} has incompatible band count or dtype")
                        item_bbox = item.get("bbox")
                        if not item_bbox or len(item_bbox) < 4:
                            continue
                        bounds = transform_bounds("EPSG:4326", dst_crs, *item_bbox[:4], densify_pts=21)
                        window = _bounded_window(bounds, dst_transform, width, height)
                        if window is None:
                            continue
                        vrt_options = {"crs": dst_crs, "transform": dst_transform,
                                       "width": width, "height": height,
                                       "resampling": Resampling.bilinear, "nodata": 0}
                        with WarpedVRT(source, **vrt_options) as vrt:
                            incoming = vrt.read([1, 2, 3, 4], window=window)
                        current = destination.read([1, 2, 3, 4], window=window)
                        incoming_valid = np.any(incoming != 0, axis=0)
                        current_empty = np.all(current == 0, axis=0)
                        use = incoming_valid & current_empty
                        if np.any(use):
                            current[:, use] = incoming[:, use]
                            destination.write(current, window=window)
        if staged_path.stat().st_size == 0:
            raise RuntimeError("NAIP mosaic output is empty")
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"Output appeared during download: {output_path}")
        os.replace(staged_path, output_path)
    return output_path


def _tnm_page(bbox, *, endpoint=DEFAULT_TNM_PRODUCTS, dataset=TNM_1M_DATASET,
              offset=0, page_size=50, timeout=60):
    bbox = _validate_bbox(bbox)
    params = {"bbox": ",".join(f"{value:.8f}" for value in bbox),
              "datasets": dataset, "prodFormats": "GeoTIFF",
              "max": int(page_size), "offset": int(offset), "format": "json"}
    request = Request(endpoint + "?" + urlencode(params),
                      headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError) as exc:
        raise RuntimeError(f"TNMAccess query failed: {exc}") from exc


def probe_3dep(bbox, *, endpoint=DEFAULT_TNM_PRODUCTS, dataset=TNM_1M_DATASET,
               max_products=50, timeout=60):
    """Query The National Map product metadata only; download no DEM files."""
    result = _tnm_page(bbox, endpoint=endpoint, dataset=dataset,
                       page_size=max_products, timeout=timeout)
    items = result.get("items", [])
    summaries = [{"title": row.get("title"), "sizeInBytes": row.get("sizeInBytes"),
                  "publicationDate": row.get("publicationDate"),
                  "extent": row.get("extent"), "boundingBox": row.get("boundingBox"),
                  "sourceId": row.get("sourceId"), "metaUrl": row.get("metaUrl"),
                  "downloadURL": row.get("downloadURL")}
                 for row in items]
    return {"endpoint": endpoint, "dataset": dataset,
            "total": int(result.get("total", len(items))),
            "returned": len(items), "truncated": int(result.get("total", len(items))) > len(items),
            "items": summaries}


def _tnm_products(bbox, *, endpoint, dataset, max_products, timeout):
    if max_products < 1:
        raise ValueError("max_products must be positive")
    first = _tnm_page(bbox, endpoint=endpoint, dataset=dataset,
                      page_size=min(max_products, 1000), timeout=timeout)
    total = int(first.get("total", len(first.get("items", []))))
    if total > max_products:
        raise ValueError(f"3DEP search found {total} products, above limit {max_products}; use a smaller bbox")
    products = list(first.get("items", []))
    while len(products) < total:
        page = _tnm_page(bbox, endpoint=endpoint, dataset=dataset,
                         offset=len(products), page_size=min(max_products, 1000), timeout=timeout)
        next_items = page.get("items", [])
        if not next_items:
            break
        products.extend(next_items)
    return products


def _download_file(url, destination, *, max_bytes, timeout=120):
    request = Request(url, headers={"User-Agent": USER_AGENT})
    size = 0
    try:
        with urlopen(request, timeout=timeout) as response, Path(destination).open("wb") as output:
            declared = response.headers.get("Content-Length")
            if declared and int(declared) > max_bytes:
                raise ValueError(f"Remote file is {int(declared)} bytes, above remaining limit {max_bytes}")
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_bytes:
                    raise ValueError(f"Download exceeded byte limit ({max_bytes})")
                output.write(chunk)
    except (HTTPError, URLError) as exc:
        raise RuntimeError(f"Could not download {url}: {exc}") from exc
    return size


def _extract_geotiffs(archive_path, output_dir, *, prefix, remaining_bytes):
    paths = []
    unpacked = 0
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            member = PurePosixPath(info.filename)
            if member.is_absolute() or ".." in member.parts or member.suffix.lower() not in (".tif", ".tiff"):
                continue
            unpacked += info.file_size
            if unpacked > remaining_bytes:
                raise ValueError("Unpacked 3DEP tiles exceed the configured extracted-size limit")
            dest = Path(output_dir) / f"{prefix}_tile_{len(paths):04d}{member.suffix.lower()}"
            with archive.open(info) as source, dest.open("wb") as target:
                shutil.copyfileobj(source, target)
            paths.append(dest)
    return paths, unpacked


def _write_clipped_dtm(raster_paths, bbox, resolution_m, output_path):
    """Warp source DEM tiles onto a bounded grid covering only ``bbox``."""
    dst_crs, dst_transform, width, height = _target_grid(bbox, resolution_m)
    nodata = -9999.0
    profile = {
        "driver": "GTiff", "height": height, "width": width,
        "count": 1, "dtype": "float32", "crs": dst_crs,
        "transform": dst_transform, "nodata": nodata, "tiled": True,
        "blockxsize": 512, "blockysize": 512, "compress": "deflate",
    }
    written_pixels = 0
    with rasterio.open(output_path, "w+", **profile) as destination:
        for row in range(0, height, 512):
            for col in range(0, width, 512):
                window = Window(col, row, min(512, width - col), min(512, height - row))
                destination.write(
                    np.full((int(window.height), int(window.width)), nodata, dtype=np.float32),
                    1, window=window,
                )
        for source_path in raster_paths:
            with rasterio.open(source_path) as source:
                if source.crs is None:
                    raise ValueError(f"3DEP raster has no CRS: {source_path}")
                bounds = transform_bounds(source.crs, dst_crs, *source.bounds, densify_pts=21)
                window = _bounded_window(bounds, dst_transform, width, height)
                if window is None:
                    continue
                vrt_options = {
                    "crs": dst_crs, "transform": dst_transform,
                    "width": width, "height": height,
                    "resampling": Resampling.bilinear, "nodata": nodata,
                }
                if source.nodata is not None:
                    vrt_options["src_nodata"] = source.nodata
                with WarpedVRT(source, **vrt_options) as vrt:
                    incoming = vrt.read(1, window=window)
                current = destination.read(1, window=window)
                incoming_valid = np.isfinite(incoming) & (incoming != nodata)
                current_empty = ~np.isfinite(current) | (current == nodata)
                use = incoming_valid & current_empty
                if np.any(use):
                    current[use] = incoming[use]
                    written_pixels += int(use.sum())
                    destination.write(current, 1, window=window)
    if written_pixels == 0:
        raise RuntimeError("3DEP source rasters did not cover the requested bbox")


def fetch_3dep_dtm(bbox, output_path, *, resolution_m=1.0,
                   endpoint=DEFAULT_TNM_PRODUCTS, dataset=TNM_1M_DATASET,
                   max_raster_km2=25.0, max_raster_pixels=70_000_000,
                   max_download_bytes=2_000_000_000,
                   max_extracted_bytes=8_000_000_000, max_products=200, timeout=120,
                   overwrite=False):
    """Download intersecting 3DEP 1 m products, merge them, publish atomically.

    Area, product-count, compressed-byte, and extracted-byte limits protect
    against accidentally requesting a full-region mosaic. No API key is needed.
    """
    bbox = _validate_bbox(bbox)
    area = _area_km2(bbox)
    if area > max_raster_km2:
        raise ValueError(f"Raster bbox is about {area:.1f} km2, above the configured "
                         f"limit of {max_raster_km2:.1f} km2; use a smaller bbox or raise the limit explicitly")
    _grid_crs, _grid_transform, width, height = _target_grid(bbox, resolution_m)
    _check_grid_size(width, height, max_raster_pixels)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}; pass overwrite=True to replace it")
    products = _tnm_products(bbox, endpoint=endpoint, dataset=dataset,
                             max_products=max_products, timeout=min(timeout, 60))
    if not products:
        raise FileNotFoundError("TNMAccess found no 3DEP DEM products for the bbox")
    advertised = 0
    for row in products:
        try:
            advertised += int(row.get("sizeInBytes") or 0)
        except (TypeError, ValueError):
            # Some product records omit a usable size; the streamed byte cap
            # still applies while downloading those files.
            continue
    if advertised and advertised > max_download_bytes:
        raise ValueError(f"3DEP products advertise {advertised} bytes, above the configured "
                         f"download limit of {max_download_bytes}; use a smaller bbox or raise the limit")

    with tempfile.TemporaryDirectory(prefix="trailnet-3dep-", dir=output_path.parent) as temp_dir:
        temp_dir = Path(temp_dir)
        raster_paths = []
        downloaded = unpacked = 0
        for index, product in enumerate(products):
            url = product.get("downloadURL")
            if not url:
                continue
            remaining = max_download_bytes - downloaded
            suffix = Path(urlsplit(url).path).suffix.lower()
            archive_path = temp_dir / f"product_{index:04d}{suffix}"
            downloaded += _download_file(url, archive_path, max_bytes=remaining, timeout=timeout)
            if zipfile.is_zipfile(archive_path):
                paths, size = _extract_geotiffs(archive_path, temp_dir, prefix=f"product_{index:04d}",
                                                remaining_bytes=max_extracted_bytes-unpacked)
                raster_paths.extend(paths); unpacked += size
                archive_path.unlink()
            elif archive_path.suffix.lower() in (".tif", ".tiff"):
                raster_paths.append(archive_path)
            else:
                raise ValueError(f"Unexpected 3DEP product format for {product.get('title')!r}")
        if not raster_paths:
            raise RuntimeError("3DEP products contained no GeoTIFF tiles")
        staged_path = temp_dir / output_path.name
        try:
            _write_clipped_dtm(raster_paths, bbox, resolution_m, staged_path)
        except Exception as exc:
            raise RuntimeError("Could not warp 3DEP tiles onto the requested bbox grid") from exc
        if not staged_path.exists() or staged_path.stat().st_size == 0:
            raise RuntimeError("3DEP mosaic output is empty")
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"Output appeared during download: {output_path}")
        os.replace(staged_path, output_path)
    return output_path


def build_overpass_query(bbox):
    west, south, east, north = _validate_bbox(bbox)
    box_text = f"{south:.8f},{west:.8f},{north:.8f},{east:.8f}"
    return ("[out:json][timeout:60];\n(\n"
            f'  way["highway"~"^(path|footway|cycleway|track)$"]({box_text});\n'
            ");\nout geom;\n")


def parse_overpass_response(payload, *, area_name=None):
    """Convert an Overpass JSON response into tagged EPSG:4326 line features."""
    features = []
    for element in payload.get("elements", []):
        if element.get("type") != "way":
            continue
        coords = [(point["lon"], point["lat"]) for point in element.get("geometry", [])
                  if "lon" in point and "lat" in point]
        if len(coords) < 2:
            continue
        tags = element.get("tags", {})
        highway = tags.get("highway")
        if highway not in {"path", "footway", "cycleway", "track"}:
            continue
        footway = tags.get("footway")
        if isinstance(footway, str):
            footway = footway.strip().lower() or None
        else:
            footway = None
        # Preserve OSM footway=* (sidewalk/crossing). Do not treat unnamed footway as sidewalk.
        if highway == "footway" and footway in {"sidewalk", "crossing"}:
            trail_type = footway
            class_group = "sidewalk"
        elif highway == "track":
            trail_type = highway
            class_group = "track"
        else:
            trail_type = highway
            class_group = "path"
        features.append({"osm_id": f"way/{element['id']}", "highway": highway,
                         "footway": footway, "trail_type": trail_type, "class_group": class_group,
                         "label_weight": 1.0, "cartocode": None,
                         "name": tags.get("name"), "area": area_name,
                         "source": "OpenStreetMap (ODbL)", "geometry": LineString(coords)})
    return gpd.GeoDataFrame(features, columns=["osm_id", "highway", "footway", "trail_type", "class_group",
                                               "label_weight", "cartocode", "name", "area", "source", "geometry"],
                            geometry="geometry", crs="EPSG:4326")


def fetch_osm_trails(bbox, output_path, *, area_name=None,
                     endpoint=DEFAULT_OVERPASS_URL, token_env=None, timeout=90,
                     max_bbox_km2=5000, overwrite=False, cache_dir=None, refresh=False):
    """Fetch OSM path/footway/track ways and write a staged GeoPackage layer."""
    bbox = _validate_bbox(bbox)
    area = _area_km2(bbox)
    if area > max_bbox_km2:
        raise ValueError(f"OSM bbox is about {area:.1f} km2, above configured limit "
                         f"{max_bbox_km2:.1f} km2")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}; pass overwrite=True to replace it")
    token = os.environ.get(token_env) if token_env else None
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json",
               "Content-Type": "application/x-www-form-urlencoded"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    payload = None
    cache_path = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha256(json.dumps([endpoint, list(bbox)], separators=(",", ":")).encode()).hexdigest()[:20]
        cache_path = cache_dir / f"osm_{key}.json"
        if cache_path.exists() and not refresh:
            with cache_path.open(encoding="utf-8") as stream:
                payload = json.load(stream)
    if payload is None:
        body = urlencode({"data": build_overpass_query(bbox)}).encode("utf-8")
        try:
            with urlopen(Request(endpoint, data=body, headers=headers, method="POST"),
                         timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read(1000).decode("utf-8", errors="replace")
            raise RuntimeError(f"Overpass returned HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"Could not reach Overpass endpoint {endpoint}: {exc.reason}") from exc
        if cache_path is not None:
            stage_path = cache_path.with_suffix(".tmp")
            with stage_path.open("w", encoding="utf-8") as stream:
                json.dump(payload, stream, separators=(",", ":"))
            os.replace(stage_path, cache_path)
    trails = parse_overpass_response(payload, area_name=area_name)
    if trails.empty:
        raise ValueError("Overpass returned no matching trail ways; existing output was preserved")
    with tempfile.TemporaryDirectory(prefix="trailnet-osm-", dir=output_path.parent) as temp_dir:
        staged_path = Path(temp_dir) / output_path.name
        trails.to_file(staged_path, layer="trails", driver="GPKG")
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"Output appeared during request: {output_path}")
        os.replace(staged_path, output_path)
    return output_path


def probe_sources(bbox, *, naip_catalog=DEFAULT_NAIP_STAC,
                  tnm_endpoint=DEFAULT_TNM_PRODUCTS, overpass_endpoint=DEFAULT_OVERPASS_URL,
                  timeout=60):
    """Run bounded metadata-only probes for NAIP and 3DEP, with no asset downloads."""
    return {
        "bbox": list(_validate_bbox(bbox)),
        "approximate_km2": round(_area_km2(bbox), 2),
        "naip": probe_naip(bbox, catalog_url=naip_catalog, max_items=100, timeout=timeout),
        "3dep": probe_3dep(bbox, endpoint=tnm_endpoint, max_products=50, timeout=timeout),
        "osm": {"endpoint": overpass_endpoint,
                "query": "highway=path|footway|track", "network_request": False},
    }
