"""Standalone Leaflet preview maps from local NAIP, 3DEP derivatives, and trails.

The HTML file plus sibling PNGs open from disk with no public map-tile server.
Predicted trails are blaze orange; existing OSM/USFS/UGRC trails are steel blue.
"""
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject, transform_bounds
from skimage.io import imsave

from trailnet.config import resolve_path
from trailnet.data.terrain import derive_terrain

PREVIEW_MAX_EDGE = 2048
PREDICTED_COLOR = "#e24a16"
EXISTING_COLOR = "#5ea4d4"

LEAFLET_CSS = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
LEAFLET_JS = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"

_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{title}</title>
<link rel="stylesheet" href="{css}" />
<script src="{js}"></script>
<style>
  :root {{
    --ink: #d8d0c2;
    --muted: #8a8478;
    --panel: rgba(18, 20, 16, 0.92);
    --line: rgba(216, 208, 194, 0.16);
    --blaze: {pred_color};
    --steel: {exist_color};
  }}
  html, body {{ height: 100%; margin: 0; background: #0e100c; color: var(--ink);
    font: 14px/1.4 "Segoe UI", "Candara", "Gill Sans", sans-serif; }}
  #map {{ height: 100%; background: #0e100c; }}
  .hud {{
    position: absolute; z-index: 500; top: 12px; left: 12px; width: 272px;
    background: var(--panel); border: 1px solid var(--line); color: var(--ink);
    padding: 14px 14px 12px; backdrop-filter: blur(8px);
  }}
  .hud h1 {{
    font-family: Palatino, "Palatino Linotype", "Iowan Old Style", serif;
    font-size: 1.15rem; font-weight: 600; margin: 0 0 2px; letter-spacing: 0.01em;
  }}
  .hud .sub {{ color: var(--muted); font-size: 0.78rem; margin: 0 0 12px; }}
  .group {{ border-top: 1px solid var(--line); padding: 8px 0 2px; }}
  .group h2 {{
    margin: 0 0 6px; font-size: 0.68rem; font-weight: 600; color: var(--muted);
    letter-spacing: 0.04em;
  }}
  label.opt {{
    display: flex; align-items: center; gap: 8px; margin: 0 0 5px;
    cursor: pointer; font-size: 0.86rem;
  }}
  label.opt input {{ margin: 0; accent-color: var(--blaze); }}
  .swatch {{ width: 18px; height: 0; border-top: 3px solid; flex: 0 0 18px; }}
  .swatch.dash {{ border-top-style: dashed; }}
  .note {{ color: var(--muted); font-size: 0.72rem; margin: 8px 0 0; }}
  .leaflet-control-attribution {{ background: rgba(14,16,12,0.7); color: #7a7468; }}
  .leaflet-control-attribution a {{ color: #a89f90; }}
</style>
</head>
<body>
<div id="map"></div>
<aside class="hud">
  <h1>{heading}</h1>
  <p class="sub">{subtitle}</p>
  <div class="group">
    <h2>Imagery</h2>
    <label class="opt"><input type="radio" name="base" value="rgb" checked> RGB</label>
    <label class="opt"><input type="radio" name="base" value="cir"> Color infrared</label>
    <label class="opt"><input type="radio" name="base" value="nir"> Near infrared</label>
  </div>
  <div class="group">
    <h2>Terrain (3DEP DTM)</h2>
    <label class="opt"><input type="radio" name="base" value="hillshade"> Hillshade</label>
    <label class="opt"><input type="radio" name="base" value="slope"> Slope</label>
    <label class="opt"><input type="radio" name="base" value="relief"> Local relief</label>
  </div>
  <div class="group">
    <h2>Trails</h2>
    <label class="opt"><input type="checkbox" id="toggle-existing" checked>
      <span class="swatch dash" style="border-color: var(--steel)"></span> Existing trails</label>
    <label class="opt"><input type="checkbox" id="toggle-predicted" checked>
      <span class="swatch" style="border-color: var(--blaze)"></span> Predicted trails</label>
    <label class="opt"><input type="checkbox" id="toggle-probability"> Model probability</label>
  </div>
  <p class="note">Hillshade, slope, and local relief come from the 3DEP 1 m DTM (lidar-derived), not a LAS point cloud. No public map tiles are loaded.</p>
</aside>
<script>
  var bounds = {bounds};
  var map = L.map('map', {{ zoomControl: false, attributionControl: true }})
    .fitBounds(bounds, {{ padding: [24, 24] }});
  L.control.zoom({{ position: 'topright' }}).addTo(map);
  var bases = {{
    rgb: L.imageOverlay({rgb}, bounds),
    cir: L.imageOverlay({cir}, bounds),
    nir: L.imageOverlay({nir}, bounds),
    hillshade: L.imageOverlay({hillshade}, bounds),
    slope: L.imageOverlay({slope}, bounds),
    relief: L.imageOverlay({relief}, bounds)
  }};
  var currentBase = bases.rgb;
  currentBase.addTo(map);
  document.querySelectorAll('input[name="base"]').forEach(function (input) {{
    input.addEventListener('change', function () {{
      if (!this.checked) return;
      map.removeLayer(currentBase);
      currentBase = bases[this.value];
      currentBase.addTo(map);
    }});
  }});
  var predicted = L.geoJSON({predicted}, {{
    style: {{ color: '{pred_color}', weight: 2.4, opacity: 0.92 }}
  }}).addTo(map);
  var existing = L.geoJSON({existing}, {{
    style: {{ color: '{exist_color}', weight: 2, opacity: 0.9, dashArray: '7 5' }}
  }}).addTo(map);
  var probability = L.imageOverlay({probability}, bounds, {{ opacity: 0.55 }});
  document.getElementById('toggle-predicted').addEventListener('change', function () {{
    this.checked ? map.addLayer(predicted) : map.removeLayer(predicted);
  }});
  document.getElementById('toggle-existing').addEventListener('change', function () {{
    this.checked ? map.addLayer(existing) : map.removeLayer(existing);
  }});
  document.getElementById('toggle-probability').addEventListener('change', function () {{
    this.checked ? map.addLayer(probability) : map.removeLayer(probability);
  }});
</script>
</body>
</html>
"""


def _preview_hw(height, width, max_edge=PREVIEW_MAX_EDGE):
    long_edge = max(height, width)
    if long_edge <= max_edge:
        return height, width
    scale = max_edge / long_edge
    return max(1, int(round(height * scale))), max(1, int(round(width * scale)))


def _stretch_u8(band, lo=2, hi=98):
    finite = np.asarray(band, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.zeros(np.shape(band), dtype=np.uint8)
    p2, p98 = np.percentile(finite, [lo, hi])
    if p98 <= p2:
        return np.zeros(np.shape(band), dtype=np.uint8)
    stretched = (np.asarray(band, dtype=np.float32) - p2) / (p98 - p2)
    return np.clip(stretched * 255, 0, 255).astype(np.uint8)


def _save_png(path, array):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.ascontiguousarray(array)
    imsave(path, array)


def _stack_rgb(*bands):
    return np.stack([_stretch_u8(b) for b in bands], axis=-1)


def _gray_rgb(band):
    gray = _stretch_u8(band)
    return np.stack((gray, gray, gray), axis=-1)


def _slope_rgb(slope01):
    gray = _stretch_u8(np.clip(slope01, 0, 1) * 255.0)
    return np.stack((gray, (gray.astype(np.uint16) * 180 // 255).astype(np.uint8),
                     (gray.astype(np.uint16) * 40 // 255).astype(np.uint8)), axis=-1)


def _probability_rgba(prob):
    prob = np.clip(np.nan_to_num(np.asarray(prob, dtype=np.float32), nan=0.0), 0, 1)
    alpha = np.clip(prob * 210, 0, 210).astype(np.uint8)
    r = np.full(prob.shape, 226, dtype=np.uint8)
    g = np.full(prob.shape, 74, dtype=np.uint8)
    b = np.full(prob.shape, 22, dtype=np.uint8)
    return np.stack((r, g, b, alpha), axis=-1)


def _empty_fc():
    return {"type": "FeatureCollection", "features": []}


def _looks_like_projected_metres(gdf):
    try:
        bounds = gdf.total_bounds
        west, south, east, north = [float(v) for v in bounds]
    except Exception:
        return False
    return max(abs(west), abs(south), abs(east), abs(north)) > 180.0


def _to_wgs84_or_raise(gdf, *, source="vector"):
    if gdf.crs is None:
        if _looks_like_projected_metres(gdf):
            raise ValueError(
                f"{source} has no CRS and coordinates look like projected metres; "
                "refusing to treat them as EPSG:4326"
            )
        raise ValueError(
            f"{source} has no CRS; refusing to treat coordinates as EPSG:4326"
        )
    epsg = gdf.crs.to_epsg() if gdf.crs is not None else None
    if epsg == 4326 and _looks_like_projected_metres(gdf):
        raise ValueError(
            f"{source} claims EPSG:4326 but coordinates look like projected metres"
        )
    if epsg != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    return gdf


def _geojson_dict(path):
    if path is None or not Path(path).exists():
        return _empty_fc(), None
    gdf = gpd.read_file(path)
    if gdf.empty or gdf.geometry.isna().all():
        return _empty_fc(), None
    gdf = _to_wgs84_or_raise(gdf, source=str(path))
    return json.loads(gdf.to_json()), gdf.total_bounds


def _write_preview_pngs(config, area, asset_dir):
    image_path = resolve_path(config, area["imagery"])
    dtm_path = resolve_path(config, area["dtm"])
    pred_path = resolve_path(config, config["paths"]["prediction_dir"]) / f"{area['name']}_probability.tif"
    radius = int(config.get("terrain", {}).get("local_relief_radius_px", 25))
    with rasterio.open(image_path) as image_ds:
        height, width = _preview_hw(image_ds.height, image_ds.width)
        rgb = image_ds.read([1, 2, 3], out_shape=(3, height, width), resampling=Resampling.average)
        nir = image_ds.read(4, out_shape=(height, width), resampling=Resampling.average) if image_ds.count >= 4 else rgb[0]
        west, south, east, north = transform_bounds(
            image_ds.crs, "EPSG:4326", *image_ds.bounds, densify_pts=21)
        dst_transform = Affine(
            image_ds.transform.a * image_ds.width / width, image_ds.transform.b, image_ds.transform.c,
            image_ds.transform.d, image_ds.transform.e * image_ds.height / height, image_ds.transform.f,
        )
        src_crs = image_ds.crs
        scaled_radius = max(1, int(round(radius * (width / image_ds.width))))
    dtm = np.full((height, width), np.nan, dtype=np.float32)
    with rasterio.open(dtm_path) as dtm_ds:
        reproject(
            source=rasterio.band(dtm_ds, 1), destination=dtm,
            src_transform=dtm_ds.transform, src_crs=dtm_ds.crs, src_nodata=dtm_ds.nodata,
            dst_transform=dst_transform, dst_crs=src_crs, dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
    hillshade, slope, relief = derive_terrain(dtm, dst_transform.a, dst_transform.e, scaled_radius)
    _save_png(asset_dir / "rgb.png", _stack_rgb(rgb[0], rgb[1], rgb[2]))
    _save_png(asset_dir / "nir.png", _gray_rgb(nir))
    _save_png(asset_dir / "cir.png", _stack_rgb(nir, rgb[0], rgb[1]))
    _save_png(asset_dir / "hillshade.png", _gray_rgb(hillshade * 255.0))
    _save_png(asset_dir / "slope.png", _slope_rgb(slope))
    _save_png(asset_dir / "relief.png", _gray_rgb(relief * 255.0))
    if pred_path.exists():
        prob = np.zeros((height, width), dtype=np.float32)
        with rasterio.open(pred_path) as pred_ds:
            reproject(
                source=rasterio.band(pred_ds, 1), destination=prob,
                src_transform=pred_ds.transform, src_crs=pred_ds.crs, src_nodata=pred_ds.nodata,
                dst_transform=dst_transform, dst_crs=src_crs, dst_nodata=0,
                resampling=Resampling.bilinear,
            )
        _save_png(asset_dir / "probability.png", _probability_rgba(prob))
    else:
        _save_png(asset_dir / "probability.png", _probability_rgba(np.zeros((height, width), np.float32)))
    return [[float(south), float(west)], [float(north), float(east)]]


def build_map(config, area, trails_path=None, existing_trails=None, output_path=None):
    """Write a Leaflet map for one area. Returns the HTML path."""
    vector_dir = resolve_path(config, config["paths"]["vector_dir"])
    trails_path = Path(trails_path) if trails_path else vector_dir / f"{area['name']}_trails.geojson"
    existing_path = Path(existing_trails) if existing_trails else resolve_path(config, area["trails"])
    output_path = Path(output_path) if output_path else (
        resolve_path(config, config["paths"]["output_dir"]) / "maps" / f"{area['name']}_map.html"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    asset_dir = output_path.parent / area["name"]
    leaflet_bounds = _write_preview_pngs(config, area, asset_dir)
    predicted, _pred_bounds = _geojson_dict(trails_path if trails_path.exists() else None)
    existing, _exist_bounds = _geojson_dict(existing_path if existing_path.exists() else None)
    rel = area["name"]
    title = f"TrailNet — {area['name']}"
    heading = area["name"].replace("_", " ")
    subtitle = area.get("description") or "NAIP RGB / CIR over 3DEP terrain"
    html = _HTML.format(
        title=title,
        heading=heading,
        subtitle=subtitle,
        css=LEAFLET_CSS,
        js=LEAFLET_JS,
        bounds=json.dumps(leaflet_bounds),
        rgb=json.dumps(f"{rel}/rgb.png"),
        cir=json.dumps(f"{rel}/cir.png"),
        nir=json.dumps(f"{rel}/nir.png"),
        hillshade=json.dumps(f"{rel}/hillshade.png"),
        slope=json.dumps(f"{rel}/slope.png"),
        relief=json.dumps(f"{rel}/relief.png"),
        probability=json.dumps(f"{rel}/probability.png"),
        predicted=json.dumps(predicted),
        existing=json.dumps(existing),
        pred_color=PREDICTED_COLOR,
        exist_color=EXISTING_COLOR,
    )
    output_path.write_text(html, encoding="utf-8")
    return output_path


def build_map_area(config, area):
    return build_map(config, area)
