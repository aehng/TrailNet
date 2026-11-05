"""Build a self-contained Leaflet HTML map from predicted trail GeoJSON.

The map opens by double-clicking it in a browser (it loads Leaflet and map
tiles from the internet). Predicted trails are drawn in orange-red; existing
OSM/USFS trails, when available, are drawn as blue dashed lines for comparison.
"""
import json
from pathlib import Path

import geopandas as gpd

from trailnet.config import resolve_path

LEAFLET_CSS = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
LEAFLET_JS = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
PREDICTED_COLOR = "#e6550d"
EXISTING_COLOR = "#3182bd"

_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<title>{title}</title>
<link rel="stylesheet" href="{css}" />
<script src="{js}"></script>
<style>
  html, body {{ height: 100%; margin: 0; }}
  #map {{ height: 100%; }}
  .legend {{ background: white; padding: 6px 10px; line-height: 1.6; }}
  .legend i {{ display: inline-block; width: 18px; height: 3px; vertical-align: middle; margin-right: 6px; }}
</style>
</head>
<body>
<div id="map"></div>
<script>
  var map = L.map('map').fitBounds({bounds});
  L.tileLayer('https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
    maxZoom: 19,
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
  }}).addTo(map);
  var predicted = {predicted};
  L.geoJSON(predicted, {{ style: {{ color: '{pred_color}', weight: 3 }} }}).addTo(map);
  var existing = {existing};
  if (existing) {{
    L.geoJSON(existing, {{ style: {{ color: '{exist_color}', weight: 2, dashArray: '6 4' }} }}).addTo(map);
  }}
  var legend = L.control({{ position: 'bottomright' }});
  legend.onAdd = function () {{
    var div = L.DomUtil.create('div', 'legend');
    div.innerHTML = '<i style="background:{pred_color}"></i> Predicted trails<br>' +
      '<i style="background:{exist_color}"></i> Existing OSM/USFS trails';
    return div;
  }};
  legend.addTo(map);
</script>
</body>
</html>
"""


def _geojson_dict(path):
    gdf = gpd.read_file(path)
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    return json.loads(gdf.to_json()), gdf.total_bounds


def build_map(config, area, trails_path=None, existing_trails=None, output_path=None):
    """Write a Leaflet map for one area's predicted trails. Returns the path."""
    vector_dir = resolve_path(config, config["paths"]["vector_dir"])
    trails_path = Path(trails_path) if trails_path else vector_dir / f"{area['name']}_trails.geojson"
    if not trails_path.exists():
        raise FileNotFoundError(f"No predicted trails file found: {trails_path}")
    predicted, bounds = _geojson_dict(trails_path)
    existing = None
    existing_path = Path(existing_trails) if existing_trails else resolve_path(config, area["trails"])
    if existing_path.exists():
        existing, _ = _geojson_dict(existing_path)
    if len(bounds) == 4 and all(map(lambda v: v == v, bounds)):
        leaflet_bounds = [[float(bounds[1]), float(bounds[0])], [float(bounds[3]), float(bounds[2])]]
    else:
        west, south, east, north = area["bbox"]
        leaflet_bounds = [[south, west], [north, east]]
    output_path = Path(output_path) if output_path else (
        resolve_path(config, config["paths"]["output_dir"]) / "maps" / f"{area['name']}_map.html"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    title = f"TrailNet — {area['name']}: {area.get('description', '')}"
    html = _HTML.format(
        title=title,
        css=LEAFLET_CSS,
        js=LEAFLET_JS,
        bounds=json.dumps(leaflet_bounds),
        predicted=json.dumps(predicted),
        existing=json.dumps(existing) if existing else "null",
        pred_color=PREDICTED_COLOR,
        exist_color=EXISTING_COLOR,
    )
    output_path.write_text(html, encoding="utf-8")
    return output_path


def build_map_area(config, area):
    return build_map(config, area)
