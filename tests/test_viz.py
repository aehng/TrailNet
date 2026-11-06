"""Preview maps: local NAIP/3DEP layers, no OSM public tiles, auto-built after predict."""
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString

from trailnet.postprocess import vectorize_area
from trailnet.viz import build_map_area


def _write_raster(path, array, *, crs="EPSG:32612", transform=None, nodata=None):
    array = np.asarray(array)
    if array.ndim == 2:
        array = array[None]
    with rasterio.open(
        path, "w", driver="GTiff", width=array.shape[2], height=array.shape[1],
        count=array.shape[0], dtype=array.dtype, crs=crs,
        transform=transform or from_origin(500000, 4500000, 0.6, 0.6), nodata=nodata,
    ) as dst:
        dst.write(array)


def _mini_area(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    yy, xx = np.mgrid[:64, :64]
    imagery = data / "naip.tif"
    dtm = data / "dtm.tif"
    trails = data / "trails.gpkg"
    _write_raster(
        imagery,
        np.stack([xx * 40, yy * 40, (xx + yy) * 20, 200 + xx], axis=0).astype("uint16"),
    )
    _write_raster(dtm, (100 + xx * 0.4 + yy * 0.2).astype("float32"))
    gpd.GeoDataFrame(
        [{"highway": "path", "class_group": "path", "source": "test",
          "geometry": LineString([(500005, 4499980), (500030, 4499980)])}],
        crs="EPSG:32612",
    ).to_file(trails, driver="GPKG")
    probability = np.zeros((64, 64), dtype=np.float32)
    probability[30:34, 8:56] = 0.92
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir()
    _write_raster(pred_dir / "probe_probability.tif", probability)
    config = {
        "_root": tmp_path,
        "paths": {
            "output_dir": "outputs",
            "prediction_dir": "predictions",
            "vector_dir": "vectors",
        },
        "terrain": {"local_relief_radius_px": 4, "channel_gain": 1.85},
        "postprocess": {"threshold": 0.5, "simplify_m": 0.1, "dedup_tolerance_m": 0.01},
        "inference": {"threshold": 0.5},
    }
    area = {
        "name": "probe",
        "description": "viz fixture",
        "bbox": [-111.6, 40.6, -111.5, 40.7],
        "imagery": "data/naip.tif",
        "dtm": "data/dtm.tif",
        "trails": "data/trails.gpkg",
    }
    return config, area


def test_preview_map_writes_local_raster_layers_without_osm_tiles(tmp_path):
    config, area = _mini_area(tmp_path)
    html_path = build_map_area(config, area)
    html = html_path.read_text(encoding="utf-8")
    assets = html_path.parent / "probe"

    assert html_path.name == "probe_map.html"
    assert "tile.openstreetmap.org" not in html
    assert "openstreetmap.org/{z}/{x}/{y}" not in html
    for name in ("rgb.png", "nir.png", "cir.png", "hillshade.png", "slope.png",
                 "relief.png", "probability.png"):
        png = assets / name
        assert png.exists(), name
        assert png.stat().st_size > 0
    for label in ("RGB", "Near infrared", "Color infrared", "Hillshade", "Slope",
                  "Local relief", "Existing trails", "Predicted trails",
                  "Model probability"):
        assert label in html
    # Georeference must be lon/lat, not UTM metres.
    assert "-111." in html
    assert "500000" not in html


def test_vectorize_rebuilds_preview_map_with_predicted_lines(tmp_path):
    config, area = _mini_area(tmp_path)
    vector_path = vectorize_area(config, area)
    html = (tmp_path / "outputs" / "maps" / "probe_map.html").read_text(encoding="utf-8")
    assert vector_path.exists()
    assert "Predicted trails" in html
    assert '"type": "FeatureCollection"' in html


def test_geojson_dict_refuses_projected_metres_without_crs(tmp_path):
    from trailnet.viz import _geojson_dict
    path = tmp_path / "metres.geojson"
    gpd.GeoDataFrame(
        {"geometry": [LineString([(500000.0, 4500000.0), (500100.0, 4500100.0)])]},
        crs=None,
    ).to_file(path, driver="GeoJSON")
    try:
        _geojson_dict(path)
    except ValueError as exc:
        message = str(exc).lower()
        assert "projected metres" in message or "crs" in message
        assert "4326" in message or "epsg" in message
    else:
        raise AssertionError("projected metres without CRS were treated as lon/lat")
