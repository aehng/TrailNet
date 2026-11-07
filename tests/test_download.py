"""Offline tests for source metadata parsing and export attributes."""
import xml.etree.ElementTree as ET

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString

from trailnet.data.download import fetch_naip, parse_overpass_response, probe_3dep
from trailnet.data.labels import _drop_unnamed_osm_footway
from trailnet.export import export_geojson, export_gpx, export_kml


def test_3dep_probe_preserves_product_footprints_and_resolution_metadata(monkeypatch):
    from trailnet.data import download

    monkeypatch.setattr(download, "_tnm_page", lambda *args, **kwargs: {
        "total": 1,
        "items": [{"title": "USGS one meter tile", "sizeInBytes": 123,
                   "publicationDate": "2025-01-01", "extent": "10000 x 10000 meter",
                   "boundingBox": {"minX": -111.9, "minY": 40.6,
                                   "maxX": -111.8, "maxY": 40.7},
                   "sourceId": "source-1", "metaUrl": "https://example.test/meta",
                   "downloadURL": "https://example.test/dem.tif"}],
    })

    result = probe_3dep([-112, 40.5, -111, 41])
    product = result["items"][0]
    assert product["boundingBox"]["minX"] == -111.9
    assert product["extent"] == "10000 x 10000 meter"
    assert product["sourceId"] == "source-1"


def test_fetch_naip_marks_fourth_output_band_undefined_not_alpha(tmp_path, monkeypatch):
    import numpy as np
    import rasterio
    from rasterio.enums import ColorInterp
    from rasterio.transform import from_origin
    from rasterio.warp import transform_bounds
    from trailnet.data import download

    source_path = tmp_path / "naip-source.tif"
    source_profile = {
        "driver": "GTiff", "width": 100, "height": 100, "count": 4,
        "dtype": "uint8", "crs": "EPSG:32612",
        "transform": from_origin(500000, 4500000, 0.6, 0.6), "nodata": 0,
    }
    source_values = np.stack([np.full((100, 100), value, dtype=np.uint8)
                              for value in (20, 40, 60, 80)])
    with rasterio.open(source_path, "w", **source_profile) as source:
        source.write(source_values)
        source.colorinterp = (ColorInterp.red, ColorInterp.green,
                              ColorInterp.blue, ColorInterp.undefined)
    bounds_4326 = transform_bounds(
        "EPSG:32612", "EPSG:4326", 500000, 4499940, 500060, 4500000,
        densify_pts=21,
    )
    bbox = [bounds_4326[0], bounds_4326[1], bounds_4326[2], bounds_4326[3]]
    monkeypatch.setattr(download, "_target_grid", lambda *_: (
        "EPSG:32612", from_origin(500000, 4500000, 0.6, 0.6), 100, 100,
    ))
    monkeypatch.setattr(download, "_stac_search", lambda *_, **__: ([{
        "id": "fixture-2021", "properties": {"datetime": "2021-06-01T00:00:00Z"},
        "assets": {"image": {"href": str(source_path)}},
        "bbox": list(bounds_4326),
    }], False))
    monkeypatch.setattr(download, "_pc_signed_href", lambda href, *_args, **_kwargs: href)

    output = fetch_naip(bbox, tmp_path / "naip-output.tif", max_raster_km2=25)
    with rasterio.open(output) as result:
        assert result.colorinterp == (ColorInterp.red, ColorInterp.green,
                                      ColorInterp.blue, ColorInterp.undefined)
        np.testing.assert_array_equal(result.read(), source_values)


def test_overpass_response_becomes_tagged_trail_lines():
    payload = {
        "elements": [
            {
                "type": "way",
                "id": 123,
                "tags": {"highway": "path", "name": "Creek Trail"},
                "geometry": [
                    {"lon": -111.8, "lat": 40.7},
                    {"lon": -111.799, "lat": 40.701},
                ],
            }
        ]
    }
    trails = parse_overpass_response(payload, area_name="northern_utah")
    assert len(trails) == 1
    assert trails.crs.to_epsg() == 4326
    assert trails.iloc[0]["osm_id"] == "way/123"
    assert trails.iloc[0]["highway"] == "path"
    assert trails.iloc[0]["name"] == "Creek Trail"


def test_all_export_formats_preserve_name_area_source_and_confidence(tmp_path):
    trails = gpd.GeoDataFrame(
        [{"trail_id": 7, "name": "Creek Trail", "area": "northern_utah",
          "source": "TrailNet prediction", "confidence": 0.87,
          "geometry": LineString([(-111.8, 40.7), (-111.799, 40.701)])}],
        crs="EPSG:4326",
    )

    geojson_path = tmp_path / "trails.geojson"
    export_geojson(trails, geojson_path, "northern_utah")
    exported = gpd.read_file(geojson_path).iloc[0]
    assert exported["name"] == "Creek Trail"
    assert exported["area"] == "northern_utah"
    assert exported["source"] == "TrailNet prediction"
    assert abs(float(exported["confidence"]) - 0.87) < 1e-6

    gpx_path = tmp_path / "trails.gpx"
    export_gpx(trails, gpx_path, "northern_utah")
    gpx = ET.parse(gpx_path).getroot()
    ns = {"g": "http://www.topografix.com/GPX/1/1"}
    assert gpx.findtext(".//g:trk/g:name", namespaces=ns) == "Creek Trail"
    assert "confidence=0.87" in gpx.findtext(".//g:trk/g:desc", namespaces=ns)
    assert "source=TrailNet prediction" in gpx.findtext(".//g:trk/g:desc", namespaces=ns)

    kml_path = tmp_path / "trails.kml"
    export_kml(trails, kml_path, "northern_utah")
    kml = ET.parse(kml_path).getroot()
    kns = {"k": "http://www.opengis.net/kml/2.2"}
    assert kml.findtext(".//k:Placemark/k:name", namespaces=kns) == "Creek Trail"
    data = {node.attrib["name"]: node.findtext("k:value", namespaces=kns)
            for node in kml.findall(".//k:Placemark/k:ExtendedData/k:Data", namespaces=kns)}
    assert data["area"] == "northern_utah"
    assert data["source"] == "TrailNet prediction"
    assert abs(float(data["confidence"]) - 0.87) < 1e-6


def test_drop_unnamed_osm_footway_keeps_named_and_non_footway():
    frame = gpd.GeoDataFrame(
        [
            {"source": "OpenStreetMap", "trail_type": "footway", "name": None,
             "highway": "path", "geometry": LineString([(0, 0), (1, 0)])},
            {"source": "OpenStreetMap", "trail_type": "footway", "name": "Jungle Trail",
             "highway": "path", "geometry": LineString([(0, 1), (1, 1)])},
            {"source": "OpenStreetMap", "trail_type": "path", "name": None,
             "highway": "path", "geometry": LineString([(0, 2), (1, 2)])},
            {"source": "Utah UGRC TrailsAndPathways", "trail_type": "footway", "name": None,
             "highway": "path", "geometry": LineString([(0, 3), (1, 3)])},
        ],
        crs="EPSG:32612",
    )
    kept = _drop_unnamed_osm_footway(frame, enabled=True)
    names = list(kept["name"].fillna("").astype(str))
    assert len(kept) == 3
    assert "Jungle Trail" in names
    assert not any(
        str(row["trail_type"]).lower() == "footway"
        and "openstreetmap" in str(row["source"]).lower()
        and str(row["name"] or "").strip() == ""
        for _, row in kept.iterrows()
    )
    unchanged = _drop_unnamed_osm_footway(frame, enabled=False)
    assert len(unchanged) == 4


def test_overpass_preserves_explicit_sidewalk_tags_not_unnamed_footway():
    payload = {
        "elements": [
            {
                "type": "way",
                "id": 1,
                "tags": {"highway": "footway", "footway": "sidewalk"},
                "geometry": [{"lon": -111.8, "lat": 40.7}, {"lon": -111.799, "lat": 40.701}],
            },
            {
                "type": "way",
                "id": 2,
                "tags": {"highway": "footway", "footway": "crossing"},
                "geometry": [{"lon": -111.8, "lat": 40.7}, {"lon": -111.7995, "lat": 40.7005}],
            },
            {
                "type": "way",
                "id": 3,
                "tags": {"highway": "footway", "name": "Jungle Trail"},
                "geometry": [{"lon": -111.81, "lat": 40.71}, {"lon": -111.809, "lat": 40.711}],
            },
            {
                "type": "way",
                "id": 4,
                "tags": {"highway": "footway"},
                "geometry": [{"lon": -111.82, "lat": 40.72}, {"lon": -111.819, "lat": 40.721}],
            },
        ]
    }
    trails = parse_overpass_response(payload, area_name="corner_canyon")
    by_id = {row.osm_id: row for _, row in trails.iterrows()}
    assert by_id["way/1"]["footway"] == "sidewalk"
    assert by_id["way/1"]["trail_type"] == "sidewalk"
    assert by_id["way/1"]["class_group"] == "sidewalk"
    assert by_id["way/2"]["trail_type"] == "crossing"
    assert by_id["way/2"]["class_group"] == "sidewalk"
    # Named / unnamed footways are NOT sidewalks without footway=sidewalk.
    assert by_id["way/3"]["trail_type"] == "footway"
    assert by_id["way/3"]["class_group"] == "path"
    assert by_id["way/4"]["trail_type"] == "footway"
    assert by_id["way/4"]["class_group"] == "path"
    assert pd.isna(by_id["way/4"]["footway"]) or by_id["way/4"]["footway"] in (None, "")
