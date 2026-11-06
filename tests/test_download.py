"""Offline tests for source metadata parsing and export attributes."""
import xml.etree.ElementTree as ET

import geopandas as gpd
from shapely.geometry import LineString

from trailnet.data.download import parse_overpass_response
from trailnet.data.labels import _drop_unnamed_osm_footway
from trailnet.export import export_geojson, export_gpx, export_kml


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
