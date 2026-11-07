import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.screen_next_sites import (
    classify_highway,
    line_length_km,
    product_bbox_coverage,
    site_bbox,
)


def test_site_bbox_is_centered_and_uses_meter_buffer():
    bbox = site_bbox(-111.0, 40.0, 3000)

    assert len(bbox) == 4
    assert bbox[0] < -111.0 < bbox[2]
    assert bbox[1] < 40.0 < bbox[3]


def test_osm_tags_separate_trail_candidate_sidewalk_and_road():
    assert classify_highway({"highway": "path"}) == "path_candidate"
    assert classify_highway({"highway": "track"}) == "track_candidate"
    assert classify_highway({"highway": "footway", "footway": "sidewalk"}) == "sidewalk"
    assert classify_highway({"highway": "footway", "footway": "crossing"}) == "sidewalk"
    assert classify_highway({"highway": "residential"}) == "road"
    assert classify_highway({"highway": "footway"}) == "path_candidate"


def test_line_length_uses_geodesic_segments():
    km = line_length_km([[0.0, 0.0], [0.0, 0.01]])

    assert 1.10 < km < 1.12


def test_3dep_product_bbox_coverage_marks_metadata_extent_as_estimate():
    result = product_bbox_coverage(
        [-111.0, 40.0, -110.9, 40.1],
        [{"boundingBox": {"minX": -111.0, "minY": 40.0,
                          "maxX": -110.9, "maxY": 40.1}}],
    )

    assert result["method"] == "TNM metadata boundingBox union"
    assert result["estimated_fraction"] == 1.0
    assert result["raster_validity_verified"] is False
