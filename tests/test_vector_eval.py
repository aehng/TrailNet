"""Synthetic verification of length-weighted vector matching and CRS guards."""
import geopandas as gpd
import pytest
from shapely.geometry import LineString

from trailnet.vector_eval import (
    vector_metrics_tolerance_sweep,
    vector_noise_metrics,
)


def _utm_line(x0, y0, x1, y1):
    return gpd.GeoDataFrame({"geometry": [LineString([(x0, y0), (x1, y1)])]}, crs="EPSG:32612")


def test_identical_lines_match_at_1m():
    a = _utm_line(500000, 4500000, 500100, 4500000)
    metrics = vector_noise_metrics(a, a, match_tolerance_m=1.0)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["pred_km"] == pytest.approx(0.1)


def test_parallel_offset_4m_misses_at_3_matches_at_5():
    pred = _utm_line(500000, 4500000, 500100, 4500000)
    label = _utm_line(500000, 4500004, 500100, 4500004)
    low = vector_noise_metrics(pred, label, match_tolerance_m=3.0)
    high = vector_noise_metrics(pred, label, match_tolerance_m=5.0)
    assert low["precision"] == pytest.approx(0.0)
    assert low["recall"] == pytest.approx(0.0)
    assert high["precision"] == pytest.approx(1.0)
    assert high["recall"] == pytest.approx(1.0)


def test_nonmatch_far_apart_stays_zero_even_at_10m():
    pred = _utm_line(500000, 4500000, 500100, 4500000)
    label = _utm_line(500000, 4500100, 500100, 4500100)
    metrics = vector_noise_metrics(pred, label, match_tolerance_m=10.0)
    assert metrics["precision"] == pytest.approx(0.0)
    assert metrics["recall"] == pytest.approx(0.0)


def test_tolerance_sweep_reports_each_value_without_picking_winner():
    pred = _utm_line(500000, 4500000, 500100, 4500000)
    label = _utm_line(500000, 4500004, 500100, 4500004)
    sweep = vector_metrics_tolerance_sweep(pred, label, tolerances_m=(1, 3, 5, 8, 10))
    assert [row["match_tolerance_m"] for row in sweep] == [1.0, 3.0, 5.0, 8.0, 10.0]
    assert sweep[0]["precision"] == pytest.approx(0.0)
    assert sweep[1]["precision"] == pytest.approx(0.0)
    assert sweep[2]["precision"] == pytest.approx(1.0)


def test_rejects_metre_coordinates_tagged_as_epsg4326():
    pred = _utm_line(500000, 4500000, 500100, 4500000)
    # Metre coordinates falsely declared as lon/lat.
    bad = gpd.GeoDataFrame({"geometry": list(pred.geometry)}, crs="EPSG:4326")
    with pytest.raises(ValueError, match="not geographic|Unable to determine UTM|metre"):
        vector_noise_metrics(bad, bad, match_tolerance_m=5.0)


def test_rejects_projected_crs_with_geographic_bounds():
    # Lon/lat numbers wrongly tagged as UTM.
    bad = gpd.GeoDataFrame(
        {"geometry": [LineString([(-111.8, 40.7), (-111.799, 40.701)])]},
        crs="EPSG:32612",
    )
    with pytest.raises(ValueError, match="look geographic|CRS mis-tag"):
        vector_noise_metrics(bad, bad, match_tolerance_m=5.0)


def test_rejects_missing_label_crs():
    pred = _utm_line(500000, 4500000, 500100, 4500000)
    labels = pred.copy()
    labels = labels.set_crs(None, allow_override=True)
    with pytest.raises(ValueError, match="no CRS"):
        vector_noise_metrics(pred, labels, match_tolerance_m=5.0)


def test_geographic_inputs_are_projected_to_metres():
    pred = _utm_line(500000, 4500000, 500100, 4500000).to_crs(4326)
    label = pred.copy()
    metrics = vector_noise_metrics(pred, label, match_tolerance_m=3.0)
    assert metrics["precision"] == pytest.approx(1.0)
    assert "326" in metrics["crs"] or "utm" in metrics["crs"].lower()
