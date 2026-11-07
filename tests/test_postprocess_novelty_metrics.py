"""Novelty filter must not feed vector metrics; unfiltered lines score at 3m."""
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString

from trailnet.postprocess import vectorize_probability
from trailnet.vector_eval import vector_noise_metrics


def _write_prob_tif(path, arr, transform, crs="EPSG:32612"):
    path = Path(path)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
    ) as ds:
        ds.write(arr.astype("float32"), 1)


def test_unfiltered_trails_match_labels_novelty_empty(tmp_path):
    # Horizontal trail along row 10, cols 5..45 in a 50x50 grid at 1m pixels.
    transform = from_origin(500000, 4500050, 1.0, 1.0)
    prob = np.zeros((50, 50), dtype=np.float32)
    prob[10, 5:46] = 0.95
    prob_path = tmp_path / "prob.tif"
    _write_prob_tif(prob_path, prob, transform)

    # Label almost on the same line (1m north) — within 3m.
    label = gpd.GeoDataFrame(
        {"geometry": [LineString([(500005, 4500039), (500045, 4500039)])]},
        crs="EPSG:32612",
    )
    label_path = tmp_path / "labels.gpkg"
    label.to_file(label_path, driver="GPKG")

    trails = tmp_path / "trails.geojson"
    novelty = tmp_path / "novelty.geojson"
    vectorize_probability(
        prob_path, trails, threshold=0.5, simplify_m=1.0,
        existing_trails=label_path, dedup_tolerance_m=3.0, area_name="t",
        novelty_filter=False, novelty_output_path=novelty,
    )
    pred = gpd.read_file(trails)
    nov = gpd.read_file(novelty)
    assert len(pred) > 0
    assert len(nov) == 0  # within 3m of labels → filtered from novelty
    metrics = vector_noise_metrics(pred.to_crs("EPSG:32612"), label, match_tolerance_m=3.0)
    assert metrics["precision"] > 0.5
    assert metrics["recall"] > 0.5


def test_legacy_novelty_filter_zeros_metric_at_dedup_tol(tmp_path):
    transform = from_origin(500000, 4500050, 1.0, 1.0)
    prob = np.zeros((50, 50), dtype=np.float32)
    prob[10, 5:46] = 0.95
    prob_path = tmp_path / "prob.tif"
    _write_prob_tif(prob_path, prob, transform)
    label = gpd.GeoDataFrame(
        {"geometry": [LineString([(500005, 4500039), (500045, 4500039)])]},
        crs="EPSG:32612",
    )
    label_path = tmp_path / "labels.gpkg"
    label.to_file(label_path, driver="GPKG")
    out = tmp_path / "novelty_only.geojson"
    vectorize_probability(
        prob_path, out, threshold=0.5, simplify_m=1.0,
        existing_trails=label_path, dedup_tolerance_m=3.0, area_name="t",
        novelty_filter=True,
    )
    nov = gpd.read_file(out)
    metrics = vector_noise_metrics(nov.to_crs("EPSG:32612"), label, match_tolerance_m=3.0)
    assert len(nov) == 0
    assert metrics["precision"] == pytest.approx(0.0)
    assert metrics["pred_km"] == pytest.approx(0.0)
