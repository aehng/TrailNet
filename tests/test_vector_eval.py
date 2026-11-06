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


def test_clip_pair_to_raster_drops_out_of_footprint_label(tmp_path):
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin
    from trailnet.vector_eval import _clip_pair_to_raster, vector_noise_metrics

    transform = from_origin(500000, 4500100, 1.0, 1.0)
    arr = np.zeros((100, 100), dtype=np.float32)
    path = tmp_path / "prob.tif"
    with rasterio.open(
        path, "w", driver="GTiff", height=100, width=100, count=1, dtype="float32",
        crs="EPSG:32612", transform=transform,
    ) as ds:
        ds.write(arr, 1)
    # Pred inside footprint; label half outside
    pred = _utm_line(500010, 4500050, 500090, 4500050)
    label_in = _utm_line(500010, 4500050, 500090, 4500050)
    label_out = _utm_line(500010, 4500200, 500090, 4500200)  # north of origin
    labels = gpd.GeoDataFrame(
        {"geometry": list(label_in.geometry) + list(label_out.geometry)},
        crs="EPSG:32612",
    )
    pred_c, lab_c = _clip_pair_to_raster(pred, labels, path)
    unclipped = vector_noise_metrics(pred, labels, match_tolerance_m=5.0)
    clipped = vector_noise_metrics(pred_c, lab_c, match_tolerance_m=5.0)
    assert unclipped["label_km"] > clipped["label_km"]
    assert clipped["precision"] == pytest.approx(1.0)


def test_assign_block_split_matches_spatial_split_hash():
    """Block polygons use the exact same hash assignment as pixel spatial_split."""
    from trailnet.data.dataset import spatial_split
    from trailnet.vector_eval import assign_block_split, unique_blocks_from_rows, split_block_polygons

    block_size_m = 2500
    seed, val_f, test_f = 42, 0.15, 0.15
    # Synthetic tile centres spanning several blocks in one area
    rows = []
    for i, (x, y) in enumerate(
        [
            (1000, 1000),
            (1200, 1100),  # same block as first
            (2600, 1000),  # bx=1
            (1000, 3000),  # by=1
            (5100, 5100),
            (7500, 1000),
            (1000, 7600),
            (9000, 9000),
        ]
    ):
        rows.append(
            {
                "area": "demo",
                "image": f"i{i}.tif",
                "mask": f"m{i}.tif",
                "class_map": f"c{i}.tif",
                "x": float(x),
                "y": float(y),
                "crs": "EPSG:32612",
                "index_dir": ".",
            }
        )
    splits = spatial_split(rows, block_size_m, val_f, test_f, seed)
    # Every tile row's block assignment must match assign_block_split
    for split_name, split_rows in splits.items():
        for row in split_rows:
            bx = int(row["x"] // block_size_m)
            by = int(row["y"] // block_size_m)
            assert assign_block_split("demo", bx, by, seed, val_f, test_f) == split_name

    blocks = unique_blocks_from_rows(rows, block_size_m)
    assert len(blocks) == 7  # eight tiles, two share a block
    polys = split_block_polygons(rows, block_size_m, val_f, test_f, seed, area_name="demo")
    # Polygons for each split must be disjoint
    from shapely.ops import unary_union

    unions = {}
    for name in ("train", "val", "test"):
        gdf = polys[name]
        if gdf.empty:
            unions[name] = None
            continue
        unions[name] = unary_union(list(gdf.geometry))
    for a, b in (("val", "test"), ("val", "train"), ("test", "train")):
        ua, ub = unions[a], unions[b]
        if ua is None or ub is None:
            continue
        assert ua.intersection(ub).area == 0.0


def test_block_clip_reduces_label_km_vs_full_footprint():
    """Val-block clip must shrink label denominator vs footprint-only geometry."""
    from shapely.geometry import box
    from trailnet.vector_eval import (
        _clip_pair_to_polygon,
        split_block_polygons,
        vector_noise_metrics,
    )

    # Long E-W label across three 2500 m blocks; pred only in middle block
    pred = _utm_line(2600, 1000, 4800, 1000)  # mostly in bx=1
    label = gpd.GeoDataFrame(
        {
            "geometry": [
                LineString([(100, 1000), (7400, 1000)]),  # bx=0,1,2
            ]
        },
        crs="EPSG:32612",
    )
    rows = [
        {"area": "demo", "x": 1000.0, "y": 1000.0, "crs": "EPSG:32612"},  # bx0
        {"area": "demo", "x": 3000.0, "y": 1000.0, "crs": "EPSG:32612"},  # bx1
        {"area": "demo", "x": 6000.0, "y": 1000.0, "crs": "EPSG:32612"},  # bx2
    ]
    # Force known splits by probing assign; we clip to whatever val blocks exist,
    # and also to an explicit single-block polygon for determinism.
    mid = gpd.GeoDataFrame({"geometry": [box(2500, 0, 5000, 2500)]}, crs="EPSG:32612")
    pred_c, lab_c = _clip_pair_to_polygon(pred, label, mid)
    full = vector_noise_metrics(pred, label, match_tolerance_m=5.0)
    clipped = vector_noise_metrics(pred_c, lab_c, match_tolerance_m=5.0)
    assert clipped["label_km"] < full["label_km"]
    assert clipped["label_km"] == pytest.approx(2.5, abs=1e-6)  # 2500 m block width

    # split_block_polygons still returns valid frames for the synthetic rows
    polys = split_block_polygons(rows, 2500, 0.15, 0.15, 42, area_name="demo")
    assert set(polys) == {"train", "val", "test"}
    assert sum(len(g) for g in polys.values()) == 3


def test_block_scope_still_enforces_metric_crs():
    from trailnet.vector_eval import vector_noise_metrics

    pred = gpd.GeoDataFrame(
        {"geometry": [LineString([(500000, 4500000), (500100, 4500000)])]},
        crs="EPSG:4326",  # metre coords falsely tagged geographic
    )
    with pytest.raises(ValueError, match="not geographic|Unable to determine UTM|metre"):
        vector_noise_metrics(pred, pred, match_tolerance_m=5.0)
