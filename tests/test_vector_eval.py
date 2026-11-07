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


def _split_params():
    return {"block_size_m": 2500.0, "val_fraction": 0.15, "test_fraction": 0.15, "seed": 42}


def _find_val_block_with_other_east():
    from trailnet.vector_eval import assign_block_split

    params = _split_params()
    for i in range(40):
        for j in range(40):
            bx, by = 200 + i, 1800 + j
            here = assign_block_split("demo", bx, by, **{
                "seed": params["seed"],
                "val_fraction": params["val_fraction"],
                "test_fraction": params["test_fraction"],
            })
            east = assign_block_split("demo", bx + 1, by, **{
                "seed": params["seed"],
                "val_fraction": params["val_fraction"],
                "test_fraction": params["test_fraction"],
            })
            if here == "val" and east != "val":
                return bx, by
    raise AssertionError("no val block with a non-val eastern neighbor")


def _tile_row(x, y):
    return {
        "area": "demo", "x": float(x), "y": float(y),
        "crs": "EPSG:32612", "index_dir": ".",
    }


def _scope(pred, labels, rows, split="val", **kwargs):
    from trailnet.vector_eval import scoped_vector_metrics

    params = _split_params()
    defaults = dict(
        tile_rows=rows,
        split=split,
        block_size_m=params["block_size_m"],
        val_fraction=params["val_fraction"],
        test_fraction=params["test_fraction"],
        seed=params["seed"],
        tolerances_m=(3.0, 5.0, 8.0, 10.0),
        report_tolerance_m=5.0,
        pixel_size_xy=(0.6, 0.6),
        tile_size_px=512,
        sidewalk_uncertain_buffer_m=2.0,
        ignore_buffer_m=5.0,
        trail_buffers={"path": 1.0, "footway": 1.0, "default": 1.0},
        area_name="demo",
        footprint_crs="EPSG:32612",
    )
    defaults.update(kwargs)
    return scoped_vector_metrics(pred, labels, **defaults)


def test_scoped_assignment_matches_spatial_split_and_scopes_are_disjoint():
    from trailnet.data.dataset import spatial_split
    from trailnet.vector_eval import assign_block_split, block_index

    params = _split_params()
    rows = []
    for i in range(6):
        for j in range(6):
            rows.append(_tile_row(500000 + i * 2500 + 100, 4_500_000 + j * 2500 + 100))
    splits = spatial_split(
        rows, params["block_size_m"], params["val_fraction"], params["test_fraction"], params["seed"],
    )
    for split_name, members in splits.items():
        for row in members:
            bx, by = block_index(row["x"], row["y"], params["block_size_m"])
            assert assign_block_split(
                "demo", bx, by, params["seed"], params["val_fraction"], params["test_fraction"],
            ) == split_name
    # A line exactly on a shared edge is counted in one split only.
    bx, by = _find_val_block_with_other_east()
    edge = (bx + 1) * 2500
    y0 = by * 2500 + 400
    line = gpd.GeoDataFrame(
        {"geometry": [LineString([(edge, y0), (edge, y0 + 100)])]},
        crs="EPSG:32612",
    )
    centers = [
        _tile_row(bx * 2500 + 1250, by * 2500 + 1250),
        _tile_row((bx + 1) * 2500 + 1250, by * 2500 + 1250),
    ]
    counted = 0.0
    for split_name in ("train", "val", "test"):
        metrics = _scope(line, line, centers, split=split_name)
        counted += metrics["label_km"] if metrics["label_km"] else 0.0
        if metrics["precision"] is not None:
            assert metrics["precision"] <= 1.0
    assert counted == pytest.approx(0.1, abs=1e-6)


def test_exact_match_and_far_wrong_prediction_inside_val_block():
    bx, by = _find_val_block_with_other_east()
    x0 = bx * 2500 + 800
    y0 = by * 2500 + 800
    good = _utm_line(x0, y0, x0 + 100, y0)
    # 80 m north is still inside the 2500 m block and outside a 10 m tolerance.
    bad = _utm_line(x0, y0 + 80, x0 + 100, y0 + 80)
    rows = [_tile_row(bx * 2500 + 1250, by * 2500 + 1250)]
    exact = _scope(good, good, rows)
    wrong = _scope(bad, good, rows)
    assert exact["empty_scope"] is False
    assert exact["precision"] == pytest.approx(1.0)
    assert exact["recall"] == pytest.approx(1.0)
    assert [row["match_tolerance_m"] for row in exact["tolerance_sweep"]] == [3.0, 5.0, 8.0, 10.0]
    assert wrong["precision"] == pytest.approx(0.0)
    assert wrong["recall"] == pytest.approx(0.0)
    assert "32612" in exact["crs"]


def test_outside_scope_and_border_are_not_matcher_inputs():
    bx, by = _find_val_block_with_other_east()
    edge = (bx + 1) * 2500
    y0 = by * 2500 + 900
    # 29 m inside the val block. Label starts 4 m outside. Distance 4 m matches at 5 m
    # only if the outside piece is still a matcher input.
    pred = _utm_line(edge - 30, y0, edge - 1, y0)
    leaked = _utm_line(edge + 3, y0, edge + 40, y0)
    rows = [_tile_row(bx * 2500 + 1250, by * 2500 + 1250)]
    metrics = _scope(pred, leaked, rows)
    assert metrics["pred_km"] == pytest.approx(0.029, abs=1e-4)
    assert metrics["label_km"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["precision"] == pytest.approx(0.0)
    assert metrics["recall"] is None
    # A line wholly in another block adds nothing to the val denominator.
    other = _utm_line((bx + 1) * 2500 + 100, y0, (bx + 1) * 2500 + 300, y0)
    inside = _utm_line(bx * 2500 + 500, y0, bx * 2500 + 620, y0)
    both_pred = gpd.GeoDataFrame(
        {"geometry": list(inside.geometry) + list(other.geometry)}, crs="EPSG:32612",
    )
    both_rows = rows + [_tile_row((bx + 1) * 2500 + 1250, by * 2500 + 1250)]
    scoped = _scope(both_pred, inside, both_rows)
    assert scoped["pred_km"] == pytest.approx(0.120, abs=1e-4)
    assert scoped["label_km"] == pytest.approx(0.120, abs=1e-4)
    assert scoped["precision"] == pytest.approx(1.0)


def test_sidewalk_buffer_ignored_unnamed_path_kept_and_empty_scope_is_null():
    from trailnet.vector_eval import resolve_sidewalk_uncertain_buffer_m

    # Actual YAML: sidewalk_uncertain_buffer_m is its own key. 2 m must not become 5 m.
    assert resolve_sidewalk_uncertain_buffer_m(2.0, ignore_buffer_m=5.0) == 2.0
    assert resolve_sidewalk_uncertain_buffer_m(None, ignore_buffer_m=5.0) == 5.0
    assert resolve_sidewalk_uncertain_buffer_m(None, ignore_buffer_m=0.0) == 2.0
    assert resolve_sidewalk_uncertain_buffer_m(0.0, ignore_buffer_m=5.0) == 0.0

    bx, by = _find_val_block_with_other_east()
    x0 = bx * 2500 + 600
    y_path = by * 2500 + 600
    y_side = y_path + 30  # farther than core 1 m + uncertain 2 m
    path = LineString([(x0, y_path), (x0 + 100, y_path)])
    sidewalk = LineString([(x0, y_side), (x0 + 100, y_side)])
    labels = gpd.GeoDataFrame(
        {
            "name": ["", "Main St sidewalk"],
            "highway": ["footway", "footway"],
            "footway": ["", "sidewalk"],
            "trail_type": ["path", "sidewalk"],
            "class_group": ["path", "sidewalk"],
            "geometry": [path, sidewalk],
        },
        crs="EPSG:32612",
    )
    preds = gpd.GeoDataFrame({"geometry": [path, sidewalk]}, crs="EPSG:32612")
    rows = [_tile_row(bx * 2500 + 1250, by * 2500 + 1250)]
    metrics = _scope(preds, labels, rows)
    assert metrics["label_km"] == pytest.approx(0.1, abs=1e-4)
    assert metrics["pred_km"] == pytest.approx(0.1, abs=1e-4)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["novelty_filter_applied"] is False
    # Sidewalk-only prediction is ignore, not a false-positive denominator.
    only_side = gpd.GeoDataFrame({"geometry": [sidewalk]}, crs="EPSG:32612")
    ignored = _scope(only_side, labels, rows)
    assert ignored["pred_km"] == pytest.approx(0.0, abs=1e-6)
    assert ignored["label_km"] == pytest.approx(0.1, abs=1e-4)
    assert ignored["precision"] == pytest.approx(0.0)
    assert ignored["recall"] == pytest.approx(0.0)
    empty = _scope(preds, labels, [], split="val")
    assert empty["empty_scope"] is True
    assert empty["precision"] is None
    assert empty["recall"] is None


def test_tile_overhang_is_reported_not_used_as_match_scope():
    from rasterio.transform import from_origin
    from rasterio.crs import CRS

    bx, by = _find_val_block_with_other_east()
    # Centre 20 m inside the west edge. A 512 px * 0.6 m tile overhangs ~153 m.
    cx = bx * 2500 + 20
    cy = by * 2500 + 800
    overhang = _utm_line(bx * 2500 - 80, cy, bx * 2500 - 10, cy)
    rows = [_tile_row(cx, cy)]
    transform = from_origin(bx * 2500 - 500, by * 2500 + 2000, 0.6, 0.6)
    grid_crs = CRS.from_epsg(32612)
    metrics = _scope(overhang, overhang, rows, footprint_crs=grid_crs)
    assert metrics["pred_km"] == pytest.approx(0.0, abs=1e-6)
    boundary = metrics["boundary_vs_tile_assignment"]
    assert boundary["pred_km_in_tiles_not_blocks"] == pytest.approx(0.07, abs=1e-3)
    assert boundary["block_pred_km"] == pytest.approx(0.0, abs=1e-6)
    # Affine of this real-ish UTM grid is accepted and is not trail truth.
    from trailnet.vector_eval import assert_metric_raster_grid
    info = assert_metric_raster_grid(grid_crs, transform, 100, 100)
    assert info["affine_ok"] is True
    assert info["pixel_size_x_m"] == pytest.approx(0.6)
    assert info["roundtrip_is_trail_truth"] is False


def test_directional_reraster_stays_within_one_and_matches_one_pixel_line():
    import numpy as np
    from rasterio.crs import CRS
    from rasterio.transform import from_origin

    from trailnet.vector_eval import assert_metric_raster_grid, directional_reraster_scores

    skel = np.zeros((20, 20), dtype=bool)
    skel[5, 2:18] = True
    burn = np.zeros_like(skel)
    burn[5, 10] = True
    scores = directional_reraster_scores(skel, burn, dilate_px=1)
    assert scores["roundtrip_precision"] <= 1.0
    assert scores["roundtrip_recall"] <= 1.0
    assert scores["legacy_precision_skel_inter_over_burn"] > 1.0
    assert scores["roundtrip_is_trail_truth"] is False
    # Identical 1 px centerline: both directional ratios are 1 at 1 px dilation.
    line = np.zeros((30, 30), dtype=bool)
    line[15, 4:26] = True
    perfect = directional_reraster_scores(line, line, dilate_px=1)
    assert perfect["roundtrip_precision"] == pytest.approx(1.0)
    assert perfect["roundtrip_recall"] == pytest.approx(1.0)
    assert perfect["dilate_px"] == 1
    utm = CRS.from_epsg(32612)
    transform = from_origin(500000, 4_500_240, 0.6, 0.6)
    assert_metric_raster_grid(utm, transform, 30, 30)
    with pytest.raises(ValueError, match="not projected"):
        assert_metric_raster_grid(CRS.from_epsg(4326), transform, 30, 30)


def test_block_scope_still_enforces_metric_crs():
    from trailnet.vector_eval import vector_noise_metrics

    pred = gpd.GeoDataFrame(
        {"geometry": [LineString([(500000, 4500000), (500100, 4500000)])]},
        crs="EPSG:4326",  # metre coords falsely tagged geographic
    )
    with pytest.raises(ValueError, match="not geographic|Unable to determine UTM|metre"):
        vector_noise_metrics(pred, pred, match_tolerance_m=5.0)
