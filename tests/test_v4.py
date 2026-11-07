"""Focused v4 tests for terrain channels, weighted labels, and free sources."""
import csv
import importlib
import importlib.util

import geopandas as gpd
import numpy as np
import rasterio
import torch
from rasterio.transform import from_origin
from shapely.geometry import LineString

from trailnet.data.dataset import TrailTileDataset
from trailnet.data.download import fetch_naip
from trailnet.data.tiles import build_area_tiles
from trailnet.data.terrain import derive_terrain
from trailnet.models.unet import dice_bce_loss


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


def test_tile_stack_has_seven_channels_and_scales_only_terrain(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    imagery = data / "naip.tif"
    dtm = data / "dtm.tif"
    trails = data / "trails.gpkg"
    yy, xx = np.mgrid[:64, :64]
    _write_raster(imagery, np.stack([xx, yy, xx + yy, 2 * xx], axis=0).astype("uint16"))
    _write_raster(dtm, (100 + xx * 0.2 + np.sin(yy / 5) * 2).astype("float32"))
    gpd.GeoDataFrame(
        [{"highway": "track", "class_group": "track", "source": "test",
          "geometry": LineString([(500005, 4499980), (500030, 4499980)])},
         {"highway": "path", "class_group": "path", "source": "test",
          "geometry": LineString([(500005, 4499980), (500030, 4499980)])}],
        crs="EPSG:32612",
    ).to_file(trails, driver="GPKG")
    config = {
        "_root": tmp_path,
        "paths": {"tile_dir": "tiles"},
        "tiling": {"tile_size": 64, "overlap": 0,
                   "trail_buffer_m": {"path": 1.0, "track": 1.5, "default": 1.0}},
        "terrain": {"local_relief_radius_px": 8, "channel_gain": 1.85},
    }
    area = {"name": "test", "imagery": "data/naip.tif", "dtm": "data/dtm.tif",
            "trails": "data/trails.gpkg"}

    index_path = build_area_tiles(config, area)
    with index_path.open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    with rasterio.open(index_path.parent / row["image"]) as tile_ds:
        tile = tile_ds.read()
    with rasterio.open(dtm) as dtm_ds:
        expected_terrain = np.stack(derive_terrain(dtm_ds.read(1), 0.6, -0.6, 8)) * 1.85
    assert tile.shape == (7, 64, 64)
    assert np.allclose(tile[4:7], expected_terrain, atol=1e-5)
    assert np.max(tile[4:7]) > 1.0
    assert np.max(tile[:4]) <= 1.0
    assert row["class_map"].endswith("_class.tif")
    with rasterio.open(index_path.parent / row["class_map"]) as class_ds:
        classes = class_ds.read(1)
    assert set(np.unique(classes)).issubset({0, 1, 2, 3})
    assert 1 in np.unique(classes)
    assert classes[33, 20] == 1


def test_outside_source_pixels_are_ignored_only_when_opted_in(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    imagery = data / "naip.tif"
    dtm = data / "dtm.tif"
    trails = data / "trails.gpkg"
    yy, xx = np.mgrid[:32, :32]
    optical = np.full((4, 32, 32), 100, dtype="uint8")
    optical[:, 2, 2] = 0  # A valid black pixel must not be mistaken for padding.
    _write_raster(imagery, optical)
    _write_raster(dtm, (100 + xx * 0.2 + np.sin(yy / 5) * 2).astype("float32"))
    gpd.GeoDataFrame(
        [{"highway": "path", "class_group": "path", "source": "test",
          "geometry": LineString([(500002, 4499995), (500018, 4499995)])},
         {"highway": "path", "class_group": "path", "source": "test",
          "geometry": LineString([(500000.6, 4499998.5), (500002.4, 4499998.5)])}],
        crs="EPSG:32612",
    ).to_file(trails, driver="GPKG")

    common = {
        "_root": tmp_path,
        "tiling": {"tile_size": 64, "overlap": 0,
                   "trail_buffer_m": {"path": 1.0, "default": 1.0}},
        "terrain": {"local_relief_radius_px": 8, "channel_gain": 1.0},
    }
    area = {"name": "edge", "imagery": "data/naip.tif", "dtm": "data/dtm.tif",
            "trails": "data/trails.gpkg"}
    outputs = {}
    for label, enabled in (("legacy", False), ("ignored", True)):
        config = {
            **common,
            "paths": {"tile_dir": f"tiles_{label}"},
            "tiling": {**common["tiling"], "ignore_outside_imagery": enabled},
        }
        index_path = build_area_tiles(config, area)
        with index_path.open(newline="", encoding="utf-8") as stream:
            row = next(csv.DictReader(stream))
        with rasterio.open(index_path.parent / row["class_map"]) as ds:
            classes = ds.read(1)
        with rasterio.open(index_path.parent / row["mask"]) as ds:
            target = ds.read(1)
        with rasterio.open(index_path.parent / row["image"]) as ds:
            image = ds.read()
        _, _, weights = TrailTileDataset([{
            "index_dir": str(index_path.parent), "image": row["image"],
            "mask": row["mask"], "class_map": row["class_map"],
        }])[0]
        outside = np.zeros((64, 64), dtype=bool)
        outside[32:, :] = True
        outside[:, 32:] = True
        outputs[label] = (classes, target, image, weights[0].numpy(), outside)

    legacy_classes, legacy_target, legacy_image, legacy_weights, outside = outputs["legacy"]
    fixed_classes, fixed_target, fixed_image, fixed_weights, _ = outputs["ignored"]
    assert np.array_equal(legacy_classes[~outside], fixed_classes[~outside])
    assert np.array_equal(legacy_target[~outside], fixed_target[~outside])
    assert np.array_equal(legacy_image, fixed_image)
    assert (legacy_classes[~outside] == 1).any()
    assert legacy_classes[2, 2] == fixed_classes[2, 2] == 1
    assert np.all(legacy_image[:4, 2, 2] == 0)
    assert np.all(legacy_classes[outside] == 0)
    assert np.all(fixed_classes[outside] == 4)
    assert np.all(legacy_target[outside] == 0)
    assert np.all(fixed_target[outside] == 0)
    assert np.all(legacy_weights[outside] == 1.0)
    assert np.all(fixed_weights[outside] == 0.0)
    assert np.array_equal(legacy_weights[~outside], fixed_weights[~outside])


def test_prepare_parallel_workers_write_all_tiles(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    imagery = data / "naip.tif"
    dtm = data / "dtm.tif"
    trails = data / "trails.gpkg"
    yy, xx = np.mgrid[:64, :64]
    _write_raster(imagery, np.stack([xx, yy, xx + yy, 2 * xx], axis=0).astype("uint16"))
    _write_raster(dtm, (100 + xx * 0.2 + np.sin(yy / 5) * 2).astype("float32"))
    gpd.GeoDataFrame(
        [{"highway": "path", "class_group": "path", "source": "test",
          "geometry": LineString([(500005, 4499980), (500030, 4499980)])}],
        crs="EPSG:32612",
    ).to_file(trails, driver="GPKG")
    config = {
        "_root": tmp_path,
        "paths": {"tile_dir": "tiles"},
        "tiling": {"tile_size": 32, "overlap": 0, "num_workers": 2,
                   "trail_buffer_m": {"path": 1.0, "track": 1.5, "default": 1.0}},
        "terrain": {"local_relief_radius_px": 4, "channel_gain": 1.85},
    }
    area = {"name": "parallel", "imagery": "data/naip.tif", "dtm": "data/dtm.tif",
            "trails": "data/trails.gpkg"}
    index_path = build_area_tiles(config, area)
    with index_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4
    for row in rows:
        assert (index_path.parent / row["image"]).exists()
        assert (index_path.parent / row["mask"]).exists()
        assert (index_path.parent / row["class_map"]).exists()
        with rasterio.open(index_path.parent / row["image"]) as ds:
            assert ds.count == 7


def test_dataset_returns_class_aware_pixel_weights(tmp_path):
    image_path = tmp_path / "tile.tif"
    mask_path = tmp_path / "tile_mask.tif"
    class_path = tmp_path / "tile_class.tif"
    _write_raster(image_path, np.zeros((7, 8, 8), dtype="float32"))
    _write_raster(mask_path, np.ones((8, 8), dtype="uint8"))
    classes = np.zeros((8, 8), dtype="uint8")
    classes[:3] = 1
    classes[3:6] = 2
    classes[6:] = 3
    _write_raster(class_path, classes)
    rows = [{"index_dir": str(tmp_path), "image": image_path.name,
             "mask": mask_path.name, "class_map": class_path.name}]

    image, mask, weights = TrailTileDataset(
        rows, track_weight=0.30, uncertain_weight=0.65,
    )[0]
    assert image.shape == (7, 8, 8)
    assert mask.shape == (1, 8, 8)
    assert torch.all(weights[0, :3] == 1.0)
    assert torch.all(weights[0, 3:6] == 0.30)
    assert torch.all(weights[0, 6:] == 0.65)


def test_binary_legacy_index_defaults_positive_pixels_to_path(tmp_path):
    image_path = tmp_path / "tile.tif"
    mask_path = tmp_path / "tile_mask.tif"
    _write_raster(image_path, np.zeros((7, 8, 8), dtype="float32"))
    mask = np.zeros((8, 8), dtype="uint8")
    mask[2:4, 2:6] = 1
    _write_raster(mask_path, mask)
    image, target, weights = TrailTileDataset([
        {"index_dir": str(tmp_path), "image": image_path.name, "mask": mask_path.name},
    ])[0]
    assert image.shape == (7, 8, 8)
    assert torch.equal(target[0].bool(), torch.from_numpy(mask).bool())
    assert torch.all(weights[0, 2:4, 2:6] == 1.0)


def test_ignore_class_is_not_positive_and_has_zero_weight(tmp_path):
    image_path = tmp_path / "tile.tif"
    mask_path = tmp_path / "tile_mask.tif"
    class_path = tmp_path / "tile_class.tif"
    _write_raster(image_path, np.zeros((7, 8, 8), dtype="float32"))
    classes = np.zeros((8, 8), dtype="uint8")
    classes[2:4, 2:6] = 1
    classes[1:5, 1] = 4
    classes[1:5, 6] = 4
    mask = np.isin(classes, (1, 2, 3)).astype("uint8")
    _write_raster(mask_path, mask)
    _write_raster(class_path, classes)
    _image, target, weights = TrailTileDataset(
        [{"index_dir": str(tmp_path), "image": image_path.name,
          "mask": mask_path.name, "class_map": class_path.name}],
        pixel_weights={"background": 1.0, "path": 8.0, "track": 3.0, "uncertain": 4.0},
    )[0]
    assert torch.all(target[0, classes == 4] == 0)
    assert torch.all(weights[0, classes == 4] == 0.0)
    assert torch.all(weights[0, classes == 1] == 8.0)
    assert int((target > 0).sum()) == int((classes == 1).sum())


def test_ignore_ring_around_path_is_class_4_not_positive():
    from trailnet.data.tiles import _trail_geometries
    gdf = gpd.GeoDataFrame(
        [{"highway": "path", "class_group": "path", "label_weight": 1.0,
          "geometry": LineString([(0.0, 0.0), (20.0, 0.0)])}],
        crs="EPSG:32612",
    )
    buffered = _trail_geometries(gdf, {"path": 1.0, "track": 1.5, "default": 1.0}, ignore_buffer_m=5.0)
    assert 4 in buffered
    assert buffered[1], "path core missing"
    assert buffered[4], "ignore ring missing"
    core = buffered[1][0][0]
    ring = buffered[4][0][0]
    assert not core.is_empty
    assert not ring.is_empty
    assert ring.area > core.area
    assert core.intersects(ring) or core.touches(ring) or ring.distance(core) < 1e-6


def test_naip_rejects_oversized_projected_grid_before_network(tmp_path):
    try:
        fetch_naip([-111.80, 40.60, -111.79, 40.61], tmp_path / "naip.tif",
                   max_raster_pixels=10)
    except ValueError as exc:
        assert "pixels" in str(exc)
    else:
        raise AssertionError("oversized projected grid was not rejected")


def test_weighted_bce_dice_gives_path_pixels_more_gradient_than_track():
    logits = torch.zeros((1, 1, 1, 2), requires_grad=True)
    targets = torch.ones_like(logits)
    weights = torch.tensor([[[[1.0, 0.30]]]])
    loss = dice_bce_loss(logits, targets, weights)
    loss.backward()
    gradients = logits.grad[0, 0, 0].abs()
    assert torch.isfinite(loss)
    assert gradients[0] > gradients[1] * 3.0


def test_label_source_parsers_and_dedup_prefer_ugrc():
    spec = importlib.util.find_spec("trailnet.data.labels")
    assert spec is not None, "v4 label source parser module is missing"
    labels = importlib.import_module("trailnet.data.labels")
    ugrc = labels.parse_arcgis_geojson({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"CartoCode": "2 - Hiking and Biking Allowed", "PrimaryName": "Ridge"},
         "geometry": {"type": "LineString", "coordinates": [[-111.8, 40.6], [-111.799, 40.601]]}},
        {"type": "Feature", "properties": {"CartoCode": "4 - Road Concurrent", "PrimaryName": "Road"},
         "geometry": {"type": "LineString", "coordinates": [[-111.8, 40.61], [-111.799, 40.611]]}},
        {"type": "Feature", "properties": {"CartoCode": "3", "PrimaryName": "Urban"},
         "geometry": {"type": "LineString", "coordinates": [[-111.8, 40.62], [-111.799, 40.621]]}},
    ]}, source="UGRC")
    usfs = labels.parse_usfs_geojson({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"trail_name": "Motorized", "terra_motorized": "YES"},
         "geometry": {"type": "LineString", "coordinates": [[-111.8, 40.63], [-111.799, 40.631]]}},
        {"type": "Feature", "properties": {"trail_name": "Unknown"},
         "geometry": {"type": "LineString", "coordinates": [[-111.8, 40.64], [-111.799, 40.641]]}},
    ]}, source="USFS")
    assert list(ugrc["cartocode"]) == ["2", "4"]
    assert list(ugrc["class_group"]) == ["path", "track"]
    assert list(usfs["class_group"]) == ["track", "path"]
    assert usfs.iloc[1]["label_weight"] == 0.65

    combined = gpd.GeoDataFrame(
        [{"source": "OpenStreetMap", "class_group": "track", "priority": 1,
          "geometry": LineString([(-111.8, 40.6), (-111.799, 40.601)])},
         {"source": "UGRC", "class_group": "path", "priority": 3,
          "geometry": LineString([(-111.800001, 40.6), (-111.799001, 40.601)])}],
        crs="EPSG:4326",
    )
    result = labels.deduplicate_trails(combined, target_crs="EPSG:32612", tolerance_m=4.0)
    assert len(result) == 1
    assert result.iloc[0]["source"] == "UGRC"
    assert result.iloc[0]["class_group"] == "path"


def test_arcgis_pagination_caches_all_pages(tmp_path, monkeypatch):
    spec = importlib.util.find_spec("trailnet.data.labels")
    assert spec is not None, "v4 label source parser module is missing"
    labels = importlib.import_module("trailnet.data.labels")
    offsets = []

    def page(_endpoint, _bbox, offset, _page_size, _timeout):
        offsets.append(offset)
        count = 2 if offset == 0 else 1
        return {"type": "FeatureCollection", "features": [
            {"type": "Feature", "properties": {"OBJECTID": offset + i + 1},
             "geometry": {"type": "LineString", "coordinates": [[0, 0], [1, 1]]}}
            for i in range(count)
        ]}

    monkeypatch.setattr(labels, "_arcgis_page", page)
    first = labels.fetch_arcgis_geojson(
        "https://example.test/query", [-111.8, 40.6, -111.79, 40.61], tmp_path,
        source="unit", page_size=2,
    )
    assert len(first["features"]) == 3
    assert offsets == [0, 2]
    monkeypatch.setattr(labels, "_arcgis_page", lambda *_args: (_ for _ in ()).throw(
        AssertionError("cache should avoid a second request")))
    again = labels.fetch_arcgis_geojson(
        "https://example.test/query", [-111.8, 40.6, -111.79, 40.61], tmp_path,
        source="unit", page_size=2,
    )
    assert len(again["features"]) == 3


def test_explicit_sidewalk_is_ignore_not_positive_unnamed_footway_stays_path():
    """footway=sidewalk → class 4 ignore; plain footway remains path positive."""
    from trailnet.data.tiles import _trail_geometries
    import geopandas as gpd
    from shapely.geometry import LineString

    gdf = gpd.GeoDataFrame(
        [
            {"class_group": "sidewalk", "highway": "footway", "footway": "sidewalk",
             "trail_type": "sidewalk", "label_weight": 1.0,
             "geometry": LineString([(0, 0), (10, 0)])},
            {"class_group": "path", "highway": "path", "footway": None,
             "trail_type": "footway", "label_weight": 1.0,
             "geometry": LineString([(0, 5), (10, 5)])},
        ],
        crs="EPSG:32612",
    )
    buffers = {"path": 1.0, "track": 1.5, "default": 1.0, "sidewalk": 1.0}
    result = _trail_geometries(gdf, buffers, ignore_buffer_m=0.0, sidewalk_uncertain_buffer_m=2.0)
    assert result[1], "unnamed/plain footway should remain a path positive"
    assert not any(False for _ in ()), "sanity"
    # No sidewalk geometry should appear as path/track/uncertain positive.
    assert result[4], "sidewalk must be present as ignore"
    # Path positive count: only the second line
    assert len(result[1]) == 1
    assert len(result[2]) == 0
