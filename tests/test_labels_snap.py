"""Relief-snap helper: reject flat scores, accept a sharp trough."""
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString

from trailnet.data.labels import snap_lines_to_relief


def _write_dtm(path, array, transform=None):
    array = np.asarray(array, dtype="float32")
    with rasterio.open(
        path, "w", driver="GTiff", width=array.shape[1], height=array.shape[0],
        count=1, dtype="float32", crs="EPSG:32612",
        transform=transform or from_origin(0, array.shape[0], 1.0, 1.0),
    ) as dst:
        dst.write(array, 1)


def test_snap_rejects_flat_score_shift(tmp_path):
    dtm_path = tmp_path / "flat.tif"
    _write_dtm(dtm_path, np.full((80, 80), 100.0, dtype="float32"))
    frame = gpd.GeoDataFrame(
        {"geometry": [LineString([(10.0, 40.0), (70.0, 40.0)])],
         "source": ["OpenStreetMap"], "highway": ["path"], "class_group": ["path"],
         "trail_type": ["path"], "name": ["flat"], "label_weight": [1.0], "priority": [1]},
        crs="EPSG:32612",
    )
    snapped = snap_lines_to_relief(frame, dtm_path, max_offset_m=8.0, step_m=1.0, sample_spacing_m=20.0)
    orig = np.array(frame.geometry.iloc[0].coords)
    new = np.array(snapped.geometry.iloc[0].coords)
    # Mid-line y should stay near 40 (no drainage/trough to lock onto).
    assert abs(float(np.mean(new[:, 1])) - 40.0) < 1.5


def test_snap_accepts_sharp_trough(tmp_path):
    dtm_path = tmp_path / "trough.tif"
    z = np.full((80, 80), 120.0, dtype="float32")
    # Incised cut 5 m north of y=40 (row 40 is y=40 with from_origin(0, 80, 1, 1)
    # because UL is (0, 80) and northing decreases with row).
    # y = 80 - row, so y=45 is row 35.
    z[33:38, :] = 80.0
    _write_dtm(dtm_path, z, transform=from_origin(0, 80, 1.0, 1.0))
    frame = gpd.GeoDataFrame(
        {"geometry": [LineString([(10.0, 40.0), (70.0, 40.0)])],
         "source": ["OpenStreetMap"], "highway": ["path"], "class_group": ["path"],
         "trail_type": ["path"], "name": ["cut"], "label_weight": [1.0], "priority": [1]},
        crs="EPSG:32612",
    )
    snapped = snap_lines_to_relief(
        frame, dtm_path, max_offset_m=12.0, step_m=1.0, sample_spacing_m=15.0,
        score_margin=0.02, sharpness_ratio=1.05,
    )
    new = np.array(snapped.geometry.iloc[0].coords)
    mean_y = float(np.mean(new[:, 1]))
    assert mean_y > 41.5, f"expected snap toward y~45 trough, got mean_y={mean_y}"
