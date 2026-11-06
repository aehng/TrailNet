import torch
from trailnet.train import trail_seg_loss
from trailnet.data.tiles import _trail_geometries, _is_explicit_sidewalk
import geopandas as gpd
from shapely.geometry import LineString

def test_all_ignored_batch_returns_zero_loss():
    logits = torch.zeros(1, 1, 8, 8)
    targets = torch.ones(1, 1, 8, 8)
    weights = torch.zeros(1, 1, 8, 8)
    loss = trail_seg_loss(logits, targets, weights, pos_boost=2.0)
    assert float(loss) == 0.0

def test_ignore_weights_excluded_from_tversky():
    logits = torch.full((1, 1, 4, 4), 10.0)  # high prob
    targets = torch.zeros(1, 1, 4, 4)
    targets[0, 0, 0, 0] = 1.0
    weights = torch.ones(1, 1, 4, 4)
    weights[0, 0, 0, 0] = 0.0  # ignore the only positive
    # remaining are background with high pred -> FP, but ignored positive shouldn't add FN
    loss = trail_seg_loss(logits, targets, weights, pos_boost=2.0)
    assert torch.isfinite(loss)

def test_soft_target_between_0_and_1_is_finite():
    logits = torch.zeros(1, 1, 4, 4)
    targets = torch.full((1, 1, 4, 4), 0.4)
    weights = torch.ones(1, 1, 4, 4)
    loss = trail_seg_loss(logits, targets, weights, pos_boost=2.0)
    assert torch.isfinite(loss) and float(loss) > 0

def test_sidewalk_geometries_only_ignore():
    gdf = gpd.GeoDataFrame([
        {"class_group":"sidewalk","highway":"footway","footway":"sidewalk","trail_type":"sidewalk",
         "label_weight":1.0,"geometry":LineString([(0,0),(5,0)])},
        {"class_group":"path","highway":"path","footway":None,"trail_type":"path",
         "label_weight":1.0,"geometry":LineString([(0,2),(5,2)])},
    ], crs="EPSG:32612")
    out = _trail_geometries(gdf, {"path":1.0,"sidewalk":1.0,"default":1.0}, ignore_buffer_m=5.0)
    assert len(out[1]) == 1
    assert out[4]  # sidewalk ignore present
