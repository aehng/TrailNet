"""Local CPU smoke test; no downloads, data sources or pretrained weights."""
import numpy as np
import rasterio
import torch
from rasterio.transform import from_origin

from trailnet.models.unet import TrailUNet
from trailnet.postprocess import vectorize_probability


def test_model_and_postprocess_smoke(tmp_path):
    torch.set_num_threads(1)
    model = TrailUNet(encoder_name="resnet34", encoder_weights=None,
                      in_channels=6, classes=1).eval()
    sample = torch.zeros((1, 6, 64, 64), dtype=torch.float32)
    with torch.inference_mode():
        logits = model(sample)
    assert logits.shape == (1, 1, 64, 64)

    probability = np.zeros((64, 64), dtype=np.float32)
    probability[30:34, 5:59] = 0.9
    raster_path = tmp_path / "probability.tif"
    with rasterio.open(raster_path, "w", driver="GTiff", height=64, width=64,
                       count=1, dtype="float32", crs="EPSG:32612",
                       transform=from_origin(500000, 4500000, 0.6, 0.6)) as dst:
        dst.write(probability, 1)
    vector_path = tmp_path / "trails.geojson"
    vectorize_probability(raster_path, vector_path, threshold=0.5, simplify_m=0.1)
    assert vector_path.exists()
    assert len(__import__("geopandas").read_file(vector_path)) >= 1
