# TrailNet

TrailNet is a small Python 3.11 research project for finding narrow hiking and mountain-bike singletrack from aerial imagery and LiDAR terrain derivatives. Version 1 is a supervised segmentation baseline: prepare georeferenced tiles and noisy trail masks, train a U-Net, predict trail probabilities, then convert likely centerlines to GeoJSON.

## Pilot areas

The configured bounding boxes are approximate and editable in `config.yaml`. Each preparation or prediction command can target one area (`--area northern_utah`) or process all configured areas sequentially (`--all-areas`). Training can pool tiles from both areas while keeping validation and test data in separate geographic blocks.

| Area | Coverage | Approximate bbox (west, south, east, north) |
| --- | --- | --- |
| `northern_utah` (primary) | Wasatch Front, Bountiful to Smithfield; Wellsville Mountains, Logan Canyon and foothills | `[-112.05, 40.85, -111.75, 41.88]` |
| `rexburg_idaho` (secondary) | Kelly Canyon and R Mountain | `[-111.85, 43.58, -111.55, 43.85]` |

## Data sources and preparation

- **Imagery:** NAIP RGB + near-infrared imagery, nominally 0.6 m. NAIP is public domain; confirm the available acquisition date, bands, resolution and coverage for each area before preparing labels.
- **Terrain:** USGS 3DEP bare-earth DTM, targeted at 1 m. Coverage and resolution vary. Hillshade and slope are derived locally; a local relief model is an optional terrain diagnostic, not an input channel in the default six-channel model.
- **Labels:** OSM `highway=path`, `footway` and `track` features; cross-check against USFS National Forest System Trails where available. These sources are incomplete and may include wide roads or omit informal trails. Review and edit labels before training.

Data download adapters intentionally do not silently fetch large areas. Configure an imagery/DTM source and implement the clearly marked adapters in `src/trailnet/data/download.py` for the chosen public catalog/service. Put source rasters and vectors under `data/` (ignored by git), and record acquisition dates, CRS, source URLs and processing choices. No API key is bundled. The pipeline expects aligned, projected GeoTIFFs and a trail vector file for each area; see `config.yaml` for paths.

## Install

From PowerShell, create a Python 3.11 virtual environment and install the pinned dependencies:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, install a compatible GDAL runtime only if using its command-line utilities; rasterio wheels include the GDAL libraries used by this project. PyTorch installation can vary by CPU/CUDA choice; use the official PyTorch selector if the pinned CPU dependency does not suit your machine.

From the repository root, scripts can be run with `python scripts/<name>.py`. They add `src/` to `sys.path` so an editable install is not required.

## Pipeline

1. Put aligned four-band NAIP imagery (`R,G,B,NIR`), a projected DTM, and OSM/USFS trail vectors in the configured paths. Implement or call the download adapters if you want to fetch source data.
2. Prepare terrain derivatives and overlapping tiles/masks:

   ```powershell
   python scripts/prepare_data.py --area northern_utah
   python scripts/prepare_data.py --area rexburg_idaho
   # Or process both areas sequentially:
   python scripts/prepare_data.py --all-areas
   ```

3. Train using geographic blocks from all configured prepared areas:

   ```powershell
   python scripts/train.py
   # Restrict training to one prepared area if desired:
   python scripts/train.py --area northern_utah
   # Explicitly pool both configured areas:
   python scripts/train.py --all-areas
   ```

   Training uses a deterministic spatial-block split. It does not randomly split adjacent tiles. Reserve the configured test fraction for final evaluation; do not repeatedly tune against it.

4. Predict one area at a time and vectorize its probability raster:

   ```powershell
   python scripts/predict.py --area northern_utah
   python scripts/vectorize.py --area northern_utah
   python scripts/predict.py --area rexburg_idaho
   python scripts/vectorize.py --area rexburg_idaho
   python scripts/evaluate.py
   ```

   Replace `--area NAME` with `--all-areas` to run the area-specific operation sequentially for both areas. Outputs and checkpoints go under the relative paths in `config.yaml` and are git-ignored.

## Viewing and exporting results

After vectorizing an area, build a map and export the trails:

```powershell
python scripts/make_map.py --area northern_utah
python scripts/export_trails.py --area northern_utah
python scripts/export_trails.py --area northern_utah --format kml
```

- **Map:** `outputs/maps/<area>_map.html` is a self-contained Leaflet page. Double-click it to open it in a browser (it loads Leaflet and map tiles from the internet). Predicted trails are orange-red; existing OSM/USFS trails, when available, are blue dashed lines for comparison.
- **GPX (default):** `outputs/exports/<area>_trails.gpx` — the format MTB Project and Trailforks accept for trail submissions. One track per predicted trail.
- **KML:** the same trails for Google Earth or GIS tools.
- **GeoJSON:** a clean re-export of the predicted trails with `name`, `area`, and `source` properties.

MTB Project and Trailforks are curated platforms: an exported file is a submission for their review process, not an automatic listing. Predicted centerlines are candidates for human review, not authoritative trail maps.

## Realistic expectations

The v1 target is F1 **0.60–0.75** on genuinely held-out geographic blocks, not a promise. Narrow trails can be hidden under canopy, labels have positional and classification noise, and NAIP/DTM dates may not match. Report metrics by area and inspect false positives and omissions on imagery. Predicted centerlines are candidates for human review, not authoritative trail maps.

## What not to do

- Do not use GAN-upscaled imagery or synthetic high-resolution data for training.
- Do not randomly split tiles. Neighboring overlapped tiles leak terrain and trail context across splits; assign entire geographic blocks to train, validation or test.
- Do not treat OSM or USFS vectors as complete ground truth, or silently count unlabelled areas as confirmed trail-free.
- Do not infer source availability from the bbox alone. Inspect CRS, band order, pixel size, acquisition date and nodata before training.
- No Sentinel-2 or GAN upscaling is part of v1.

## Development

`tests/test_smoke.py` describes a fast CPU-only synthetic forward/post-processing check. It constructs the encoder without pretrained weights and performs no downloads. Run it yourself with `python -m pytest tests/test_smoke.py` when ready.

MIT licensed; see `LICENSE`.
