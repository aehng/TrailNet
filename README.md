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
- **Terrain:** USGS 3DEP bare-earth DTM, targeted at 1 m. Coverage and resolution vary. Hillshade, slope, and local relief are derived locally and join NAIP RGB/NIR as the seven-channel model input. These are bare-earth DTM derivatives, not full LiDAR canopy height (CHM) or intensity products. Multi-year NAIP stacking is reserved for a later run.
- **Labels:** OSM `highway=path`, `footway` and `track` features; cross-check against USFS National Forest System Trails where available. These sources are incomplete and may include wide roads or omit informal trails. Review and edit labels before training.

Data acquisition is explicit. Planetary Computer provides a public NAIP STAC catalog and token-signing endpoint; The National Map's TNMAccess API searches 3DEP products; OSM path lines come from Overpass. The code issues no request or download on import. `scripts/fetch_data.py` is an offline dry run by default; `--probe` queries bounded metadata only, while each `--fetch-*` option is an explicit source request. [NAIP STAC and asset access](https://planetarycomputer.microsoft.com/docs/quickstarts/storage/), [Planetary Computer signing](https://planetarycomputer.microsoft.com/docs/concepts/sas/), [USGS TNMAccess API](https://tnmaccess.nationalmap.gov/api/v1/docs), [Overpass query guide](https://wiki.openstreetmap.org/wiki/Overpass_API/Language_Guide).

Raster downloads have a 25 km2 default area cap, a 2 GB compressed DEM cap, and a 200-product cap. The configured pilot bboxes are larger than the default raster cap; first edit an area's bbox to a manageable pilot subset. Raising the limit is explicit and can create large files. Existing outputs are preserved unless `--overwrite` is supplied. Downloading OSM writes raw OSM lines to the configured `trails.gpkg`; copy any hand-curated labels before replacing it, then cross-check/merge USFS National Forest System Trails manually. No USFS downloader is included. No API key is stored; an optional Overpass bearer token can be provided using the environment variable named by `downloads.overpass_token_env`.

The v4 label path adds `--fetch-labels`, which caches OSM and ArcGIS response data under `data/_cache/` and merges the enabled sources into the configured `trails.gpkg`. Utah UGRC CartoCode 1/2/5 records are path labels and code 4 records are track labels; the paved-only code 3 and other codes are skipped. TrailNFS motorized attributes map to track, while records without a clear motorization value receive a medium-confidence path weight. Features are clipped to the AOI, reprojected to the imagery CRS, then de-duplicated only when projected linework lies within 4 m and substantially overlaps; source preference is UGRC, USFS, then OSM. The tile writer records background/path/track/uncertain values in each `_class.tif` and applies path weight 1.0, track weight 0.30, and uncertain weight 0.65 in the binary BCE+Dice loss.

Safe acquisition commands from the repository root:

```powershell
# Offline plan only; makes no network request.
python scripts/fetch_data.py --area northern_utah

# Query NAIP and 3DEP availability metadata only; downloads no raster assets.
python scripts/fetch_data.py --area northern_utah --probe
python scripts/fetch_data.py --area rexburg_idaho --probe

# Explicitly fetch OSM trail ways to data/northern_utah/trails.gpkg.
python scripts/fetch_data.py --area northern_utah --fetch-osm
python scripts/fetch_data.py --area rexburg_idaho --fetch-osm

# Merge the label sources enabled for the selected AOI in its config.
python scripts/fetch_data.py --config config_kaggle.yaml --area kelly_canyon --fetch-labels
python scripts/fetch_data.py --config config_kaggle.yaml --area corner_canyon --fetch-labels

# For a bbox you have reduced to <=25 km2, raster fetches require explicit limits.
python scripts/fetch_data.py --area northern_utah --fetch-naip --max-raster-km2 25
python scripts/fetch_data.py --area northern_utah --fetch-3dep --max-raster-km2 25 --max-download-gb 2 --max-products 200
python scripts/fetch_data.py --area rexburg_idaho --fetch-naip --max-raster-km2 25
python scripts/fetch_data.py --area rexburg_idaho --fetch-3dep --max-raster-km2 25 --max-download-gb 2 --max-products 200
```

NAIP reads the Planetary Computer `naip` collection's four-band `image` asset and selects the latest acquisition year returned for the bbox. The 3DEP query targets the 1 m DEM product where available; it crops and warps returned source tiles onto a bounded UTM grid covering only the requested bbox. Coverage and native source resolution vary. Both raster adapters enforce the configured area and projected pixel caps before downloading. Tile preparation reprojects the DTM to the NAIP grid and writes seven channels: RGB, NIR, hillshade, slope, and local relief.

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

1. Probe sources or fetch a small configured bbox using the commands above. Manually cross-check OSM labels against USFS trails and preserve your curated source copy. The model pipeline expects four-band NAIP (`R,G,B,NIR`), a DTM, and a trail vector file under the configured paths.
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

   `predict` and `vectorize` each rebuild that area's HTML preview map under `outputs/maps/` (or the config `output_dir`). The map uses local NAIP and 3DEP preview PNGs as basemaps, so it does not load OpenStreetMap raster tiles.

## Viewing and exporting results

After predict or vectorize, open the area map in a browser:

```powershell
python scripts/make_map.py --area northern_utah
python scripts/export_trails.py --area northern_utah
python scripts/export_trails.py --area northern_utah --format kml
```

- **Map:** `outputs/maps/<area>_map.html` plus a sibling `<area>/` folder of preview PNGs. Double-click the HTML (Leaflet JS still loads from the internet; imagery does not). Switch RGB, color infrared, near infrared, hillshade, slope, and local relief in the panel. Existing trails are steel-blue dashed lines; predicted leftover linework is blaze orange; model probability is an optional overlay.
- **GPX (default):** `outputs/exports/<area>_trails.gpx` — the format MTB Project and Trailforks accept for trail submissions. One track per predicted trail.
- **KML:** the same trails for Google Earth or GIS tools.
- **GeoJSON:** a re-export with `name`, `area`, `source`, and per-line `confidence` properties. GPX tracks and KML placemarks also carry the name plus area/source/confidence metadata.

MTB Project accepts GPX/KML/KMZ tracks through its trail contribution workflow, and Trailforks accepts GPS-file geometry for trail submissions. Both platforms review contributions; Trailforks also asks the contributor to supply required trail details such as activity, location, difficulty, and surface. These exports provide geometry and TrailNet metadata, not those human-curated fields. [MTB Project mapping guide](https://www.mtbproject.com/mapping), [Trailforks GPX submission guide](https://help.trailforks.com/hc/en-us/articles/5446619643671-Add-a-Trail-using-a-GPS-File). Predicted centerlines are candidates for human review, not authoritative trail maps.

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
