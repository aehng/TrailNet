"""Safe, area-scoped acquisition CLI. Dry-run is the default; no data downloads implicitly."""
import argparse
import json

import _bootstrap  # noqa: F401
from trailnet.config import load_config, resolve_path, select_areas
from trailnet.data.download import (
    DEFAULT_NAIP_STAC,
    DEFAULT_OVERPASS_URL,
    DEFAULT_TNM_PRODUCTS,
    TNM_1M_DATASET,
    _area_km2,
    _utm_epsg_for_bbox,
    fetch_3dep_dtm,
    fetch_naip,
    fetch_osm_trails,
    probe_sources,
)
from trailnet.data.labels import UGRC_TRAILS_URL, USFS_TRAILS_URL, merge_area_labels
import rasterio


parser = argparse.ArgumentParser(description="Probe public data sources or explicitly fetch one area's inputs")
parser.add_argument("--config", help="Configuration YAML path (defaults to config.yaml)")
parser.add_argument("--area", required=True, help="One configured area; bulk raster downloads are not automatic")
mode = parser.add_mutually_exclusive_group()
mode.add_argument("--probe", action="store_true", help="Query bounded NAIP/3DEP metadata only; download no assets")
mode.add_argument("--fetch-osm", action="store_true", help="Fetch OSM path/footway/track lines to the configured trails path")
mode.add_argument("--fetch-labels", action="store_true", help="Merge configured free label sources into trails.gpkg")
mode.add_argument("--fetch-naip", action="store_true", help="Create a four-band NAIP mosaic (explicit raster download)")
mode.add_argument("--fetch-3dep", action="store_true", help="Download and mosaic 3DEP DEM tiles (explicit raster download)")
parser.add_argument("--max-raster-km2", type=float, help="Raster footprint cap; defaults to config.yaml's downloads.max_raster_km2")
parser.add_argument("--max-raster-pixels", type=int, help="Projected pixel cap; defaults to config.yaml's downloads.max_raster_pixels")
parser.add_argument("--max-download-gb", type=float, help="Compressed 3DEP download cap; defaults to config.yaml")
parser.add_argument("--max-products", type=int, help="Maximum 3DEP products; defaults to config.yaml")
parser.add_argument("--overwrite", action="store_true", help="Replace an existing configured output after a successful staged fetch")
args = parser.parse_args()

config = load_config(args.config)
area = select_areas(config, args.area)[0]
downloads = config.get("downloads", {})
bbox = area["bbox"]
max_area = (args.max_raster_km2 if args.max_raster_km2 is not None
            else float(downloads.get("max_raster_km2", 25.0)))
max_download_gb = (args.max_download_gb if args.max_download_gb is not None
                   else float(downloads.get("max_download_gb", 2.0)))
max_bytes = int(max_download_gb * 1_000_000_000)
max_extracted = int(float(downloads.get("max_extracted_gb", 8.0)) * 1_000_000_000)
max_products = (args.max_products if args.max_products is not None
                else int(downloads.get("max_3dep_products", 200)))
if max_area <= 0 or max_download_gb <= 0 or max_products <= 0:
    parser.error("raster area, download byte, and product caps must be greater than zero")
naip_url = downloads.get("naip_stac_url", DEFAULT_NAIP_STAC)
tnm_url = downloads.get("tnm_products_url", DEFAULT_TNM_PRODUCTS)
overpass_url = downloads.get("overpass_url", DEFAULT_OVERPASS_URL)
cache_dir = resolve_path(config, downloads.get("cache_dir", "data/_cache"))

if args.probe:
    result = probe_sources(bbox, naip_catalog=naip_url,
                           tnm_endpoint=tnm_url, overpass_endpoint=overpass_url)
    print(json.dumps(result, indent=2))
elif args.fetch_osm:
    path = fetch_osm_trails(
        bbox, resolve_path(config, area["trails"]), area_name=area["name"],
        endpoint=overpass_url, token_env=downloads.get("overpass_token_env"),
        max_bbox_km2=float(downloads.get("max_osm_km2", 5000)), overwrite=args.overwrite,
        cache_dir=cache_dir, refresh=args.overwrite,
    )
    print(path)
elif args.fetch_labels:
    label_sources = area.get("label_sources", config.get("label_sources", {}))
    imagery_path = resolve_path(config, area["imagery"])
    if imagery_path.exists():
        with rasterio.open(imagery_path) as image_ds:
            target_crs = image_ds.crs
    else:
        target_crs = f"EPSG:{_utm_epsg_for_bbox(bbox)}"
    label_cfg = config.get("labels", {})
    rebuild = bool(label_cfg.get("rebuild_from_sources", False))
    existing = None if rebuild else (
        resolve_path(config, area["trails"])
        if resolve_path(config, area["trails"]).exists() else None
    )
    snap_cfg = label_cfg.get("snap_to_relief") or {}
    result = merge_area_labels(
        bbox, resolve_path(config, area["trails"]), area_name=area["name"],
        target_crs=target_crs, cache_dir=cache_dir,
        include_osm=bool(label_sources.get("osm", True)),
        include_ugrc=bool(label_sources.get("ugrc", False)),
        include_usfs=bool(label_sources.get("usfs", False)),
        osm_endpoint=overpass_url,
        ugrc_endpoint=downloads.get("ugrc_trails_url") or UGRC_TRAILS_URL,
        usfs_endpoint=downloads.get("usfs_trails_url") or USFS_TRAILS_URL,
        existing_path=existing,
        refresh=args.overwrite, token_env=downloads.get("overpass_token_env"),
        drop_unnamed_osm_footway=bool(label_cfg.get("drop_unnamed_osm_footway", False)),
        snap_to_relief=snap_cfg,
        dtm_path=resolve_path(config, area["dtm"]) if area.get("dtm") else None,
    )
    print(json.dumps(result, indent=2))
elif args.fetch_naip:
    path = fetch_naip(
        bbox, resolve_path(config, area["imagery"]),
        catalog_url=naip_url,
        collection=downloads.get("naip_collection", "naip"),
        asset_key=downloads.get("naip_asset", "image"),
        resolution_m=float(config["tiling"]["resolution_m"]),
        max_raster_km2=max_area,
        max_raster_pixels=(args.max_raster_pixels if args.max_raster_pixels is not None
                           else int(downloads.get("max_raster_pixels", 70_000_000))),
        sign_url=downloads.get("planetary_computer_sign_url"),
        overwrite=args.overwrite,
    )
    print(path)
elif args.fetch_3dep:
    path = fetch_3dep_dtm(
        bbox, resolve_path(config, area["dtm"]),
        resolution_m=float(config["downloads"].get("dtm_resolution_m", 1.0)),
        endpoint=tnm_url,
        dataset=downloads.get("tnm_1m_dataset", TNM_1M_DATASET),
        max_raster_km2=max_area,
        max_raster_pixels=(args.max_raster_pixels if args.max_raster_pixels is not None
                           else int(downloads.get("max_raster_pixels", 70_000_000))),
        max_download_bytes=max_bytes,
        max_extracted_bytes=max_extracted,
        max_products=max_products,
        overwrite=args.overwrite,
    )
    print(path)
else:
    print(json.dumps({
        "mode": "offline dry-run",
        "area": area["name"],
        "bbox_wsen": bbox,
        "approximate_km2": round(_area_km2(bbox), 2),
        "imagery_output": str(resolve_path(config, area["imagery"])),
        "dtm_output": str(resolve_path(config, area["dtm"])),
        "trails_output": str(resolve_path(config, area["trails"])),
        "raster_area_cap_km2": max_area,
        "download_cap_gb": max_bytes / 1_000_000_000,
        "next_step": "Add --probe for metadata-only network requests; select one --fetch-* flag for explicit acquisition.",
    }, indent=2))
