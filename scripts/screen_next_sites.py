"""Metadata-only shortlist screen for possible future TrailNet study areas."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from pyproj import Geod, Transformer
from shapely.geometry import box
from shapely.ops import transform as transform_geometry, unary_union

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trailnet.data.download import probe_sources

USER_AGENT = "TrailNet site screen (research; contact: trailnet data maintainer)"
OVERPASS = "https://overpass-api.de/api/interpreter"
GEOD = Geod(ellps="WGS84")
SITES = (
    ("Fruita 18 Road", -108.735, 39.235, "approximate North Fruita Desert / 18 Road trail-system anchor"),
    ("Sedona Bell Rock", -111.7619, 34.7915, "Bell Rock Vista trailhead coordinate, reviewed 2026-08-10"),
    ("Crested Butte", -106.985, 38.869, "approximate Lower Loop access near Crested Butte"),
    ("Buffalo Creek", -105.27462, 39.38918, "South Buffalo Creek Road trailhead coordinate"),
    ("Tiger Mountain", -121.936128, 47.467988, "Tiger Summit Trailhead coordinate"),
    ("Copper Harbor", -87.88833, 47.46889, "approximate Copper Harbor center near principal MTB trail access"),
    ("Kingdom Trails", -71.936, 44.588, "approximate East Burke trail-system anchor"),
)
ROAD_CLASSES = {
    "motorway", "motorway_link", "trunk", "trunk_link", "primary",
    "primary_link", "secondary", "secondary_link", "tertiary",
    "tertiary_link", "unclassified", "residential", "living_street",
    "service", "track",
}
TRAIL_CLASSES = {"path", "footway", "cycleway"}


def site_bbox(lon: float, lat: float, half_width_m: float = 2500) -> list[float]:
    """Return a compact lon/lat bbox approximately half_width_m each way."""
    lat_delta = half_width_m / 110_574.0
    lon_delta = half_width_m / (111_320.0 * max(0.2, math.cos(math.radians(lat))))
    return [lon - lon_delta, lat - lat_delta, lon + lon_delta, lat + lat_delta]


def classify_highway(tags: dict) -> str:
    highway = str(tags.get("highway") or "").lower()
    footway = str(tags.get("footway") or "").lower()
    if highway == "footway" and footway in {"sidewalk", "crossing"}:
        return "sidewalk"
    if highway in ROAD_CLASSES - {"track"}:
        return "road"
    if highway == "track":
        return "track_candidate"
    if highway in TRAIL_CLASSES:
        return "path_candidate"
    return "other"


def line_length_km(points: list[list[float]]) -> float:
    total_m = 0.0
    for start, end in zip(points, points[1:]):
        _, _, distance = GEOD.inv(start[0], start[1], end[0], end[1])
        total_m += distance
    return total_m / 1000.0


def product_bbox_coverage(bbox: list[float], products: list[dict]) -> dict:
    """Estimate site coverage from TNM metadata extents; does not prove valid raster coverage."""
    west, south, east, north = bbox
    zone = max(1, min(60, int(math.floor((west + east) / 12.0 + 31))))
    epsg = (32600 if (south + north) / 2 >= 0 else 32700) + zone
    project = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True).transform
    site = transform_geometry(project, box(west, south, east, north))
    extents = []
    for product in products:
        bounds = product.get("boundingBox") or {}
        if not all(key in bounds for key in ("minX", "minY", "maxX", "maxY")):
            continue
        extent = box(float(bounds["minX"]), float(bounds["minY"]),
                     float(bounds["maxX"]), float(bounds["maxY"]))
        extents.append(transform_geometry(project, extent).intersection(site))
    union = unary_union(extents) if extents else box(0, 0, 0, 0)
    area = max(0.0, float(site.area))
    covered = max(0.0, float(union.area))
    return {
        "method": "TNM metadata boundingBox union",
        "estimated_site_area_km2": round(area / 1_000_000, 3),
        "estimated_covered_area_km2": round(covered / 1_000_000, 3),
        "estimated_fraction": round(min(1.0, covered / area), 4) if area else None,
        "product_boxes_with_footprints": len(extents),
        "raster_validity_verified": False,
        "limitation": "metadata bounding boxes estimate product coverage; download/open the selected DEM to verify pixel coverage and resolution",
    }


def osm_lengths(bbox: list[float], session: requests.Session) -> dict:
    west, south, east, north = bbox
    query = (
        "[out:json][timeout:45];"
        f'way["highway"~"^(path|footway|cycleway|track|motorway|motorway_link|'
        f'trunk|trunk_link|primary|primary_link|secondary|secondary_link|'
        f'tertiary|tertiary_link|unclassified|residential|living_street|service)$"]'
        f"({south},{west},{north},{east});out tags geom;"
    )
    response = session.post(
        OVERPASS,
        data={"data": query},
        headers={"User-Agent": USER_AGENT},
        timeout=60,
    )
    response.raise_for_status()
    totals = {"path_candidate": 0.0, "track_candidate": 0.0,
              "road": 0.0, "sidewalk": 0.0, "other": 0.0}
    ways = 0
    for element in response.json().get("elements", []):
        if element.get("type") != "way":
            continue
        category = classify_highway(element.get("tags") or {})
        totals[category] += line_length_km(
            [[point["lon"], point["lat"]] for point in element.get("geometry", [])]
        )
        ways += 1
    return {
        "endpoint": OVERPASS,
        "attribution": "© OpenStreetMap contributors, ODbL",
        "ways": ways,
        "length_km": {key: round(value, 3) for key, value in totals.items()},
    }


def screen_site(name: str, lon: float, lat: float, center_source: str,
                session: requests.Session) -> dict:
    bbox = site_bbox(lon, lat)
    result = {
        "site": name,
        "status": "review_later",
        "center": {"lon": lon, "lat": lat, "source_note": center_source},
        "bbox_wgs84": [round(value, 6) for value in bbox],
        "bbox_half_width_m": 2500,
    }

    try:
        result["sources"] = probe_sources(bbox, overpass_endpoint=OVERPASS, timeout=60)
        result["three_dep_coverage_estimate"] = product_bbox_coverage(
            bbox, result["sources"].get("3dep", {}).get("items", []))
    except (requests.RequestException, ValueError, KeyError) as error:
        result["source_probe"] = {"status": "unavailable", "reason": str(error)}

    try:
        result["osm_lengths"] = osm_lengths(bbox, session)
    except (requests.RequestException, ValueError, KeyError) as error:
        result["osm_lengths"] = {"status": "unavailable", "reason": str(error)}

    result["decision_basis"] = (
        "Metadata and OSM length are screening evidence only. Review seasonal imagery, "
        "trail truth, access, and permissions before any training-data decision."
    )
    three_dep = (result.get("sources") or {}).get("3dep", {})
    result["3dep_resolution_evidence"] = {
        "requested_dataset": three_dep.get("dataset"),
        "product_extents": sorted({item.get("extent") for item in three_dep.get("items", []) if item.get("extent")}),
        "coverage_note": "Product bounding boxes are metadata extents; actual pixel validity requires opening the raster.",
    }
    result["season_evidence"] = {
        "naip_item_dates": sorted({item.get("datetime", "")[:10] for item in (result.get("sources") or {}).get("naip", {}).get("items", []) if item.get("datetime")}),
        "snow_shadow_cloud_leaf_condition": "unknown; metadata dates alone do not establish visibility or canopy state",
    }
    result["independent_review"] = {
        "verified_target_trail_samples": None,
        "verified_distractor_samples": None,
        "status": "not_collected",
        "reason": "No independent local truth source or reviewer sample has been verified for this screen.",
    }
    result["permissions"] = {"status": "unknown", "notes": "No area-specific imagery/trail-data permission review performed."}
    result["decision_reason"] = (
        "review_later: metadata shows candidate imagery and 1 m DEM product extents, but no independently "
        "verified target-trail/distractor samples or area-specific permission review has been collected."
    )
    result["screen_status"] = "review_later"
    return result


def screen_all(output: Path, pause_s: float = 1.1) -> list[dict]:
    results = []
    with requests.Session() as session:
        for index, (name, lon, lat, center_source) in enumerate(SITES):
            if index:
                time.sleep(pause_s)
            results.append(screen_site(name, lon, lat, center_source, session))
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": "Approximate named-site anchor, 5 km x 5 km metadata probe, OSM way lengths; no rasters downloaded",
        "sites": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/overnight-20261107/next-sites.json"),
    )
    args = parser.parse_args()
    results = screen_all(args.output)
    for result in results:
        lengths = result.get("osm_lengths", {}).get("length_km", {})
        print(
            f"{result['site']}: {result['screen_status']} "
            f"paths={lengths.get('path_candidate')} km "
            f"tracks={lengths.get('track_candidate')} km "
            f"roads={lengths.get('road')} km sidewalks={lengths.get('sidewalk')} km"
        )
    print(f"SCREEN_SAVED {args.output}")


if __name__ == "__main__":
    main()
