"""Reconstruct NAIP source candidates for the exact AOI extents from STAC metadata."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trailnet.data.download import probe_naip


def trace(summary_path: Path, output_path: Path) -> dict:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    results = []
    for area in summary["areas"]:
        west, south, east, north = area["bbox_wsen"]
        bbox = [west, south, east, north]
        try:
            result = probe_naip(bbox, max_items=1000, timeout=60)
            dates = [item.get("datetime", "") for item in result.get("items", []) if item.get("datetime")]
            latest_year = max((date[:4] for date in dates), default=None)
            selected = [item for item in result.get("items", [])
                        if latest_year and item.get("datetime", "").startswith(latest_year)]
            results.append({
                "area": area["name"], "bbox_wgs84": bbox,
                "stac_catalog": result.get("catalog"), "collection": result.get("collection"),
                "item_count": result.get("item_count"), "truncated": result.get("truncated"),
                "latest_year_selected_by_fetcher": latest_year,
                "items_selected_by_latest_year_rule": selected,
                "source_identity_caveat": "Reconstructed from retained AOI bbox and current fetcher rule; raster tags did not retain source item IDs, so exact mosaic item identity is not independently recorded.",
            })
        except Exception as error:
            results.append({"area": area["name"], "bbox_wgs84": bbox,
                            "status": "unavailable", "reason": str(error)})
    payload = {"generated_at": datetime.now(timezone.utc).isoformat(),
               "method": "NAIP STAC metadata query at retained AOI bounding boxes; no assets downloaded",
               "areas": results}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("experiments/overnight-20261107/naip-source-dates.json"))
    args = parser.parse_args()
    result = trace(args.summary, args.output)
    for area in result["areas"]:
        print(f"{area['area']}: latest={area.get('latest_year_selected_by_fetcher')} selected={len(area.get('items_selected_by_latest_year_rule', []))}")


if __name__ == "__main__":
    main()
