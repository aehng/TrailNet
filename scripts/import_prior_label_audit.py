"""Copy and summarize read-only label-audit evidence from an existing checkout."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def import_audit(source: Path, output: Path) -> dict:
    measurements_path = source / "measurements.json"
    followup_path = source / "review_followup.json"
    measurements = json.loads(measurements_path.read_text(encoding="utf-8"))
    followup = json.loads(followup_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)

    copied = []
    for name in (
        "measurements.json", "review_followup.json", "km_by_source.csv",
        "offsets_park_city.csv", "offsets_corner_canyon.csv",
        "offsets_kelly_canyon.csv", "offsets_snowbasin.csv",
        "offsets_teton_pass.csv", "overlay_park_city.png",
        "overlay_corner_canyon.png", "overlay_kelly_canyon.png",
        "overlay_snowbasin.png", "overlay_teton_pass.png",
    ):
        path = source / name
        if path.is_file():
            destination = output / name
            shutil.copy2(path, destination)
            copied.append({"file": name, "sha256": sha256(destination), "bytes": destination.stat().st_size})

    alignment = measurements.get("alignment", {})
    summary = {
        "evidence_type": "pre-existing exploratory audit; copied from the preserved original checkout",
        "source_directory": str(source),
        "source_config": measurements.get("config"),
        "copied_files": copied,
        "alignment_by_area": {
            area: {
                key: value.get(key)
                for key in ("n_samples", "mean_offset_m", "abs_median_m", "share_abs_lt_2m",
                            "share_abs_gt_5m", "share_abs_gt_10m", "dtm_pixel_m")
            }
            for area, value in alignment.items()
        },
        "corner_canyon_footway_inventory": next(
            (row for row in followup.get("footway_named", []) if row.get("aoi") == "corner_canyon"), None
        ),
        "limitations": [
            "Relief-edge offsets measure alignment to a terrain cue, not whether each line is a real trail.",
            "The sampled relief response can be a road, drainage, ski cut, or other terrain feature.",
            "No blinded, independently adjudicated image-window audit or snow/cloud/shadow label was present in these artifacts.",
            "OSM and local source overlap does not establish independent ground truth.",
        ],
        "interpretation": (
            "Use this evidence to justify a blinded audit and alignment/visibility diagnostics. "
            "Do not automatically drop unnamed footways or alter masks from this audit alone."
        ),
    }
    (output / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("experiments/overnight-20261107/audit"))
    args = parser.parse_args()
    summary = import_audit(args.source, args.output)
    print(f"IMPORTED {len(summary['copied_files'])} files into {args.output}")
    for area, metrics in summary["alignment_by_area"].items():
        print(f"{area}: abs_median={metrics['abs_median_m']:.2f}m n={metrics['n_samples']}")


if __name__ == "__main__":
    main()
