"""Dry-run snap_lines_to_relief and write QA overlays. Does not overwrite v4 GPKGs."""
import argparse
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio


from trailnet.config import load_config, resolve_path, select_areas
from trailnet.data.labels import snap_lines_to_relief
from trailnet.data.terrain import derive_terrain


def overlay_png(dtm_path, original, snapped, out_png, size=900):
    with rasterio.open(dtm_path) as ds:
        transform = ds.transform
        z = ds.read(1)
        height, width = ds.height, ds.width
    hillshade, _slope, relief = derive_terrain(z, transform.a, transform.e, 15)
    centroid = original.union_all().centroid
    col, row = ~transform * (centroid.x, centroid.y)
    col0 = int(np.clip(col - size / 2, 0, max(0, width - size)))
    row0 = int(np.clip(row - size / 2, 0, max(0, height - size)))
    h = min(size, height - row0)
    w = min(size, width - col0)
    hs = hillshade[row0:row0 + h, col0:col0 + w]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(hs, cmap="gray", origin="upper")

    def draw(frame, color, lw):
        for geom in frame.geometry:
            parts = list(geom.geoms) if geom.geom_type == "MultiLineString" else [geom]
            for part in parts:
                xs, ys = zip(*part.coords)
                cols, rows = [], []
                for x, y in zip(xs, ys):
                    c, r = ~transform * (x, y)
                    cols.append(c - col0)
                    rows.append(r - row0)
                ax.plot(cols, rows, color=color, linewidth=lw)

    draw(original, "#e24a16", 1.8)
    draw(snapped, "#3dbbdb", 1.2)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_axis_off()
    ax.set_title("orange=original  cyan=snapped")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def median_abs_offset(original, snapped, spacing=25.0):
    offs = []
    for a, b in zip(original.geometry, snapped.geometry):
        if a is None or b is None or a.is_empty or b.is_empty:
            continue
        n = max(2, int(a.length // spacing))
        for i in range(1, n):
            d = i * spacing
            if d >= a.length or d >= b.length:
                continue
            pa, pb = a.interpolate(d), b.interpolate(min(d, b.length))
            offs.append(pa.distance(pb))
    if not offs:
        return None, None
    arr = np.array(offs)
    return float(np.median(arr)), float(np.mean(arr < 2.0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_v4_local.yaml")
    parser.add_argument("--areas", default="kelly_canyon,teton_pass,corner_canyon")
    parser.add_argument("--out-dir", default="data/trails_snapped")
    args = parser.parse_args()
    config = load_config(args.config)
    names = [n.strip() for n in args.areas.split(",") if n.strip()]
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    snap_cfg = (config.get("labels") or {}).get("snap_to_relief") or {}
    report = {}
    for name in names:
        area = select_areas(config, name)[0]
        trails = resolve_path(config, area["trails"])
        dtm = resolve_path(config, area["dtm"])
        original = gpd.read_file(trails)
        snapped = snap_lines_to_relief(
            original, dtm,
            max_offset_m=float(snap_cfg.get("max_offset_m", 15.0)),
            step_m=float(snap_cfg.get("step_m", 0.6)),
            sample_spacing_m=float(snap_cfg.get("sample_spacing_m", 25.0)),
            score_margin=float(snap_cfg.get("score_margin", 0.08)),
            sharpness_ratio=float(snap_cfg.get("sharpness_ratio", 1.20)),
            reconstruct=str(snap_cfg.get("reconstruct", "smooth_line")),
        )
        dest = Path(area["trails"]).with_name("trails_snapped.gpkg")
        dest = resolve_path(config, str(dest))
        dest.parent.mkdir(parents=True, exist_ok=True)
        snapped.to_file(dest, layer="trails", driver="GPKG")
        png = out_root / f"{name}_snap_qa.png"
        overlay_png(dtm, original.to_crs(snapped.crs) if original.crs != snapped.crs else original, snapped, png)
        med, share = median_abs_offset(
            original.to_crs(snapped.crs) if original.crs != snapped.crs else original, snapped
        )
        report[name] = {
            "snapped_gpkg": str(dest),
            "qa_png": str(png),
            "n": int(len(snapped)),
            "median_abs_shift_m": med,
            "share_shift_lt_2m": share,
        }
        print(f"SNAP_QA area={name} n={len(snapped)} median_abs_shift_m={med} png={png}")
    (out_root / "snap_qa.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    sys.exit(main() or 0)
