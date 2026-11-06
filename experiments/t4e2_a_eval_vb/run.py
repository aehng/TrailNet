"""Eval-only: A checkpoint → predict → unfiltered vectorize → length P/R. NO train."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

EXPERIMENT = "t4e2_a_eval_vb"
CONFIG_NAME = "config_exp_t4e2_a_ignore5.yaml"
RASTER_SLUG = "trailnet-multiaoi-v4"
CODE_SLUG = "trailnet-t4e-code"
CKPT_SLUG = "trailnet-t4e2-a-ckpt"
AREA_NAMES = ["kelly_canyon", "corner_canyon", "park_city", "teton_pass", "snowbasin"]
INPUT_ROOT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
PROJECT = WORK / "TrailNet"
LOG_PATH = WORK / "trailnet-t4e2_a-eval-vb-run.log"


def log(message: str) -> None:
    print(message, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as stream:
        stream.write(message + "\n")


def run_command(args: list[str], env: dict[str, str] | None = None) -> str:
    log("$ " + " ".join(args))
    output = []
    process = subprocess.Popen(
        args, cwd=PROJECT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, errors="replace", bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        clean = line.rstrip("\r\n")
        output.append(clean)
        log(clean)
    code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, args)
    return "\n".join(output)


def find_mount(slug: str) -> Path:
    candidate = INPUT_ROOT / slug
    if candidate.is_dir():
        return candidate
    hits = list(INPUT_ROOT.rglob(slug))
    dirs = [p for p in hits if p.is_dir()]
    if dirs:
        return dirs[0]
    visible = sorted(p.name for p in INPUT_ROOT.iterdir()) if INPUT_ROOT.is_dir() else []
    raise FileNotFoundError(f"dataset {slug} not mounted; visible={visible}")


def find_file(root: Path, name: str) -> Path | None:
    direct = root / name
    if direct.is_file():
        return direct
    hits = list(root.rglob(name))
    return hits[0] if hits else None


def list_mount(path: Path) -> None:
    try:
        names = sorted(p.name for p in path.iterdir())[:80]
        log(f"MOUNT_LIST {path} {names}")
    except Exception as exc:
        log(f"MOUNT_LIST_FAIL {path} {exc}")


def stage_inputs() -> None:
    rasters = find_mount(RASTER_SLUG)
    code_mount = find_mount(CODE_SLUG)
    ckpt_mount = find_mount(CKPT_SLUG)
    log(f"INPUT_RASTERS {rasters}")
    log(f"INPUT_CODE {code_mount}")
    log(f"INPUT_CKPT {ckpt_mount}")
    list_mount(INPUT_ROOT)
    list_mount(rasters)
    list_mount(code_mount)
    list_mount(ckpt_mount)
    PROJECT.mkdir(parents=True, exist_ok=True)
    trees = [p for p in (code_mount, code_mount / "trailnet_code") if (p / "src").is_dir()]
    archive = find_file(code_mount, "trailnet_code.tar.gz")
    if trees:
        source_root = trees[0]
        for name in ("src", "scripts", "requirements.txt", "LICENSE", "README.md", "label_overrides", "docs", "tests"):
            source = source_root / name
            if source.is_dir():
                shutil.copytree(source, PROJECT / name, dirs_exist_ok=True)
            elif source.is_file():
                shutil.copy2(source, PROJECT / name)
        yaml_root = code_mount if any(code_mount.glob("config*.yaml")) else source_root
        for yaml in yaml_root.glob("config*.yaml"):
            shutil.copy2(yaml, PROJECT / yaml.name)
        log(f"CODE_STAGED from directory {source_root}")
    elif archive is not None and archive.is_file():
        with tarfile.open(archive, "r:gz") as bundle:
            bundle.extractall(PROJECT)
        log("CODE_STAGED from tar.gz")
    else:
        raise FileNotFoundError("no TrailNet source on code mount")

    # Prove fixed vectorize is present
    post = (PROJECT / "src" / "trailnet" / "postprocess.py").read_text(encoding="utf-8")
    if "novelty_output_path" not in post or "novelty_filter=False" not in post:
        raise RuntimeError("stale postprocess.py: missing unfiltered/novelty split")
    log("POSTPROCESS_OK unfiltered trails + novelty split")
    ve = (PROJECT / "src" / "trailnet" / "vector_eval.py").read_text(encoding="utf-8")
    if "split_block_polygons" not in ve or "val_blocks" not in ve:
        raise RuntimeError("stale vector_eval.py: missing val/test block scopes")
    log("VECTOR_EVAL_OK block scopes present")

    ckpt = find_file(ckpt_mount, "trailnet_best.pt")
    if ckpt is None:
        raise FileNotFoundError("trailnet_best.pt missing on ckpt mount")
    dest = PROJECT / "checkpoints_t4e2_a"
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ckpt, dest / "trailnet_best.pt")
    log(f"CKPT_STAGED {dest / 'trailnet_best.pt'} bytes={ckpt.stat().st_size}")

    for name in AREA_NAMES:
        area_dir = PROJECT / "data" / name
        area_dir.mkdir(parents=True, exist_ok=True)
        for stem, suffix in (("naip_rgbn", ".tif"), ("3dep_dtm", ".tif"), ("trails", ".gpkg")):
            source = find_file(rasters, f"{name}_{stem}{suffix}")
            if source is None:
                raise FileNotFoundError(f"missing {name}_{stem}{suffix}")
            shutil.copy2(source, area_dir / f"{stem}{suffix}")
    overrides = PROJECT / "label_overrides"
    if not overrides.is_dir():
        for zcand in (code_mount / "label_overrides.zip", PROJECT / "label_overrides.zip"):
            if zcand.is_file():
                import zipfile
                with zipfile.ZipFile(zcand) as zf:
                    zf.extractall(PROJECT)
                break
    overrides = PROJECT / "label_overrides"
    if not overrides.is_dir():
        hits = [h for h in PROJECT.rglob("label_overrides") if h.is_dir()]
        if not hits:
            raise FileNotFoundError("label_overrides missing")
        overrides = hits[0]
    for name in AREA_NAMES:
        src = overrides / name / "trails.gpkg"
        if not src.is_file():
            raise FileNotFoundError(f"missing sidewalk-aware trails for {name}: {src}")
        shutil.copy2(src, PROJECT / "data" / name / "trails.gpkg")
        log(f"LABEL_OVERRIDE {name} <- {src}")


def apply_checkpoint_threshold() -> float:
    import torch
    import yaml
    ckpt_path = PROJECT / "checkpoints_t4e2_a" / "trailnet_best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    thr = float(ckpt.get("best_threshold", 0.9))
    cfg_path = PROJECT / CONFIG_NAME
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg.setdefault("postprocess", {})["threshold"] = thr
    cfg.setdefault("inference", {})["threshold"] = thr
    cfg.setdefault("evaluation", {})
    cfg["evaluation"]["vector_match_tolerances_m"] = [3.0, 5.0, 8.0, 10.0]
    cfg["evaluation"]["vector_report_tolerance_m"] = 5.0
    cfg["evaluation"]["clip_vector_metrics_to_prediction"] = True
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    log(f"THRESHOLD_FROM_CKPT {thr} epoch={ckpt.get('epoch')} val_f1={ckpt.get('val_f1')}")
    return thr


def ensure_deps() -> None:
    import importlib.util
    mapping = {
        "segmentation_models_pytorch": "segmentation-models-pytorch",
        "geopandas": "geopandas",
        "rasterio": "rasterio",
        "skimage": "scikit-image",
        "yaml": "pyyaml",
    }
    missing_pkgs = [pkg for mod, pkg in mapping.items() if importlib.util.find_spec(mod) is None]
    if missing_pkgs:
        run_command([sys.executable, "-m", "pip", "install", "-q", *missing_pkgs])



def write_vector_metrics() -> dict:
    sys.path.insert(0, str(PROJECT / "src"))
    from trailnet.config import load_config
    from trailnet.vector_eval import length_weighted_aggregate, vector_metrics_for_area
    config = load_config(PROJECT / CONFIG_NAME)
    metrics = {}
    val_by_area = {}
    test_by_area = {}
    full_by_area = {}
    for area in config["areas"]:
        row = vector_metrics_for_area(config, area)
        metrics[area["name"]] = row
        scopes = row.get("scopes") or {}
        full = scopes.get("full_footprint") or row
        val = scopes.get("val_blocks") or {}
        test = scopes.get("test_blocks") or {}
        full_by_area[area["name"]] = full
        val_by_area[area["name"]] = val
        test_by_area[area["name"]] = test
        log(f"VECTOR_METRICS {area['name']} {json.dumps(full, default=str)}")
        log(f"VAL_BLOCK_VECTOR_METRICS {area['name']} {json.dumps(val, default=str)}")
        log(f"TEST_BLOCK_VECTOR_METRICS {area['name']} {json.dumps(test, default=str)}")
    report_tol = float(config.get("evaluation", {}).get("vector_report_tolerance_m", 5.0))
    val_agg = length_weighted_aggregate(val_by_area, report_tolerance_m=report_tol)
    test_agg = length_weighted_aggregate(test_by_area, report_tolerance_m=report_tol)
    full_agg = length_weighted_aggregate(full_by_area, report_tolerance_m=report_tol)
    log(f"VAL_BLOCK_AGG_5M {json.dumps(val_agg, default=str)}")
    log(f"TEST_BLOCK_AGG_5M {json.dumps(test_agg, default=str)}")
    log(f"FULL_FOOTPRINT_AGG_5M {json.dumps(full_agg, default=str)}")
    metrics["_aggregates"] = {
        "val_blocks": val_agg,
        "test_blocks": test_agg,
        "full_footprint": full_agg,
        "report_tolerance_m": report_tol,
    }
    return metrics



def re_rasterize_roundtrip_check() -> dict:
    """Compare vectorized lines re-rasterized vs hard probability mask (same CRS/footprint)."""
    import geopandas as gpd
    import numpy as np
    import rasterio
    from rasterio import features
    from skimage.morphology import skeletonize

    sys.path.insert(0, str(PROJECT / "src"))
    from trailnet.config import load_config, resolve_path
    config = load_config(PROJECT / CONFIG_NAME)
    thr = float(config["postprocess"]["threshold"])
    out = {}
    for area in config["areas"]:
        name = area["name"]
        prob_path = resolve_path(config, config["paths"]["prediction_dir"]) / f"{name}_probability.tif"
        vec_path = resolve_path(config, config["paths"]["vector_dir"]) / f"{name}_trails.geojson"
        if not prob_path.exists() or not vec_path.exists():
            out[name] = {"skipped": "missing prob or vectors"}
            continue
        with rasterio.open(prob_path) as ds:
            prob = ds.read(1)
            transform, crs = ds.transform, ds.crs
            shape = prob.shape
        hard = (prob >= thr)
        skel = skeletonize(hard)
        gdf = gpd.read_file(vec_path).to_crs(crs)
        burned = features.rasterize(
            [(geom, 1) for geom in gdf.geometry if geom is not None and not geom.is_empty],
            out_shape=shape, transform=transform, fill=0, dtype="uint8",
        ).astype(bool)
        # Dilate vector burn by 1px for fair centerline compare
        from skimage.morphology import binary_dilation, disk
        burned_d = binary_dilation(burned, disk(1))
        skel_n = int(skel.sum())
        burn_n = int(burned.sum())
        inter = int((skel & burned_d).sum())
        precision = (inter / burn_n) if burn_n else 0.0
        recall = (inter / skel_n) if skel_n else 0.0
        row = {
            "threshold": thr,
            "crs": str(crs),
            "skel_px": skel_n,
            "vector_burn_px": burn_n,
            "inter_skel_vs_vecdilate1": inter,
            "roundtrip_precision": round(precision, 4),
            "roundtrip_recall": round(recall, 4),
            "n_vector_lines": int(len(gdf)),
            "meaningful": bool(skel_n > 0 and burn_n > 0 and recall > 0.1),
        }
        out[name] = row
        log(f"RERASTER_CHECK {name} {json.dumps(row)}")
    return out


def main() -> None:
    if LOG_PATH.exists():
        LOG_PATH.unlink()
    started = time.time()
    os.environ.setdefault("MPLBACKEND", "Agg")
    log(f"TrailNet {EXPERIMENT} starting (NO TRAIN)")
    stage_inputs()
    ensure_deps()
    thr = apply_checkpoint_threshold()
    sys.path.insert(0, str(PROJECT / "src"))
    import torch
    log(f"CUDA_COUNT {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    # prepare sidewalk-aware tiles (CPU-ish) then GPU predict
    run_command([sys.executable, "scripts/prepare_data.py", "--config", CONFIG_NAME, "--all-areas"])
    run_command([sys.executable, "scripts/predict.py", "--config", CONFIG_NAME, "--all-areas"])
    run_command([sys.executable, "scripts/vectorize.py", "--config", CONFIG_NAME, "--all-areas"])
    # Confirm novelty product exists and trails are unfiltered (novelty count <= trails)
    for name in AREA_NAMES:
        trails = PROJECT / "outputs_t4e_a" / "vectors" / f"{name}_trails.geojson"
        novelty = PROJECT / "outputs_t4e_a" / "vectors" / f"{name}_novelty.geojson"
        if trails.exists() and novelty.exists():
            import geopandas as gpd
            nt, nn = len(gpd.read_file(trails)), len(gpd.read_file(novelty))
            log(f"NOVELTY_SPLIT {name} trails={nt} novelty={nn}")
            if nn > nt:
                raise RuntimeError(f"novelty count {nn} > trails {nt} for {name}")
    vectors = write_vector_metrics()
    reraster = re_rasterize_roundtrip_check()
    # also run held-out pixel evaluate for continuity (uses same ckpt threshold via evaluate)
    eval_output = run_command([
        sys.executable, "scripts/evaluate.py", "--config", CONFIG_NAME,
        "--thresholds", "0.90",
    ])
    aggregates = (vectors or {}).get("_aggregates") or {}
    compact = {
        "run": EXPERIMENT,
        "config": CONFIG_NAME,
        "architecture": "dual_terrain",
        "no_train": True,
        "eval_threshold": thr,
        "val_block_agg_5m": aggregates.get("val_blocks"),
        "test_block_agg_5m": aggregates.get("test_blocks"),
        "full_footprint_agg_5m": aggregates.get("full_footprint"),
        "reraster_check": {
            k: {kk: vv for kk, vv in (v or {}).items() if kk in (
                "roundtrip_precision", "roundtrip_recall", "meaningful", "crs", "threshold"
            )}
            if isinstance(v, dict) else v
            for k, v in (reraster or {}).items()
        },
        "elapsed_seconds": round(time.time() - started, 1),
    }
    # Per-area val/test headlines (small) for downloadable gate table
    val_block_metrics = {
        "aggregates": aggregates,
        "per_area": {
            name: {
                "val_blocks": (row.get("scopes") or {}).get("val_blocks"),
                "test_blocks": (row.get("scopes") or {}).get("test_blocks"),
                "full_footprint": (row.get("scopes") or {}).get("full_footprint"),
            }
            for name, row in (vectors or {}).items()
            if name != "_aggregates" and isinstance(row, dict)
        },
    }
    metrics = {
        **compact,
        "vector_metrics": vectors,
        "reraster_check": reraster,
        "eval_tail": eval_output[-2000:],
    }
    metrics_text = json.dumps(metrics, indent=2, default=str)
    compact_text = json.dumps(compact, indent=2, default=str)
    val_text = json.dumps(val_block_metrics, indent=2, default=str)
    # Root of /kaggle/working so files API retains them (not buried under project/)
    (WORK / "metrics.json").write_text(compact_text, encoding="utf-8")
    (WORK / "val_block_metrics.json").write_text(val_text, encoding="utf-8")
    (WORK / "metrics_full.json").write_text(metrics_text, encoding="utf-8")
    out_dir = PROJECT / "outputs_t4e_a"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(compact_text, encoding="utf-8")
    (out_dir / "val_block_metrics.json").write_text(val_text, encoding="utf-8")
    log("METRICS_WRITTEN " + json.dumps({"root": str(WORK / "metrics.json"), "val_block": str(WORK / "val_block_metrics.json")}))
    log("RUN_COMPLETE " + json.dumps(compact, default=str))


if __name__ == "__main__":
    main()
