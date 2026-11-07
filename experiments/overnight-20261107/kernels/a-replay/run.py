"""Eval-only: A checkpoint → predict → unfiltered vectorize → length P/R. NO train."""
from __future__ import annotations

import json
import csv
import hashlib
import math
import os
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

EXPERIMENT = "t4e2_a_eval"
CONFIG_NAME = "config_exp_t4e2_a_ignore5.yaml"
RASTER_SLUG = "trailnet-multiaoi-v4"
CODE_SLUG = "trailnet-t4e-code"
CKPT_SLUG = "trailnet-t4e2-a-ckpt"
AREA_NAMES = ["kelly_canyon", "corner_canyon", "park_city", "teton_pass", "snowbasin"]
INPUT_ROOT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
PROJECT = Path("/kaggle/temp/TrailNet")
LOG_PATH = WORK / "trailnet-t4e2_a-eval-run.log"
DEFAULT_COMMAND_TIMEOUT_SECONDS = 60 * 60
RUN_TIMEOUT_SECONDS = 90 * 60


def log(message: str) -> None:
    print(message, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as stream:
        stream.write(message + "\n")


def write_callback(record: dict, name: str = "command") -> Path:
    callback_dir = WORK / "callbacks"
    callback_dir.mkdir(parents=True, exist_ok=True)
    path = callback_dir / f"{time.time_ns()}-{name}.callback.json"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)
    return path


def run_command(
    args: list[str], env: dict[str, str] | None = None,
    timeout_seconds: float = DEFAULT_COMMAND_TIMEOUT_SECONDS,
) -> str:
    started_at = time.time()
    started_monotonic = time.monotonic()
    log("$ " + " ".join(args))
    timed_out = False
    output = ""
    launch_error = None
    try:
        process = subprocess.Popen(
            args, cwd=PROJECT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, errors="replace",
        )
        try:
            output, _ = process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            output, _ = process.communicate()
        code = process.returncode
    except OSError as exc:
        launch_error = exc
        code = None
        output = str(exc)
        status = "launch_error"
    else:
        status = "timed_out" if timed_out else ("complete" if code == 0 else "failed")

    for line in output.splitlines():
        log(line)
    ended_at = time.time()
    callback = {
        "executable": args[0],
        "arguments": args[1:],
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started_at)),
        "ended_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ended_at)),
        "elapsed_seconds": round(time.monotonic() - started_monotonic, 3),
        "timeout_seconds": timeout_seconds,
        "status": status,
        "exit_code": None if timed_out else code,
        "timed_fallback_triggered": timed_out,
        "output_tail": output[-4000:],
    }
    callback_path = write_callback(callback, Path(args[0]).stem or "command")
    log(f"COMMAND_CALLBACK {callback_path} status={status}")
    if timed_out:
        raise subprocess.TimeoutExpired(args, timeout_seconds, output=output)
    if launch_error is not None:
        raise launch_error
    if code:
        raise subprocess.CalledProcessError(code, args, output=output)
    return output


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
    log("VECTOR_EVAL_SKIPPED pixel parity replay only")

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
    import torch
    mapping = {
        "segmentation_models_pytorch": "segmentation-models-pytorch",
        "geopandas": "geopandas",
        "rasterio": "rasterio",
        "skimage": "scikit-image",
        "yaml": "pyyaml",
    }
    missing_pkgs = [pkg for mod, pkg in mapping.items() if importlib.util.find_spec(mod) is None]
    if missing_pkgs:
        version, cuda_build = torch.__version__, torch.version.cuda
        constraint = WORK / "torch-constraints.txt"
        constraint.write_text(f"torch=={version}\n", encoding="utf-8")
        run_command([sys.executable, "-m", "pip", "install", "-q", "-c", str(constraint), *missing_pkgs])
        probe = (
            "import json, torch; "
            "print(json.dumps({'version': torch.__version__, 'cuda': torch.version.cuda}))"
        )
        installed_build = json.loads(run_command([sys.executable, "-c", probe]).splitlines()[-1])
        if installed_build["version"] != version or installed_build["cuda"] != cuda_build:
            raise RuntimeError("dependency installation changed the preinstalled PyTorch/CUDA build")
        log(f"PYTORCH_BUILD_PRESERVED version={version} cuda={cuda_build}")



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
            all_touched=False,
        ).astype(bool)
        from trailnet.vector_eval import assert_metric_raster_grid, directional_reraster_scores
        try:
            grid = assert_metric_raster_grid(crs, transform, shape[1], shape[0])
        except ValueError as exc:
            out[name] = {
                "skipped": "bad_affine_or_crs",
                "error": str(exc),
                "roundtrip_is_trail_truth": False,
            }
            log(f"RERASTER_CHECK {name} {json.dumps(out[name])}")
            continue
        # Directional 1px counts. Recall uses dilated burn over skeleton pixels.
        # Precision uses dilated skeleton over vector-burn pixels. The old ratio
        # (skeleton∩dilated-burn) / burn_px can exceed 1 and is not reported as precision.
        scores = directional_reraster_scores(skel, burned, dilate_px=1)
        recall = scores["roundtrip_recall"] or 0.0
        row = {
            "threshold": thr,
            "n_vector_lines": int(len(gdf)),
            "meaningful": bool(scores["skel_px"] > 0 and scores["vector_burn_px"] > 0 and recall > 0.1),
            **grid,
            **scores,
        }
        out[name] = row
        log(f"RERASTER_CHECK {name} {json.dumps(row)}")
    return out


EXPECTED_INDEX_SHA256 = {
    "corner_canyon": "277eac2dad6bdc33386eba744838a82fccea0eea0cf9e90f69a83faee7ed2411",
    "kelly_canyon": "60bb9f2c139cfaa47c6de7fda1121d325bfb4807ff1c056ad2dbb2d295a3f63d",
    "park_city": "b288e2b19654911aee4b81b5d51c1679affc3227a75b9964c0eb359a834a8ba2",
    "snowbasin": "250e16679bacb0a0d20d1160ad85e1724718a20576e3b8a19535eeb126732500",
    "teton_pass": "f04eff6be5a833ad08d2a882e7c3a88eaf16b53cef4a948eea69ab735b50984f",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _audit_split(rows):
    import numpy as np
    result = {"train": [], "val": [], "test": []}
    for row in rows:
        bx, by = int(np.floor(float(row["x"]) / 2500)), int(np.floor(float(row["y"]) / 2500))
        key = f"42:{row['area']}:{bx}:{by}".encode()
        value = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / 2**64
        result["test" if value < 0.15 else ("val" if value < 0.30 else "train")].append(row)
    return result


def _split_exclusive_footprints(rows, tile_width_m):
    from shapely.geometry import box
    from shapely.ops import unary_union

    if tile_width_m <= 0:
        raise ValueError("tile_width_m must be positive")
    grouped = {}
    half = tile_width_m / 2.0
    for row in rows:
        footprint = box(float(row["x"]) - half, float(row["y"]) - half,
                        float(row["x"]) + half, float(row["y"]) + half)
        grouped.setdefault(row["area"], {}).setdefault(row["split"], []).append(footprint)
    exclusive = {}
    for area, split_geometries in grouped.items():
        unions = {name: unary_union(split_geometries[name]) if split_geometries.get(name) else None
                  for name in ("train", "val", "test")}
        for split_name, own in unions.items():
            others = [geom for name, geom in unions.items()
                      if name != split_name and geom is not None]
            other_union = unary_union(others) if others else None
            exclusive[(area, split_name)] = (
                own.difference(other_union) if own is not None and other_union is not None else own
            )
    return exclusive


def _window_is_split_exclusive(row, split_name, crop_m, exclusive):
    from shapely.geometry import box

    if crop_m <= 0:
        raise ValueError("crop_m must be positive")
    footprint = exclusive.get((row["area"], split_name))
    if footprint is None or footprint.is_empty:
        return False
    half = crop_m / 2.0
    crop = box(float(row["x"]) - half, float(row["y"]) - half,
               float(row["x"]) + half, float(row["y"]) + half)
    return footprint.covers(crop)


def write_audit_windows(index_root: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import rasterio
    from rasterio.windows import Window

    audit_area_order = ("park_city", "corner_canyon", "kelly_canyon", "snowbasin", "teton_pass")
    rows = []
    for area in audit_area_order:
        with (index_root / area / "index.csv").open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                row["index_dir"] = str(index_root / area)
                rows.append(row)
    split = _audit_split(rows)
    split_rows = [{**row, "split": split_name} for split_name, bucket in split.items()
                  for row in bucket]
    exclusive_footprints = _split_exclusive_footprints(split_rows, tile_width_m=512 * 0.6)
    allocation = {"park_city": {"train": 12, "val": 12},
                  "corner_canyon": {"train": 3, "val": 3},
                  "kelly_canyon": {"train": 3, "val": 3},
                  "snowbasin": {"train": 3, "val": 3},
                  "teton_pass": {"train": 3, "val": 3}}
    selected, shortages = [], []
    for area in audit_area_order:
        for split_name in ("train", "val"):
            wanted = allocation[area][split_name]
            candidates = [
                r for r in split[split_name]
                if r["area"] == area
                and _window_is_split_exclusive(r, split_name, crop_m=128.0,
                                               exclusive=exclusive_footprints)
            ]
            pool = sorted(candidates, key=lambda r: hashlib.sha256(
                f"42:{r['area']}:{r['image']}".encode()).hexdigest())[:max(wanted*4, wanted)]
            positive, empty = [], []
            for row in pool:
                base = Path(row["index_dir"])
                with rasterio.open(base / row["class_map"]) as ds:
                    crop = round(128 / math.sqrt(abs(ds.transform.a * ds.transform.e)))
                    win = Window(max(0,(ds.width-crop)//2), max(0,(ds.height-crop)//2),
                                 min(crop,ds.width), min(crop,ds.height))
                    labels = ds.read(1, window=win)
                (positive if np.isin(labels,[1,2,3]).any() else empty).append(row)
            chosen = positive[:math.ceil(wanted/2)] + empty[:math.ceil(wanted/2)]
            used = {r["image"] for r in chosen}
            chosen.extend(r for r in pool if r["image"] not in used and len(chosen)<wanted)
            if len(chosen)<wanted:
                shortages.append({"area":area,"split":split_name,"requested":wanted,"available":len(chosen)})
            selected.extend((r,split_name) for r in chosen)

    audit_dir = WORK / "audit"
    audit_dir.mkdir(parents=True,exist_ok=True)
    key_rows, panels = [], []
    for number,(row,split_name) in enumerate(selected,1):
        base=Path(row["index_dir"]); image_path=base/row["image"]; class_path=base/row["class_map"]
        with rasterio.open(image_path) as ds:
            crop=round(128/math.sqrt(abs(ds.transform.a*ds.transform.e)))
            win=Window(max(0,(ds.width-crop)//2),max(0,(ds.height-crop)//2),min(crop,ds.width),min(crop,ds.height))
            image=ds.read(window=win).astype(np.float32); valid=np.all(ds.read_masks(window=win)>0,axis=0)
            transform=ds.window_transform(win); cx,cy=transform*(win.width/2,win.height/2)
        with rasterio.open(class_path) as ds: labels=ds.read(1,window=win)
        sid=f"W{number:03d}"
        key_rows.append({"sample_id":sid,"area":row["area"],"split":split_name,"image":row["image"],
            "center_x":cx,"center_y":cy,"crop_width_px":int(win.width),"crop_height_px":int(win.height),
            "resolution_m":0.6,"image_mask_valid_fraction":round(float(valid.mean()),6),
            "image_sha256":sha256(image_path),"class_map_sha256":sha256(class_path),
            "class_1_px":int((labels==1).sum()),"class_2_px":int((labels==2).sum()),
            "class_3_px":int((labels==3).sum()),"ignore_px":int((labels==4).sum()),
            "background_px":int((labels==0).sum()),
            "selection_stratum":"reference_positive" if np.isin(labels,[1,2,3]).any() else "apparently_empty"})
        rgb=np.moveaxis(image[:3],0,-1); rgb=np.clip(rgb/max(1.0,float(np.nanpercentile(rgb,99))),0,1)
        cir=np.moveaxis(image[[3,0,1]],0,-1); cir=np.clip(cir/max(1.0,float(np.nanpercentile(cir,99))),0,1)
        panels.append((sid,rgb,cir,image[6]))
    with (audit_dir/"audit_samples.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(key_rows[0])); writer.writeheader(); writer.writerows(key_rows)
    for sheet,start in enumerate(range(0,len(panels),16),1):
        batch=panels[start:start+16]; fig,axes=plt.subplots(len(batch),3,figsize=(11,max(3,len(batch)*2.1)),squeeze=False)
        for i,(sid,rgb,cir,relief) in enumerate(batch):
            for ax,array,title in zip(axes[i],(rgb,cir,relief),("RGB","NIR/R/G","3DEP relief")):
                ax.imshow(array,cmap=None if array.ndim==3 else "gray"); ax.set_title(title,fontsize=8); ax.set_xticks([]); ax.set_yticks([])
            axes[i,0].set_ylabel(sid,rotation=0,labelpad=20,fontsize=8)
        fig.tight_layout(); fig.savefig(audit_dir/f"contact_sheet_{sheet:02d}.png",dpi=130); plt.close(fig)
    summary={"seed":42,"block_size_m":2500,"crop_target_m":128,"requested_windows":48,
             "sampled_windows":len(key_rows),"shortages":shortages,
             "tile_pixel_identity_check":"compare image_sha256 and class_map_sha256 to local provisional samples",
             "limitations":["no independent human adjudication","no labels or predictions appear on contact sheets"]}
    (audit_dir/"audit_window_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    return summary


def main() -> None:
    if LOG_PATH.exists():
        LOG_PATH.unlink()
    started = time.time()
    os.environ.setdefault("MPLBACKEND", "Agg")
    log(f"TrailNet {EXPERIMENT} starting (NO TRAIN; A PIXEL REPLAY + TILE AUDIT)")
    stage_inputs()
    ensure_deps()
    threshold = apply_checkpoint_threshold()
    sys.path.insert(0, str(PROJECT / "src"))
    import torch
    log(f"CUDA_COUNT {torch.cuda.device_count() if torch.cuda.is_available() else 0}")

    run_command([sys.executable, "scripts/prepare_data.py", "--config", CONFIG_NAME, "--all-areas"])
    tile_root = PROJECT / "data" / "tiles_t4e_a"
    index_hashes = {}
    for area, expected in EXPECTED_INDEX_SHA256.items():
        path = tile_root / area / "index.csv"
        actual = sha256(path)
        index_hashes[area] = {"sha256": actual, "expected_recovered_a_sha256": expected,
                              "matches": actual == expected}
    if not all(item["matches"] for item in index_hashes.values()):
        raise RuntimeError(f"A regenerated tile index mismatch: {index_hashes}")
    audit = write_audit_windows(tile_root)

    run_command([sys.executable, "scripts/predict.py", "--config", CONFIG_NAME, "--all-areas"])
    eval_output = run_command([sys.executable, "scripts/evaluate.py", "--config", CONFIG_NAME,
                               "--thresholds", "0.90"])
    result = {
        "run": EXPERIMENT, "status": "complete", "config": CONFIG_NAME,
        "no_train": True, "checkpoint_threshold": threshold, "pixel_eval_threshold": 0.90,
        "eval_tail": eval_output[-8000:], "regenerated_a_index_hashes": index_hashes,
        "audit_windows": audit, "elapsed_seconds": round(time.time() - started, 1),
        "source_datasets": [RASTER_SLUG, CODE_SLUG, CKPT_SLUG],
    }
    (WORK / "metrics.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    (WORK / "preflight.json").write_text(json.dumps({
        "checkpoint_sha256": sha256(PROJECT / "checkpoints_t4e2_a" / "trailnet_best.pt"),
        "checkpoint_handoff_sha256": "73b9f79ce0d20b9056a36c37cea8645f8a128eaea313672dae13c2f96ff16e4a",
        "checkpoint_matches_handoff": sha256(PROJECT / "checkpoints_t4e2_a" / "trailnet_best.pt") == "73b9f79ce0d20b9056a36c37cea8645f8a128eaea313672dae13c2f96ff16e4a",
        "regenerated_a_index_hashes": index_hashes,
        "audit_sample_tile_hashes": "audit/audit_samples.csv",
        "pixel_identity_verified_for_all_sampled_tiles": False,
    }, indent=2), encoding="utf-8")
    log("RUN_COMPLETE " + json.dumps({"elapsed_seconds": result["elapsed_seconds"],
                                      "threshold": threshold, "audit_windows": audit.get("sampled_windows")}))


if __name__ == "__main__":
    run_started_at = time.time()
    run_finished = threading.Event()

    def run_timeout_fallback() -> None:
        if run_finished.is_set():
            return
        write_callback({
            "status": "timed_out",
            "timed_fallback_triggered": True,
            "timeout_seconds": RUN_TIMEOUT_SECONDS,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(run_started_at)),
            "ended_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "elapsed_seconds": RUN_TIMEOUT_SECONDS,
            "reason": "top-level replay exceeded the timed fallback deadline",
        }, "trailnet-run-timeout")
        os._exit(124)

    watchdog = threading.Timer(RUN_TIMEOUT_SECONDS, run_timeout_fallback)
    watchdog.daemon = True
    watchdog.start()
    run_status = "failed"
    run_error = None
    try:
        main()
        run_status = "complete"
    except BaseException as exc:
        run_error = repr(exc)
        raise
    finally:
        run_finished.set()
        watchdog.cancel()
        write_callback({
            "status": run_status,
            "timed_fallback_triggered": False,
            "timeout_seconds": RUN_TIMEOUT_SECONDS,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(run_started_at)),
            "ended_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "elapsed_seconds": round(time.time() - run_started_at, 3),
            "error": run_error,
        }, "trailnet-run")
