"""A-checkpoint VAL input ablation only. No optimizer or held-out evaluation."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

WORK = Path("/kaggle/working")
INPUT_ROOT = Path("/kaggle/input")
PROJECT = Path("/kaggle/temp/TrailNet")
CONFIG_NAME = "config_exp_t4e2_a_ignore5.yaml"
RASTER_SLUG = "trailnet-multiaoi-v4"
CODE_SLUG = "trailnet-t4e-code"
CKPT_SLUG = "trailnet-t4e2-a-ckpt"
AREA_NAMES = ["kelly_canyon", "corner_canyon", "park_city", "teton_pass", "snowbasin"]
RUN_TIMEOUT_SECONDS = 45 * 60
COMMAND_TIMEOUT_SECONDS = 30 * 60
FIXED_THRESHOLD = 0.90
EXPECTED_INDEX_SHA256 = {
    "corner_canyon": "277eac2dad6bdc33386eba744838a82fccea0eea0cf9e90f69a83faee7ed2411",
    "kelly_canyon": "60bb9f2c139cfaa47c6de7fda1121d325bfb4807ff1c056ad2dbb2d295a3f63d",
    "park_city": "b288e2b19654911aee4b81b5d51c1679affc3227a75b9964c0eb359a834a8ba2",
    "snowbasin": "250e16679bacb0a0d20d1160ad85e1724718a20576e3b8a19535eeb126732500",
    "teton_pass": "f04eff6be5a833ad08d2a882c3e7a88eaf16b53cef4a948eea69ab735b50984f",
}


def log(message: str) -> None:
    print(message, flush=True)
    with (WORK / "val-ablation.log").open("a", encoding="utf-8") as stream:
        stream.write(message + "\n")


def write_callback(stage: str, status: str, **extra) -> None:
    callback_dir = WORK / "callbacks"
    callback_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "stage": stage,
        "status": status,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **extra,
    }
    final = callback_dir / f"val-ablation-{stage}.callback.json"
    temporary = final.with_suffix(final.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2), encoding="utf-8")
    temporary.replace(final)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_command(args: list[str], timeout_seconds: int = COMMAND_TIMEOUT_SECONDS) -> str:
    started = time.monotonic()
    process = subprocess.Popen(
        args, cwd=PROJECT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, errors="replace",
    )
    try:
        output, _ = process.communicate(timeout=timeout_seconds)
        timed_out = False
    except subprocess.TimeoutExpired:
        process.kill()
        output, _ = process.communicate()
        timed_out = True
    status = "timed_out" if timed_out else ("complete" if process.returncode == 0 else "failed")
    for line in output.splitlines():
        log(line)
    write_callback(
        f"command-{Path(args[0]).stem}", status,
        arguments=args[1:], elapsed_seconds=round(time.monotonic() - started, 3),
        timeout_seconds=timeout_seconds, exit_code=process.returncode,
        timed_fallback_triggered=timed_out, output_tail=output[-4000:],
    )
    if timed_out:
        raise subprocess.TimeoutExpired(args, timeout_seconds, output=output)
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, args, output=output)
    return output


def find_mount(slug: str) -> Path:
    direct = INPUT_ROOT / slug
    if direct.is_dir():
        return direct
    matches = [path for path in INPUT_ROOT.rglob(slug) if path.is_dir()]
    if matches:
        return matches[0]
    visible = sorted(path.name for path in INPUT_ROOT.iterdir()) if INPUT_ROOT.is_dir() else []
    raise FileNotFoundError(f"dataset {slug} not mounted; visible={visible}")


def find_file(root: Path, name: str) -> Path | None:
    direct = root / name
    if direct.is_file():
        return direct
    matches = list(root.rglob(name))
    return matches[0] if matches else None


def stage_inputs() -> None:
    rasters, code_mount, ckpt_mount = (find_mount(slug) for slug in
                                       (RASTER_SLUG, CODE_SLUG, CKPT_SLUG))
    PROJECT.mkdir(parents=True, exist_ok=True)
    trees = [path for path in (code_mount, code_mount / "trailnet_code")
             if (path / "src").is_dir()]
    archive = find_file(code_mount, "trailnet_code.tar.gz")
    if trees:
        source_root = trees[0]
        for name in ("src", "scripts", "requirements.txt", "LICENSE", "README.md",
                     "label_overrides", "docs", "tests"):
            source = source_root / name
            if source.is_dir():
                shutil.copytree(source, PROJECT / name, dirs_exist_ok=True)
            elif source.is_file():
                shutil.copy2(source, PROJECT / name)
        yaml_root = code_mount if any(code_mount.glob("config*.yaml")) else source_root
        for yaml_path in yaml_root.glob("config*.yaml"):
            shutil.copy2(yaml_path, PROJECT / yaml_path.name)
    elif archive is not None:
        with tarfile.open(archive, "r:gz") as bundle:
            bundle.extractall(PROJECT)
    else:
        raise FileNotFoundError("no TrailNet source on code mount")

    postprocess = (PROJECT / "src" / "trailnet" / "postprocess.py").read_text(encoding="utf-8")
    if "novelty_output_path" not in postprocess or "novelty_filter=False" not in postprocess:
        raise RuntimeError("stale TrailNet source on code mount")
    checkpoint = find_file(ckpt_mount, "trailnet_best.pt")
    if checkpoint is None:
        raise FileNotFoundError("trailnet_best.pt missing on checkpoint mount")
    checkpoint_dir = PROJECT / "checkpoints_t4e2_a"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, checkpoint_dir / "trailnet_best.pt")

    for area in AREA_NAMES:
        area_dir = PROJECT / "data" / area
        area_dir.mkdir(parents=True, exist_ok=True)
        for stem, suffix in (("naip_rgbn", ".tif"), ("3dep_dtm", ".tif"), ("trails", ".gpkg")):
            source = find_file(rasters, f"{area}_{stem}{suffix}")
            if source is None:
                raise FileNotFoundError(f"missing {area}_{stem}{suffix}")
            shutil.copy2(source, area_dir / f"{stem}{suffix}")

    overrides = PROJECT / "label_overrides"
    if not overrides.is_dir():
        import zipfile
        for candidate in (code_mount / "label_overrides.zip", PROJECT / "label_overrides.zip"):
            if candidate.is_file():
                with zipfile.ZipFile(candidate) as bundle:
                    bundle.extractall(PROJECT)
                break
    if not overrides.is_dir():
        matches = [path for path in PROJECT.rglob("label_overrides") if path.is_dir()]
        if matches:
            overrides = matches[0]
    if not overrides.is_dir():
        raise FileNotFoundError("label_overrides missing")
    for area in AREA_NAMES:
        source = overrides / area / "trails.gpkg"
        if not source.is_file():
            raise FileNotFoundError(f"missing sidewalk-aware labels for {area}: {source}")
        shutil.copy2(source, PROJECT / "data" / area / "trails.gpkg")
    log("INPUTS_STAGED; no external file dependencies in kernel code")


def ensure_deps() -> None:
    import torch
    missing = [package for module, package in {
        "segmentation_models_pytorch": "segmentation-models-pytorch",
        "geopandas": "geopandas", "rasterio": "rasterio",
        "skimage": "scikit-image", "yaml": "pyyaml",
    }.items() if importlib.util.find_spec(module) is None]
    if not missing:
        return
    version, cuda = torch.__version__, torch.version.cuda
    constraint = WORK / "torch-constraints.txt"
    constraint.write_text(f"torch=={version}\n", encoding="utf-8")
    run_command([sys.executable, "-m", "pip", "install", "-q", "-c", str(constraint), *missing])
    probe = "import json,torch; print(json.dumps({'version':torch.__version__,'cuda':torch.version.cuda}))"
    installed = json.loads(run_command([sys.executable, "-c", probe]).splitlines()[-1])
    if installed["version"] != version or installed["cuda"] != cuda:
        raise RuntimeError("dependency install changed the existing PyTorch/CUDA build")


def apply_checkpoint_threshold() -> float:
    import torch
    import yaml
    checkpoint = torch.load(PROJECT / "checkpoints_t4e2_a" / "trailnet_best.pt",
                            map_location="cpu", weights_only=False)
    threshold = float(checkpoint.get("best_threshold", 0.9))
    config_path = PROJECT / CONFIG_NAME
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config.setdefault("postprocess", {})["threshold"] = threshold
    config.setdefault("inference", {})["threshold"] = threshold
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return threshold


def neutralize_terrain(images):
    """Zero only hillshade, slope, and local-relief channels (4:7)."""
    if images.ndim != 4 or images.shape[1] != 7:
        raise ValueError(f"expected NCHW seven-channel input, got {tuple(images.shape)}")
    result = images.clone()
    result[:, 4:7] = 0
    return result


def register_optical_zero_hook(model):
    """Zero RGB+NIR after RGB normalization, leaving terrain inputs untouched."""
    optical_branch = getattr(model, "optical_net", None)
    if optical_branch is None:
        optical_branch = getattr(model, "network", None)
    if optical_branch is None:
        raise TypeError("model exposes neither an optical_net nor a shared network")

    def zero_optical(_module, inputs):
        if not inputs or inputs[0].ndim != 4 or inputs[0].shape[1] < 4:
            raise ValueError("optical branch must receive an NCHW tensor with at least four channels")
        value = inputs[0].clone()
        value[:, :4] = 0
        return (value, *inputs[1:])

    return optical_branch.register_forward_pre_hook(zero_optical)


def run_bounded_command(args: list[str]) -> str:
    return run_command(args, timeout_seconds=COMMAND_TIMEOUT_SECONDS)


def sha256_rows(rows) -> str:
    payload = "\n".join(
        f"{row['area']}\t{row['image']}\t{row['mask']}\t{row.get('class_map', '')}"
        for row in sorted(rows, key=lambda item: (item["area"], item["image"]))
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _counts_by_threshold(model, dataset, rows, device, variant, batch_size, thresholds):
    import torch
    from trailnet.train import confusion_from_binary, metrics_from_counts

    totals = {threshold: [0, 0, 0] for threshold in thresholds}
    by_area = {}
    area_sizes = {}
    hook = register_optical_zero_hook(model) if variant == "optical_neutralized" else None
    try:
        model.eval()
        with torch.inference_mode():
            for start in range(0, len(rows), batch_size):
                chunk = rows[start:start + batch_size]
                items = [dataset[start + offset] for offset in range(len(chunk))]
                images = torch.stack([item[0] for item in items]).to(device)
                masks = torch.stack([item[1] for item in items])
                weights = torch.stack([item[2] for item in items])
                if variant == "terrain_neutralized":
                    images = neutralize_terrain(images)
                probabilities = torch.sigmoid(model(images)).cpu()
                truth = masks >= 0.5
                ignore = weights == 0
                for offset, row in enumerate(chunk):
                    area = row["area"]
                    area_sizes[area] = area_sizes.get(area, 0) + 1
                    by_area.setdefault(area, {t: [0, 0, 0] for t in thresholds})
                    for threshold in thresholds:
                        pred = probabilities[offset:offset + 1] >= threshold
                        target = truth[offset:offset + 1]
                        ignored = ignore[offset:offset + 1]
                        tp, fp, fn = confusion_from_binary(pred, target, ignore=ignored)
                        totals[threshold][0] += tp
                        totals[threshold][1] += fp
                        totals[threshold][2] += fn
                        by_area[area][threshold][0] += tp
                        by_area[area][threshold][1] += fp
                        by_area[area][threshold][2] += fn
    finally:
        if hook is not None:
            hook.remove()

    def summarize(counts):
        return {
            f"{threshold:.2f}": {
                "tp": values[0], "fp": values[1], "fn": values[2],
                "iou": metrics_from_counts(*values)[0],
                "f1": metrics_from_counts(*values)[1],
            }
            for threshold, values in counts.items()
        }

    return {
        "thresholds": summarize(totals),
        "by_area": {
            area: {"tile_count": area_sizes[area], "thresholds": summarize(counts)}
            for area, counts in sorted(by_area.items())
        },
    }


def main() -> None:
    started = time.monotonic()
    log("START A checkpoint VAL ablation; no training; held-out split excluded")
    write_callback("start", "running", timeout_seconds=RUN_TIMEOUT_SECONDS)

    stage_inputs()
    ensure_deps()
    threshold_from_checkpoint = apply_checkpoint_threshold()
    import sys
    sys.path.insert(0, str(PROJECT / "src"))

    import torch
    from trailnet.config import load_config, resolve_path
    from trailnet.data.dataset import TrailTileDataset, read_indexes, spatial_split
    from trailnet.models.unet import build_model

    config = load_config(PROJECT / CONFIG_NAME)
    run_command([sys.executable, "scripts/prepare_data.py", "--config", CONFIG_NAME,
                 "--all-areas"])
    write_callback("prepare-data", "complete")
    thresholds = tuple(sorted({float(value) for value in config["training"]["val_thresholds"]}
                              | {FIXED_THRESHOLD}))
    if not thresholds or min(thresholds) <= 0 or max(thresholds) >= 1:
        raise ValueError(f"invalid VAL threshold grid: {thresholds}")
    index_paths = [resolve_path(config, config["paths"]["tile_dir"]) / area["name"] / "index.csv"
                   for area in config["areas"]]
    tile_hashes = {}
    for area, expected in EXPECTED_INDEX_SHA256.items():
        path = resolve_path(config, config["paths"]["tile_dir"]) / area / "index.csv"
        actual = sha256(path)
        tile_hashes[area] = {"sha256": actual, "expected_sha256": expected,
                             "matches": actual == expected}
    if not all(row["matches"] for row in tile_hashes.values()):
        raise RuntimeError(f"A index hash mismatch: {tile_hashes}")

    rows = read_indexes(index_paths)
    split = spatial_split(
        rows,
        block_size_m=config["training"]["block_size_m"],
        val_fraction=config["training"]["val_fraction"],
        test_fraction=config["training"]["test_fraction"],
        seed=config.get("seed", 42),
    )
    val_rows = split["val"]
    if len(val_rows) != 172:
        raise RuntimeError(f"expected frozen A VAL tile count 172, got {len(val_rows)}")
    if len(split["test"]) != 358:
        raise RuntimeError(f"expected frozen A held-out tile count 358, got {len(split['test'])}")
    val_hash = sha256_rows(val_rows)
    write_callback("preflight", "complete", val_tile_count=len(val_rows),
                   heldout_tile_count=len(split["test"]), val_membership_sha256=val_hash,
                   index_hashes=tile_hashes)
    log(f"PREFLIGHT_OK val=172 heldout=358 val_sha256={val_hash}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Kaggle CUDA device is unavailable; refusing a slow/unexpected CPU ablation")
    device_name = torch.cuda.get_device_name(0)
    model = build_model(config["model"], encoder_weights=None).to(device)
    checkpoint_path = resolve_path(config, config["training"]["checkpoint_path"])
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    dataset = TrailTileDataset(val_rows, pixel_weights=config.get("loss", {}).get("pixel_weights", {}))
    outputs = {}
    batch_size = 2
    for variant in ("full", "terrain_neutralized", "optical_neutralized"):
        write_callback(variant, "running", batch_size=batch_size)
        log(f"ABLATION_START {variant} batch_size={batch_size}")
        outputs[variant] = _counts_by_threshold(
            model, dataset, val_rows, device, variant, batch_size, thresholds)
        write_callback(variant, "complete", val_tile_count=len(val_rows))
        log(f"ABLATION_COMPLETE {variant}")

    reference = outputs["full"]["thresholds"][f"{FIXED_THRESHOLD:.2f}"]
    deltas = {}
    for variant in ("terrain_neutralized", "optical_neutralized"):
        candidate = outputs[variant]["thresholds"][f"{FIXED_THRESHOLD:.2f}"]
        deltas[variant] = {
            key: round(candidate[key] - reference[key], 6)
            for key in ("iou", "f1")
        }
        deltas[variant]["tp_delta"] = candidate["tp"] - reference["tp"]
        deltas[variant]["fp_delta"] = candidate["fp"] - reference["fp"]
        deltas[variant]["fn_delta"] = candidate["fn"] - reference["fn"]

    result = {
        "status": "complete",
        "experiment": "A checkpoint VAL modality sensitivity",
        "no_train": True,
        "heldout_scored": False,
        "label_truth_note": "All pixel metrics measure agreement with existing A labels; no independent human trail/distractor truth is implied.",
        "config": CONFIG_NAME,
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_threshold": threshold_from_checkpoint,
        "thresholds": list(thresholds),
        "fixed_comparison_threshold": FIXED_THRESHOLD,
        "split_seed": config.get("seed", 42),
        "block_size_m": config["training"]["block_size_m"],
        "val_tile_count": len(val_rows),
        "val_membership_sha256": val_hash,
        "val_tile_ids_by_area": {
            area: sorted(row["image"] for row in val_rows if row["area"] == area)
            for area in sorted({row["area"] for row in val_rows})
        },
        "regenerated_a_index_hashes": tile_hashes,
        "device": device_name,
        "batch_size": batch_size,
        "ignore_mask": "same class_map/weight-derived mask for all three passes; weights == 0 are ignored",
        "ablation_semantics": {
            "full": "original seven-channel model input",
            "terrain_neutralized": "zero channels 4:7 in the original [0,1] input before forward",
            "optical_neutralized": "zero RGB+NIR channels 0:4 at the model optical-network pre-hook, after RGB normalization; terrain branch unchanged",
        },
        "results": outputs,
        "fixed_0_90_deltas_vs_full": deltas,
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }
    output = WORK / "val-modality-ablation.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_callback("run", "complete", output=str(output),
                   val_tile_count=len(val_rows), elapsed_seconds=result["elapsed_seconds"])
    log("RUN_COMPLETE " + json.dumps({"output": str(output), "elapsed_seconds": result["elapsed_seconds"]}))


if __name__ == "__main__":
    started_at = time.time()
    finished = threading.Event()

    def timeout_fallback() -> None:
        if finished.is_set():
            return
        write_callback("run", "timed_out", timed_fallback_triggered=True,
                       timeout_seconds=RUN_TIMEOUT_SECONDS,
                       elapsed_seconds=round(time.time() - started_at, 2))
        os._exit(124)

    timer = threading.Timer(RUN_TIMEOUT_SECONDS, timeout_fallback)
    timer.daemon = True
    timer.start()
    status = "failed"
    error = None
    try:
        main()
        status = "complete"
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        finished.set()
        timer.cancel()
        write_callback("run-final", status, timed_fallback_triggered=False,
                       timeout_seconds=RUN_TIMEOUT_SECONDS, error=error)
