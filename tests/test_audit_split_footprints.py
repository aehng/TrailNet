import importlib.util
from pathlib import Path
import csv
import hashlib


AUDIT_SCRIPT = Path(__file__).parents[1] / "scripts" / "make_audit_windows.py"
KAGGLE_REPLAY_SCRIPT = (
    Path(__file__).parents[1]
    / "experiments"
    / "overnight-20261107"
    / "kernels"
    / "a-replay"
    / "run.py"
)


def load_audit_module():
    spec = importlib.util.spec_from_file_location("trailnet_audit_windows", AUDIT_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pairwise_tile_overlap_uses_union_area_and_index_centers():
    audit = load_audit_module()
    rows = [
        {"area": "park_city", "image": "train_a", "x": 5.0, "y": 5.0, "split": "train"},
        {"area": "park_city", "image": "train_duplicate", "x": 5.0, "y": 5.0, "split": "train"},
        {"area": "park_city", "image": "val_a", "x": 10.0, "y": 5.0, "split": "val"},
    ]

    report = audit.pairwise_tile_overlap_report(rows, tile_width_m=10.0)

    assert report["park_city"]["train_val_intersection_m2"] == 50.0


def test_audit_window_must_stay_inside_its_assigned_split_blocks():
    audit = load_audit_module()
    tile_rows = [
        {"area": "park_city", "image": "train", "x": 5.0, "y": 5.0, "split": "train"},
        {"area": "park_city", "image": "val", "x": 10.0, "y": 5.0, "split": "val"},
    ]
    exclusive = audit.split_exclusive_footprints(tile_rows, tile_width_m=10.0)
    safe_window = {"area": "park_city", "x": 2.0, "y": 5.0}
    overlap_window = {"area": "park_city", "x": 4.5, "y": 5.0}

    assert audit.window_is_split_exclusive(safe_window, "train", crop_m=2.0, exclusive=exclusive)
    assert not audit.window_is_split_exclusive(overlap_window, "train", crop_m=2.0, exclusive=exclusive)


def test_split_preflight_writes_membership_ids_hashes_and_overlap(tmp_path):
    audit = load_audit_module()
    index_root = tmp_path / "indexes"
    area_dir = index_root / "park_city"
    area_dir.mkdir(parents=True)
    index_path = area_dir / "index.csv"
    rows = None
    for block_x in range(-8, 8):
        left = {"area": "park_city", "image": "left.tif", "x": (block_x + 1) * 2500 - 1,
                "y": 1250.0, "crs": "EPSG:32612"}
        right = {"area": "park_city", "image": "right.tif", "x": (block_x + 1) * 2500 + 1,
                 "y": 1250.0, "crs": "EPSG:32612"}
        left_split = next(name for name, bucket in audit.frozen_spatial_split([left]).items() if bucket)
        right_split = next(name for name, bucket in audit.frozen_spatial_split([right]).items() if bucket)
        if {left_split, right_split} == {"train", "val"}:
            rows = [left, right]
            break
    assert rows is not None
    with index_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["area", "image", "mask", "class_map", "x", "y", "crs"])
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "mask": "mask.tif", "class_map": "class.tif"})

    report = audit.build_split_preflight(index_root, tmp_path / "out")

    assert report["tile_counts"]["total"] == 2
    assert report["cross_split_tile_footprints"]["park_city"]["train_val_intersection_m2"] > 0
    ids = sorted(f"park_city/{row['image']}" for row in rows)
    expected_train_ids = [tile_id for tile_id in ids if tile_id in report["tile_ids_by_split"]["train"]]
    expected_hash = hashlib.sha256("\n".join(expected_train_ids).encode()).hexdigest()
    assert report["membership_sha256"]["train"] == expected_hash
    assert (tmp_path / "out" / "split_membership.csv").is_file()


def test_kaggle_replay_excludes_audit_crops_outside_split_exclusive_footprint():
    spec = importlib.util.spec_from_file_location("trailnet_kaggle_replay", KAGGLE_REPLAY_SCRIPT)
    assert spec is not None and spec.loader is not None
    replay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(replay)
    tile_rows = [
        {"area": "park_city", "image": "train", "x": 5.0, "y": 5.0, "split": "train"},
        {"area": "park_city", "image": "val", "x": 10.0, "y": 5.0, "split": "val"},
    ]
    exclusive = replay._split_exclusive_footprints(tile_rows, tile_width_m=10.0)

    assert replay._window_is_split_exclusive(
        {"area": "park_city", "x": 2.0, "y": 5.0}, "train", 2.0, exclusive
    )
    assert not replay._window_is_split_exclusive(
        {"area": "park_city", "x": 4.5, "y": 5.0}, "train", 2.0, exclusive
    )
