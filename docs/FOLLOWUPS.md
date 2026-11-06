# Follow-ups (not in this checkpoint)

Documented findings from the 2026-11-05 overnight runs. **Not implemented here.**

## Keep as baseline
- **T4-D dual encoder** (`config_exp_t4d.yaml`): RGB/NIR ResNet34 + terrain ResNet18 (hillshade/slope/local relief), `pos_boost` 2. Best held-out pixel F1@0.90 **0.5335** / IoU **0.3638** among overnight Kaggle runs (see `docs/overnight/comparison.md`).
- Threshold sweeps belong on **validation** only. Do **not** choose decision thresholds on the held-out test set.

## Labels / sidewalks
- **Code in this branch:** `parse_overpass_response` keeps `footway=*`; explicit `sidewalk`/`crossing` become `class_group=sidewalk` and rasterize as **ignore (class 4)** with an uncertain buffer (default 2 m). Unnamed footway stays path. Corner Canyon is not dropped.
- **Still required operationally:** re-fetch/rebuild GPKGs+tiles so existing Corner caches pick up preserved tags (old GPKGs lack `footway`).
- See `docs/LABEL_MISALIGNMENT_OPTIONS.md` for GPS-offset training options (ignore bands, bounded alignment, trusted-label weights).

## XPU-C ReliefSnap
- Snap QA + local train ran (`experiments/xpuc_snap/`). Best **val** F1 **0.2059 @ 0.90** (epoch 8). Training ends without running held-out eval.
- **No same-GT held-out comparison** vs T4-D. Snap can pull labels onto drainages/terrain edges; terrain response ≠ proof of true tread.
- Optional later: eval snap checkpoint and T4-D on one fixed GT, or park snap until a drainage QA gate.

## Vector metrics
- Overnight P/R at 3 m was **0.0**; synthetic checks show coincident lines score 1.0, while a **4 m parallel offset scores 0.0 at 3 m and 1.0 at 5 m** — consistent with measured 4–7 m label offsets, not a fragment-count bug.
- Matcher now requires metric CRS (rejects metre-as-4326), uses length-weighted symmetric P/R, and exposes a **tolerance sweep** `{1,3,5,8,10}` without selecting a test winner.

## Repo hygiene
- Push code/configs/docs only. Never commit rasters, GPKGs, checkpoints, `_agent-runs/`, or secrets (Overpass token stays in env `TRAILNET_OVERPASS_TOKEN`).

## T4E (2026-11-06)
- Both ignore5/soft5 Kaggle kernels **ERROR** pre-train (`label_overrides` not copied in directory staging). No metrics. See `docs/experiments/t4e2_followup.md` for the fixed relaunch pair.
