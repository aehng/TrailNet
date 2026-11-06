# Follow-ups (not in this checkpoint)

Documented findings from the 2026-11-05 overnight runs. **Not implemented here.**

## Keep as baseline
- **T4-D dual encoder** (`config_exp_t4d.yaml`): RGB/NIR ResNet34 + terrain ResNet18 (hillshade/slope/local relief), `pos_boost` 2. Best held-out pixel F1@0.90 **0.5335** / IoU **0.3638** among overnight Kaggle runs (see `docs/overnight/comparison.md`).
- Threshold sweeps belong on **validation** only. Do **not** choose decision thresholds on the held-out test set.

## Labels / sidewalks
- OSM Overpass cache still carries `footway=sidewalk` and `footway=crossing`, but `parse_overpass_response` currently stores only `trail_type=<highway>` and **drops** those tags.
- Prefer preserving sidewalk/crossing as an **ignore mask or separate class** at parse time. Unnamed OSM `footway` is **not** equivalent to sidewalk.
- Do **not** drop Corner Canyon wholesale: UGRC singletrack and named mountain trails remain useful. Overnight T4-B (unnamed-footway drop) underperformed baseline.

## XPU-C ReliefSnap
- Snap QA + local train ran (`experiments/xpuc_snap/`). Best **val** F1 **0.2059 @ 0.90** (epoch 8). Training ends without running held-out eval.
- **No same-GT held-out comparison** vs T4-D. Snap can pull labels onto drainages/terrain edges; terrain response ≠ proof of true tread.
- Optional later: eval snap checkpoint and T4-D on one fixed GT, or park snap until a drainage QA gate.

## Vector metrics
- Overnight vector precision/recall at `match_tolerance_m=3.0` read **0.0** for A2/B/D. Matcher tolerance and/or CRS need a dedicated eval fix (e.g. 5–8 m) before using P/R as a win condition. That is **eval plumbing**, not a model upgrade.

## Repo hygiene
- Push code/configs/docs only. Never commit rasters, GPKGs, checkpoints, `_agent-runs/`, or secrets (Overpass token stays in env `TRAILNET_OVERPASS_TOKEN`).
