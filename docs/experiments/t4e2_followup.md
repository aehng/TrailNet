# T4E2 follow-up: ignore-5m vs soft-5m (staging fix)

Written: 2026-11-06 (America/Denver). Authorized one follow-up pair after T4E.

## T4E result (terminal, no metrics)

Kernels `elikearl/trailnet-t4e-a-ignore5` and `elikearl/trailnet-t4e-b-soft5` both reached **ERROR** before `prepare_data` / train.

- Failure: `FileNotFoundError: label_overrides missing after unzip`
- Cause: Kaggle `stage_inputs()` copied `src`/`scripts`/yaml from the code dataset directory but **not** `label_overrides/`, so sidewalk-aware GPKGs never replaced raster-mount trails.
- Comparability vs T4-D: **unavailable** (no held-out or vector metrics). Do not rank T4E.

## Hypotheses (unchanged science; packaging fixed)

Shared with T4-D: `architecture: dual_terrain`, `pos_boost: 2`, seed 42, geo splits, sidewalk-aware GPKGs (class sidewalk → ignore), epochs 16 / patience 5.

| Slug | Config | Isolated lever |
|------|--------|----------------|
| `elikearl/trailnet-t4e2-a-ignore5` | `config_exp_t4e2_a_ignore5.yaml` | `ignore_buffer_m=5.0`, `soft_label_band_m=0` — hard GPS positives; 5 m annulus ignored |
| `elikearl/trailnet-t4e2-b-soft5` | `config_exp_t4e2_b_soft5.yaml` | `soft_label_band_m=5.0`, `ignore_buffer_m=0` — hard GPS positives; 5 m soft decaying band |

Both still keep a **hard GPS positive core**, so neither directly fixes foliage wrongly labeled positive on the offset line. Goal is to see which annulus treatment improves precision/fragmentation without inventing a drainage snap.

## Config diffs vs T4-D

- Checkpoint dirs: `checkpoints_t4e2_a` / `checkpoints_t4e2_b` (do not overwrite T4-D).
- A: `ignore_buffer_m` 0→5; B: `soft_label_band_m` 0→5.
- Sidewalk uncertain buffer remains 2 m (fully ignore sidewalks).

## Decision criteria (same GT)

- Same geographic split and fixed evaluation GT as T4-D.
- Validation-only checkpoint / threshold selection; never tune threshold on held-out test.
- Confirm sidewalk exclusion and that loss/metrics honor ignore.
- Vector eval: real metric CRS; length-weighted P/R at 1/3/5/8/10 m; **do not** widen matching tolerance to claim a win.
- Report unavailable metrics honestly; unmatched lines are not proven noise.
- Compare mountain vs urban precision/recall and fragmentation with coverage limits.
- Do **not** rank incompatible F1 across label regimes.

## Kernel staging fix

Directory staging now also copies `label_overrides` (plus `docs`/`tests`). Error text clarifies directory-or-unzip failure.
