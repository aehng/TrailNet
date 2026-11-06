# TrailNet overnight comparison

Baseline **v4_local**: IoU **0.2214** / F1 **0.3626** @ t=0.60; frozen eval F1 **0.5000** @ t=0.90 (same checkpoint, no retrain).

Before T4-D the model was a **single shared ResNet34 encoder** on 7 channels (RGB, NIR, hillshade, slope, local relief). T4-D splits that into an RGB/NIR Unet fused with a dedicated terrain CNN.

XPU-C ReliefSnap was left alone (`experiments/xpuc_snap/`). No git push or merge.

## Pixel metrics (held-out 358 tiles)

| run | lever | best epoch | best val F1 | ckpt t | IoU@0.60 | F1@0.60 | IoU@0.90 | F1@0.90 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v4_local | shared 7ch | 20 | 0.0951 | 0.60 | 0.2214 | 0.3626 | 0.3333 | 0.5000 |
| T4-A2 | pos_boost 2 | 25 | 0.2264 | 0.90 | 0.2318 | 0.3764 | 0.3416 | 0.5092 |
| T4-B | unnamed footway drop + 5 m ignore | 7 | 0.2658 | 0.90 | 0.1026 | 0.1862 | 0.2153 | 0.3543 |
| **T4-D** | **dual terrain CNN + A2 settings** | **14** | **0.3394** | **0.90** | **0.2797** | **0.4371** | **0.3638** | **0.5335** |

T4-B `legacy_f1` @ 0.90 = 0.2732 (ignore-as-background). Below baseline, so T4-D did not mix in the footway filter.

## Vector noise (match_tolerance_m=3.0)

Precision/recall are **0.0** on A2, B, and D at 3 m. The matcher is too tight (or CRS-mismatched) to use as a win condition. Fragment/noise km still rank the runs:

| run | Corner pred_km | Corner noise_fragments | Corner noise_km | Kelly noise_km |
| --- | ---: | ---: | ---: | ---: |
| T4-A2 | 75.17 | 5564 | 30.39 | 5.39 |
| T4-B | 41.49 | 4153 | 22.64 | 9.04 |
| **T4-D** | **51.22** | **3337** | **18.91** | **2.66** |

T4-B’s lower Corner pred_km is partly the dropped ~197 km unnamed footway labels, not a fair pixel-set compare. T4-D uses the same labels as A2 and still cuts Corner noise km from 30.4 to 18.9 and Kelly from 5.39 to 2.66.

## Winner

**T4-D dual terrain CNN** (resnet34 RGB/NIR + resnet18 hillshade/slope/relief, A2 pos_boost 2, original v4 tiles).

It beats A2 and v4_local on F1@0.60 (0.437 vs 0.376 vs 0.363) and F1@0.90 (0.534 vs 0.509 vs 0.500), with the highest val F1 (0.339) and less short-fragment noise than A2. T4-B lost to baseline.

Kernels:

- A2 https://www.kaggle.com/code/elikearl/trailnet-t4-a2-posboost2
- B https://www.kaggle.com/code/elikearl/trailnet-t4-b-unnamedfootway-ignore
- D https://www.kaggle.com/code/elikearl/trailnet-t4-d-terraincnn

Per-run notes: `experiments/t4a2/`, `experiments/t4b/`, `experiments/t4d/`.

## Acceptance

(a) Labels: T4-D/A2 keep the stacked OSM/UGRC/USFS GPKGs. T4-B dropped unnamed OSM footway (Corner unnamed km=0, named 2.30 km) and still underperformed.
(b) Trail vs noise: T4-D raises pixel F1 and lowers A2-matched noise fragments. Vector precision at 3 m is still 0.0 for every run.
(c) Orchestration stayed on this PC / Kaggle; one GPU kernel at a time for T4-D.
(d) New configs/checkpoints only (`config_exp_t4d.yaml`, `checkpoints_t4d` on Kaggle). v4_local untouched.
(e) Metrics: T4-D is the most balanced of the three Kaggle runs; still FP-heavy at t=0.60 (FP/TP ≈ 2.3) and cleaner at t=0.90 (FP/TP ≈ 1.02).

## Next step

Keep dual-encoder + A2 pos_boost 2 as the working head. Raise vector match tolerance to 5–8 m (or fix CRS) so precision/recall stop reading as 0.0. XPU-C snap remains a separate local lever. Do not mix T4-B’s footway drop into the next train unless an ignore-only follow-up is isolated.
