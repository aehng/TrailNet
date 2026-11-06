# T4-A2 PosBoost2

Kernel: https://www.kaggle.com/code/elikearl/trailnet-t4-a2-posboost2  
Logs: `_agent-runs/overnight/t4a-metrics/kernel.logs.txt`  
Lever: `pos_boost=2`, `pos_oversample=2`, val thresholds through 0.90, original v4 tiles.

| metric | value |
| --- | ---: |
| BATCH_FIT | requested=12 chosen=12, n_gpus=2 |
| best_epoch | 25 |
| best_validation_f1 | 0.2264 @ t=0.90 |
| held-out tiles | 358 |
| held-out IoU @ 0.90 | 0.3416 |
| held-out F1 @ 0.90 | 0.5092 |
| held-out IoU @ 0.60 | 0.2318 |
| held-out F1 @ 0.60 | 0.3764 |
| elapsed | 2660.2 s |

Vector (match_tolerance_m=3.0; precision/recall 0.0 — matching too tight / CRS):

| area | pred_km | label_km | noise_fragments | noise_km |
| --- | ---: | ---: | ---: | ---: |
| kelly_canyon | 15.63 | 67.39 | 1084 | 5.39 |
| corner_canyon | 75.17 | 328.93 | 5564 | 30.39 |
| park_city | 10.49 | 90.44 | 700 | 3.83 |
| teton_pass | 4.15 | 25.96 | 475 | 2.65 |
| snowbasin | 27.26 | 133.03 | 2444 | 12.80 |
