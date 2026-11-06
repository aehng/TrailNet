# T4-B UnnamedFootway-Ignore

Kernel: https://www.kaggle.com/code/elikearl/trailnet-t4-b-unnamedfootway-ignore  
Pull: `_agent-runs/overnight/t4b-complete`  
Lever: drop unnamed OSM footway + 5 m ignore ring. Corner unnamed footway km=0 after filter; named footway 2.30 km.

| metric | value |
| --- | ---: |
| BATCH_FIT | requested=12 chosen=12, n_gpus=2 |
| best_epoch | 7 |
| best_validation_f1 | 0.2658 |
| held-out tiles | 358 |
| held-out IoU @ 0.90 | 0.2153 |
| held-out F1 @ 0.90 | 0.3543 |
| held-out IoU @ 0.60 | 0.1026 |
| held-out F1 @ 0.60 | 0.1862 |
| legacy_f1 @ 0.90 | 0.2732 |
| elapsed | 1851.5 s |

Vector (match_tolerance_m=3.0; precision/recall 0.0):

| area | pred_km | label_km | noise_fragments | noise_km |
| --- | ---: | ---: | ---: | ---: |
| kelly_canyon | 20.80 | 67.39 | 1568 | 9.04 |
| corner_canyon | 41.49 | 132.16 | 4153 | 22.64 |
| park_city | 11.81 | 89.11 | 1092 | 5.81 |
| teton_pass | 4.18 | 25.96 | 540 | 2.86 |
| snowbasin | 24.66 | 132.10 | 2126 | 12.27 |
