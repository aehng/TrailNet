# T4-D TerrainCNN (dual encoder)

Kernel: https://www.kaggle.com/code/elikearl/trailnet-t4-d-terraincnn  
Logs: `_agent-runs/overnight/t4d-complete/kernel.logs.txt`  
Lever: dedicated terrain CNN (resnet18 on hillshade/slope/relief) fused with RGB/NIR resnet34, plus T4-A2 pos_boost 2 / original v4 tiles.

`DUAL_ENCODER_OK optical=ResNetEncoder terrain=ResNetEncoder`

| metric | value |
| --- | ---: |
| BATCH_FIT | requested=12 chosen=12, n_gpus=2 |
| best_epoch | 14 (early stop 21) |
| best_validation_f1 | 0.3394 @ t=0.90 |
| held-out tiles | 358 |
| held-out IoU @ 0.90 | 0.3638 |
| held-out F1 @ 0.90 | 0.5335 |
| held-out IoU @ 0.60 | 0.2797 |
| held-out F1 @ 0.60 | 0.4371 |
| elapsed | 2960.3 s |

Vector (match_tolerance_m=3.0; precision/recall 0.0):

| area | pred_km | label_km | noise_fragments | noise_km |
| --- | ---: | ---: | ---: | ---: |
| kelly_canyon | 11.77 | 67.39 | 409 | 2.66 |
| corner_canyon | 51.22 | 328.93 | 3337 | 18.91 |
| park_city | 13.06 | 90.44 | 587 | 3.84 |
| teton_pass | 2.03 | 25.96 | 152 | 0.93 |
| snowbasin | 10.91 | 133.03 | 572 | 3.45 |
