# XPU-C ReliefSnap

## Snap QA (before train)

Snapped GPKGs written beside originals as `data/<aoi>/trails_snapped.gpkg`. Original v4 GPKGs untouched.

| AOI | n | median |shift| m | share \|shift\|<2 m |
| --- | ---: | ---: | ---: |
| kelly_canyon | 86 | 2.12 | 0.48 |
| teton_pass | 24 | 1.98 | 0.50 |
| corner_canyon | 2691 | 0.05 | 0.62 |
| park_city | 292 | 0.97 | 0.61 |
| snowbasin | 196 | 2.44 | 0.46 |

PNG review (Kelly / Teton / Corner): cyan tracks orange; Kelly/Teton corridors stay on hillside cuts, not the main drainage. Corner urban footways barely move.

## Train

- Config: `config_exp_xpuc.yaml`
- Tiles: `data/tiles_v4_snap`
- Checkpoints: `checkpoints_v4_snap` (v4_local immutable)
- BATCH_FIT requested=16 chosen=14 peak_gb=5.79 (allowed)
- AMP off, pos_boost 4, oversample 4.0x, thresholds to 0.90, patience 7 / 25 epochs
- Run 1 interrupted during val epoch 5 (session/process killed, no traceback). Best so far epoch 3 val_f1=0.1778 @ t=0.85. Copied to `checkpoints_v4_snap/trailnet_best_ep3.pt`. Restarted same config as `train_run2.log`.
