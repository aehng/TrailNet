# TrailNet overnight status

- **State:** done. A2, B, and T4-D Kaggle COMPLETE. XPU-C left alone. comparison.md written.
- **Started:** 2026-11-05T23:55:20.9494176-07:00
- **Frozen eval:** F1@0.60=0.3626, F1@0.90=0.5000; gate **T4-A2 PosBoost2**
- **Encoder (code):** v4/A2/B use a **single shared ResNet encoder** (`smp.Unet` resnet34) on all 7 channels. T4-D is `DualTrailUNet`: RGB/NIR resnet34 fused with a dedicated terrain resnet18 (hillshade, slope, local relief).
- **Experiments:** T4-A2 PosBoost2, T4-B UnnamedFootway-Ignore, T4-D TerrainCNN, XPU-C ReliefSnap (untouched)
- **Kernel slugs:**
  - T4-A2: https://www.kaggle.com/code/elikearl/trailnet-t4-a2-posboost2 (COMPLETE)
  - T4-B: https://www.kaggle.com/code/elikearl/trailnet-t4-b-unnamedfootway-ignore (COMPLETE)
  - T4-D: https://www.kaggle.com/code/elikearl/trailnet-t4-d-terraincnn (COMPLETE v1)
- **T4-A2:** BATCH_FIT 12; best_epoch=25 val_f1=0.2264 @ t=0.90; held-out F1@0.90=0.5092 IoU=0.3416; F1@0.60=0.3764. Logs: `_agent-runs/overnight/t4a-metrics/kernel.logs.txt`
- **T4-B:** BATCH_FIT 12; best_epoch=7 val_f1=0.2658; held-out F1@0.90=0.3543 IoU=0.2153; F1@0.60=0.1862 legacy_f1@0.90=0.2732 (below baseline). Pull: `_agent-runs/overnight/t4b-complete`
- **T4-D:** DUAL_ENCODER_OK; BATCH_FIT 12; early stop 21; best_epoch=14 val_f1=0.3394 @ t=0.90; held-out F1@0.90=0.5335 IoU=0.3638; F1@0.60=0.4371. Logs: `_agent-runs/overnight/t4d-complete/kernel.logs.txt`
- **Winner:** T4-D (see `_agent-runs/overnight/comparison.md`)
- **XPU-C:** leave `experiments/xpuc_snap/` alone
- **Baseline:** v4_local IoU 0.2214 / F1 0.3626 @ t=0.60; frozen F1 0.50 @ t=0.90
- **Git:** no push, no merge
