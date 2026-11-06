> Current handoff (Nov 6, 2026): see [CLI_HANDOFF.md](CLI_HANDOFF.md) and [EXPERIMENT_LEDGER.md](EXPERIMENT_LEDGER.md). Older run notes below are historical, not the current decision protocol.

# Checkpoint: trailnet-overnight-20261106

Branch snapshot of verified overnight training/label work after the local v4 collapse fix and Kaggle A2/B/D runs.

## Included
- Loss/device/train fixes in `src/trailnet/train.py` (focal BCE + batch Tversky, XPU-aware device, threshold sweep to 0.90, geographic spatial split).
- Dual-encoder model path in `src/trailnet/models/unet.py` used by T4-D.
- Label/tile/download stack under `src/trailnet/data/` (Python only).
- Named experiment configs: `config_exp_t4a.yaml`, `config_exp_t4b.yaml`, `config_exp_t4d.yaml`, `config_exp_xpuc.yaml`, plus local/kaggle/smoke configs.
- Concise results under `docs/overnight/` and `experiments/*/RESULTS.md` (+ metrics JSON, snap QA JSON only).

## Omitted on purpose
- Raw NAIP/DEM/GPKG/tiles, checkpoints (`*.pt`), prediction outputs, `_agent-runs/` pulls, snap QA PNGs, `*.bak` backups.
- New sidewalk parser and vector-matcher tolerance fixes (see `docs/FOLLOWUPS.md`).

## Box test run (2026-11-06)

- Environment: `/home/box/.venvs/trailnet-smoke` (Python 3.13, torch 2.15.0+cpu).
- Command: `PYTHONPATH=src python -m pytest tests/ -v`
- Result: **48 passed**.
- No Intel XPU on the box. `test_select_device_uses_xpu_before_cpu` uses mocks / availability checks and passed on CPU; it does **not** exercise a real Arc XPU path.
