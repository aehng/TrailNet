# Direct Grok / CLI handoff

## Git truth and safe PC restoration
GitHub account `aehng`; remote `https://github.com/aehng/TrailNet.git`. PR1 open, branch `checkpoint/trailnet-overnight-20261106`, base master. No merge, no force-push. Latest source code implementation b25e04e; handoff doc commit is recorded in export SOURCE_COMMIT.txt.

Original PC checkout `C:\Users\elijk\Documents\Projects\TrailNet`: dirty master at97d3e5c with many tracked edits and untracked experiments/data/code. **Preserved untouched.** Dated export `TrailNet-handoff-20261106-1650` is a NEW snapshot, not the original Git working folder. `source/` has no .git; `TrailNet-history.bundle` has complete committed Git history/branch, but not uncommitted work in old checkout.

User/Grok can run these manually in PowerShell (NOT executed by handoff):
```powershell
# Read-only inspect old checkout before doing anything else:
git -C "$HOME\Documents\Projects\TrailNet" status --short
# The export is already backed by its ZIP. For further changes to old checkout,
# additionally back up its tracked diff AND untracked code deliberately; bundle does not cover them.
# Restore committed history into a NEW folder (stop if it already exists):
$handoff = "$HOME\Documents\TrailNet-handoff-20261106-1650"
git clone --branch checkpoint/trailnet-overnight-20261106 "$handoff\TrailNet-history.bundle" "$HOME\Documents\TrailNet-cli-20261106"
Set-Location "$HOME\Documents\TrailNet-cli-20261106"
git remote set-url origin https://github.com/aehng/TrailNet.git
git fetch origin
git status -sb
# Do NOT reset/rebase/switch the old dirty repo. Reconcile its unique work manually later.
```

## Verified prerequisites and validation
PC: Git2.55.0, Python3.11.9, KaggleCLI2.2.4, Grok1.0.77, gh2.98.0. `gh auth status` confirms `aehng`; credentials not included in export. Existing PC XPU environment was NOT replaced. Do not reinstall torch from pinned requirements over a working Intel-XPU build without checking compatibility.

Box CPU validation: 37 tests passed: vector_eval, postprocess_novelty_metrics, soft_ignore_loss, labels_snap, v4. Syntax/diff checks passed. GitHub has no checks in PR status rollup; this is local validation, not a claim of CI/XPU coverage.

From new source/Git checkout with suitable Python environment:
```powershell
# Existing PC environment may already have these; inspect before installing.
python -m venv .venv
# CPU/new env only; pinned torch2.4.1 is not a promise of Intel XPU support:
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest tests/test_vector_eval.py tests/test_postprocess_novelty_metrics.py tests/test_soft_ignore_loss.py tests/test_labels_snap.py tests/test_v4.py -q
```
Run script names below exist. Commands require input paths/artifacts staged first:
```powershell
python scripts/prepare_data.py --config config_exp_t4e2_a_ignore5.yaml --all-areas
python scripts/predict.py --config config_exp_t4e2_a_ignore5.yaml --checkpoint checkpoints_t4e2_a/trailnet_best.pt --all-areas
# Set postprocess.threshold=.90 from checkpoint.best_threshold BEFORE vectorizing.
# Static YAML still contains historical .60; train writes it back, but inference alone does not.
python scripts/vectorize.py --config config_exp_t4e2_a_ignore5.yaml --all-areas
python scripts/evaluate.py --config config_exp_t4e2_a_ignore5.yaml --checkpoint checkpoints_t4e2_a/trailnet_best.pt --thresholds 0.90
```
`evaluate.py` is heldout pixel evaluation. For scoped vectors call `trailnet.vector_eval.vector_metrics_for_area(config, area)`; scopes `val_blocks/test_blocks` and per-tolerance rows. See `experiments/t4e2_a_eval_vb/run.py` for aggregate printing. No train.py command is needed for handoff or re-evaluation. Don't launch either archived eval runner blind; mounts/code assertions need current code dataset.

## Artifacts / staging
Export `assets/` contains sidewalk-aware label_overrides, recovered A unfiltered GeoJSON + tile indexes, corrected val_block_table, A/B metrics, final eval-vb log, 4 PNG visuals. A checkpoint transfer is separate: native transfer of its two parts failed (computer temporarily unreachable). Do NOT assume checkpoint is present. Retrieve from the checkpoint dataset below, or transfer the box file later; verify SHA256 in CHECKPOINT_SHA256.txt. No huge imagery/DTM included. No B checkpoint/probability raster/visual comparison supplied. No actual A probability rasters local; regenerate inference only if needed.

A checkpoint must be copied to `checkpoints_t4e2_a/trailnet_best.pt` in chosen new checkout. Input paths are `data/<area>/naip_rgbn.tif`, `3dep_dtm.tif`, `trails.gpkg`; use sidewalk-aware overrides for trails, NOT stale original sidewalk-unaware GPKGs. Tile paths separate; rebuild if moving to a new environment. Data AOIs: Kelly, Corner, Park City, Teton, Snowbasin. Either intentionally copy imagery/DTM from original PC data (don't modify originals) or retrieve Kaggle dataset:
```powershell
kaggle datasets download -d elikearl/trailnet-multiaoi-v4 -p INPUTS --unzip
```
Dataset filenames are `<area>_naip_rgbn.tif`, `<area>_3dep_dtm.tif`; map/copy to configured area paths. `label_overrides/<area>/trails.gpkg` supersedes original dataset trails. Set input absolute paths in YOUR local config if avoiding duplicate rasters; keep output/tile/checkpoint paths in the new work folder.

Sources:
- https://www.kaggle.com/datasets/elikearl/trailnet-multiaoi-v4
- https://www.kaggle.com/datasets/elikearl/trailnet-t4e-code (mutable dataset; Git snapshot is code truth)
- https://www.kaggle.com/datasets/elikearl/trailnet-t4e2-a-ckpt
- https://www.kaggle.com/code/elikearl/trailnet-t4e2-a-ignore5
- https://www.kaggle.com/code/elikearl/trailnet-t4e2-b-soft5
- https://www.kaggle.com/code/elikearl/trailnet-t4e2-a-eval
- https://www.kaggle.com/code/elikearl/trailnet-t4e2-a-eval-vb
All completed as of Nov6 4:50PM MT; no new models/experiments launched for handoff.

## Operating rules
Use VAL only for thresholds/model decisions. Frozen5m report convention, sweep3/5/8/10m. Ignore sidewalks/crossing and uncertain buffer, retain unnamed paths. Report empty scopes as null. Unfiltered vectors for P/R, novelty separate. A positive GPS core may sit in foliage; no proof of alignment from terrain response. Missing labels mean unmatched predictions are not automatically false trails.

Read START_HERE and ledger first in Grok. Original historic docs/results stay preserved, not rewritten as current truth. The prior autonomous TrailNet routine is paused; owned watcher has terminated, no continued intermediary work.
