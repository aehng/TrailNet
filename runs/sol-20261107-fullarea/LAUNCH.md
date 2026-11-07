# Full-area tests launched

Both version1 kernels were confirmed **RUNNING** using the Kaggle CLI. Agent stops here; a separate script will collect outputs. No completion polling, output download, or new training.

| Test | URL | Hypothesis and controlled change |
|---|---|---|
| Compare | https://www.kaggle.com/code/elikearl/trailnet-sol-fullarea-compare-20261107 | A vs TTA4 on all five full AOIs, identical native tiling and unfiltered vectorization; ensemble is the only change. VAL support: vector F1 +.02 and recall no worse than A−.02. |
| Phase | https://www.kaggle.com/code/elikearl/trailnet-sol-fullarea-phase-20261107 | TTA4 native vs192px shifted tile context/ownership grid, all five AOIs. VAL support: vector P/R each within.02 and pooled declared-valid VAL binary Jaccard≥.60. No shifted-grid selection. |

Controls: unchanged A checkpoint and hash-verified original optical/DTM/sidewalk-aware labels/indexes; threshold0.90; seed42/block2500; tile512/overlap128; terrain gain1.85; vector simplify1m; 3/5/8/10m matching, report5m. Sidewalk/crossing and uncertainty excluded from both sides of scoped metrics. Frozen bounds-only historical evaluation unchanged. Export optical and terrain declared-valid masks separately, no zero/black heuristic; their intersection is explicitly supplemental for grid consistency. VAL alone supports decisions. Test-block results are descriptive only (heldout already used); original split footprints overlap. Full AOIs include train geography. No independent truth, weather, registration or false-trail claim.

Device plan: CLI advertises NvidiaTeslaT4 as T4×2; server metadata and successful launches accept that accelerator. **Actual allocated device counts/names have not been observed by this agent.** Each runtime logs torch.cuda.device_count() and every name, then spawns one process per visible device. Ten area/arm jobs are round-robin sharded across all devices, with per-device assignment/progress/completed tile counts/timing. No unused visible GPU is silently accepted. Batch4; TTA views sequential. GPU inference precedes CPU vectorization.

Quota before launch:27.36 GPU-hours remaining. Requested NvidiaTeslaT4 explicitly; both platform timeouts10800seconds. User lifted3h reserve; no paid compute and no PC. Outputs include runtime/cost estimate ($0 user compute cost; allocated-GPU-hour estimate labeled non-authoritative), dependency freeze and actual hardware logs.

Exports under `/kaggle/working/runs/sol-20261107-fullarea/{compare|phase}/`: full probability TIFFs, unfiltered `*_trails.geojson`, separate `*_novelty.geojson`, modality-validity TIFFs, nine fixed3×3 geographic quantile overlays per AOI/arm, all scoped vector tables, consistency metrics, frozen config/input hashes, per-device logs, hardware.json and results.json. Overlays are geography-selected before seeing outputs, with unverified reference labels clearly identified.

Reproducibility: source/config/test commit `bcfecc4`; shared payload SHA256 `874191841ce7f7f3e20f7b3775679035b72641ec51fb7784eb7084205bb22fa0`. Both generated self-contained kernel scripts, metadata, source manifests, launch/status logs, detailed PLAN.md and **119 passing CPU tests** are committed on `sol/attempt-20261107`. No merge or force-push.
