# TrailNet — direct Grok/CLI handoff (Nov 6, 2026, MT)

**Start with [docs/CLI_HANDOFF.md](docs/CLI_HANDOFF.md) and [docs/EXPERIMENT_LEDGER.md](docs/EXPERIMENT_LEDGER.md).** These supersede older checkpoint/planning prose; historical experiments remain preserved.

- GitHub: https://github.com/aehng/TrailNet — branch `checkpoint/trailnet-overnight-20261106`, PR https://github.com/aehng/TrailNet/pull/1 is OPEN against master, not merged. No GitHub checks configured/reported; local CPU tests are not CI.
- Current model: **T4E2-A ignore5**. B-soft5 lost on validation. A's frozen validation threshold is **0.90**.
- Corrected geographic VAL vector P/R at5m: **0.4709 / 0.5522**, four blocks across three AOIs. Kelly/Teton have no val blocks (null, not perfect scores).
- No H1/clDice or H2/bootstrap training occurred. R<0.50 launch trigger did not hold. That is NOT proof the product goal is met.
- All Kaggle jobs COMPLETE at handoff; no background training. Autonomous TrailNet routine paused by parent. No future intermediary work scheduled.
- The dated PC export is a source snapshot plus Git bundle, **not** an overwrite/update of the old dirty working checkout. Do not reset, rebase, switch or replace that checkout blindly.

Ready-to-paste Grok prompt:
> Read START_HERE.md, docs/CLI_HANDOFF.md and docs/EXPERIMENT_LEDGER.md. First inspect Git status and preserve all local work. Use the provided corrected A artifacts and frozen split/threshold; audit representative visible tread, canopy and missing-label cases. Distinguish label agreement from real trail truth. Do not train, auto-pseudolabel, snap to terrain, merge PR1, or overwrite the original dirty checkout without my explicit next instruction.
