"""Record the completed bounded attempt without further selection or tuning."""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
val=json.loads((OUT/'val-output/tta-val-results.json').read_text())
held=json.loads((OUT/'heldout-output/tta-heldout-results.json').read_text())
assert val['winner']=='TTA4' and held['heldout_candidate_count']==1 and held['threshold']==.9
assert hashlib.sha256((OUT/'val-output/tta-val-results.json').read_bytes()).hexdigest()==held['selection']['val_results_sha256']
assert held['status']=='complete' and not held['training_launched']
def row(name,values):
    return '| '+name+' | '+' | '.join(f'{v:.6f}' for v in values)+' |\n'
table='| Arm at 0.90 | VAL coverage F1 | VAL legacy F1 | Heldout coverage F1 | Heldout legacy F1 |\n|---|---:|---:|---:|---:|\n'
for a in ('A','TTA4'):
    table+=row(a,[val['scores'][a]['coverage']['0.9']['f1'],val['scores'][a]['legacy']['0.9']['f1'],held['scores'][a]['coverage']['f1'],held['scores'][a]['legacy']['f1']])
table+='| TTA8 | '+f"{val['scores']['TTA8']['coverage']['0.9']['f1']:.6f} | {val['scores']['TTA8']['legacy']['0.9']['f1']:.6f} | Not opened | Not opened |\n"
a=held['scores']['A']['coverage'];c=held['scores']['TTA4']['coverage'];v=val['scores']['TTA4']['coverage']['0.9'];va=val['scores']['A']['coverage']['0.9']
verdict='passed' if held['heldout_veto_passed'] else 'failed'
summary=f'''Completed one bounded attempt on `sol/attempt-20261107`, based on Luna 23bcaf2. Two predeclared inference arms, no new training, one selected-candidate heldout evaluation, no PC use. TTA4 won VAL and {verdict} the inherited heldout numerical veto. No more arms or heldout tuning follow.

{table}
TTA4 uses the unchanged A checkpoint and averages four aligned views; threshold0.90 was frozen before heldout. VAL coverage F1 improved by {v['f1']-va['f1']:.6f}; label precision changed {va['precision']:.6f}→{v['precision']:.6f}, and label recall {va['recall']:.6f}→{v['recall']:.6f}. Heldout coverage F1 changed by {c['f1']-a['f1']:.6f}; precision {a['precision']:.6f}→{c['precision']:.6f}, recall {a['recall']:.6f}→{c['recall']:.6f}. Exact TTA4 heldout TP/FP/FN: {c['tp']:,}/{c['fp']:,}/{c['fn']:,}. The veto requires coverage F1≥A, P≥.53, R≥.33 and legacy F1≥.4311. All counts refer to existing labels.

M1 fixed legacy comparison (candidate original-target F1 .342786526 vs A .342788397; original .2884 invalid), froze original sampler 993/752 with strict identity/hash checks, made coverage training fail without it, and corrected stale no-training statements. Nine original sampler changes are documented. M2 separated optical metadata validity from terrain validity; Teton's declared optical gap has no VAL membership, and six source-to-tensor traces match exactly. Frozen evaluation masks were unchanged.

This demonstrates a label-agreement inference gain; it does not verify real-trail accuracy. Independent trail/distractor truth and the trusted-reference guardrails remain unavailable, unmatched predictions are not proven false, and existing split footprints overlap. A remains the default. TTA4 is opt-in (`inference.geometric_ensemble: tta4`) and costs four forward passes per tile. The ready inference configuration is `runs/sol-20261107/config_tta4_inference.yaml`; full-AOI inference/vectorization was not run. All 107 CPU tests pass. The CPU environment, immutable Kaggle bundles, source hashes, configs, full threshold tables, per-area counts and logs are retained under `runs/sol-20261107/`.
'''
(OUT/'SUMMARY.md').write_text(summary)
Path('/workspace/trailnet-sol-final.md').write_text(summary)
p=ROOT/'docs/EXPERIMENT_LEDGER.md';p.write_text(p.read_text()+'\n## Completed Sol inference attempt, Nov 7 2026\n\n'+summary)
p=ROOT/'experiments/overnight-20261107/REPORT.md';s=p.read_text();s=s.replace('**Status (2026-11-07):**','**Status (2026-11-07; Sol attempt complete):** TTA4 won the predeclared VAL inference screen and '+verdict+' the one-shot heldout numerical veto; see the final Sol table below. ');p.write_text(s+'\n## Sol final bounded result (M4)\n\n'+summary)
result=dict(stage='M4_complete',val_result='val-output/tta-val-results.json',heldout_result='heldout-output/tta-heldout-results.json',winner='TTA4',threshold=.9,heldout_veto_passed=held['heldout_veto_passed'],new_training_runs=0,candidate_heldout_evaluations=1,default_unchanged=True,independent_truth_verified=False,tests_passed=107)
(OUT/'outcome.json').write_text(json.dumps(result,indent=2)+'\n')
p=Path('/workspace/trailnet-sol-attempt-state.json');state=json.loads(p.read_text());state.update(stage='M4_complete',outcome=result,next_action='Commit and push final code/results, verify clean branch, mark complete')
for kernel in state['kaggle_kernels']:kernel['status']='complete'
p.write_text(json.dumps(state,indent=2)+'\n')
print(summary)
