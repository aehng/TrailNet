# Experiment ledger — authoritative handoff, Nov 6 2026 (MT)

| Run | Best val F1 | Epoch | Threshold (val) | Heldout pixel F1 / IoU | Interpretation |
|---|---:|---:|---:|---|---|
| T4E2-A ignore5 | .3427 | 7 | .90 | .4311 / .2748 | Current pair winner |
| T4E2-B soft5 | .3176 | 10 | .90 | .3764 / .2319 | Lost on same pair protocol |
| Historical T4-D | see experiments/t4d | — | historical | F1 .5335 | Different label GT; NOT a comparable baseline |
| XPU-C ReliefSnap | .2059 | historical | historical | no same-GT heldout | Terrain response does not prove real tread alignment |

A/B same seed42, geographic blocks2500m, val/test fractions .15, heldout358 tiles; pos993/empty752. Both have hard positive core; A ignores5m outside core, B softens a5m band. Sidewalk/crossing class4 ignored, unnamed paths retained. Hard cores can still label foliage when GPS is shifted. Neither experiment solves label registration by itself.

## Evaluation repair and final results
Original vectorization removed all lines within3m of references (novelty filtering) BEFORE matching. P/R=0 at3m was therefore by construction, not model accuracy. Unfiltered trails and novelty products are now separate. Full-AOI numbers mix train geography and cannot select models.

Corrected **local sidewalk-aware** A block-scoped metrics (b25e04e):

| Tolerance | VAL precision | VAL recall |
|---:|---:|---:|
|3m|.4614|.4875|
|5m|.4709|.5522|
|8m|.4745|.5923|
|10m|.4767|.6125|

5m is a provisional engineering reporting convention, NOT optimized on heldout. Four VAL blocks /172 center-assigned tiles across Corner Canyon, Park City, Snowbasin. Kelly/Teton have no VAL blocks -> null. At5m: pred9.630km, reference13.468km. Test report-only P/R=.6177/.5445 (pred33.518km, ref58.603km); no decisions from test.

Scope: exact spatial_split hash/seed/fractions, disjoint block geometry intersected with original inference footprint; split lines at boundaries; remove explicit sidewalk/crossing cores PLUS configured uncertain buffer from BOTH prediction and reference. Pixel metrics use full overlapping center-assigned tiles; geometric block metrics differ near boundaries, with deltas recorded in table.

### Reconcile completed Kaggle eval-vb
`elikearl/trailnet-t4e2-a-eval-vb` COMPLETE, ~1196.8s, NO TRAIN. It mounted pre-b25e04e code. VAL5m=.470882/.552156 agrees (these val blocks have effectively no excluded sidewalks). Its TEST5m=.628693/.202853 includes sidewalk reference lengths: ref164.597km, versus58.603km in corrected local table. Do NOT combine those different denominators or call that a model regression. Prefer local corrected table for the sidewalk-excluded protocol.

Old real-raster roundtrip recall~.974-.982 indicates affine/centerline agreement, not tread truth. Old reported precision used the wrong directional numerator. b25e04e fixes it with synthetic tests; corrected directional precision has NOT been rerun on actual saved probability rasters. Do not quote old precision as audited.

## Decisions and unresolved science
Grok4.7 medium proposed clDice(alpha .2) / soft-bootstrap Dice(beta .3) only if corrected VAL R5<.50. Actual .5522 does not trigger it; **no new training**. Potential losses remain hypotheses, not implemented/proven improvements. Bootstrap can confirm bad foliage labels; clDice can enforce connectivity in wrong corridors.

Next human-directed work: representative visible-tread/canopy/snow/missing-label audit, trusted validation subset with no test tuning, assess false discoveries vs missing reference labels, then pre-register a controlled trial if warranted. No auto-pseudolabel cascade or ungated GPS-to-terrain snapping.

Literature methods (not TrailNet outcomes): Henry ICPR2021 noise-aware/bootstrapped Dice https://doi.org/10.1109/ICPR48806.2021.9412054 ; Shit CVPR2021 clDice https://arxiv.org/abs/2003.07311 ; weak OSM pretraining https://arxiv.org/abs/2309.07823 .

## Sol bounded attempt — M1 review corrections, Nov 7 2026

Existing coverage-ignore candidate rescored on original A VAL supervision: legacy F1 .342786526 (TP75,170/FP186,333/FN101,909; positives177,079); coverage F1 .352590487 (same TP/FP, FN89,714; positives164,884). CPU A reproduces original legacy .342788397 and coverage .353654146 exactly. Prior candidate .2884 is invalid due to altered targets. Historical candidate GPU coverage .352618809 and this CPU replay differ slightly; both fail gate .373654146. Nine sampler eligibility changes confounded the diagnostic. Original TRAIN eligibility993/752 is now frozen by tile identity/hash. No new training or heldout scoring in M1. Full evidence: runs/sol-20261107/.
