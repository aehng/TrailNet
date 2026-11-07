# TrailNet overnight audit and data quality report

**Status (2026-11-07):** A-only replay completed and matches the published held-out F1. No model training ran. The imagery screen exposed substantial zero-filled response in the 2018 candidate and unresolved band/mask semantics in the 2021 NAIP source. No independent label audit has been completed, so there is no evidence yet to justify a training change.

## Completed and verified

- Preserved the original checkout; work is in the isolated `codex/overnight-failure-tests` worktree.
- Verified checkpoint SHA-256 `73b9f79ce0d20b9056a36c37cea8645f8a128eaea313672dae13c2f96ff16e4a` against the handoff. The five regenerated A index hashes match their recovered A indexes.
- Corrected the staged Teton index hash typo. The final evaluation-only Kaggle replay completed in 854 seconds with callback status complete. At threshold 0.90 it scored 358 held-out tiles: IoU 0.2748, F1 0.4311, TP 301,931, FP 325,818 and FN 471,116. Park City held-out IoU/F1 were 0.2133/0.3516. These reproduce published pixel agreement; the cross-split footprint overlap below limits spatial-independence claims. No training kernel ran.
- The replay regenerated 45 audit windows. Their IDs/centers do not fully match the earlier local provisional crops: center matches are 35/45 overall (18/24 Park City), and image/class-map pixel hashes match 0/45. Treat prior local contact-sheet interpretations as provisional. The municipal comparison below instead used 24 exact Park City windows from the replay manifest.
- Ran a bounded CPU retrieval of those same 24 windows from current 2021 NAIP and the Park City/Summit County PC2018 ImageServer. The four imagery-only sheets are in `park-city-imagery-final-output/park-city-imagery/`; no labels or predictions appear.
- Screened Fruita 18 Road, Sedona Bell Rock, Crested Butte, Buffalo Creek, Tiger Mountain, Copper Harbor and Kingdom Trails. All remain `review_later` in `next-sites.json`; product metadata and OSM lengths are discovery evidence, not independent trail truth.

## What the imagery shows and what remains uncertain

The current Park City mosaic is reconstructed from four NAIP items dated September 24 and November 21, 2021. In the imagery-only sheets, many windows contain large bright or blue-white regions consistent with snow or bright exposed surfaces. Their condition is not independently classified: do not label them snow from brightness alone. Keep snow, mixed cover, bare ground, cloud, deep shadow, canopy, bright bare rock/pavement, invalid and unknown as separate review tags. The original sample imagery and its canonical pair are linked below for blinded review.

The municipal PC2018 service reports 0.5 m four-band U8 imagery flown June 24 and July 1, 2018; those acquisition dates make it a useful visibility candidate, not proof of snow-free tread or favorable illumination. In the exact 24-window retrieval, 13 candidate RGB chips were entirely zero and the mean RGB zero-filled fraction was 57.1854%. Yet the returned masks report 100% validity. The service therefore has partial usable coverage, and its masks cannot be accepted without pixel/footprint validation. Remaining nonzero windows look clearer and less snow-covered than several 2021 windows, but that is only an imagery-only visual screen, not a trail-visibility or label comparison.

There is a separate input-contract concern: Rasterio reports the current NAIP mosaic's color interpretation as `red,green,blue,alpha` and `nodata=0`, while TrailNet reads four channels as optical input. The fourth channel has continuous values, not a binary alpha pattern. Its NIR role is plausible from NAIP conventions, but the exact reconstructed STAC item band metadata and the mosaic writer's color tags must be checked before asserting it. The Planetary Computer documentation describes NAIP as red/green/blue/NIR, while the local raster tag labels the fourth channel alpha ([Planetary Computer NAIP example](https://planetarycomputer.microsoft.com/docs/quickstarts/storage/)). Resolve the discrepancy with the specific STAC items and compare raster masks/alpha/zeros against actual footprint edges. Never use numeric zero alone as nodata or trust a full mask when returned RGB is black.

PC2018's public service metadata confirms the flight dates, 0.5 m pixels and four U8 bands, but does not name band roles or establish machine-learning reuse rights ([PC2018Imagery service](https://gis.parkcity.org/server/rest/services/PC2018Imagery/ImageServer)). All 24 sample centers falling inside its published extent did not ensure valid image pixels. The municipal source's use terms remain unknown.

## Labels and split limits

No independent, dated geometry has been adjudicated against imagery. UGRC is not independent for the existing Park City labels: the source inventory already includes 55 UGRC features. Park City points to Mountain Trails Foundation as its most accurate online local trail resource, but a dated downloadable GIS/as-built/maintenance layer, its overlap with existing vectors, positional uncertainty, trail construction/reroute dates and ML permission remain unverified. A public municipal WalkingRoutes layer's published extent is east of this sample AOI; its date and MTB-singletrack semantics are also unknown. Consequently there are zero verified target-trail or distractor samples and no trusted VAL reference set.

Recovered A split membership is 1,745 train, 172 VAL and 358 held-out. The geometric pairwise cross-split footprint overlap sums to about 2.34 km², including 0.401 km² between Park City VAL and held-out. These are bounding-footprint areas; nodata-valid overlap was not established. Historical and replay F1 values measure agreement with the existing labels and do not prove independent spatial generalization.

Earlier audits remain hypotheses, not label corrections: the provisional imagery/relief check saw Park City relief-response offset median absolute 4.2 m over 500 samples (46.2% above 5 m; 24.2% above 10 m); the Corner Canyon inventory had 2,199 unnamed OSM footways totaling about 196.8 km of 2,229 features. Neither finding alone proves a trail label is wrong or right.

## Training decision and next work

All five candidate arms remain ineligible in `launch_decisions.json`. The A replay gate is met for published held-out pixel parity, but the following data gates remain open:

1. Resolve current NAIP and PC2018 band identity, source masks and nodata behavior; freeze one source-to-training-tensor example.
2. Blinded-review valid canonical image pairs, scoring snow, bare/mixed, shadow, cloud, canopy, saturation and tread visibility separately; measure stable-landmark registration. Remove zero-return candidate areas from denominators.
3. Obtain dated independent Park City trail records and documented reuse permission; verify source lineage, tread type, positional accuracy and construction/reroute dates. Freeze trusted VAL geometry before exposing predictions.
4. Then run A VAL modality diagnostics and only reopen a training arm if the preregistered thresholds are met. Preserve snow/shadow/canopy as separate evaluation slices; do not drop snow-affected samples or let a date/brightness heuristic determine labels.
5. Keep the seven new sites at `review_later` until independent target/distractor sample counts, actual 3DEP raster coverage, seasonal image evidence and permissions are established. Do not add Bentonville or train on new areas under this plan.

**Recommended next action:** resolve the band/mask contract, then complete the blinded review and request dated independent Park City trail references. Do not launch training yet.

## Artifacts

- Plan and handoff: `docs/plans/2026-11-07-overnight-failure-tests.md`
- Replay metrics and audit crops: `kaggle-a-replay-final-output/metrics.json` and `audit/audit_samples.csv`
- Canonical paired-image metadata and blinded sheets: `park-city-imagery-final-output/park-city-imagery/paired-imagery-screen.json` and `paired-contact-sheet-01.png` through `paired-contact-sheet-04.png`
- Decisions: `results.json`, `diagnostics.json`, `launch_decisions.json`, `park-city-followup-screen.json`, `next-sites.json`

## Verification record

Fresh verification completed: `pytest -q` passed 87 tests with 146 warnings in the normal Windows context. All experiment JSON parsed; the imagery inventory has five rows; the canonical comparison has 24 windows, 13 allzero candidate windows, and matching hashes for all five A indexes. `git diff --check` passed before commit.
