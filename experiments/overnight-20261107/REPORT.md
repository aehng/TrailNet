# TrailNet overnight audit and experiment report

**Status:** Early stop after three Kaggle staging failures across two kernels. The permitted A replay staging retry is spent. No A inference or tile-pixel audit completed, and no training arm launched.

## What was completed

- Preserved the original checkout. Work was staged in the isolated `codex/overnight-failure-tests` worktree.
- Verified the A checkpoint SHA-256 against the handoff manifest: `73b9f79ce0d20b9056a36c37cea8645f8a128eaea313672dae13c2f96ff16e4a`.
- Compared 15 local source files (five NAIP rasters, five 3DEP rasters, and five trail vectors) with the Kaggle raster dataset's published SHA-256 manifest. All 15 matched.
- Compared all five recovered A tile-index CSV hashes with the local `tiles_v4` index hashes. All matched. Tile raster and class-map pixel identities have not been verified, so the local contact-sheet sample is explicitly provisional.
- Built a per-area imagery/terrain inventory, reconstructed NAIP STAC item dates using the retained AOI extents and current latest-year selection rule, and created 45 no-overlay RGB, NIR/R/G, and relief windows. The 48-window target was short by six because Kelly Canyon and Teton Pass have no VAL tiles in the frozen 2,500 m, seed-42 split.
- Ran the local Arc 130V batch-12 fit smoke check. It failed for memory; no local full training run was scheduled.
- Submitted an isolated A evaluation replay to Kaggle, using a new private kernel slug. Both replay attempts failed before model inference: the first had stale vector-evaluation imports; the one permitted retry reached dependency setup, then failed when its installer reloaded the already-imported PyTorch module and duplicated Triton's `TORCH_LIBRARY` registration. A separate audit-only kernel also failed at startup because Kaggle did not expose its auxiliary modules on the import path. Neither retry nor audit kernel generated tiles, audit pixels, or model scores. No further push was made because the A replay's staging-retry allowance is spent.
- Reproduced the exact PyTorch reload failure locally with the installed `2.14.0+xpu` build. Fixed the staged replay package to verify the post-install PyTorch/CUDA build from a fresh subprocess instead of reloading the live module. Child commands now wait on process completion without polling, write callback JSON on completion/failure, and are killed with a timeout callback after 60 minutes; a 90-minute top-level watchdog records a timed fallback and exits if the replay itself gets stuck. This package fix is locally tested but was not submitted because the plan's single staging retry is exhausted.
- Screened all seven requested next-site areas against NAIP STAC, USGS TNM 1 m DEM product metadata, and OSM line lengths. Each remains `review_later` because independent truth samples and area-specific permissions have not been verified.

## Data and label evidence

The retained raster dataset has 0.6 m NAIP and 1 m 3DEP source grids. Its summary and raster tags do not retain the exact NAIP source-item IDs or 3DEP acquisition dates. The NAIP dates below were reconstructed from STAC using the current fetcher behavior, which mosaics items from the latest acquisition year; they are not embedded source tags.

Park City has four reconstructed 2021 NAIP items: two dated September 24 and two dated November 21. The contact sheets show extensive snow-like or very bright cover across many Park City samples, but the screen did not trace snow pixels or corridor visibility. It also includes exposed ground, canopy, dark shadow, and invalid edge crops. Snow, bright pavement/rock, cloud, and roofs must remain separate review categories.

The strongest follow-up imagery candidate is the complete four-tile NAIP set from 2018, dated September 9 and September 27. It is a later, same-season full-area alternative to the mixed 2021 mosaic. A complete 2016 August set is a fallback. These dates suggest better snow visibility, but they do not prove the pixels are snow-free; review the affected trail corridors directly and exclude trails built or rerouted after the candidate acquisition. Item identity remains reconstructed from the STAC index because the source mosaic did not retain item IDs.

In the 45-window provisional sample, 17 windows have less than 99% valid imagery, 8 have less than 50%, and 5 have less than 10%. These are image-coverage cases, not weather failures. Use each window's valid mask in all future weather denominators.

An existing exploratory relief-offset audit found Park City's median absolute relief-response offset was 4.2 m over 500 samples; 46.2% exceeded 5 m and 24.2% exceeded 10 m. This measures alignment to a terrain response, not whether each line is a true trail. Corner Canyon's inventory had 2,199 unnamed OSM footways totaling about 196.8 km out of 2,229 footway features; naming alone is not a valid inclusion rule. Neither finding justifies automatic relabeling or dropping unnamed trails.

The local contact-sheet screen was not fully blinded: the separate sample key was inspected before image review. Treat its visual observations as exploratory. Independent adjudication and an independent dated trail source are still needed before label edits or weather-arm eligibility.

For Park City reference data, UGRC's public Utah Trails and Pathways service is useful for source attribution and trail/use classes, but it is not independent for the existing Park City labels: the retained AOI summary says 55 UGRC features were already included in those source vectors. Park City's local Mountain Trails Foundation (MTF) guide/map is the strongest next candidate; the city points visitors to it as its most accurate online trail resource, and MTF builds and maintains local trails. I did not verify downloadable geometries, their lineage against OSM/UGRC, historical alignment to the 2018/2021 imagery, or ML reuse permissions. Treat the map as context until MTF supplies dated GIS/as-built or maintenance records and permission. Park City GIS is another candidate for city-maintained assets, but coverage across county, resort and federal land is unknown.

## Next-site screen

All seven candidate areas have NAIP STAC candidates and 1 m 3DEP metadata extents. The estimated TNM metadata-footprint coverage is about 100% of each 5 km screen box except Copper Harbor at about 98.2%; this does not prove raster pixel validity. The metadata records no snow, shadow, cloud, or leaf-state evidence. Verified trail and distractor sample counts are null, and permissions remain unknown. `next-sites.json` records each site as `review_later` with the evidence gap.

## Experiment decision

No preregistered training arm qualified. The current evidence does not separate false positive labels, omitted tread, terrain/imagery misalignment, or obscured snow trails well enough to pick labelmask, labeladd, hard-negative, or weather-dropout training. The Arc batch-12 smoke failure and Kaggle staging failures prevent local/GPU training and A parity replay. Preserve A; do not interpret the failed staging attempt as a model result.

For Park City, request dated MTF GIS/as-built or trail-maintenance records, provenance, positional uncertainty, construction/reroute dates and ML permission; use only features not already present in A labels for independent evaluation. In parallel, review aligned, verified snow-free imagery over affected corridors. Review complete September 2018 coverage first, then the August 2016 set if 2018 visibility is poor. Preserve the 2021 late-fall imagery as its own snow/visibility evaluation slice. Keep canopy and deep-shadow slices separate, using leaf-off imagery or independent reference evidence where available.

## Recommended next action

Fix the dependency installer in a separately reviewed package, then request a new run authorization before submitting another Kaggle kernel because the plan's one staging retry has been consumed. After a successful replay, regenerate the exact A tiles and review the 45-window sample against independent trail references with image-only visibility notes recorded before labels or predictions are revealed. This resolves the remaining tile-pixel identity and audit-blinding gaps before any training arm is chosen.

## Verification

The focused callback/dependency tests passed: 4 tests; focused screen/download tests passed: 9 tests. The full suite passed: 83 tests, with 146 library/deprecation warnings. The first sandboxed full-suite invocation could not load the installed Intel XPU DLL; rerunning with normal Windows access completed successfully. Kaggle failure diagnoses above come from remote kernel logs; the PyTorch reload root cause was also reproduced locally.

---

## Summary

- **Inputs:** Source rasters/vectors, A tile indexes, and checkpoint are hash-verified; local tile pixels still need parity verification.
- **Labels:** Existing audits show measurable alignment uncertainty and many unnamed paths, but neither establishes true label errors.
- **Imagery:** Park City mixes September and late-November 2021 NAIP; snow-like cover and invalid tile edges warrant separate visibility/validity slices.
- **Sites:** All seven are `review_later`; metadata is promising, but independent sample counts and permissions are unknown.
- **Experiments:** No training arm is qualified. The A replay failed in dependency setup before inference; a corrected callback-enabled package is prepared locally but not rerun under the exhausted retry allowance. Local batch 12 did not fit on the Arc 130V.
