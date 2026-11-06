# Vector metric novelty-filter fix (2026-11-06)

## Bug
`vectorize_area` always passed label GPKG into `vectorize_probability`, which kept only
lines farther than `postprocess.dedup_tolerance_m` (3 m) from labels. Those novelty-only
GeoJSONs were scored by `vector_metrics_for_area`, so length-weighted P/R at 3 m was ~0
by construction. T4E2 A/B showed P/R=0 at 1/3 m and small nonzero from 5 m — matching a
4 m residual offset after 3 m dedup.

## Fix
- `{area}_trails.geojson` = unfiltered skeleton lines (for metrics)
- `{area}_novelty.geojson` = optional novelty product
- `evaluation.vector_report_tolerance_m` default 5 m for headline (not dedup tol)
- Optional clip to probability raster footprint before length sums

## Re-eval
Re-vectorize from saved `*_probability.tif` then re-run vector metrics. Do not re-score
old novelty GeoJSON. Pixel held-out metrics are unaffected.
