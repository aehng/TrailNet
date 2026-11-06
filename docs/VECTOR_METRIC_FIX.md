> Current handoff (Nov 6, 2026): see [CLI_HANDOFF.md](CLI_HANDOFF.md) and [EXPERIMENT_LEDGER.md](EXPERIMENT_LEDGER.md). Older run notes below are historical, not the current decision protocol.

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

## Geographic val/test scope
`scoped_vector_metrics` keeps the unfiltered lines and does not apply the novelty
filter. Val and test use the same block hash as `spatial_split` (seed, `block_size_m`,
fractions, tile-centre floor). Lines are split on those block borders before matching,
then intersected with the raster footprint. Overlapping tile windows are not the match
scope; the length difference versus tile assignment is reported as
`boundary_vs_tile_assignment`.

Explicit `sidewalk` / `crossing` centerlines and `tiling.sidewalk_uncertain_buffer_m`
(default follows `tiles.py`: the key itself, else `ignore_buffer_m` if positive, else
2 m) are removed from both prediction and reference lengths. That removal is ignore,
not a background false positive. Unnamed paths stay. An empty split returns null
precision and recall.

## Re-raster round trip
Precision and recall use different numerators. Recall is skeleton pixels covered by
the vector burn dilated 1 px, over skeleton pixels. Precision is vector-burn pixels
covered by the skeleton dilated 1 px, over vector-burn pixels. Dividing the
skeleton-side intersection by the undilated vector pixel count can exceed 1 and is
not a precision. The check validates a projected metre affine and is not trail truth.
