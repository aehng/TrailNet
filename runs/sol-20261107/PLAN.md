# Predeclared bounded recognition attempt

M1 complete at 504036e; 104 CPU tests pass; pushed branch sol/attempt-20261107. No training.

M2: scan all five original optical and terrain sources separately using GDAL's explicit nodata/mask metadata; trace two deterministic original VAL windows per represented area to actual tensors. Original optical metadata tags band4 alpha although recovered NAIP item metadata identifies NIR; numerical tiling reads band4 as NIR and does not apply alpha. Nodata shadows alpha according to GDAL. Four AOIs have no declared in-grid optical invalid pixels; Teton has21,846,520/62,401,437 invalid pixels in every band. All source terrain rasters declare -9999 and have zero invalid cells. All six traced optical/terrain tensors exactly match reconstruction. Reprojected terrain nonfinite counts are reported separately. Source metadata does not prove independent trail truth or registration. Teton has no VAL membership; keep frozen bounds-only mask unchanged. No image adjudication is needed for the geometric ensemble hypothesis.

M3 hypothesis: the trained model is not exactly rotation/reflection equivariant, despite geometric augmentation. Averaging aligned outputs may improve segmentation agreement without changing supervision or fitting a new model. This is a testable inference change, not an established real-trail improvement.

Exactly two candidate arms, in one Kaggle kernel, no training:
1. TTA4: identity, horizontal reflection, 180-degree rotation, vertical reflection; arithmetic mean of inverse-aligned probabilities.
2. TTA8: all four90-degree rotations, each with/without horizontal reflection; arithmetic mean of inverse-aligned probabilities.

Keep all seven channels together under each transform, without recomputing terrain derivatives. A checkpoint SHA25673b9f79ce0d20b9056a36c37cea8645f8a128eaea313672dae13c2f96ff16e4a. All original A source rasters, label overrides, indexes and split seed42 remain unchanged. Original target/weight tensors are immutable for both legacy and coverage scoring. Sampler is unused at inference and remains frozen to993/752 for future controlled training.

Primary: pooled coverage-aware VAL F1 over172 original VAL IDs, same14 thresholds .25:.05:.90, frozen bounds-only mask SHA2566350928231908df0e7092003486735ecb697721a1d6904b026ba4ed7be9bfc52. A .3536541461658466 at .90; candidate must reach .3736541461658466. Always show legacy TP/FP/FN and fixed .90 alongside selected-threshold results. Ties choose fewer transforms, then higher threshold. Require exact positive-denominator parity (legacy177079, coverage164884) and baseline replay F1 within .0001; fail closed on input/hash/membership/mask mismatch. Results measure agreement with existing labels; no independent-truth or weather claim.

CPU smoke completed before GPU: real checkpoint64x64 inputs, both arms finite and correctly shaped; tests verify inverse alignment/channel preservation. Kaggle budget: one kernel maximum, hard45-minute run timeout, at most2GPU-hours expected including setup, at least3 GPU-hours reserve. Current quota must be checked immediately before launch. No second training or additional arm follows failures.

M4: only if either arm passes VAL, freeze winner+threshold first and score selected candidate and A once on original358 heldout tiles. No other candidate heldout evaluation or threshold tuning. Report historical legacy and same frozen-mask coverage metrics. Without independent references, call a gain label agreement, never independently verified real-trail accuracy. Otherwise record both negatives, retain A and complete this attempt.
