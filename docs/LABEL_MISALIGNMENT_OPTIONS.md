# Three grounded options for GPS / OSM label misalignment

Rendering bugs (Leaflet declaring UTM metres as EPSG:4326) are separate from **true**
label–imagery offset. Pixel overlays on 3DEP still show typical |offset| ≈ 4–7 m on
these AOIs, so a hard 1 m path buffer can paint the visible tread as background and
foliage/soil as trail. Prefer approaches that do **not** blind-snap into drainages.

## Option A — Ignore bands / soft distance targets (recommended next train lever)
**Idea:** Keep vector lines where they are. Dilate a small **ignore ring** around each
positive (and optionally use a distance-transform soft target decaying from the line)
so slightly shifted GPS does not create hard false negatives on the real tread or hard
false positives on adjacent foliage.

**Evidence:** Henry et al., *Aerial Road Segmentation in the Presence of Topological
Label Noise* (ISPRS / DLR; Massachusetts + DeepGlobe roads) — noise-aware / soft
bootstrapped losses for registration and omission noise from OSM-style labels
(https://elib.dlr.de/136343/1/Henry_et_al_Aerial_Road_Segmentation_Topological_Label_Noise.pdf).
SpaceNet / CosmiQ practice of buffering skeletons to a fixed width similarly treats
vector width as uncertain (NVIDIA SpaceNet road writeup).

**Tradeoffs:** Cheap; no drainage snap risk; can hide thin true trails if ignore is too
wide. Does not *move* labels onto the cut.

**Fixed experiment variables (one GT, clean geo val):**
- GT: fixed `trails.gpkg` after sidewalk ignore (no snap).
- Val: mountain-only geographic blocks (exclude Corner urban blocks) + full geo val.
- Vars: `ignore_buffer_m ∈ {0, 2, 3, 5}`; optional soft radius same set; loss ∈ {current
  focal-Tversky, + soft-bootstrapped Dice β∈{0.4,0.6}}.
- Report: val F1/IoU @ threshold swept on **val only**; vector P/R at **all** of
  {1,3,5,8,10} m (no picking test-best tolerance as a model win).

## Option B — Bounded latent / learned alignment (no relief snap)
**Idea:** Predict a small per-tile or per-segment translation (and optionally rotation)
that re-registers labels to imagery features, then supervise with the corrected mask.
Clamp the shift (e.g. ≤ 8–10 m) so lines cannot jump into drainages.

**Evidence:** *Learning to segment from misaligned and partial labels* (ACM COMPASS /
SIGCAS 2020) — Alignment Correction Network for translation errors in OSM-like
annotations before segmentation (https://doi.org/10.1145/3378393.3402254).

**Tradeoffs:** Can recover systematic GPS bias; more moving parts; still needs a clamp
and a clean val GT. Unbounded or relief-maximizing snaps remain unsafe (XPU-C QA:
cyan lines can lock onto gullies).

**Fixed experiment variables:**
- Same GT/val as A.
- Max shift ∈ {5, 8, 10} m; alignment input ∈ {RGB+NIR, +hillshade/slope}; freeze vs
  joint train.
- Ablate: reject updates whose shift lands in local-relief troughs below a threshold.
- Metrics as in A, plus median |shift| on val and % shifts rejected by drain gate.

## Option C — Trusted-label subset / confidence-weighted supervision
**Idea:** Do not treat all vectors equally. Up-weight UGRC singletrack / named path /
USFS non-motor; down-weight or ignore uncertain OSM; optionally require agreement
between sources within N m before a pixel is a hard positive.

**Evidence:** Crowdsourced GPS + imagery fusion literature stresses noise, waiting
areas, and misalignment in GPS traces (e.g. Sun et al., CVPR 2019 workshop poster on
crowdsourced GPS for road extraction; DF-DRUNet GPS+imagery fusion, *Int. J. Applied
Earth Obs.*, 2023, https://doi.org/10.1016/j.jag.2023.103632). Our audit already shows
source priority UGRC > USFS > OSM and Corner unnamed footway ≠ sidewalk.

**Tradeoffs:** Reduces sidewalk/urban pull without dropping Corner Canyon; may shrink
recall where only OSM exists (Teton). Complements A.

**Fixed experiment variables:**
- Same GT family with sidewalk ignore on.
- Positive policy ∈ {all path/track, named∪UGRC∪USFS only, dual-source agree @ 4 m};
  weights path/track/uncertain as now vs path=8 only on trusted.
- Metrics as in A; break out mountain vs Corner urban tiles.

## What not to do next
- Blind ReliefSnap into local-relief minima (drainage lock risk).
- Drop all unnamed footways or the whole Corner Canyon AOI.
- Declare a model win by choosing the vector tolerance that maximizes held-out P/R.
