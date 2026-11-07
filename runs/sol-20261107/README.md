# Bounded Sol attempt, 2026-11-07

Base: Luna 23bcaf2. No PC checkout or compute used.

M1 scripts reconstruct original A supervision and sampler from read-only external inputs. `prepare_review.py` uses A-era local rasters, original recovered indexes, and sidewalk-aware overrides at paths recorded in source. `rescore.py --arm A` and `--arm prior` replay existing checkpoints on VAL only. NPY probabilities, regenerated TIFFs, and checkpoints are local artifacts excluded from Git; scripts and their identities make reconstruction possible. No candidate heldout scores have been opened.

CPU: `/home/box/.venvs/trailnet-smoke/bin/python`. Kaggle: `/home/box/.venvs/kaggle/bin/kaggle`.

Coverage training now fails closed unless both `training.frozen_sampler_manifest` and `training.frozen_sampler_sha256` are provided. Use `sampler-original.json` and SHA256 `e7d4f41185ffc0d08af9512fce331c6de099d8e0adb9be8d2c0fa6c53bb240e0`. The new ensembles are inference only. Their launched bundle is immutable and recorded separately from subsequent changes to the unused training guard.

The frozen four-view ensemble can be used by `scripts/predict.py` through `inference.geometric_ensemble: tta4`; omit it or use `none` for baseline inference. `config_tta4_inference.yaml` records exact box source references and separate output paths; it is an inference configuration, with only VAL tiles staged. Do not use it as a training configuration. The optional full-AOI inference command is:

```
/home/box/.venvs/trailnet-smoke/bin/python scripts/predict.py --config runs/sol-20261107/config_tta4_inference.yaml --all-areas
```

This command has not been run; full-AOI/vector scores are not implied by tile evaluation. Expect four forward passes per tile. `frozen-selection.json` fixes TTA4 at0.90 before the sole heldout comparison. The VAL and heldout bundles each record their exact source snapshot and external input hashes. No PC run was requested or performed.
