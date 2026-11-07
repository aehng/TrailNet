# Bounded Sol attempt, 2026-11-07

Base: Luna 23bcaf2. No PC checkout or compute used.

M1 scripts reconstruct original A supervision and sampler from read-only external inputs. `prepare_review.py` uses A-era local rasters, original recovered indexes, and sidewalk-aware overrides at paths recorded in source. `rescore.py --arm A` and `--arm prior` replay existing checkpoints on VAL only. NPY probabilities, regenerated TIFFs, and checkpoints are local artifacts excluded from Git; scripts and their identities make reconstruction possible. No candidate heldout scores have been opened.

CPU: `/home/box/.venvs/trailnet-smoke/bin/python`. Kaggle: `/home/box/.venvs/kaggle/bin/kaggle`.

Coverage training now fails closed unless both `training.frozen_sampler_manifest` and `training.frozen_sampler_sha256` are provided. Use `sampler-original.json` and SHA256 `e7d4f41185ffc0d08af9512fce331c6de099d8e0adb9be8d2c0fa6c53bb240e0`. The new ensembles are inference only. Their launched bundle is immutable and recorded separately from subsequent changes to the unused training guard.
