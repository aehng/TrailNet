# Bounded Sol attempt, 2026-11-07

Base: Luna 23bcaf2. No PC checkout or compute used.

M1 scripts reconstruct original A supervision and sampler from read-only external inputs. `prepare_review.py` uses A-era local rasters, original recovered indexes, and sidewalk-aware overrides at paths recorded in source. `rescore.py --arm A` and `--arm prior` replay existing checkpoints on VAL only. NPY probabilities, regenerated TIFFs, and checkpoints are local artifacts excluded from Git; scripts and their identities make reconstruction possible. No candidate heldout scores have been opened.

CPU: `/home/box/.venvs/trailnet-smoke/bin/python`. Kaggle: `/home/box/.venvs/kaggle/bin/kaggle`.
