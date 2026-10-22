# Experiment configurations

One JSON file per experiment in `docs/experiments/INDEX.md`. A run is defined by
**git commit + config + seed + evaluation protocol**.

- `train_args`: `train.py` arguments by argparse name (`dual_lr`, not `--dual-lr`).
  `python train.py --config <file> --method M --seed S --output runs/<group>/M-sS`
  uses them as defaults; flags on the command line override.
- `extends`: a base file whose `train_args` are merged underneath (`safety/base.json`).
- `code`: the branch/commit that produced the results. A historical config may use
  arguments that only exist on that commit (e.g. `eval_chunk` on the archived branch1, tag `archive/branch1`); `train.py`
  refuses unknown arguments instead of silently ignoring them.
- `unknown`: settings that could not be verified from saved metadata. Never fill these
  in from current defaults.
- Seeds and layouts: `seeds.train` are training seeds; evaluation layouts are derived
  from fixed environment seeds (see `evaluation`). Layout seeds also depend on the
  evaluation chunk size (simulators alive at once), so the chunk is part of the protocol.
