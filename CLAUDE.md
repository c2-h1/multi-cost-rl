# Instructions for coding-agent sessions

Multi-cost safe RL research repo (one undergraduate researcher). Keep it simple.

## Before changing anything
- Read `docs/experiments/INDEX.md` and the config of the experiment you are touching
  (`configs/`). Read PLAN.md for open decisions.
- Check `git status`, the current branch, and running jobs (`pgrep -af train.py`,
  `tmux ls`). Never modify, move, or delete files that an active run uses.
- `branch1` is **archived** (tag `archive/branch1`): read it for history (it produced
  E3/E4), never develop on it or merge it. Current work builds on `main`/`pi0-weights`.

## Code
- One trainer: `train.py` (multi-cost PPO-Lagrangian). The paper baseline `baseline_ppo.py`
  is intentionally separate. Don't create per-experiment copies of either.
- Reuse the launchers (`run_pilot.py`, `run_paper_repro.py`), `evaluate.py`, and
  `scripts/summarize_runs.py`. Extend them with a flag before writing a new script.
- Don't silently change existing baselines: algorithms, reward/cost definitions,
  hyperparameters, parser defaults (`--dual-lr` default 0.15 is the toy setting), or
  evaluation seeds/chunk sizes. New behavior goes behind a flag whose default keeps old
  runs reproducible. Check bit-identity against the previous commit on a short toy run.
- Evaluation layouts depend on the env seed **and** the chunk size (`--eval-num-envs`;
  archived branch1 used 10, main's run_pilot uses 2). Keep it fixed within an experiment.

## Experiments
- A new experiment gets an ID in `docs/experiments/INDEX.md` and a config in `configs/`
  (`extends` a base; settings that can't be verified go in `unknown`, never filled from
  current defaults). Run with `train.py --config ...` or `run_pilot.py --extra "--config ..."`.
- Keep tuning/validation seeds separate from final-report seeds.
- Report both the method's native constraint and the per-cost "both ≤ 1" check.
  Don't call a run converged without evidence; a slack constraint may have λ ≈ 0.
- Don't launch long training unless asked.

## Results and git
- Raw outputs go in `runs/` (ignored). Only small curated summaries/figures go in `reports/`.
  Stage files by name; never force-add `runs/` or `wandb/`.
- Never delete experimental artifacts (runs, checkpoints, W&B data) without explicit permission.
- Run `python -m pytest test_core.py` before every commit. Keep code commits separate from
  result commits (`feat:`/`fix:`/`refactor:` vs `results:`/`docs:`). No amend, no history
  rewrite, no push unless asked.
