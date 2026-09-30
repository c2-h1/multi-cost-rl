# Multi-cost safe RL internship starter

This workspace contains a research plan, source-checked literature briefing, Korean mathematical notes, and executable PPO experiments. The completed session ran 17 training jobs, including a 12-run toy comparison and a real H1 pilot. Start with [training-report.md](training-report.md) for measured outcomes and limitations. The local runs used CPU PyTorch; W&B and local JSONL/checkpoints are both supported.

## Read first

- [Experiment setup](exp-setup.md): HumanoidBench costs, comparisons, budgets, compute and evaluation.
- [Recent work](recent-work.md): exact paper references, close prior art and possible research gaps.
- [Mathematical background (Korean)](background.md): CMDPs, feasible sets, duality, critics/GAE and multi-cost PPO.
- [Toy experiment results](results/toy-comparison-v1.md): completed comparison, uncertainty and learning curves.
- [Training report](training-report.md): actual runs, hardware measurements and limitations.

Splitting costs adds independent safety budgets and multipliers. It does **not** increase the actor's parameter dimension or automatically improve optimal return. Adding constraints to an otherwise unchanged CMDP shrinks its feasible set. Scalar aggregation can either hide violations or enforce a more conservative feasible set; compare individual costs in every baseline.

## Environment

The session created `.venv-train` using Python 3.11 and saved installed versions to `requirements-lock.txt`. To reproduce on this Mac:

```bash
source .venv-train/bin/activate
PYTHONPATH=external/humanoid-bench python -m unittest -v test_core test_humanoid_costs
python calibrate_toy.py
```

To create another environment:

```bash
python3.11 -m venv .venv-train
.venv-train/bin/python -m pip install -r requirements.txt
```

`requirements-lock.txt` records the complete tested environment; package wheels and accelerator builds differ on Linux. For CUDA choose the appropriate PyTorch build on the cluster. The `jax[cpu]` dependency here satisfies an upstream import; this trainer does not use MJX or GPU simulation.

HumanoidBench source is checked out at `external/humanoid-bench`, commit `cb1189039151c8aadaaa987b442da54383c87fab`. To recreate it:

```bash
git clone https://github.com/carlosferrazza/humanoid-bench.git external/humanoid-bench
git -C external/humanoid-bench checkout cb1189039151c8aadaaa987b442da54383c87fab
```

The adapter loads that checkout directly. We install the tested minimal runtime dependencies instead of the entire upstream research stack; the upstream setup also pins older Torch and includes unused MJX/Brax algorithms. No upstream files are modified.

## Train the toy comparison

```bash
.venv-train/bin/python run_suite.py --updates 600 --seeds 0 1 2 \
  --group toy-comparison-v1 --wandb-mode online
.venv-train/bin/python analyze.py --group toy-comparison-v1
```

Existing completed runs are skipped. Incomplete directories are preserved and cause an error; choose a new group for a clean rerun. A single run:

```bash
.venv-train/bin/python train.py --method multi --seed 3 --updates 600 \
  --wandb-mode offline --output runs/my-multi-s3
```

`TwoCostNavigation` is a custom, continuous 2D point-navigation toy inspired by safety navigation tasks. It is **not Safety Gym, Safety-Gymnasium, or HumanoidBench**. Observations contain position, goal displacement, distance from the central hazard, remaining-time information and the active flag. Tanh-squashed Gaussian actions move the point. Costs are central-hazard occupancy and mean squared action. Goal arrival is absorbing with zero reward/cost thereafter; the 64-step horizon is intrinsic and observed.

Per-episode budgets are `[1.92, 7.68]`, equivalent to fixed-horizon mean costs `[0.03, 0.12]`. The hand-coded controller in `calibrate_toy.py` verifies a feasible goal-reaching behavior, and is never used for policy initialization or imitation. The task reward is `2 * distance_progress - 0.015 + 2 * first_goal_arrival` while active.

All four methods use the same actor and reward + two cost critic heads, trained on the same targets:

| Method | Policy penalty | Constraint for multiplier update |
|---|---|---|
| `unconstrained` | none | none |
| `aggregate_loose` | one multiplier times summed normalized cost advantages | sum of expected normalized costs ≤ 2 |
| `aggregate_conservative` | same scalar form | sum of expected normalized costs ≤ 1 |
| `multi` | one multiplier per normalized cost advantage | each expected normalized cost ≤ 1 |

Because cost heads are retained in every method, this pilot controls auxiliary value-learning capacity. It does not yet implement the additional true single-critic ablation described in the research plan. The scalar methods have different feasible sets from `multi`; the suite illustrates this distinction, rather than establishing a novel optimizer advantage.

## Actual HumanoidBench pilot

```bash
.venv-train/bin/python train.py --env humanoid --humanoid-id h1-walk-v0 \
  --method multi --seed 0 --num-envs 8 --horizon 256 --updates 300 --dual-lr 0.02 \
  --eval-episodes 8 --final-eval-episodes 32 --eval-every 50 \
  --wandb-mode online --group humanoid-pilot \
  --output runs/humanoid-pilot-multi-s0
```

This is a short **modified finite-horizon pilot** on the real upstream simulator and robot without hands. The benchmark's main task is `h1hand-walk-v0`. See the report for the exact command actually run. Physical termination is padded to the pilot horizon with zero reward and continued fall-risk cost; time and absorption are observed. The collector deliberately gathers complete episodes, so `gamma=1` and zero terminal bootstrap match the implemented finite-horizon objective. It is not a generic collector for time-truncated continuing tasks.

Simulation is CPU MuJoCo and the batch adapter is sequential. `--device cuda` moves the actor/critics to a CUDA device but does not turn MuJoCo into a GPU simulator. Parallel independent seed jobs are the straightforward next use of a small cluster; benchmark the actual collector before increasing the training budget. There is no resume flag: checkpoints are usable for evaluation, while exact interrupted-run continuation would also require simulator and random-generator state.

## W&B and output files

The authorized session uses the existing W&B account at [multi-cost-rl](https://wandb.ai/cch-eck-postech/multi-cost-rl). No credential is copied into this workspace. Choose `--entity`/`--project` for your own destination or `--wandb-mode offline` to retain logs locally. An online initialization failure falls back to offline logging. A later sync requires an authenticated SDK:

```bash
wandb sync runs/RUN_NAME/wandb/offline-run-*
```

W&B supports [offline mode and environment configuration](https://docs.wandb.ai/models/track/environment-variables). The trainer also writes independent local files:

- `config.json`: parameters, package versions, cost budgets and source hashes.
- `source/`: exact source files used for the run.
- `metrics.jsonl`: returns, individual costs/violations, multipliers, training exposure, KL and value losses.
- `checkpoint.pt`: final actor/critic, optimizer and multiplier states.
- `evaluation.npz`: held-out per-episode returns and cost vectors, for stochastic and deterministic policies.
- `summary.json`, `wandb-info.json`: final evaluation and dashboard link.

Final stochastic-policy evaluation is primary. Mean-action deterministic evaluation can change cost behavior and is reported separately. Expected-cost feasibility does not imply per-episode or physical safety. This code is an experimental baseline, with no safe-exploration or hardware-deployment guarantee.
