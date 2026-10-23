# Experiment index

One row per experiment, in the order they were run. Each has a config in `configs/` and,
when results exist, a summary in `reports/`. Status values: planned / running / completed /
failed / superseded / **unknown** (could not be checked from this checkout).

**Evidence levels.** None of the raw run directories (`config.json`, `metrics.jsonl`,
W&B files) for E2–E6 are in this checkout: they were produced on a Vast.ai instance and the
titanxp cluster. Entries say where each fact comes from: *code* (launcher defaults at the
recorded commit), *report* (a committed report), *notes* (lab-meeting notes 2026-09-30,
[`docs/2026-09-30-lab-meeting.pdf`](../2026-09-30-lab-meeting.pdf)). Nothing marked unknown has been filled in from
current defaults.

"branch1" below means the archived branch, kept as tag `archive/branch1` (archived 2026-10-22).

The lab-meeting notes number the stages differently: Task 1 = E1, Task 2 = E2, Task 3 = E3+E4.

| ID | Name | Question | Status | Config | Code | Output / W&B | Report |
|---|---|---|---|---|---|---|---|
| E1 | toy-comparison-v1 | Does separate budgeting behave differently from an aggregate one in a 2-cost toy? | completed 2026-09-20 | `configs/toy/toy-comparison-v1.json` | 4ac4491 | W&B group `toy-comparison-v1` | `reports/toy-comparison-v1/`, `reports/2026-09-21-training-report.md` |
| E2 | paper-repro | Does our Safety-Gymnasium setup reproduce Ray et al. (2019) PPO / PPO-Lag? | completed (notes) | `configs/safety/paper-repro.json` | unverified | `runs/paper-repro*/` (not here) | `reports/paper-repro/` |
| E3 | sg-calibration | Per-cost budgets for SafetyPointGoal2 | completed | `configs/safety/sg-calibration.json` | branch1 | `runs/dev/unconstrained-s100`, `runs/pipeline/status.json` (Vast.ai) | budgets 31.9 / 73.9 |
| E4 | sg-pilot-v1 | One shared budget vs two separate budgets on SafetyPointGoal2 | completed (notes) | `configs/safety/sg-pilot-v1.json` | branch1 | `runs/sg-pilot-v1/` (Vast.ai), W&B disabled | `reports/sg-pilot-v1/` |
| E5a | sg-pilot-v1-duallr004 | Do separate multipliers converge with dual lr 0.04? | unknown | `configs/safety/sg-pilot-v1-duallr004.json` | branch1 b240033 | `runs/sg-pilot-v1-duallr004/` | — |
| E5b | sg-pilot-v1-conservative | Strict shared budget x_h + x_v ≤ 1 as the matched control | unknown | `configs/safety/sg-pilot-v1-conservative.json` | branch1 b240033 | `runs/sg-pilot-v1-conservative/` | — |
| E6 | Lagrangian hparam search | Successive-halving search over multi / aggregate / aggregate_loose | unknown — no code or results found | — | not found | not found | — |
| E7 | humanoid-pilot | Can the multi-cost trainer run on HumanoidBench H1? | failed (no walking) | `configs/humanoid/humanoid-pilot.json` | 4ac4491 | W&B run `jfn5p0c9` | `reports/2026-09-21-training-report.md` §4 |
| E8 | pi0-weights | Weight costs by the untrained policy's costs instead of the budgets (from the session's task text; not the professor's suggestion, see E10) | planned; π_0 calibration done | `configs/safety/pi0-weights.json` | pi0-weights | `runs/calibration/` | — |
| E9 | hierarchical / priority safe RL | How to prioritize constraints that cannot all be met (CSED499II topic) | planned | — | — | — | notes p. 9 |
| E10 | sg-pilot-v2 | E4 at 5M steps; multi dual lr 0.04, aggregate_loose 0.02. Do both converge? | running (started 2026-10-23) | `configs/safety/sg-pilot-v2.json` | pi0-weights afe85f2 | `runs/sg-pilot-v2-loose/`, `runs/sg-pilot-v2-multi/` (lab machine) | — |

## Dual learning rate: which value was used where

The `train.py` default `--dual-lr` is 0.15 and has never changed. It was the setting used
in the toy experiment; every Safety Gym launcher overrides it.

| Experiment | dual lr | Evidence |
|---|---|---|
| E1 toy main comparison | 0.15 | report ("The multiplier learning rate was 0.15") |
| E1 toy development seeds 101 / 102 | 0.02 | report §3 |
| E3, E4 Safety Gym | 0.02 | code (`run_pilot.py` FROZEN, `slurm_safety_*.sh`) and notes ("dual lr 0.02") |
| E5a | 0.04 | code (`run_followups.sh` on branch1) |
| E7 humanoid | 0.02 | report §4 |
| E6 hparam search | unknown | — |
| E10 | multi 0.04, aggregate_loose 0.02 | config |

Saved `config.json` files for E3–E5 were not available to confirm this directly.

---

## E1 toy-comparison-v1 (completed)
- **Setup:** `toy_env.TwoCostNavigation` (64 steps; hazard occupancy budget 1.92, effort budget 7.68).
  Four methods × seeds 0/1/2, 2.46M transitions each, dual lr 0.15. 512 stochastic held-out episodes.
- **Findings:** all methods reach the goal. aggregate_loose satisfies its sum ≤ 2 but violates the
  effort budget in all 3 seeds (≈ 1.58× its budget): the motivating failure case. multi and
  aggregate_conservative meet both mean budgets in 1/3 seeds each. Multipliers oscillated.
- **Limitations:** 3 seeds; feasible sets differ between methods; dual lr 0.02 on dev seeds did not
  fix feasibility.

## E2 paper-repro (completed, per notes)
- **Setup:** `baseline_ppo.py` (mirrors safety-starter-agents), scalar cost, limit 25, 10M steps, seeds 0/1/2.
- **Findings:** PPO 22.70 return / 191.1 cost (paper ≈ 22.5 / 200). PPO-Lag 1.67 / 8.5 (paper ≈ 1.0 / 30).
  Reproduces the paper's trends; our PPO-Lag is more conservative than the paper's.
- **History:** the first attempt (all 6 runs at once, 30 envs each) ran the host out of memory → memory-aware
  launchers (`run_paper_repro.py` on main, `memguard.py` on branch1).
- **Unknown:** which branch's code produced the numbers; run directory.

## E3 sg-calibration (completed)
- Unconstrained, seed 100, 500k steps; 30 layouts (environment seed 500000, chunk 10).
  d_i = max(1, round(0.5 × mean C_i, 1)) → **d_hazard = 31.9, d_vase = 73.9**.

## E4 sg-pilot-v1 (completed, per notes)
- **Setup:** unconstrained / aggregate_loose / multi × seeds 0/1/2, 2M steps, dual lr 0.02;
  50 held-out layouts (900000 / 900001).
- **Findings:** aggregate: return 12.2, x = (0.59, 0.81), both ≤ 1 in 2/3 seeds, its own constraint
  in 3/3, so seed 0 meets sum ≤ 2 while x_vase = 1.1. multi: return 19.3, x = (1.59, 1.37), 0/3.
  Multipliers still rising at 2M steps (not converged).
- **Next:** E5a (faster duals), E5b (matched strict aggregate), E8.

## E5a / E5b follow-ups (unknown)
Queued by `run_followups.sh` on branch1 after the meeting ("experiments in progress"). Check
`runs/sg-pilot-v1-duallr004/` and `runs/sg-pilot-v1-conservative/` on the machine where they ran,
then run `python scripts/summarize_runs.py --root <dir>` and update this entry.

## E6 Lagrangian hyperparameter search (unknown)
A successive-halving search over multi / aggregate / aggregate_loose was described, but on
2026-10-22 no branch (local or GitHub: main, branch1, pi0-weights), commit message, file or run
directory in this checkout mentions it. **No results are recorded here.** To consolidate it:
copy its code onto a branch and its run directories (or summaries) under `runs/hparam/`, then
run `python scripts/summarize_runs.py --root runs/hparam` and record per rung which runs finished.
When comparing methods, report native feasibility and both ≤ 1 separately, keep tuning seeds
apart from final-report seeds, and don't treat a near-zero multiplier on a slack constraint as
non-convergence.

## E7 humanoid-pilot (failed)
h1-walk-v0, 3 costs (fall risk, joint limit, actuator load), horizon 256, 300 updates, seed 0.
The robot fell in every episode (physical steps 21.8% of scheduled); raising the fall-risk
multiplier did not help. Next: reward-only walking at the native horizon first, or a pretrained policy.

## E8 pi0-weights (planned)
- **Idea:** w_i = 1/max(C_i^0, floor), where C_i^0 is the mean episode cost of the untrained policy π_0.
  The budgets d_i stay the same; evaluation is unchanged. Implemented on branch `pi0-weights`
  (`--cost-weights pi0`); the default `budget` mode is bit-identical to the old trainer.
- **Calibration** (`calibrate_pi0.py`, 30 layouts of E3, chunk 10; files in `runs/calibration/`):

  | seed | C_h^0 | zero-cost episodes | C_v^0 | zero-cost episodes |
  |---|---:|---:|---:|---:|
  | 0 | 20.5 ± 82.8 | 90% | 13.2 ± 36.6 | 87% |
  | 1 | 11.3 ± 49.0 | 93% | 13.6 ± 38.9 | 80% |
  | 2 | 40.0 ± 113.8 | 87% | 19.0 ± 49.7 | 80% |
  | 100 | 6.3 ± 29.8 | 93% | 11.9 ± 33.6 | 83% |

  (mean ± std over 30 episodes; `runs/calibration/chunk2/` holds an earlier run on the wrong
  layouts, chunk 2.)
- **Open issues:** each mean comes from 2–6 non-zero episodes, so the weights vary a lot
  across seeds (C_h^0 from 6.3 to 40.0). w_i·d_i ranges from 0.8 to 6.2: for seed 2, π_0 already
  exceeds the hazard budget; for the other seeds the budgets are looser than π_0. dual lr 0.02 acts
  several times faster in π_0 units than in budget units. Decide on more calibration episodes or
  pooled seeds, and on the dual lr, before running.

## E10 sg-pilot-v2 (running)
- **Why:** E4 did not converge in 2M steps (λ still rising). Follows interpretation A of the
  professor's suggestion: fixed weights 1/d_i, x_h + x_v ≤ 2. Interpretation B (weights that follow
  the violation) is open for the next lab meeting.
- **Setup:** aggregate_loose (dual lr 0.02) and multi (0.04) × seeds 0/1/2, 5M steps (1250 updates).
  Training evals use chunk 2; held-out uses `--eval-num-envs 10`, the same 50 layouts as E4.
- **Cost:** ≈ 28 s per update on the lab machine → ≈ 12 h per run, 3 at a time, two groups in a row.

## E9 hierarchical / priority safe RL (planned)
CSED499II topic: priorities among constraints that cannot all be met (people > expensive
parts > replaceable parts). Reading list in the notes: lexicographic RL, LexiSafe (2026),
Hierarchical Relaxation (IROS 2023), POCE (CVPR 2024).
