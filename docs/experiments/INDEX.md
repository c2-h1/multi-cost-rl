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
| E5a | sg-pilot-v1-duallr004 | Do separate multipliers converge with dual lr 0.04? | closed: did not converge (user, 2026-10-23) | `configs/safety/sg-pilot-v1-duallr004.json` | branch1 b240033 | `runs/sg-pilot-v1-duallr004/` | — |
| E5b | sg-pilot-v1-conservative | Strict shared budget x_h + x_v ≤ 1 as the matched control | closed: did not converge (user, 2026-10-23) | `configs/safety/sg-pilot-v1-conservative.json` | branch1 b240033 | `runs/sg-pilot-v1-conservative/` | — |
| E6 | Lagrangian hparam search | Successive-halving search over multi / aggregate / aggregate_loose | closed 2026-10-23 (not pursued) | — | not found | not found | — |
| E7 | humanoid-pilot | Can the multi-cost trainer run on HumanoidBench H1? | failed (no walking) | `configs/humanoid/humanoid-pilot.json` | 4ac4491 | W&B run `jfn5p0c9` | `reports/2026-09-21-training-report.md` §4 |
| E8 | pi0-weights | (dropped 2026-10-23: weights from the untrained policy; not the professor's suggestion) | dropped | — | tag `archive/pi0-weights` | — | — |
| E9 | hierarchical / priority safe RL | How to prioritize constraints that cannot all be met (CSED499II topic) | planned | — | — | — | notes p. 9 |
| E10 | spg-v2 | E4 at 5M steps; multi dual lr 0.04, aggregate_loose 0.02. Do both converge? | completed 2026-10-24 (both at budget, not converged) | `configs/safety/spg-v2.json` | spg-v2 fe872d5 / b0441c0 | `runs/spg-v2-aggregate-loose/`, `runs/spg-v2-multi/` (lab machine), W&B groups `spg-v2-*` | `reports/spg-v2/` |
| E11 | spg-v2-pi | multi: does K_I 0.04 → 0.02, and a proportional term (K_P 1, 4), reduce the multiplier oscillation? | running (started 2026-10-24) | `configs/safety/spg-v2-pi.json` | spg-v2 f3dd700 | `runs/spg-v2-pi-*/` (lab machine), W&B groups `spg-v2-pi-*` | — |

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
| E11 | multi, PI controller: K_I 0.04 or 0.02, K_P 0 / 1 / 4 | config |

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
- **Next:** E5a (faster duals), E5b (matched strict aggregate), then E10 (5M steps).

## E5a / E5b follow-ups (closed)
Run by `run_followups.sh` on branch1 (2M steps). Per the user (2026-10-23), neither converged;
the run files are not available, so no numbers are recorded. Closed; E10 continues with more steps.

## E6 Lagrangian hyperparameter search (closed)
A successive-halving search over multi / aggregate / aggregate_loose was described, but no code
or results were found. Closed on 2026-10-23: hyperparameters are chosen by hand for now (E10).

## E7 humanoid-pilot (failed)
h1-walk-v0, 3 costs (fall risk, joint limit, actuator load), horizon 256, 300 updates, seed 0.
The robot fell in every episode (physical steps 21.8% of scheduled); raising the fall-risk
multiplier did not help. Next: reward-only walking at the native horizon first, or a pretrained policy.

## E10 spg-v2 (completed 2026-10-24)
- **Why:** E4 did not converge in 2M steps (λ still rising). Interpretation A of the professor's
  suggestion: fixed weights 1/d_i, x_h + x_v ≤ 2. Interpretation B is open for the next lab meeting.
- **Setup:** aggregate_loose (dual lr 0.02) and multi (0.04) × seeds 0/1/2, 5M steps (1250 updates).
  Training evals use chunk 2; held-out uses `--eval-num-envs 10`, the same 50 layouts as E4.
- **Code:** aggregate_loose runs at fe872d5; multi runs at b0441c0 (same Python as ac53cc6, pi0 code
  removed, verified bit-identical on toy runs). Run directories renamed from `sg-pilot-v2-*` after training.
- **Findings (held-out, means of 3 seeds):** aggregate_loose return 16.9, x = (0.84, 1.01);
  multi 14.6, x = (0.88, 0.97); both sums 1.85. Both ≤ 1: aggregate_loose 2/3, multi 1/3, but the
  per-cost standard error is 0.1–0.15, so per-seed pass/fail is within noise. aggregate_loose
  trades costs (seed 2 x_vase 1.30). Multipliers peaked at 1.3–2.5M steps and are still moving
  (damped oscillation): **not converged**. No method difference established.
- **Next:** damp the dual (PID Lagrangian or lower dual lr after the peak), more seeds and
  held-out episodes, matched feasible sets, interpretation B.

## E11 spg-v2-pi (running)
- **Why:** E10's multipliers oscillated slowly (peak 2–2.5M steps, still moving at 5M).
- **Conditions (multi only, seeds 0/1/2, 5M steps, I = λ = 0 at start):** A K_I 0.04 (E10's gain),
  B K_I 0.02, C K_I 0.02 + K_P 1, D K_I 0.02 + K_P 4. Projected PI with a signed proportional
  error (`train.py pi_update`), not a reproduction of a paper's code. A→B = integral gain effect,
  B→C/D = proportional term effect.
- **Baseline reuse:** none. E10 multi started at λ = 0.1, lacks 4.25M / 4.75M checkpoints and the
  controller logs, so A is rerun (12 runs, 60M steps).
- **Evaluation:** validation 200 episodes (env 700000, action 700001, chunk 10) on the 4.0–5.0M
  checkpoints; held-out 500 episodes (env 950000, action 950001) only after selecting on validation.

## E9 hierarchical / priority safe RL (planned)
CSED499II topic: priorities among constraints that cannot all be met (people > expensive
parts > replaceable parts). Reading list in the notes: lexicographic RL, LexiSafe (2026),
Hierarchical Relaxation (IROS 2023), POCE (CVPR 2024).
