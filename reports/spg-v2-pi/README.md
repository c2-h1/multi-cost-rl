# E11 spg-v2-pi: projected PI multipliers for multi (SafetyPointGoal2)

**Status (2026-10-24 22:55 KST): running. No results yet.** Training started 2026-10-24 22:54 on the
lab machine (tmux `spg-v2-pi`); expected ≈ 24 h for training (3 waves of 4 runs, ≈ 22 s/update)
plus ≈ 5 h of validation. Config: [`configs/safety/spg-v2-pi.json`](../../configs/safety/spg-v2-pi.json).

| Condition | K_I | K_P | K_D | Group | Seeds 0 / 1 / 2 |
|---|---:|---:|---:|---|---|
| A: E10 baseline gain | 0.04 | 0 | 0 | `spg-v2-pi-a-ki04` | running / queued / queued |
| B: slower integral | 0.02 | 0 | 0 | `spg-v2-pi-b-ki02` | running / queued / queued |
| C: weak PI | 0.02 | 1 | 0 | `spg-v2-pi-c-ki02-kp1` | running / queued / queued |
| D: strong PI | 0.02 | 4 | 0 | `spg-v2-pi-d-ki02-kp4` | running / queued / queued |

Held-out evaluation (500 episodes) has not been run; it happens only after a condition is selected on validation.

## Controller

Projected PI variant with a signed proportional error (`train.py` `pi_update`, `--dual-controller pi`).
It is not a reproduction of a paper's implementation (cf. Stooke et al., 2020). Per cost i, rollout k:

```text
e_i[k]       = mean_episode_cost_i[k] / budget_i - 1
I_i[k+1]     = max(0, I_i[k] + K_I * e_i[k])
lambda_i[k+1] = max(0, I_i[k+1] + K_P * e_i[k])
I_i[0] = lambda_i[0] = 0
```

- I and λ are separate states; the next integral step starts from I, never from λ.
- Rollout k's PPO update uses λ[k]; λ[k+1] is used by the next update (same order as before).
- No D term, cost EMA, λ cap or gain decay.
- Logged every update: `dual/error_i`, `dual/integral_i`, `dual/proportional_i`, `dual/lambda_i`, `dual/ki`, `dual/kp`,
  plus the existing raw normalized costs, return, approx KL, KL early stop and advantage magnitudes.
- Checkpoints store `dual_controller` = {type, ki, kp, kd, integral, lambdas}.

## Checks before launch

- Unit tests (`test_core.py`): K_P = 0 matches the integral update exactly over 500 random steps
  (multi and both aggregates, including the 0 boundary); signs, clipping of I and of λ, and that the
  next step starts from I.
- Toy runs (multi, aggregate_loose; 8 updates): the new default path and PI with K_P = 0 give
  metrics and summaries identical to the previous commit.
- A 3-update SafetyPointGoal2 smoke run with K_P = 4: logs, the off-grid checkpoint, and evaluate.py on it.
- Parsed arguments of each condition vs the E10 multi-s0 `config.json`: only `lambda_init` (0.1 → 0),
  `dual_controller`, `dual_ki`, `dual_kp`, `checkpoint_steps` (and `dual_lr`, set equal to K_I and
  unused by the PI path) differ. Episodes per update, PPO settings, budgets, network and evaluation are unchanged.

## Baseline reuse

E10's multi runs (dual lr 0.04) are **not reused**: they started at λ = 0.1 (here I = λ = 0), have no
4.25M / 4.75M checkpoints, and did not log the controller state. Condition A is rerun, so training
is 12 runs, 60M steps.

## Fixed settings

SafetyPointGoal2-v0; budgets 31.9 / 73.9; 4 episodes (4,000 steps) per update; 1,250 updates (5M
steps); PPO lr 3e-4, 6 epochs, minibatch 512, clip 0.2, target KL 0.03; 64×64 networks; costs
normalized by the budgets; advantages mixed then normalized once. Seeds 0, 1, 2 (exploration
seeds; no strong statistical claims from 3 seeds).

## Evaluation plan

- **Validation:** checkpoints 4.0M, 4.252M, 4.5M, 4.752M, 5.0M, 200 episodes each,
  environment seed 700000, action seed 700001, chunk 10, stochastic, in a separate process after
  training (training RNG untouched). Same layouts and action seed for every condition.
- 4.25M and 4.75M are not multiples of 4,000 steps; the checkpoints are the first updates past them
  (update 1063 = 4,252,000 and update 1188 = 4,752,000 steps).
- **Held-out (after selection only):** final 5M checkpoint, 500 episodes, environment seed 950000,
  action seed 950001, chunk 10, for A, B and the selected PI condition. Held-out results are not
  used to re-select gains.
- **Criteria:** 0.25M-step windows over 4–5M (cost, λ, return means and trends; window-to-window
  swing separated from raw per-update noise), cumulative positive excess, validation expected costs
  near or below budget, return loss / over-conservatism, λ still trending (= training may be unfinished).
  Joint pass rate is auxiliary. Episode uncertainty and seed-to-seed spread are reported separately.
- Tables and figures: `python scripts/analyze_pi.py --output reports/spg-v2-pi`.

## Next (only after the first results)

- K_P = 4 still oscillates slowly and adds little noise → add K_P = 16.
- K_P = 1 stable but K_P = 4 jumps every update → try K_P = 2.
- PI helps → validate on more seeds, then apply to aggregate. K_D stays 0 if PI suffices.
