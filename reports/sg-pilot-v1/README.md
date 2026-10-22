# E4 sg-pilot-v1: one shared budget vs two separate budgets (SafetyPointGoal2)

**Source:** lab-meeting notes 2026-09-30 (`260930.pdf`, Task 3), transcribed. Raw outputs
(`runs/sg-pilot-v1/`, `analysis/`) are on the Vast.ai instance, not in this checkout.
Config: `configs/safety/sg-pilot-v1.json` (code: branch1). Budgets from E3:
d_hazard = 31.9, d_vase = 73.9 steps/episode; x_i = C_i / d_i.

2M steps per run, seeds 0/1/2, dual lr 0.02, λ init 0.1. Evaluation: final checkpoint,
stochastic policy, 50 held-out layouts (environment seed 900000, action seed 900001).

| Method (`--method`) | Return | Goals/ep | x_hazard | x_vase | x_h + x_v | Seeds with both x ≤ 1 | Native constraint met |
|---|---:|---:|---:|---:|---:|---:|---|
| Unconstrained (`unconstrained`) | 22.4 | 8.6 | 1.76 | 1.84 | 3.60 | 0/3 | n/a |
| Aggregate, λ × 1 (`aggregate_loose`, x_h + x_v ≤ 2) | 12.2 | 5.0 | 0.59 | 0.81 | 1.41 | 2/3 | 3/3 (see note) |
| Separate, λ × 2 (`multi`, each x_i ≤ 1) | 19.3 | 7.6 | 1.59 | 1.37 | 2.96 | 0/3 | 0/3 |

Values are means over the 3 seeds. "Native constraint met" is the method's own training
constraint; for `multi` it is the same as both x ≤ 1.

Note on native feasibility: the notes state seed 0 of aggregate ended at (x_h, x_v) = (0.7, 1.1),
sum 1.8 ≤ 2, so it satisfies its own constraint while violating the vase budget. The other two
aggregate seeds lie inside the both ≤ 1 square on the held-out scatter plot (read off the
figure, not from saved numbers).

## Findings recorded at the meeting
- Aggregate looks over-constrained: large return loss; videos show it avoiding hazards and
  vases excessively. Its single λ is driven by the sum of both violations, so per cost it
  grows about twice as fast as each separate λ.
- Separate (`multi`) does not yet satisfy its constraints; it keeps entering hazards.
- Multipliers had **not converged** by 2M steps: separate λ_h, λ_v ≈ 5–6.5 and still rising;
  aggregate λ ≈ 3–8, flattening or falling for some seeds. Low aggregate cost is therefore not
  evidence of converged constrained optimization.
- Open design question from the meeting: the feasible sets differ (sum ≤ 2 vs each ≤ 1), so how
  should the shared budget be matched (equal area? sum ≤ 1 = `aggregate_conservative`)?

## Follow-ups
E5a (`multi`, dual lr 0.04) and E5b (`aggregate_conservative`) were queued with
`run_followups.sh` on branch1; their status is unknown here. E8 (pi0 weights) is planned.
