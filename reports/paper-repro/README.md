# E2 paper-repro: Safety Gym PPO / PPO-Lagrangian on SafetyPointGoal2

**Source:** lab-meeting notes 2026-09-30 ([`docs/2026-09-30-lab-meeting.pdf`](../../docs/2026-09-30-lab-meeting.pdf), Task 2), transcribed. Raw outputs
(`runs/paper-repro*/`, `paper-comparison.{md,png,json}` from `compare_paper.py`) are not in
this checkout. Config: `configs/safety/paper-repro.json`.

3 seeds (0, 1, 2), 10M steps each, mean ± std across seeds. Final values average the last 5
epochs; cost rate = cumulative training cost / total steps. Paper values are read off Fig. 7
of Ray et al. (2019) and are approximate (`docs/reference/safetygym-fig7-pointgoal2.png`).

| Method | Episode return | Episode cost (limit 25) | Cost rate |
|---|---:|---:|---:|
| PPO, ours | 22.70 ± 0.16 | 191.1 ± 6.8 | 0.191 ± 0.003 |
| PPO, paper | ≈ 22.5 | ≈ 200 | ≈ 0.200 |
| PPO-Lagrangian, ours | 1.67 ± 1.74 | 8.5 ± 3.3 | 0.032 ± 0.006 |
| PPO-Lagrangian, paper | ≈ 1.0 | ≈ 30 | ≈ 0.037 |

**Reading:** PPO matches the paper. PPO-Lagrangian reproduces the paper's qualitative result
(cost and return both collapse; the paper's PPO-Lag also barely learns PointGoal2), but our
final episode cost (8.5) is well below the limit of 25, i.e. our run is more conservative than
the paper's. Not investigated further.

**Unknown:** which branch's `baseline_ppo.py` produced these numbers (main and branch1 differ
in environment pooling), and the run directory.
