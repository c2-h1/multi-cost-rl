# Plan: one aggregate safety budget vs. two separate budgets

Status and next action are in [§6 Status](#6-status-and-handoff). A new session should read this file, then README.md.

## 1. Objective

A safe-RL agent often has more than one way to be unsafe. Here a point robot has to reach goals while avoiding two things:

- **hazards:** zones on the floor it shouldn't enter
- **vases:** fragile objects it shouldn't bump or push

The usual Safety-Gym approach combines all violations into **one scalar cost** with one limit. The question is what changes when each kind of violation gets **its own budget** instead of sharing one.

**Why it matters.** With one shared budget, the agent can trade the two violations against each other. It could run through hazards a lot and never touch a vase, and the total still passes. Separate budgets forbid that trade.

**The two-day objective** is a working, validated pipeline plus honest three-seed preliminary results that feed the semester proposal. The goal is not to prove separate budgets win. A null or negative result is fine if it is measured properly.

## 2. Machine and software (measured 2026-09-30)

| | |
|---|---|
| Host | this SSH box, `/home/cyan/multi-cost-rl` |
| Hardware | 4× RTX 2080 Ti (11 GB), 28 CPU cores, 62 GB RAM |
| Env | `.venv` via `./setup_env.sh`: Python 3.10.21 (uv), torch 2.14+cu126, MuJoCo 2.3.3, Gymnasium 0.28.1, Safety-Gymnasium 1.0.0 (PyPI). Exact versions in `requirements-lock.txt` |
| Throughput | Simulation is CPU-bound; CPU and CUDA give the same speed. Single-env sim ≈ 415 steps/s. One job (4 envs) ≈ 15 s per 4,000-step update. **9 concurrent jobs ≈ 25 s/update each, ~1,450 transitions/s total** |

Always `source .venv/bin/activate; export MUJOCO_GL=egl OMP_NUM_THREADS=2`.

**Why Python 3.10:** Safety-Gymnasium 1.0.0 pins `mujoco==2.3.3` and `pygame==2.1.0`, which have no Python 3.12 wheels. GitHub tags v1.1/v1.2 have the same pins and only cosmetic cost-code changes. The unofficial `LIRA-illinois/safety-gymnasium2` fork was created 2026-05 and has 5 commits on a `dev` branch, no releases, and 0 stars. It ports to Gymnasium 1.x, Python ≥ 3.11, and unpinned MuJoCo 3.x. Its cost code is unchanged, but MuJoCo 3 contact physics would break comparability with published results, so it is **not used**.

## 3. Environment and costs

**`SafetyPointGoal2-v0`:**

- **Robot and actions:** a point robot with 2 actions, forward and turn, in [-1, 1].
- **Observation:** 60 dims, mostly lidar to the goal, hazards, and vases.
- **Reward:** progress toward the goal plus a bonus on reaching it. A new goal then appears, so several goals per episode are possible.
- **Episode:** 1000 steps. Verified: `SafeTimeLimit` = `task.num_steps`.
- **Layout:** 10 hazards and 10 vases, randomly placed each reset.

**Two costs.** Each is a per-step indicator from `info`:

```text
c_hazard = 1[cost_hazards > 0]
c_vase   = 1[cost_vases_contact > 0 OR cost_vases_velocity > 0]
```

- **Episode cost** is the sum over 1000 steps, so it counts affected timesteps, not incidents.
- **No double counting within a cost:** vase contact plus vase motion on the same step counts as 1.
- **A step can count for both costs.**
- **The environment's scalar `cost`** is 1[any component > 0] (`constrain_indicator=True`, same as the original Safety Gym). It is used only for the paper reproduction (§5).

**Episode handling** in our trainer (`safety_adapter.py`):

- Every episode is complete: 1000 steps, `gamma=1`, zero bootstrap at the end.
- **Observations:** we append the elapsed fraction and an absorbed flag (62 dims).
- **Early termination:** the episode becomes absorbing; the env is not stepped again, and reward and cost are 0.
- **Hard errors:** a simulator exception, an unexpected truncation, or a missing cost field raises. None is ever recorded as zero cost.

## 4. Main experiment: three methods

All three use the same PPO and the same networks: an actor, plus a critic with three outputs (reward, hazard cost, vase cost). Keeping the network identical means differences come from the constraint setup. Costs are divided by their budgets, `x_i = C_i / d_i`, so 1.0 means exactly on budget.

| `--method` | Constraint | Enforcement |
|---|---|---|
| `unconstrained` | none | reference point: how unsafe it is when it doesn't care |
| `aggregate_loose` | x_hazard + x_vase ≤ 2 | **one** multiplier λ on (A_hazard + A_vase) |
| `multi` | x_hazard ≤ 1 **and** x_vase ≤ 1 | **two** multipliers, one per cost advantage |

- **Multiplier updates.** After each batch, each λ changes by `dual_lr × (batch-mean normalized cost − target)`, projected to ≥ 0. The actor uses the reward advantage minus the λ-weighted cost advantages, normalized once after mixing.
- **Selective pressure.** In `multi`, if the robot violates only the hazard budget, only λ_hazard rises. In `aggregate_loose`, one λ pushes on both costs, and it stops pushing once the sum is under 2, even if one cost alone is 1.5× its budget.
- **Caveat:** the feasible sets differ. (1.5, 0.5) passes `aggregate_loose` and fails `multi`. So if `multi` ends up safer per cost, that is partly by construction; it does not prove better optimization. The experiment measures the practical consequences of the specification.

**Frozen settings** (`run_pilot.py:FROZEN`; change only with a new `--group`):

- **Networks:** 64×64 tanh actor and critic.
- **Batch:** 4 envs × 1000 steps per update.
- **PPO:** lr 3e-4, 6 epochs, minibatch 512, clip 0.2, GAE λ 0.95, γ 1, target KL 0.03, value coefficient 1, entropy 0.
- **Multipliers:** initialized at 0.1, learning rate 0.02, target 1.0.
- **Seeds:** development 100/101, main 0/1/2.

**Measured for every policy:**

- return and goals per episode
- mean C_hazard/d_hazard and C_vase/d_vase, and whether both are ≤ 1
- per-episode exceedance rates
- cumulative training cost
- multiplier trajectories (settle, grow, or oscillate)

## 5. Stage R: reproduce the Safety Gym paper baseline first

Before the main experiment, reproduce the published PPO and PPO-Lagrangian results on PointGoal2. This validates the environment, the scalar cost, and a canonical training setup against known numbers.

**Reference.** Ray, Achiam & Amodei (2019), *Benchmarking Safe Exploration in Deep RL*. The code is `openai/safety-starter-agents`: `scripts/experiment.py`, `safe_rl/pg/{run_agent,buffer,network,agents,algos}.py`. The settings below were read from that source:

| Setting | Paper / reference code |
|---|---|
| Env steps | 1e7 (Point/Car); 30,000 steps per epoch → 333 epochs; max episode length 1000 |
| Networks | separate π, V_r, V_c MLPs, 256×256 tanh |
| Policy | Gaussian, state-independent log_std initialized at −0.5, **no tanh squashing** (env clips actions) |
| Discounting | γ = 0.99, λ_GAE = 0.97 for both reward and cost; value bootstrap at the 1000-step timeout |
| Advantages | reward advantage normalized (mean 0, std 1); cost advantage **centered only** |
| Policy update | Adam lr 3e-4, 80 full-batch iterations, clip 0.2, early stop when KL > 1.2 × 0.01 |
| Value update | Adam lr 1e-3, 80 full-batch iterations on (V_r loss + V_c loss) |
| Cost | Safety Gym scalar indicator cost, cost_lim = 25 per episode |
| PPO-Lagrangian | penalty = softplus(p), initialized to 1. **Once per epoch, before the policy update:** one Adam step (lr 5e-2) on −p · (mean EpCost − 25). Objective: (surr_adv − penalty · surr_cost) / (1 + penalty), with surr_cost = mean(ratio · A_c) unclipped |
| Unconstrained PPO | same, with no cost term in the objective |

**Differences we cannot remove.** The paper used `safety_gym` with mujoco-py (MuJoCo 2.0) and TF1 with MPI. Here it is Safety-Gymnasium 1.0.0 (a reimplementation) with MuJoCo 2.3.3 and PyTorch. Expect agreement in trends and rough magnitude, not exact numbers.

**Implementation:** a separate script, `baseline_ppo.py`, that mirrors the reference code and does not modify `train.py`. It runs 3 seeds × {ppo, ppo_lagrangian} at 10M steps each.

**Compare against the paper** on its three metrics for PointGoal2:

- final average episode return J_r
- final average episode cost J_c (the target is ≤ 25 for PPO-Lagrangian)
- cost rate ρ_c: cumulative training cost / total steps

Plot the learning curves against the paper's figures and table.

**Success criterion:** PPO-Lagrangian's final episode cost lands near 25 while PPO's stays far above it, with returns in the paper's range for each. If the numbers are far off, find out why before the main experiment.

## 6. Status and handoff

### Done

- [x] Environment built and versions recorded (§2).
- [x] `safety_adapter.py`; `train.py` takes `--env safety --budget-hazard --budget-vase`, saves step-named checkpoints, logs advantage scale, and uses fixed eval layouts.
- [x] `validate_safety.py --video` passes. A scripted controller forces hazard and vase contact because random rollouts never touch vases. 93 steps with vase contact and motion together each counted as 1. A missing field raises. The video renders headless. Output is in `runs/validation/`.
- [x] `python -m unittest test_core`: 7/7, including the multiplier cases. (1.5, 0.5) raises only λ_hazard and leaves the aggregate exactly on target; (0.5, 0.5) lowers every λ.
- [x] Training smokes on CPU, on CUDA, and on the toy env; throughput measured (§2).
- [x] Cleanup: removed the toy-only scripts, `run_suite.py`, empty files, and the old lockfile. Docs are in `docs/`.
- [x] Reference hyperparameters extracted (§5).

### Next

- [x] **R1:** `baseline_ppo.py` written following §5 and smoke tested for 2 epochs. Env stepping uses 4 worker processes per run; each epoch is still 30 × 1000-step episodes.
- [x] **R2:** throughput is ~37 s/epoch alone and ~50 s/epoch with 6 runs concurrent, so each run takes **~4.6 h**.
- [ ] **R3 (running since 2026-10-01 00:44 KST, ETA ~05:30 KST):** `./run_paper_repro.sh` started PPO and PPO-Lagrangian × seeds 0, 1, 2 at 10M steps, writing to `runs/paper-repro/<algo>-s<seed>/`.
  - Monitor: `tail -n1 runs/paper-repro/*.log | cut -c1-200`
  - Compare: `python compare_paper.py` writes `paper-comparison.{md,png,json}`. It works mid-run.
  - Paper targets, read off the end of Fig. 7 (approximate; crop saved at `docs/reference/safetygym-fig7-pointgoal2.png`):

    | | EpRet | EpCost | CostRate |
    |---|---|---|---|
    | PPO | ≈ 22.5 | ≈ 200 | ≈ 0.20 |
    | PPO-Lagrangian | ≈ 1 | ≈ 30 | ≈ 0.037 |

    The paper's PPO-Lagrangian barely learns the task on PointGoal2; that is the published result, not a bug.
  - Early observation: the policy update usually stops after 1–2 of 80 iterations (KL > 0.012). Our code follows the reference: unnormalized observations into a 256-wide layer make one Adam step already exceed the KL limit. If PPO's final return falls far short of ≈ 22, check this first.
  - If results match, continue with Stage 1. If not, diagnose before the main experiment.
- [ ] **Stage 1** (~1–3 h): unconstrained development pilot with seed 100, 500k steps:

  ```bash
  python train.py --env safety --method unconstrained --seed 100 --budget-hazard 1 --budget-vase 1 \
    --updates 125 --dual-lr 0.02 --eval-every 25 --checkpoint-every 25 --eval-episodes 10 \
    --final-eval-episodes 50 --wandb-mode offline --group dev --output runs/dev/unconstrained-s100
  python evaluate.py runs/dev/unconstrained-s100/checkpoint.pt --episodes 30 --environment-seed 500000 \
    --output runs/dev/unconstrained-s100/calibration.npz
  ```

  - **Learning check:** goals/episode must rise and a video must look sensible. If PPO doesn't learn the task, fix that first.
  - **Budgets:** `d_i = max(1, 0.5 × unconstrained mean C_i)` over 30 layouts. If a cost is almost always 0, it is not an active constraint; investigate.
  - **Short constrained pilots:** `aggregate_loose` and `multi` with seed 100 and `--updates 50`.
  - **One adjustment allowed:** budgets to 0.75 × mean, **or** multiplier learning rate 0.005 if multipliers oscillate. Not both.
  - **Then freeze.**
- [ ] **Stage 2** (~4 h unattended): 3 methods × seeds 0/1/2, 2M steps each, all 9 concurrent:

  ```bash
  nohup python run_pilot.py --group sg-pilot-v1 --budget-hazard <D_H> --budget-vase <D_V> \
    --transitions 2000000 > runs/sg-pilot-v1.out 2>&1 &     # --dry-run to preview
  ```

  - Eval on 10 fixed layouts and save a checkpoint every 100k steps.
  - Nothing changes after seeing results; a bug fix means a new group and a full rerun.
  - Write the semester proposal while it runs.
- [ ] **Stage 3:** held-out evaluation of the final checkpoint of each run on the same 50 layouts:

  ```bash
  for d in runs/sg-pilot-v1/*/; do python evaluate.py $d/checkpoint.pt --episodes 50 \
    --environment-seed 900000 --action-seed 900001 --output $d/heldout.npz > $d/heldout.json; done
  ```

  - Use the stochastic policy. Deterministic runs are only for labeled videos.
  - **Analysis script still to write:** learning curves with budget lines, a cost-space scatter showing the "both ≤ 1" square and the x + y = 2 line, and multiplier trajectories.
  - Per-seed table: `method | seed | steps | return | goals/ep | C_h/d_h | C_v/d_v | both within?`, plus exceedance rates and cumulative training cost.
  - A 20–40 s video of all three methods on the same layout.
- [ ] **Stage 4:** meeting package.
  - Materials: one-page brief, figures, table, video, and the 2–3 page semester proposal.
  - Meeting outline:
    1. why separate safety concerns deserve separate measurement
    2. what was built and verified
    3. what the pilot shows
    4. limitations: different feasible sets, 3 seeds, short training, budgets calibrated from one pilot
    5. semester plan: single-critic ablation, budget/weight sweeps, 5 seeds, a second task, then humanoid if justified

### Open decisions

- Main run length: 2M (default) vs. 1M.
- W&B logging: online vs. offline (default).

## 7. Interpreting outcomes

| Observation | Defensible reading |
|---|---|
| Aggregate passes its sum but breaks one budget | The shared budget hid a violation; this is the motivating failure case, not proof that vector optimization is better |
| `multi` meets both, lower return | Expected price of stricter constraints |
| `multi` meets both, higher return | Promising; check critic effects and seed variance before generalizing |
| Methods look the same | One constraint may be inactive or the costs correlated; still a valid result |
| Multipliers grow, costs stay high | Infeasible budget, too little training, or a bug; diagnose, don't guess |
| PPO doesn't reach goals | Present pipeline progress; fix task learning first |

With 3 seeds, report per-seed results and make no significance claims.
