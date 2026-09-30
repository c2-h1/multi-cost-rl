# Multi-cost safe RL on HumanoidBench: experimental setup

Prepared 2026-09-20. This document specifies a research experiment; measured outcomes are recorded separately in `results/` and the run summaries. `background.md` develops the mathematics in Korean; `recent-work.md` surveys prior work.

## 1. What is worth testing?

**Yes: start locally with the real, hands-free HumanoidBench walking environment, then scale a controlled comparison on the cluster.** Use a small navigation task to debug the multi-cost optimizer first. The implemented `humanoid_costs.py` adds three explicit cost channels to genuine HumanoidBench physics, so the project is not limited to a toy.

The professor's first intuition is a useful hypothesis: distinguishing failure modes can prevent a scalar average from hiding a serious violation, and can give the optimizer better credit assignment. The second needs a qualification:

> 비용별로 독립적인 multiplier를 두면 **안전 요구사항을 조정하는 자유도**는 늘어난다. 그러나 같은 policy architecture의 표현력이 늘어나는 것은 아니며, 제약을 추가하면 일반적으로 feasible policy set은 작아진다. 성능 향상은 실험으로 확인할 가설이다.

Let $x_i=J_{c_i}/d_i\ge0$ be each expected episode cost divided by its budget. Three specifications have different feasible sets:

$$
\underbrace{\{\pi:\sum_i x_i\le1\}}_{\text{conservative aggregate}}
\subseteq
\underbrace{\{\pi:x_i\le1\ \forall i\}}_{\text{vector constraints}}
\subseteq
\underbrace{\{\pi:\sum_i x_i\le m\}}_{\text{loose aggregate}}.
$$

Thus, beating a conservative scalar constraint can simply mean allowing a larger feasible set. Beating the loose aggregate on safety can simply mean forbidding compensation between costs. Both are useful specification results, but neither alone demonstrates a superior optimization algorithm. The central question is:

**At the same separately audited safety budgets, does independent adaptation of cost multipliers improve feasible return, sample efficiency, or robustness compared with a well-tuned scalar baseline?**

This is a multi-constraint CMDP with one task reward. The supplied Safety Gym report already defines a set of cost functions and budgets, and recommends evaluating task return, final constraint satisfaction, and costs incurred during learning. Multiple costs are therefore an established formulation; the contribution should concern learning behavior or the humanoid application. [Ray, Achiam & Amodei, 2019, §§3.1, 5](https://cdn.openai.com/safexp-short.pdf).

## 2. Task and implementation provenance

Pin the upstream repository at **`cb1189039151c8aadaaa987b442da54383c87fab`**. It is checked out under `external/humanoid-bench/`.

| Stage | Task | Purpose |
|---|---|---|
| Optimizer debugging | Local `TwoCostNavigation` | Separate hazard contact and control effort; cheap replicated learning curves |
| First physical pilot | `h1-walk-v0` | Genuine H1 MuJoCo locomotion without dexterous hands; locally verified observation size 51, action size 19 |
| Main experiment | `h1hand-walk-v0` | Published benchmark morphology; repeat every baseline under the same cost wrapper |
| Transfer | `h1hand-run-v0`, then `h1hand-hurdle-v0` | Faster motion and obstacle interaction; recalibrate task-dependent specifications before freezing them |

The official repository registers the H1 walking tasks and provides a CPU MuJoCo PPO route. The project includes more complex manipulation tasks, but they are unnecessary for the first hypothesis test. [Official repository README](https://github.com/carlosferrazza/humanoid-bench/blob/cb1189039151c8aadaaa987b442da54383c87fab/README.md), [registration source](https://github.com/carlosferrazza/humanoid-bench/blob/cb1189039151c8aadaaa987b442da54383c87fab/humanoid_bench/__init__.py), [Sferrazza et al., 2024](https://arxiv.org/abs/2403.10506).

The walking reward is the product of standing/upright, small-control, and movement terms. The H1 walking speed target is 1 m/s; termination occurs when root height `data.qpos[2] < 0.2`. Existing `info` keys such as `small_control` and `stand_reward` are **reward components**, not supplied constrained-RL costs. Keep the upstream reward unchanged in the main comparison, and report forward speed and falls separately so that shaped return is interpretable. [Pinned walking implementation, lines 12–97](https://github.com/carlosferrazza/humanoid-bench/blob/cb1189039151c8aadaaa987b442da54383c87fab/humanoid_bench/envs/basic_locomotion_envs.py#L12-L97).

The default task uses 1,000 control steps, frame skip 10, and a 0.002-second physics timestep for H1 walking: 0.02 seconds per control step. The local physical pilot uses an explicitly different 256-step horizon (5.12 seconds). Its returns must not be compared directly with the official 1,000-step benchmark success threshold. [Task base class](https://github.com/carlosferrazza/humanoid-bench/blob/cb1189039151c8aadaaa987b442da54383c87fab/humanoid_bench/tasks.py), [walking XML](https://github.com/carlosferrazza/humanoid-bench/blob/cb1189039151c8aadaaa987b442da54383c87fab/humanoid_bench/assets/envs/h1_pos_walk.xml).

## 3. Three costs to implement first

Use three distinct failure mechanisms. More channels can follow after the first three demonstrably work; start with enough separation to test the idea without a large tuning problem.

### C1. Loss of upright posture / fall-risk exposure

$$
c_t^{\rm fall}=\mathbf1\{h_{\rm head,t}<1.35\ \lor\ z_{\rm torso,t}^{\top}z_{\rm world}<\cos45^\circ\ \lor\ \mathrm{terminated}_t\}.
$$

Read `env.unwrapped.robot.head_height()` and `.torso_upright()`. They respectively read the `head` site height and the torso/world vertical-axis projection. This detects substantial posture loss earlier than the upstream root-height termination. It is a **fall-risk proxy**, not an estimate of fall probability. [Robot accessors](https://github.com/carlosferrazza/humanoid-bench/blob/cb1189039151c8aadaaa987b442da54383c87fab/humanoid_bench/robots.py#L1-L47).

Initial 1.35 m and 45° thresholds distinguish upright H1 walking from a large collapse/tilt. They are provisional research choices, not published safety limits. A crouching or crawling task needs a different definition. Always log actual physical termination/fall frequency, minimum head height, and maximum torso tilt as well.

### C2. Joint-stop proximity

For robot hinge/slide joints with finite model limits, define fractional distance to the closest stop:

$$
\rho_{j,t}=\frac{\min(q_{j,t}-q_j^{\min},\ q_j^{\max}-q_{j,t})}{q_j^{\max}-q_j^{\min}},\qquad
c_t^{\rm joint}=\frac1{|\mathcal J|}\sum_{j\in\mathcal J}\mathbf1\{\rho_{j,t}<0.05\}.
$$

Use `model.jnt_type`, `jnt_limited`, `jnt_range`, and `jnt_qposadr`; exclude the free root and object joints. Do not assume that a joint index equals its `qpos` address. This channel measures the fraction of robot joints in the outer 5% of their modeled range. The wrapper uses 19 limited H1 joints in the first task. Joint ranges come from the simulator model; they do not certify a real joint's injury or damage threshold. [Pinned H1 model](https://github.com/carlosferrazza/humanoid-bench/blob/cb1189039151c8aadaaa987b442da54383c87fab/humanoid_bench/assets/robots/h1_pos.xml).

Because an average can hide one repeatedly stressed joint, audit `min_joint_margin_fraction`, per-joint histograms, and the probability that **any** joint crosses its margin. A later ablation can constrain that any-joint indicator instead; do not change this aggregation halfway through a comparison.

### C3. Actuator load

$$
c_t^{\rm load}=\frac1{n_u}\sum_{a=1}^{n_u}
\left(\frac{f_{a,t}}{f_a^{\max}}\right)^2,
\qquad
f_a^{\max}=\max(|f_a^{\min}|,|f_a^{\max,\rm XML}|).
$$

Read actual `data.actuator_force`, normalized by the enforced `model.actuator_forcerange`. The H1 position-controller model provides actuator force limits; the wrapper rejects a model with missing/unenforced limits. **The normalized policy action is a position command, so $\|a\|^2$ is not motor torque or energy in this environment.** The implemented load channel is an RMS-type actuator utilization proxy, not a thermal model. The same H1 XML specifies the position servos and their force limits. [Controller/model source](https://github.com/carlosferrazza/humanoid-bench/blob/cb1189039151c8aadaaa987b442da54383c87fab/humanoid_bench/assets/robots/h1_pos.xml#L14-L35).

Also record absolute mechanical power $P_t=\sum_a|f_{a,t}v_{a,t}^{\rm actuator}|$ using `actuator_velocity`, and approximate mechanical work $\sum_tP_t\Delta t$. This is neither electrical battery energy nor signed net work. MuJoCo distinguishes actuator-space force/velocity from generalized joint force/velocity. [MuJoCo 3.1.6 data definitions](https://mujoco.readthedocs.io/en/3.1.6/APIreference/APItypes.html#mjdata).

### Costs to add later

| Candidate | Observable implementation | Why defer it? |
|---|---|---|
| Non-foot body impact | Iterate `data.contact[:data.ncon]`; map `geom1/geom2` through `model.geom_bodyid`; reject allowed foot-floor contacts; obtain normal force with `mujoco.mj_contactForce` | Must define allowed contact pairs per task, avoid double counting, and sample physics substeps |
| Excessive foot-ground impact | Normal contact impulse $\sum_k F_{n,k}\Delta t_{\rm physics}$, plus per-contact peak | A single control-step endpoint can miss a short impact |
| Foot slip | Tangential foot/site velocity while that foot is in contact | Requires a robust contact condition; foot motion during swing is legitimate |
| Action/command jerk | Difference of position commands divided by control time; include previous command in state | Smoothness is not automatically a safety requirement; commands differ from physical acceleration |
| Object damage / human contact | Task-specific protected geom contacts, force/impulse and spatial zones | Flat-ground walking contains no modeled humans or fragile task objects |

`mj_contactForce` returns a contact-frame force/torque vector. Resolve contact frames and aggregate over the 10 physics substeps if impact safety becomes an actual constraint. [MuJoCo 3.1.6 contact-force API](https://mujoco.readthedocs.io/en/3.1.6/APIreference/APIfunctions.html#mj-contactforce). The current wrapper samples endpoint signals, and makes no claim about between-sample peaks. Do not add a generic “collision cost” penalizing normal foot support.

## 4. Freeze definitions and calibrate budgets before the comparison

Use a finite-horizon, undiscounted definition consistently:

$$
J_{c_i}(\pi)=\mathbb E_\pi\left[\sum_{t=0}^{H-1}c_{i,t}\right],\qquad
d_i=H b_i,\qquad \widetilde c_{i,t}=c_{i,t}/d_i.
$$

The executable H1 pilot starts at budget rates

$$
(b_{\rm fall},b_{\rm joint},b_{\rm load})=(0.05,0.05,0.20).
$$

At $H=256$, raw episode budgets are $(12.8,12.8,51.2)$; at $H=1000$, they would be $(50,50,200)$. These numbers express a provisional operating specification: at most 5% of scheduled control time in posture risk, 5% joint-margin exposure averaged over joints/time, and mean squared normalized actuator load at most 0.2. **Their joint feasibility for successful walking is not established by choosing them.** Satisfying these expectations would still permit some failures, including a fall near the episode end.

Before the main study:

1. Collect reference trajectories from reset-pose control, random actions, an unconstrained learned walker, and a deliberately conservative walker, using separate calibration seeds. Validate each cost with state injections and inspect video. A stationary controller only demonstrates a budget is attainable without walking.
2. Inspect distributions, correlations, saturation, and rare events for all channels. Locate successful, reasonably conservative trajectories and record their operating envelope. If model limits are the only available physical basis, call the resulting thresholds simulator specifications.
3. Choose a nonempty, nontrivial budget region: at least one useful reference behavior should satisfy it, and a high-return reference should violate at least one channel. If neither exists, report infeasibility or a trivial constraint and revise the pilot before freezing it. Avoid choosing budgets after seeing which method wins.
4. Freeze cost code, normalization, thresholds, and budgets for all methods. Predeclare loose/nominal/tight operating points, for example $1.25b,b,0.75b$, restricted to the calibration-supported range. A changed threshold changes the problem, so version it.

Finite-horizon handling matters. On physical termination, the local trainer pads the remaining scheduled horizon with zero reward and costs $(1,0,0)$, and exposes time plus an absorbing-state flag. This prevents early death from escaping the **fall-risk** budget; it does not prevent reducing accumulated joint/load cost by falling, which is why all constraints and actual fall rates must be evaluated together. The wrapper itself preserves native reward and termination; padding belongs to the adapter. Report both scheduled transitions and actual MuJoCo transitions.

At the intrinsic horizon, value bootstrap is zero. A mere data-collection cutoff inside an ongoing episode must instead bootstrap from the next value. Do not mix discounted critic targets with undiscounted budget estimates without explicitly deriving that surrogate choice.

## 5. Comparison matrix and what each result would mean

All methods use the same actor architecture, observations, action distribution, reward, training steps, initialization seeds, optimizer schedule, and reward/cost critic supervision. Even the scalar and unconstrained controls retain the same cost heads in the executable pilot. This makes differences less likely to come from extra auxiliary tasks or network capacity.

| Method | Training objective/specification | Interpretation |
|---|---|---|
| Unconstrained PPO | Maximize original reward | Performance reference and check that safety is nontrivial |
| Fixed scalar penalty | $J_r-\sum_iw_iJ_{c_i}/d_i$; validation sweep of nonnegative $w_i$ | Necessary control for the claim that adaptive separate penalties help; planned cluster comparison |
| Scalar loose PPO-Lag | One multiplier, $\sum_ix_i\le m$ | Shows compensation between channels; larger feasible set |
| Scalar conservative PPO-Lag | One multiplier, $\sum_ix_i\le1$ | Sufficient for individual constraints with nonnegative costs; smaller feasible set |
| Vector PPO-Lag | Separate $\lambda_i\ge0$, each $x_i\le1$ | Main method: separate budgets and adaptive multipliers |
| Scalar feasible-envelope control | Tune scalar weights/thresholds on validation; keep only policies meeting every vector budget | Stronger practical comparison at matched measured feasibility |

The automated local suite implements the unconstrained, loose aggregate, conservative aggregate, and vector methods. The fixed-weight sweep and feasible-envelope study are planned work, not completed results.

For all methods, evaluate the **same cost vector**, even if training uses a scalar. Plot reward against worst normalized violation $\max_i(x_i-1)$, with a separate reward-versus-$x_i$ plot for every channel. Rank return only among policies satisfying every declared budget; otherwise report no feasible policy. State whether feasibility means the point estimate or a conservative upper-confidence-bound criterion.

For a stronger test of learning rather than specification, compare vector PPO-Lag with the best scalar feasible envelope at equal tuning budget. In a second control, keep the exact same scalar constraint and compare one cost critic against a decomposed multi-head estimator whose weighted sum feeds the **same one multiplier**. That isolates representation/credit-assignment effects from changing constraints. Separately adding constraints to an unconstrained problem cannot increase its optimal reward.

An exact algebraic scalar representation of vector feasibility is $\max_i(J_{c_i}/d_i-1)\le0$. It acts on expected returns. Replacing each transition by $\max_i c_{i,t}/d_i$ generally defines a stricter, different problem; do not call this an equivalent matched control.

## 6. Training protocol, compute stages, and stopping rules

The local PPO prototype uses a 64–64 tanh actor, a reward-plus-cost value network, tanh-bounded Gaussian actions, complete episode batches, $\gamma=1$, GAE parameter 0.95, PPO clip 0.2, and projected multiplier updates. It normalizes the combined advantage once, after mixing reward and normalized cost advantages; it does not separately standardize away cost units. Exact run parameters and changed pilot settings belong in each saved `config.json`.

1. **Tests and measured smoke:** reset/step the actual environment headlessly; inject low-head and joint-margin states; check actual actuator loads; verify the wrapper leaves reward, observations and termination unchanged. Five physics integration tests are provided in `test_humanoid_costs.py`.
2. **Toy learning:** complete replicated short runs to expose multiplier sign, cost shape, feasibility and logging errors. The local toy is custom 2-D navigation inspired by Safety Gym, not Safety Gym itself and not a humanoid model. It uses hazard occupancy and control effort, $H=64$, and budgets $(1.92,7.68)$. Toy results provide implementation evidence, not humanoid transfer evidence.
3. **H1 pilot:** start with 8 environments, $H=256$, and roughly 0.5–1 million scheduled transitions for each exploratory run. Measure actual simulation steps, training wall time, and validation behavior. A short run can validate learning infrastructure without solving locomotion.
   **Walking-competence gate:** the completed short pilot falls in all final evaluation episodes. Before the main safety comparison, establish a repeatable reward-only walking policy at the native horizon, inspect its gait, and calibrate costs on its trajectories. If using that policy to initialize safety training, initialize every competing method from the same checkpoints and report pretraining cost separately. Otherwise inability to walk will dominate the safety comparison.
4. **Main H1 comparison:** after calibration, restore $H=1000$. Allocate an initial 2-million-transition screening budget per method and a 10-million-transition main budget, extending every method equally to 20 million only if learning curves still improve and resources permit. These are proposed budgets, not claims that the task will be solved in that many steps.
5. **Published morphology / transfer:** repeat the selected configurations on `h1hand-walk-v0`, then add run/hurdle. Do not transfer a “safe” label solely because a cost formula still executes.

Use training seeds 0, 1, 2 for an engineering pilot; use at least 5 independently trained seeds (0–4) for the main study, preferably 10 when feasible. Keep calibration/tuning seeds separate from final evaluation seeds. Allocate equal hyperparameter trials, environment steps, and device-hours to each competing family, and disclose all failed runs. Early stopping may respond to NaNs, numerical instability, or a predefined compute cap, not unfavorable method rankings.

For scheduled evaluation, use at least 100 fresh episodes per checkpoint and 500 final episodes per trained seed where affordable. Report the stochastic policy that the CMDP describes, and deterministic-mean deployment behavior separately. Each training seed is the unit of replication for algorithmic comparisons; many evaluation episodes of one network are not many training seeds. Use paired seed plots and a bootstrap over training seeds for a 95% interval on mean differences. For a small pilot, label intervals exploratory. For safety claims, also report episode-level intervals and a simultaneous one-sided bound across the $m$ mean-cost estimates; finite samples cannot prove zero failure probability.

## 7. Required metrics and W&B layout

Use the existing project **[multi-cost-rl](https://wandb.ai/cch-eck-postech/multi-cost-rl)** for this session; the launcher saves the actual W&B run URL in each run directory. Suggested grouping: task, cost-specification version, budget level, horizon, and experiment phase; method and seed identify the run.

| Category | Quantities |
|---|---|
| Performance | Original episode return; forward velocity/displacement; episode survival; published success only at the published horizon |
| Safety | Every raw $J_{c_i}$; $J_{c_i}/d_i$; each violation; worst violation; joint feasibility; physical fall frequency; min joint margin and max load |
| Learning safety | Total accumulated raw cost per channel; cost per actual interaction; mean normalized violation over updates; first feasible checkpoint |
| Optimization | Every multiplier; policy/value losses; separate cost value errors; entropy; KL estimate; clip fraction; gradient norms |
| Compute | Scheduled transitions; actual physics/control steps; simulation and end-to-end steps/second; wall time; memory/device |
| Provenance | Code commit/hash; upstream SHA; environment/task/horizon; cost formulas and thresholds; seeds; budgets; full hyperparameters; dependency versions |

Retain all per-episode evaluation costs so feasibility can be re-audited. Upload final policy and best validation-feasible policy as separately named artifacts, plus configuration, normalization state if used, checkpoints, and a few fixed-seed videos. Select checkpoints on validation only; use an untouched final evaluation set. A low average aggregate cost must never replace per-channel reporting.

If online tracking is unavailable, run with `--wandb-mode offline` and keep local JSON/CSV logs; sync later. W&B supports offline mode via `WANDB_MODE=offline`. [W&B environment-variable reference](https://docs.wandb.ai/models/track/environment-variables). The current project has already verified online logging; credentials must remain outside saved configurations and logs.

## 8. Can this run locally or on a small GPU cluster?

**Local answer: yes for headless environment operation, implementation tests, the toy comparison, and short real-H1 training.** On this Apple M2 machine the actual `h1-walk-v0` reset/step smoke worked. A short 1,000-step random-action timing observed about 2,252 real control steps/s, including resets, before PPO overhead. This is one measured simulator smoke result, not a full training benchmark or a projection for the 69-actuator hands model. Use run summaries for measured end-to-end timings.

The environment physics in the current path runs on CPU. A small MLP may also train efficiently on CPU; CUDA on a Linux worker can accelerate neural-network updates but does not automatically move classical MuJoCo simulation to the GPU. Run independent method/seed jobs across cluster workers first. Give each job enough CPU workers and avoid CPU oversubscription; profile 1, 4, 8, and 16 environments before choosing parallelism.

The upstream repository has a separate MJX path for low-level reaching policies. Treat a GPU-vectorized walking implementation as additional engineering, not a flag that accelerates the current Gym environment. Verify reward, cost and termination parity against the CPU implementation before using it for a comparison. [Official MJX reaching example](https://github.com/carlosferrazza/humanoid-bench/blob/cb1189039151c8aadaaa987b442da54383c87fab/humanoid_bench/mjx/envs/reach_continual.py), [MJX PPO entry point](https://github.com/carlosferrazza/humanoid-bench/blob/cb1189039151c8aadaaa987b442da54383c87fab/humanoid_bench/mjx/ppo_continuous_action.py).

The full upstream installer pins a broad older dependency stack, including packages unnecessary for this PPO path. This workspace therefore uses an isolated `.venv-train` with a checked `requirements-lock.txt` and the pinned local source checkout; it does not install into the user's base environment. Rendering is disabled for training. [Upstream dependency pins](https://github.com/carlosferrazza/humanoid-bench/blob/cb1189039151c8aadaaa987b442da54383c87fab/setup.py).

No cluster hostname, scheduler, or credentials were supplied, so no remote jobs are claimed. Once the local pilot is profiled, estimate each worker's cost as

$$
\mathrm{hours}\approx\frac{\mathrm{real\ training\ transitions}}{\mathrm{measured\ end\!\!\;to\!\!\;end\ transitions/s}\times3600}
+\mathrm{evaluation\ hours},
$$

then multiply by configurations and training seeds and retain headroom for longer surviving episodes. Absorbing padding is cheap; a poorly performing policy can misleadingly appear to achieve high scheduled steps/second.

## 9. Reproduce the physical pilot

From this workspace, with the provided environment and upstream checkout:

```bash
PYTHONPATH=external/humanoid-bench .venv-train/bin/python -m unittest test_humanoid_costs -v

.venv-train/bin/python train.py \
  --env humanoid --humanoid-id h1-walk-v0 --method multi \
  --num-envs 8 --horizon 256 --updates 300 --seed 0 --dual-lr 0.02 \
  --eval-episodes 8 --final-eval-episodes 32 --eval-every 50 \
  --device cpu --wandb-mode online --project multi-cost-rl \
  --group humanoid-pilot --output runs/humanoid-reproduce-multi-s0
```

The adapter locates `external/humanoid-bench` automatically for training. These settings match the completed physical pilot, with a fresh output directory. Inspect [`training-report.md`](training-report.md) and the saved run configuration for the results. Passing a test or producing a checkpoint establishes that training ran, not that all safety budgets were met or the professor's hypothesis was confirmed.
