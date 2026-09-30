# Recent work: multi-cost safe RL for HumanoidBench

Literature checked **20 September 2026**. This is a focused briefing, not an exhaustive systematic review. Links point to papers, official proceedings, or author-maintained sources. Publication status is stated explicitly; an arXiv identifier alone does not establish peer review. The proposed study and recommendations below are our synthesis, not results already established on HumanoidBench.

## 1. What is established, and what could be new?

**One task reward plus several separately constrained costs is an established multi-constraint constrained Markov decision process (CMDP).** Even the Safety Gym report linked in the original request defines a feasible policy set using costs \(c_1,\ldots,c_k\) and individual limits in Section 3.1, Eq. (2). Its benchmark experiments then specialize to one aggregate cost constraint. Thus, “replace one scalar cost with a cost vector” is a sensible implementation project, but is not by itself a new algorithmic contribution. [Ray, Achiam, and Amodei, 2019, §§3.1 and 5.1](https://cdn.openai.com/safexp-short.pdf)

The professor's performance intuition is a useful **empirical hypothesis**, with one mathematical qualification. Adding separate constraints to an otherwise unchanged problem cannot enlarge its feasible policy set. Compared with a fixed scalar penalty, however, separate budgets can allow *selective* pressure: relax a satisfied energy constraint while addressing a violated collision constraint. A crude scalar penalty may keep penalizing both. Improvements can therefore come from the specification, adaptive multipliers, gradient interference, or critic estimation—not from automatically increasing the dimension or freedom of the policy.

For positive weights \(w_i\), all individual constraints imply the aggregate constraint:

\[
\left(\forall i,\ J_{C_i}(\pi)\le d_i\right)
\implies
\sum_i w_iJ_{C_i}(\pi)\le\sum_i w_id_i.
\]

The converse fails: one violated cost can be offset by slack in another. Hence a higher-return aggregate-cost baseline might simply be less safe under the actual per-cost requirements. A credible contribution would test **when semantic decomposition improves return at matched per-cost feasibility, and when additional constraints instead harm optimization**. See [background.md](background.md) for the derivation and [exp-setup.md](exp-setup.md) for the experimental design.

## 2. Most relevant recent papers

### Gradient Shaping / GradS — the closest algorithmic prior art

**Yihang Yao, Zuxin Liu, Zhepeng Cen, Peide Huang, Tingnan Zhang, Wenhao Yu, and Ding Zhao. “Gradient shaping for multi-constraint safe reinforcement learning.” L4DC 2024, PMLR 242:25–39.** Initial preprint December 2023, arXiv:2312.15127. [Official proceedings](https://proceedings.mlr.press/v242/yao24a.html) · [paper](https://proceedings.mlr.press/v242/yao24a/yao24a.pdf)

GradS explicitly studies **redundant and conflicting constraint gradients** and modifies Lagrangian-based safe RL updates using a multi-objective optimization perspective. It reports improvements in reward, constraint satisfaction, and scaling with the number of constraints. This directly overlaps an explanation that decomposing safety changes the useful policy-update directions; it must be cited when positioning that claim.

**Implication for this project:** vector PPO-Lagrangian is the simple baseline, while GradS is an appropriate subsequent comparison. Log cosine similarities between cost gradients, constraint violations, and active multipliers. Include duplicated-cost and conflicting-cost controls: merely increasing the vector length should not be confused with supplying useful information. These are proposed diagnostics, not claims that GradS has already validated HumanoidBench.

### Safe and Balanced — multiple objectives as well as constraints

**Shangding Gu, Bilgehan Sel, Yuhao Ding, Lu Wang, Qingwei Lin, Alois Knoll, and Ming Jin. “Safe and Balanced: A Framework for Constrained Multi-Objective Reinforcement Learning.” IEEE Transactions on Pattern Analysis and Machine Intelligence 47(5):3322–3331, 2025. DOI:10.1109/TPAMI.2025.3528944.** Preprint May 2024, arXiv:2405.16390. [Paper](https://arxiv.org/abs/2405.16390) · [author-institution publication record](https://portal.fis.tum.de/de/publications/safe-and-balanced-a-framework-for-constrained-multi-objective-rei-2/) · [DOI](https://doi.org/10.1109/TPAMI.2025.3528944)

This work handles **multiple reward objectives under safety constraints**, using natural-gradient manipulation and corrective optimization when constraints are violated. Its preprint states convergence and constraint-violation guarantees for the tabular setting. This is relevant to gradient conflicts, but broader than the proposed one-reward/multiple-cost problem.

**Implication:** do not call a vector of safety costs “multi-objective RL” without explaining the distinction. Walking speed can remain the objective, while joint limits and contact events are constraints. If energy is instead treated as a competing objective, the scientific question changes to selecting reward trade-offs or a Pareto frontier. Neural humanoid policies also do not inherit tabular guarantees simply by implementing the update.

### Multi-constraint control barrier functions — a different safety mechanism

**Chenggang Wang, Xinyi Wang, Yutong Dong, Lei Song, and Xinping Guan. “Multi-Constraint Safe Reinforcement Learning via Closed-form Solution for Log-Sum-Exp Approximation of Control Barrier Functions.” L4DC 2025, PMLR 283:698–710.** arXiv:2505.00671. [Official proceedings](https://proceedings.mlr.press/v283/wang25c.html) · [paper](https://raw.githubusercontent.com/mlresearch/v283/main/assets/wang25c/wang25c.pdf)

The authors combine multiple barrier constraints into a smooth composite control barrier function and derive a closed-form safety-layer solution to reduce differentiable-optimization overhead. The formulation concerns safety for control-affine systems, rather than simply bounding expected episodic cost with multipliers.

**Implication:** this is adjacent work on handling many constraints, not an interchangeable PPO-Lagrangian baseline. Its aggregation approximates a conjunction of safety conditions; this differs from allowing costs to compensate in a weighted sum. A humanoid safety filter would require appropriate dynamics, barrier functions, feasibility conditions, and treatment of contact. Do not claim that a generic learned cost critic offers the same guarantee. A filter could be a separate later experiment.

### Constraint decomposition for human following — recent direct empirical overlap

**Shiting Gong, Jianpeng Yao, Jinfeng Wang, Marco Pavone, and Jiachen Li. “Navigating the Proximity-Safety Balance: Constraint Decomposition for Human Following in Pedestrian Crowds.” 2026, arXiv:2608.10056v1, submitted 10 August 2026.** The author-supplied arXiv comments identify IROS 2026; a proceedings DOI was not verified here. [arXiv record](https://arxiv.org/abs/2608.10056) · [full text](https://arxiv.org/html/2608.10056v1)

This paper decomposes human following into a sparse task reward and independent constraints for following distance, human safety, and obstacle safety. It also incorporates uncertainty in human-motion prediction, and reports evaluation under distribution shifts and real-robot deployment. The direct overlap is the use of behaviorally meaningful thresholds instead of opaque reward-weight ratios.

**Implication:** “decomposition improves interpretability and tunability” already has very recent robotics precedent. HumanoidBench could extend this question to whole-body dynamics, actuator demand, and contact, with controlled aggregation/representation ablations. The navigation paper does not establish that decomposition will improve humanoid locomotion, so avoid transferring its empirical conclusions without testing.

### Dedicated critics — a 2026 item to read before claiming architectural novelty

**Yue Yang, Chenghao Huang, and Hao Wang. “Why Dedicated Critics: Eliminating Target Drift in Multi-Constraint RL.” 2026.** The authors' lab lists this as ICML 2026. [Author-maintained research listing](https://www.maincode.com/research) · [OpenReview record](https://openreview.net/forum?id=fgUo5WbsYj)

The lab's summary reports analysis of bias induced by changing multipliers when a critic learns a mixed penalty, contrasted with separate reward/constraint critics, and experiments in a multi-constraint power-system environment. **Verification limit:** the author listing was accessible, but the OpenReview full text could not be retrieved during this review. Treat the mechanism as author-reported; no theorem assumptions, quantitative gains, proceedings pagination, or detailed implementation are asserted here.

**Implication:** preserve individual cost targets and combine their advantages only during the policy update. A shared network with separate outputs and fully independent critics are different architecture choices; the accessible summary does not justify claiming that shared trunks are inherently invalid. Include an architecture-matched control before attributing gains to the number of costs.

### Uniformly Safe RL / Objective Suppression — stronger safety semantics

**Zihan Zhou, Jonathan Booher, Khashayar Rohanimanesh, Wei Liu, Aleksandr Petiushko, and Animesh Garg. “Uniformly Safe RL with Objective Suppression for Multi-Constraint Safety-Critical Applications.” 2024, arXiv:2402.15650; revised 28 August 2024.** [Paper record](https://arxiv.org/abs/2402.15650)

The paper introduces a uniformly constrained MDP formulation that places constraints on all reachable states, and suppresses task-reward optimization according to a safety critic. It evaluates multi-constraint domains including autonomous driving. This briefing cites the 2024 preprint; an earlier related workshop version should not be confused with an independently verified main-conference publication of this exact version.

**Implication:** small average cost can conceal unsafe rare states. Report fall probability and severe-event statistics alongside mean cost. Expected-cost PPO-Lagrangian remains a useful starting point, but the word “safe” should always be accompanied by the actual mathematical condition being measured.

## 3. Foundations that determine the baselines

### CPO

**Joshua Achiam, David Held, Aviv Tamar, and Pieter Abbeel. “Constrained Policy Optimization.” ICML 2017, PMLR 70:22–31.** arXiv:1705.10528. [Official proceedings](https://proceedings.mlr.press/v70/achiam17a.html)

CPO constructs trust-region policy updates with explicit constraints and theoretical bounds underlying the approximate update. It is the foundational reference for optimizing performance subject to safety limits, rather than tuning a fixed penalty. Its practical approximations, finite samples, and neural function approximation matter; “CPO” is not a blanket guarantee of zero unsafe actions. For a small internship study, its implementation complexity makes it a useful later baseline rather than a prerequisite to the first working experiment.

### Safety Gym and PPO-Lagrangian

**Alex Ray, Joshua Achiam, and Dario Amodei. “Benchmarking Safe Exploration in Deep Reinforcement Learning.” OpenAI technical report, 2019.** [The exact report supplied by the professor](https://cdn.openai.com/safexp-short.pdf)

Section 5.2 benchmarks PPO-Lagrangian and TRPO-Lagrangian alongside unconstrained methods and CPO. Its evaluation distinguishes final reward, final constraint violation, and costs incurred throughout learning. This is the source to cite for the baseline convention in this project, without treating PPO-Lagrangian as the title of a separate original PPO paper.

**Project consequence:** log reward and every cost during both training and held-out evaluation. A feasible final checkpoint does not erase expensive or unsafe exploration earlier in the run. The report's particular cost limit is tied to its environment and return convention; do not transplant that number unchanged into HumanoidBench.

### PID Lagrangian

**Adam Stooke, Joshua Achiam, and Pieter Abbeel. “Responsive Safety in Reinforcement Learning by PID Lagrangian Methods.” ICML 2020, PMLR 119:9133–9143.** [Official proceedings](https://proceedings.mlr.press/v119/stooke20a.html)

This paper interprets the usual multiplier update through control theory and adds proportional/derivative components to improve responsiveness and reduce oscillations. It makes a strong practical comparison when vanilla multiplier learning repeatedly overshoots its budget.

**Project consequence:** establish the ordinary vector Lagrangian baseline first. Then, if needed, apply a controller per constraint, documenting gains and smoothing. That vector extension is an implementation choice here; the paper is not proof that independently tuned controllers will solve every interaction among many constraints.

### SDAC — multi-constraint locomotion already exists

**Dohyeong Kim, Kyungjae Lee, and Songhwai Oh. “Trust Region-Based Safe Distributional Reinforcement Learning for Multiple Constraints.” NeurIPS 2023, Advances in Neural Information Processing Systems 36.** arXiv:2301.10923. [Official proceedings](https://papers.nips.cc/paper/2023/hash/3f20f2b0315c72201e23512fdbd1ee91-Abstract-Conference.html) · [official code](https://github.com/rllab-snu/Safe-Distributional-Actor-Critic)

SDAC combines gradient integration for multiple constraints with distributional critics for risk-averse safety measures. Its evaluation includes multi-constraint locomotion, so neither “multiple robot safety costs” nor “multiple constraints for legged locomotion” is an adequate novelty claim. The official repository includes Cheetah, Laikago, and Cassie experiments and optional W&B logging.

**Project consequence:** consult its cost design and feasibility-recovery ideas before inventing new terminology. Match the risk measure when comparing algorithms: an expectation constraint and a tail-risk constraint are different problems, even when they share the same immediate cost signal.

## 4. The two suggested benchmarks fit different roles

**HumanoidBench:** Carmelo Sferrazza, Dun-Ming Huang, Xingyu Lin, Youngwoon Lee, and Pieter Abbeel. “HumanoidBench: Simulated Humanoid Benchmark for Whole-Body Locomotion and Manipulation.” Robotics: Science and Systems, 2024. arXiv:2403.10506. [Official RSS paper](https://www.roboticsproceedings.org/rss20/p061.pdf) · [project](https://humanoid-bench.github.io/) · [code](https://github.com/carlosferrazza/humanoid-bench)

HumanoidBench supplies challenging MuJoCo whole-body tasks, with proprioceptive, visual, and tactile observation options. The original work reports difficulty for flat RL and benefits from hierarchical structure supported by capable low-level skills. It is a **control benchmark**, while the proposed semantic safety taxonomy, budgets, and evaluation protocol are additions we must define and validate. Start from locomotion and state observations so that learning the base task does not overwhelm the safety comparison.

**Safety-Gymnasium:** Jiaming Ji, Borong Zhang, Jiayi Zhou, Xuehai Pan, Weidong Huang, Ruiyang Sun, Yiran Geng, Yifan Zhong, Josef Dai, and Yaodong Yang. “Safety Gymnasium: A Unified Safe Reinforcement Learning Benchmark.” NeurIPS 2023, Datasets and Benchmarks Track. arXiv:2310.12567. [Official proceedings](https://papers.nips.cc/paper/2023/hash/3c557a3d6a48cc99444f85e924c66753-Abstract-Datasets_and_Benchmarks.html) · [official code](https://github.com/PKU-Alignment/safety-gymnasium)

This is a later benchmark in the Safety Gym line, with safety-oriented tasks and a Safe Policy Optimization library. It is useful for validating the learner before expensive humanoid studies. It is not the same environment as the 2019 report, and a modern benchmark's presence of several hazard types does not automatically mean the learner receives a cost vector; inspect the environment API and preserve each channel explicitly.

## 5. A defensible internship contribution

The following are **research proposals based on the reviewed literature**, not claims of proven novelty:

1. **Cost semantics in whole-body control.** Build and document channels that correspond to distinguishable failure mechanisms, for example unsafe body contact, joint-limit proximity, and actuator demand. Audit whether a channel is a physical risk indicator, a proxy, or merely a motion-quality preference. Validate it by inspecting trajectories.
2. **A fair decomposition experiment.** Compare unconstrained PPO, a fixed weighted penalty, one aggregate constraint, and separate constraints using the same reward, trajectories, action space, compute budget, and evaluation seeds. Evaluate every method with all original per-cost budgets. Tune scalar weights enough that the scalar baseline is credible.
3. **Disentangle representation from constraints.** Hold critic capacity approximately constant; compare scalar and vector targets, shared-trunk and independent critics, and matched controller settings. “More heads won” is otherwise ambiguous.
4. **Test meaningful versus artificial decomposition.** Contrast semantic channels with duplicate or randomly grouped channels. Add correlated and conflicting costs. GradS and SDAC make gradient interactions a particularly relevant explanation to investigate.
5. **Measure the feasible frontier.** Sweep budgets and plot reward against individual violations, all-constraints-feasible rate, and training violations. Report multiple seeds and uncertainty. Never choose the highest-return checkpoint without also checking all constraints on separate evaluation episodes.
6. **Keep the scope honest.** A toy example establishes that the implementation can learn and exposes an aggregation failure. A short HumanoidBench run establishes integration and training behavior. Neither alone establishes a new method, solved locomotion, real-robot safety, or statistically reliable superiority.

A practical reading order is **Safety Gym → PPO-Lagrangian/PID → GradS → SDAC → the 2026 decomposition paper**, followed by dedicated-critic work if architecture becomes the focus. The control-barrier-function and uniformly constrained formulations become necessary when the research question shifts from expected budgets toward stronger statewise or runtime safety.
