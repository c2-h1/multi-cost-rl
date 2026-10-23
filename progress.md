---
tags: [research, safe-rl, multi-cost]
source: "github.com/c2-h1/multi-cost-rl @ pi0-weights (ae0ed14)"
related: "[[260930]]"
updated: 2026-10-23
---

# Multi-cost Safe RL: SafetyPointGoal2-v0 (hazard + vase)
> [!summary] TL;DR
> - **질문**: safety cost를 하나로 합치지 않고 종류별(hazard, vase)로 나눠서 각각 제약을 걸면, 같은 safety 수준에서 return이 더 높아질까?
> - **방법**: PPO + Lagrangian(primal-dual). cost에 weight $w_i$를 곱해 단위를 맞춘 뒤, 하나의 $\lambda$(aggregate)와 cost별 $\lambda_i$(multi)를 비교.
> - **지금까지 결과**: 2M step 안에 수렴하지 않음. aggregate는 overshoot 때문에 over-constrained, multi는 아직 budget violation.
> - **이번 세션 (10/22–23)**: $w_i$를 학습 전 policy $\pi_0$ 기준으로 바꾸는 옵션(`--cost-weights pi0`)을 구현하고 $\pi_0$ calibration까지 완료. repo 정리(experiment index, config, CLAUDE.md), branch1 archive. **E8 본 실험은 아직 안 돌림**: 결정할 것 2개가 남음 (§6).

---

## 0. 실험 ID (E1–E9)

이번 세션에서 모든 실험에 번호를 붙였다 (전체 목록: `docs/experiments/INDEX.md`, 설정: `configs/`). 미팅 노트의 Task 번호와의 관계: **Task 1 = E1, Task 2 = E2, Task 3 = E3 + E4**. 이 문서는 Task 3 이후(E3~E8)만 다룬다.

| ID | 무엇 | 상태 |
|---|---|---|
| E1, E2 | Task 1 (toy), Task 2 (논문 재현) | 완료 (이 문서에서는 생략) |
| **E3** | SafetyPointGoal2 budget calibration: unconstrained agent로 $d_h = 31.9$, $d_v = 73.9$를 정함 | 완료 |
| **E4** | SafetyPointGoal2 본 실험: unconstrained / aggregate_loose / multi × seed 3개, 2M step (§3.1의 표) | 완료 (수렴 안 함) |
| E5a | E4 follow-up: dual lr 0.04로 multi가 수렴하는지 | Vast.ai에서 돌렸지만 결과를 이 컴퓨터에서 못 찾음 |
| E5b | E4 follow-up: aggregate_conservative ($x_h + x_v \le 1$), multi와 feasible set을 맞춘 비교 대상 | 위와 같음 |
| E6 | Lagrangian hyperparameter search (successive halving) | 코드와 결과를 못 찾음 |
| E7 | HumanoidBench pilot | 실패 (생략) |
| **E8** | $\pi_0$ weights (이번 세션에서 구현. 교수님 제안이 아님, §2.2 참고) | calibration만 완료, 본 실험 전 |
| E9 | hierarchical / priority safe RL (CSED499II 주제) | 계획 |
| **E10** | E4를 5M step으로 다시 (해석 A): aggregate_loose dual lr 0.02, multi 0.04, seed 3개 | **실행 중** (10/23 시작, 약 하루) |

---

## 1. 연구 질문

기존 Safety Gym 논문은 hazard와 vase를 **하나의 scalar cost**로 합친다.

$$
c_t = \mathbb{1}[\text{hazard 안에 있음} \;\lor\; \text{vase를 건드림}]
$$

이렇게 하면 $\lambda$가 하나라서 두 종류의 violation을 구분하지 못한다. 가설은 두 가지:

1. **Exploit 문제**: 합만 제약하면, vase는 전혀 안 건드리는 대신 hazard에 계속 들어가는 식으로 한쪽을 희생할 수 있다.
2. **Selective pressure**: cost별 $\lambda_i$를 쓰면 violation이 있는 cost에만 penalty가 커지므로, 같은 safety 수준에서 return이 더 높을 수 있다.

---

## 2. 수학적 설정

### 2.1 CMDP

finite horizon $T$ (Safety Gym: $T = 1000$), $\gamma = 1$.

$$
\max_\pi \; J_r(\pi) = \mathbb{E}_\pi\Big[\sum_{t=0}^{T-1} r_t\Big]
\quad \text{s.t.} \quad
J_{c_i}(\pi) = \mathbb{E}_\pi\Big[\sum_{t=0}^{T-1} c_{i,t}\Big] \le d_i, \quad i = 1,\dots,m
$$

- $c_{i,t}$: step $t$에서 cost $i$ (0 또는 1의 indicator)
- $d_i$: cost $i$의 episode당 budget (단위: "violation이 있었던 step 수 / episode")

### 2.2 Cost weight $w_i$ (단위 맞추기)

hazard와 vase는 발생 빈도가 달라서 그대로 더하면 의미가 없다. 그래서 각 cost에 weight $w_i$를 곱한다. 코드에서는 **critic이 학습하는 cost 신호 자체를** $w_i c_{i,t}$로 바꿔서, advantage도 처음부터 weight 단위가 된다 (`train.py`: `cost / cost_scale`, $w_i = 1/\text{cost\_scale}_i$).

두 가지 선택이 있다 (`--cost-weights`):

| 모드 | $w_i$ | 의미 |
|---|---|---|
| `budget` (기본값, 지금까지의 모든 실험) | $1/d_i$ | $x_i = J_{c_i}/d_i$. $x_i = 1$이면 budget에 딱 맞음 |
| `pi0` (이번 세션, E8) | $1/\max(C_i^0, \text{floor})$ | $C_i^0 = J_{c_i}(\pi_0)$: **학습 전** policy의 평균 episode cost. floor 기본 1.0 |

- `budget` 모드는 예전 코드와 **bit-identical** (test로 확인).
- **evaluation은 모드와 무관**: 항상 각 cost를 자기 $d_i$와 비교 ($x_i = J_{c_i}/d_i$).
- 선택 옵션 `--budget-from-pi0 ALPHA`: $d_i = \alpha\, C_i^0$로 budget 자체를 $\pi_0$에서 정함.

> [!question] $\pi_0$가 뭔가?
> **학습을 한 번도 하지 않은, 초기화 직후의 policy**다 (update 0). `train.py --seed s`가 처음 만드는 network 그대로:
> - actor의 마지막 layer가 gain 0.01로 초기화돼서 action 평균 ≈ 0, std = $e^{-0.7}$ ≈ 0.5인 Gaussian → **사실상 제자리에서 랜덤하게 떠는 policy**.
> - 그래서 robot은 시작 위치 근처에만 있고, cost는 시작 위치가 우연히 hazard/vase 근처였던 episode에서만 생긴다 (§3.2에서 80~93% episode가 cost 0인 이유).
> - $C_i^0$ = 이 policy를 고정된 30개 layout에서 돌렸을 때 cost $i$의 평균 episode cost. `calibrate_pi0.py`가 측정한다. seed마다 초기 network가 다르므로 $\pi_0$도, $C_i^0$도 seed마다 다르다.
> - 학습을 하지 않으므로 계산이 싸다 (rollout만 30 episode).

> [!warning] $\pi_0$ weighting의 출처 (10/23 정정)
> 세션 첫 작업 지시문에 "교수님 제안"이라고 적혀 있어서 구현했지만, 교수님이 실제로 하신 말은 **"hazard와 vase의 violation에 따라 weighting해서 cost 합이 2가 되도록 aggregation"**이다. 500K step으로 budget을 정하는 방식이 문제라는 말은 없었다. 따라서 $\pi_0$ weighting은 근거가 약한 대안이다 (`--cost-weights` 기본값은 `budget`이라 기존 실험에는 영향 없음). 교수님 제안의 해석은 §6 참고.

### 2.3 비교하는 방법 4가지 (`train.py --method`)

일반형 ($w_i$ 단위). `budget` 모드에서는 $w_i d_i = 1$이라 괄호 안 값이 된다.

| 코드 이름 | 제약 | $\lambda$ 개수 | 슬라이드 이름 |
|---|---|---|---|
| `unconstrained` | 없음 ($\lambda = 0$) | 0 | Unconstrained |
| `aggregate_loose` | $\sum_i w_i J_{c_i} \le \sum_i w_i d_i$ &nbsp; ($x_h + x_v \le 2$) | 1 | Aggregate (sum ≤ 2) |
| `aggregate_conservative` | $\sum_i w_i J_{c_i} \le \min_i w_i d_i$ &nbsp; ($x_h + x_v \le 1$) | 1 | (아직 실험 안 함) |
| `multi` | $w_i J_{c_i} \le w_i d_i \;\forall i$ &nbsp; ($x_h \le 1,\; x_v \le 1$) | 2 | Separate |

**Feasible set 포함 관계** (budget 모드):

$$
\underbrace{\{x_h + x_v \le 1\}}_{\text{conservative}}
\;\subset\;
\underbrace{\{x_h \le 1,\, x_v \le 1\}}_{\text{multi (정사각형)}}
\;\subset\;
\underbrace{\{x_h + x_v \le 2\}}_{\text{loose (삼각형)}}
$$

> [!note] 직관
> - loose의 삼각형은 multi의 정사각형을 포함하므로 **loose가 더 느슨한 제약**이다. 예: $(x_h, x_v) = (0.7, 1.1)$은 loose는 통과, multi는 violation.
> - "같은 safety"에서 비교하려면 feasible set을 맞춰야 한다. loose와 multi의 return을 그대로 비교하면 공정하지 않다.
> - **일반화**: 제약을 합치면 feasible set은 항상 넓어지거나 같다. $\forall i,\, x_i \le 1 \Rightarrow \sum_i x_i \le m$ 이지만 역은 성립하지 않는다.
> - conservative의 $\min_i w_i d_i$는 "모든 $J_{c_i} \le d_i$를 보장하는 가장 큰 shared bound"다: $w_j J_{c_j} \le \sum_i w_i J_{c_i} \le \min_i w_i d_i \le w_j d_j$.

> [!important] $w_i$를 바꾸면 무엇이 바뀌나
> - **multi**: $w_i J_{c_i} \le w_i d_i \iff J_{c_i} \le d_i$. **feasible set은 그대로**이고 학습 dynamics(advantage에서 penalty 크기, $\lambda$ 속도)만 바뀐다.
> - **aggregate**: 합에서 hazard와 vase의 상대 비중이 $d_v/d_h$에서 $C_v^0/C_h^0$로 바뀐다. 즉 **feasible set 자체가 바뀐다** (삼각형의 기울기가 달라짐).

### 2.4 Lagrangian과 primal-dual

제약 문제를 $\lambda \ge 0$을 가진 min-max 문제로 바꾼다.

**multi**:
$$
\mathcal{L}(\pi, \boldsymbol\lambda) = J_r(\pi) - \sum_i \lambda_i \big(w_i J_{c_i}(\pi) - w_i d_i\big)
$$

**aggregate (loose / conservative)**:
$$
\mathcal{L}(\pi, \lambda) = J_r(\pi) - \lambda \Big(\sum_i w_i J_{c_i}(\pi) - k\Big), \qquad k = \textstyle\sum_i w_i d_i \text{ (loose)},\; \min_i w_i d_i \text{ (conservative)}
$$

$$
\min_{\lambda \ge 0} \; \max_\pi \; \mathcal{L}(\pi, \lambda)
$$

번갈아 업데이트한다:
- **Primal step**: $\lambda$를 고정하고 PPO로 $\pi$를 업데이트 (reward는 올리고, $\lambda$만큼 cost는 내림)
- **Dual step**: $\pi$를 고정하고 $\lambda$를 업데이트 (violation이 있으면 $\lambda$가 커짐)

### 2.5 Primal step: PPO

**(a) GAE.** reward와 각 weighted cost 신호 $s \in \{r, w_h c_h, w_v c_v\}$마다 따로 계산 ($\gamma = 1$, $\lambda_{\text{GAE}} = 0.95$, 마지막 $V = 0$):

$$
\delta_t^{s} = s_t + V^{s}(o_{t+1}) - V^{s}(o_t), \qquad
\hat A_t^{s} = \sum_{k \ge 0} (\lambda_{\text{GAE}})^k \, \delta_{t+k}^{s}
$$

critic은 하나의 네트워크에 출력 3개 ($V^r, V^h, V^v$).

**(b) Advantage 섞기** (`combine_advantages`):

$$
\text{multi:}\quad \hat A_t = \hat A^r_t - \lambda_h \hat A^h_t - \lambda_v \hat A^v_t
\qquad
\text{aggregate:}\quad \hat A_t = \hat A^r_t - \lambda \big(\hat A^h_t + \hat A^v_t\big)
$$

섞은 뒤에 한 번만 표준화: $\tilde A_t = (\hat A_t - \text{mean}) / \text{std}$.

**(c) Clipped surrogate** ($\rho_t = \pi_\theta(a_t|o_t) / \pi_{\text{old}}(a_t|o_t)$, $\epsilon = 0.2$):

$$
L^{\text{CLIP}}(\theta) = -\,\mathbb{E}_t\Big[\min\big(\rho_t \tilde A_t,\; \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\,\tilde A_t\big)\Big]
$$

total loss = $L^{\text{CLIP}} + 1.0 \cdot \tfrac12 \lVert V - \text{target} \rVert^2$ (3개 head 모두). 6 epoch, minibatch 512, approx KL > 0.03이면 early stop, grad norm clip 0.5.

### 2.6 Dual step: projected gradient ascent (`update_multipliers`)

PPO 업데이트가 끝난 뒤, 이번 batch(4 episodes)의 평균 weighted cost $\hat y_i = w_i \hat J_{c_i}$로:

$$
\text{multi:}\quad \lambda_i \leftarrow \max\big(0,\; \lambda_i + \eta\,(\hat y_i - w_i d_i)\big)
\qquad
\text{aggregate:}\quad \lambda \leftarrow \max\Big(0,\; \lambda + \eta\,\big(\textstyle\sum_i \hat y_i - k\big)\Big)
$$

budget 모드에서는 $\hat y_i = \hat x_i$, $w_i d_i = 1$이라 예전 식 $\lambda_i + \eta(\hat x_i - 1)$과 같다.

$\eta$ = `dual_lr`, $\lambda_0 = 0.1$. **어떤 dual lr이 어디서 쓰였는지** (이번 세션에서 정리; `train.py` 기본값 0.15는 SafetyPointGoal2에서는 안 씀):

| 실험 | dual lr | 근거 |
|---|---|---|
| SafetyPointGoal2 calibration, 본 실험 (E3/E4) | **0.02** | `run_pilot.py` FROZEN, slurm 스크립트, 미팅 노트 |
| follow-up E5a | 0.04 | branch1의 `run_followups.sh` |
| hparam search (E6) | unknown | 코드도 결과도 이 repo에 없음 |

> [!tip] 직관: $\lambda$는 violation의 누적합이다 (integral controller)
> $$\lambda_K = \lambda_0 + \eta \sum_{k=1}^{K} (\hat x_k - 1) \quad (\text{0 아래로 안 내려가는 것만 빼면})$$
> - violation이 계속되는 동안 $\lambda$는 **직선으로** 계속 오른다.
> - violation이 해소된 순간 $\lambda$는 이미 크다. 그래서 cost가 budget보다 한참 아래로 내려가고(**over-constrained**), $\lambda$가 다시 줄어드는 데 오래 걸린다 (**overshoot**).
> - $\lambda$의 반응 속도는 $\eta \times$(update 횟수)$\times$(violation의 크기)로 정해진다.

> [!warning] $\pi_0$ 모드에서 dual lr의 의미가 바뀐다
> $k_i = w_i d_i = d_i / C_i^0$로 두면 ($\pi_0$ calibration에서 **0.8 ~ 6.2**, budget 모드에서는 1):
> - dual step: $\hat y_i - w_i d_i = k_i(\hat x_i - 1)$ → $\lambda_i$가 violation 1당 $k_i$배 빨리 오름.
> - advantage: cost 신호가 $w_i c_i = k_i (c_i/d_i)$라서 $\lambda_i \hat A^{w_i c_i} = (k_i\lambda_i)\,\hat A^{c_i/d_i}$ → 같은 $\lambda_i$가 $k_i$배 큰 penalty.
> - 그래서 policy가 실제로 느끼는 penalty $\lambda_i^{\text{eff}} = k_i \lambda_i$는 $\lambda_i^{\text{eff}} \leftarrow \lambda_i^{\text{eff}} + \eta\,k_i^2(\hat x_i - 1)$로 움직인다. **속도는 $k_i^2$배** (0.6 ~ 38배), 시작값도 $k_i \lambda_0$. → §6 결정 2.

---

## 3. SafetyPointGoal2-v0 설정

**환경**: Safety-Gymnasium `SafetyPointGoal2-v0`, episode 1000 step, Point robot이 goal에 도달하면 reward.

**cost 정의** (`safety_adapter.py`, 환경의 `cost_sum`은 쓰지 않음):
$$
c_{h,t} = \mathbb{1}[\texttt{cost\_hazards} > 0], \qquad
c_{v,t} = \mathbb{1}[\texttt{cost\_vases\_contact} > 0 \;\lor\; \texttt{cost\_vases\_velocity} > 0]
$$
- 한 step이 두 cost에 동시에 들어갈 수 있음
- observation: 원래 60차원 + (경과 시간 비율, absorbed flag) = 62차원

**budget (E3)**: unconstrained agent, seed 100, 500K step → 30개 layout(env seed 500000, chunk 10)에서 평균 cost의 절반:
$d_i = \max(1, \text{round}(0.5 \times \bar C_i, 1))$ → $d_h = 31.9$, $d_v = 73.9$

**본 실험 설정 (E4, `run_pilot.py` FROZEN)**: 4 env × 1000 step × 500 update = 2M step, dual lr 0.02, $\lambda_0 = 0.1$, 25 update마다 eval 10 episodes, 끝나면 final eval 50 episodes, held-out 평가는 `evaluate.py --environment-seed 900000 --action-seed 900001` (50 layouts).

> [!warning] Evaluation layout은 env seed **와** chunk 크기로 정해진다
> `collect_chunked`가 chunk마다 `make_env(seed + start, size)`를 부르기 때문. branch1(E3/E4)은 chunk 10, main의 `run_pilot.py`는 `--eval-num-envs 2`. **E4와 비교하려면 `--eval-num-envs 10`** (E8 config에 이미 들어 있음). 한 실험 안에서는 절대 바꾸지 말 것.

### 3.1 E4 결과 (held-out 평가, 미팅 노트 260930)

| | return | goals/ep | $x_h$ | $x_v$ | 합 | 둘 다 ≤ 1인 seed |
|---|---:|---:|---:|---:|---:|---:|
| Unconstrained | 22.4 | 8.6 | 1.76 | 1.84 | 3.60 | 0/3 |
| Aggregate (loose) | 12.2 | 5.0 | 0.59 | 0.81 | 1.41 | 2/3 |
| Separate (multi) | 19.3 | 7.6 | 1.59 | 1.37 | 2.96 | 0/3 |

- aggregate는 자기 제약(합 ≤ 2)은 3/3 seed 모두 만족. 그런데 seed 0은 합은 만족하면서 $x_v = 1.1$ → 이것이 exploit 문제의 예.

**$\lambda$ 그래프 해석** (2.6의 integral controller 관점):

- **Separate**: $\lambda_h, \lambda_v$가 2M step 끝까지 **직선으로 증가** → 아직 violation 중이고, 수렴 전.
  - update 500번 × $\eta = 0.02$ × 평균 violation 약 0.5 ≈ 5. 그래프의 5~6과 맞는다.
  - $\lambda$가 충분히 커지기 전에 학습이 끝났다. **step 수가 부족하거나 $\eta$가 작다.**
- **Aggregate**: violation이 $\hat x_h + \hat x_v - 2$라서 초반에 약 $3.6 - 2 = 1.6$으로 크다 → $\lambda$가 multi보다 약 2배 빨리 오름.
  - $\lambda \approx 8$까지 올라간 뒤에야 violation이 해소되고, 그때는 이미 너무 커서 합이 1.41까지 내려감 → **overshoot로 인한 over-constrained**. 이후 $\lambda$가 천천히 내려오는 중이었음.

> [!warning] 해석 시 주의
> loose(삼각형)와 multi(정사각형)는 feasible set이 달라서, 이 표만으로 "어느 쪽이 return이 높다"고 결론 내리면 안 된다. 두 방법 모두 수렴하지 않았다. 결과를 쓸 때는 항상 **native constraint 만족 여부**와 **per-cost "둘 다 ≤ 1"**을 따로 보고할 것.

### 3.2 $\pi_0$ calibration 결과 (이번 세션, E8 준비)

`calibrate_pi0.py`, E3와 같은 30개 layout (env seed 500000, chunk 10). 파일: `runs/calibration/pi0_SafetyPointGoal2-v0_<seed>.json`.

| seed | $C_h^0$ | zero-cost episode | $C_v^0$ | zero-cost episode | $w_h d_h$ | $w_v d_v$ |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 20.5 ± 82.8 | 90% | 13.2 ± 36.6 | 87% | 1.56 | 5.60 |
| 1 | 11.3 ± 49.0 | 93% | 13.6 ± 38.9 | 80% | 2.82 | 5.43 |
| 2 | 40.0 ± 113.8 | 87% | 19.0 ± 49.7 | 80% | 0.80 | 3.89 |
| 100 | 6.3 ± 29.8 | 93% | 11.9 ± 33.6 | 83% | 5.06 | 6.21 |

(mean ± std over 30 episodes)

- 둘 다 0 근처는 아님 → floor 1.0은 실제로 걸리지 않음.
- **문제**: random Point robot은 거의 안 움직여서 30 episode 중 2~6개만 cost가 0이 아니다. 그래서 평균이 매우 noisy하다 ($C_h^0$가 seed에 따라 6.3 ~ 40.0).
- seed 2는 $\pi_0$가 이미 hazard budget을 넘음 ($40.0 > 31.9$). 나머지 seed는 budget이 $\pi_0$보다 느슨함.
- `runs/calibration/chunk2/`는 처음에 실수로 잘못된 layout(chunk 2)으로 돌린 결과. 비교용으로만 남겨둠.

---

## 4. 코드 구조 (`pi0-weights` branch 기준)

### 4.1 SafetyPointGoal2-v0 실험에 쓰는 파일

| 파일 | 역할 |
|---|---|
| `train.py` | **유일한 학습 코드.** PPO + scalar/vector Lagrangian. 주요 함수: `collect`(rollout, 신호 = `[r, cost/cost_scale]`), `generalized_advantage`, `combine_advantages`, `update_multipliers`(dual step), `collect_chunked`(고정 layout eval), `train`. 출력: `config.json`, `metrics.jsonl`, checkpoint, `evaluation.npz`, `summary.json` |
| `safety_adapter.py` | Safety-Gymnasium을 감싸서 hazard, vase cost를 **따로** 꺼냄 |
| `calibrate_pi0.py` | **(신규)** 학습 전 policy $\pi_0$의 cost별 평균 episode cost 측정. `train.py`와 같은 seeding 순서로 network를 만들어서, `train.py --seed s`의 $\pi_0$와 정확히 같은 policy |
| `run_pilot.py` | method × seed matrix launcher. Safety Gym 설정은 FROZEN, 이미 끝난 run은 건너뜀(resume 가능). **(신규)** `--extra "..."`로 train.py flag 추가 (`{seed}` 치환) |
| `evaluate.py` | 저장된 checkpoint를 처음 보는 layout(held-out)에서 평가 |
| `scripts/summarize_runs.py` | **(신규)** `runs/` 아래 결과로 표 재생성 (학습 안 함). status, return, $x_i$, native feasible vs 둘 다 ≤ 1, 마지막 $\lambda$, 마지막 1/4 동안의 $\lambda$ 변화(수렴 판단용) |
| `slurm_safety_calibrate_titanxp.sh`, `slurm_safety_multicost_titanxp.sh` | titanxp cluster에서 budget calibration / 본 실험 |
| `validate_safety.py` | cost 신호가 제대로 나오는지 검증 (영상 포함) |
| `test_core.py` | unit test 15개. 이번 세션에 추가: budget 모드가 예전 dual update와 bit-identical, weighted violation = $\sum w_i(C_i - d_i)$, cost_scale이 신호에 반영됨, native feasibility ≠ per-cost, config `extends`/CLI override, repo의 config 전부 load 가능 |

### 4.2 실험 기록 체계 (이번 세션에 만듦)

| 위치 | 내용 |
|---|---|
| `docs/experiments/INDEX.md` | **실험 목록 E1–E9**: 질문, 상태, config, 어느 코드(commit/branch)로 돌렸는지, 결과 위치, 근거(code / report / notes). 모르는 것은 unknown으로 둠 |
| `configs/safety/*.json` | 실험마다 JSON 하나. `base.json`(FROZEN 설정) + `extends`로 차이만. 확인 못 한 값은 `unknown`. SafetyPointGoal2 관련: `sg-calibration`(E3), `sg-pilot-v1`(E4), `sg-pilot-v1-duallr004`(E5a), `sg-pilot-v1-conservative`(E5b), `pi0-weights`(E8) |
| `train.py --config X.json --experiment-id ...` | config의 `train_args`가 기본값이 되고 CLI flag가 덮어씀. 모르는 key가 있으면 에러 |
| run의 `config.json` | 이제 git commit/branch/dirty, 사용한 experiment config, evaluation protocol(eval seed, chunk)까지 저장 |
| `reports/sg-pilot-v1/` | 미팅 노트에서 옮긴 E4 결과 요약 |
| `docs/2026-09-30-lab-meeting.pdf` | 260930 미팅 노트 |
| `CLAUDE.md` | 앞으로 coding agent가 지킬 규칙 (기존 baseline을 몰래 바꾸지 않기, 새 기능은 flag 뒤에, chunk 크기 고정, 결과 삭제 금지, commit 전 pytest 등) |
| `README.md`, `PLAN.md` | 다시 씀 / 현재 상태 추가 |

### 4.3 한 update에서 일어나는 일 (`train.py`)

```text
cost_scale = d (budget 모드)  또는  max(C0, floor) (pi0 모드)      # w = 1 / cost_scale
for update in 1..500:
    # 1) rollout: 4개 env에서 1000 step씩 (= episode 4개)
    signals_t = [r_t, c_hazard_t / cost_scale_h, c_vase_t / cost_scale_v]

    # 2) GAE: 신호 3개 각각 advantage 계산
    A_r, A_h, A_v = GAE(signals, critic values)

    # 3) advantage 섞기 + 표준화
    A = A_r - λ_h A_h - λ_v A_v          # multi
    A = A_r - λ (A_h + A_v)              # aggregate
    A = (A - mean) / std

    # 4) PPO: 6 epoch, minibatch 512, KL > 0.03이면 중단
    # 5) dual update: λ += η * (weighted cost - weighted budget), 0 아래로 clip
    # 6) 25 update마다 eval 10 episodes (seed 10000+seed), checkpoint 저장
# 끝나면 final eval 50 episodes (seed 200000+seed), evaluation은 항상 C_i vs d_i
```

### 4.4 이번 세션 commit (`pi0-weights`, d5b7932..ae0ed14, 12개, GitHub에 push됨)

- **pi0 weighting**: `calibrate_pi0.py`, `train.py --cost-weights pi0`, test (d5b7932–f0c381e)
- **실험 기록**: `--config`, git 정보 기록, `summarize_runs.py`, `run_pilot.py --extra`, INDEX/configs/README/PLAN/CLAUDE.md, 보고서와 미팅 노트 (08a5f95–896a117)
- **정리**: 잘못 생긴 파일 삭제, `.gitignore`, branch1 archive (945336a, 094cda2, ae0ed14)
- 3-update smoke test: multi / aggregate_loose / aggregate_conservative 모두 pi0 모드에서 loss가 유한하고 $\lambda$가 움직임. $\lambda$ 업데이트 한 번을 손으로 계산해서 확인.
- **branch1 archive**: tag `archive/branch1` (b240033, GitHub에도 push). GitHub의 `branch1`, `claude/fervent-wozniak-pamdhy` branch는 삭제. 옛 코드는 `git show archive/branch1:pipeline.py`로 볼 수 있음.

---

## 5. 남은 일 (구체적으로)

0. **E10 실행 중** (tmux `sg-pilot-v2`, `runs/sg-pilot-v2.sh`): aggregate_loose 3개 (약 12h) → multi 3개 (약 12h) → held-out 평가(E4와 같은 50 layout, `--eval-num-envs 10`) → `runs/sg-pilot-v2-summary.md`. 진행 확인: `tail -n1 runs/sg-pilot-v2-*/*.log | cut -c1-200`, `python scripts/summarize_runs.py --root runs/sg-pilot-v2-loose`. 끝나면 볼 것: $\lambda$가 평평해졌는지(`lambda_change_last_quarter`), aggregate가 여전히 over-constrained인지, multi가 둘 다 ≤ 1이 되는지.
   - **W&B**: tmux `wandb-tail`이 `metrics.jsonl`을 60초마다 업로드. project `multi-cost-rl`, group `spg-v2-aggregate-loose`, `spg-v2-multi` (W&B 쪽은 이미 새 이름).
   - [ ] **TODO (E10이 모두 끝난 뒤) 이름 변경 `sg-pilot-v2` → `spg-v2`**. 조건: tmux `sg-pilot-v2`에 "all done", tmux `wandb-tail`이 종료됨. 학습 중에는 절대 옮기지 말 것.
     1. `mv runs/sg-pilot-v2-loose runs/spg-v2-aggregate-loose`, `mv runs/sg-pilot-v2-multi runs/spg-v2-multi`
     2. `mv runs/sg-pilot-v2.sh runs/spg-v2.sh` (`.out`도), 표 다시 만들기: `python scripts/summarize_runs.py --root runs --output runs/spg-v2-summary.md`
     3. `git mv configs/safety/sg-pilot-v2.json configs/safety/spg-v2.json`, 안의 name/commands/output 수정. `docs/experiments/INDEX.md`, 이 문서도 수정 → commit
     4. 각 run의 `config.json`, `manifest.json` 안의 옛 group 이름은 기록이므로 **고치지 않음** (config에 "renamed from sg-pilot-v2-*" 메모만)

1. **E8 결정 2개** (§6) → 결정하면 config `configs/safety/pi0-weights.json` 수정.
2. **(결정 1에 따라) calibration 다시 돌리기**: 예) `python calibrate_pi0.py --seed s --episodes 300`.
3. **E8 본 실험 실행** (README의 명령):
   ```bash
   python run_pilot.py --group sg-pi0-v1 --methods multi aggregate_conservative aggregate_loose \
     --budget-hazard 31.9 --budget-vase 73.9 --transitions 2000000 --device cpu \
     --extra "--config configs/safety/pi0-weights.json --eval-num-envs 10 --pi0-calibration runs/calibration/pi0_SafetyPointGoal2-v0_{seed}.json"
   ```
   그다음 held-out 평가(`evaluate.py ... --environment-seed 900000 --action-seed 900001`), `scripts/summarize_runs.py --root runs/sg-pi0-v1`, `reports/sg-pi0-v1/README.md` 작성, INDEX 업데이트.
4. **E5a / E5b 결과 찾기** (dual lr 0.04, conservative): Vast.ai에서 돌렸고 이 컴퓨터에는 없음. `runs/sg-pilot-v1-duallr004/`, `runs/sg-pilot-v1-conservative/`를 가져와서 `summarize_runs.py`로 정리. E8과 비교할 baseline이 됨 (특히 conservative는 아직 결과가 하나도 없음).
5. **E6 hparam search 찾기**: successive halving 코드와 결과가 어떤 branch, commit, 파일에도 없음. 있는 곳(다른 컴퓨터?)을 찾아서 `runs/hparam/`에 넣기.
6. **`pi0-weights`를 `main`에 merge** (GitHub PR). 아직 안 함.
7. GitHub의 PR #1이 자동으로 닫혔는지 확인 (`gh`가 없어서 여기서는 확인 못 함).
8. 미팅 노트의 개선 항목 (아직 안 함):
   - [ ] step 수 2M → 5~10M (지금 $\lambda$가 수렴하지 않음)
   - [ ] seed 5개 이상 (tuning seed와 최종 보고 seed는 분리)
   - [ ] $\alpha$ sweep ($d_i = \alpha C_i^0$, `--budget-from-pi0`)로 **return vs. joint pass rate** 곡선 비교 (feasible set을 맞춘 공정한 비교)

---

## 6. 내가 결정해야 할 것

### 결정 0: 교수님 제안의 해석 → **일단 A로 진행 (10/23)**, B는 다음 랩미팅에서 상의
교수님 말: "hazard와 vase의 violation에 따라 weighting해서 cost 합이 2가 되도록 aggregation".

- **해석 A (고정 weight)**: $w_i = 1/d_i$로 $x_h + x_v \le 2$. **지금의 `aggregate_loose`(E4)와 같다.** 할 일은 E4를 수렴할 때까지 돌리는 것 (step 수↑, dual lr↑). → 진행 중.
- **해석 B (violation에 따라 바뀌는 weight)**: 아래.

#### 해석 B 정리 (랩미팅 상의용)

**식.** $w_h + w_v = 2$를 유지하면서, 매 update마다 violation이 큰 cost 쪽으로 weight를 옮긴다. 제약은 $\sum_i w_i x_i \le 2$, $\lambda$는 하나.
$$
w_i = 2\,\frac{\max(\hat x_i - 1, 0) + \epsilon}{\sum_j \left(\max(\hat x_j - 1, 0) + \epsilon\right)}
$$
- $x_i = J_{c_i}/d_i$ (§2.2의 정규화 cost, cost $i$가 budget의 몇 배인지). 학습 중에는 진짜 $J_{c_i}$를 모르므로 dual step(§2.6)과 같은 batch 추정치 $\hat x_i$ (이번 update의 4 episode 평균 cost $/ d_i$)를 쓴다.
- 둘 다 violation이 없으면 $w = (1, 1)$ → 지금의 aggregate. 예: $\hat x = (0.7, 1.1)$이면 violation $(0, 0.1)$ → $w \approx (0, 2)$, 제약은 $2\hat x_v \le 2$. 합만 맞추는 exploit(seed 0)을 막으려는 것.
- A와의 차이: A는 weight가 학습 전에 정해져서 고정, B는 $\lambda$처럼 weight도 feedback으로 움직임. (둘 다 on-policy. on/off-policy의 차이가 아님.)

**B의 사실상의 feasible set은 multi와 같다.** 한쪽이 넘으면 weight가 그쪽으로 몰려서 $x_i \le 1$이 되므로, 결국 정사각형 $\{x_h \le 1, x_v \le 1\}$. 즉 B는 "feasible set은 multi, $\lambda$는 하나를 같이 씀".

**걱정: overshoot가 multi보다 쉽게 생긴다.**
1. **$\lambda$ 공유**: vase만 계속 violation이면 $w \approx (0, 2)$, $\lambda$는 vase 때문에 계속 커지고 hazard penalty는 거의 0. hazard에 들어가는 순간 weight가 넘어가면서 **이미 커진 $\lambda$가 hazard에 그대로 걸림** → hazard 쪽이 갑자기 over-constrained. multi는 $\lambda_i$가 따로라서 이런 일이 없다.
2. **방치된 cost가 다시 나빠짐**: $w_h \approx 0$인 동안 policy는 hazard를 신경 쓰지 않음 → hazard 증가 → weight가 넘어가면 이번엔 vase 방치 → 두 cost 사이를 왔다 갔다 하는 oscillation.
3. **batch noise**: $\hat x_i$는 4 episode 평균이라 noisy하고(특히 hazard는 가끔 몰아서 생김), $\max(\cdot)$ 때문에 한쪽이 조금만 넘어도 weight가 거의 전부 그쪽으로 감 → noise만으로도 weight가 매 update 뒤집힐 수 있음.

**완화 방법 (B를 한다면)**
- weight smoothing: $\hat x$ 대신 여러 update의 EMA로 weight 계산
- weight 하한 (예: $w_i \ge 0.5$) → 한 cost를 완전히 방치하지 않음
- winner-take-all 대신 softmax(temperature)
- two-timescale: weight를 $\lambda$보다 느리게 업데이트
- PID Lagrangian (Stooke et al. 2020): integrator overshoot 자체를 줄임 (A, B, multi 모두에 적용 가능)

**교수님께 여쭤볼 것**
1. weight를 고정하는 건지(A, 이미 E4에서 한 것), violation에 따라 바꾸는 건지(B)?
2. B라면, feasible set이 multi와 같은데 "scalar $\lambda$ 하나 + weight 배분"이 multi(cost별 $\lambda_i$)와 무엇이 다른 걸 보려는 건지? (예: 해석 가능성, 파라미터 수, cost가 많아질 때의 확장성?)
3. "합이 2"는 $\sum w_i = 2$ (weight의 합)인지, $\sum x_i \le 2$ (제약의 오른쪽)인지?

(참고) $\pi_0$ weighting(E8)은 A, B 어느 것도 아니므로 결정 1, 2는 보류.

### 결정 1: $\pi_0$ cost 추정이 noisy한 문제
30 episode 중 80~93%가 cost 0이라 평균이 2~6개 episode로 정해진다. 선택지:
- **(a) episode 수를 늘림** (예: 300). seed별로 따로 $w_i$. 비용은 학습 없는 rollout이라 작음.
- **(b) seed를 합쳐서 $w_i$ 하나로 고정** (예: seed 0/1/2/100 + 추가 episode의 평균). 모든 run이 같은 constraint를 가지므로 seed 간 비교가 깔끔함. 대신 "그 seed의 $\pi_0$"라는 의미는 약해짐.
- (c) 그대로 진행 (seed마다 weight가 최대 6배 차이 → aggregate의 feasible set이 seed마다 다름. 비추천).
- 부수적으로: floor 값 (지금 1.0, 실제로는 안 걸림).

### 결정 2: $\pi_0$ 단위에서의 dual lr
§2.6 warning: 같은 $\eta$라도 policy가 느끼는 penalty $k_i\lambda_i$는 budget 모드보다 $k_i^2$배 빠르게 움직인다 ($k_i = d_i/C_i^0$).

**"budget 모드와 같은 속도"의 뜻**: $\eta_i = \eta / k_i^2$, $\lambda_{0,i} = \lambda_0 / k_i$로 두면 $k_i\lambda_i$가 E4의 $\lambda_i$와 **매 update 똑같이** 움직인다. 예: seed 0 hazard는 $k = 1.56$ → $\eta = 0.02/1.56^2 \approx 0.008$.

> [!important] 이러면 multi에서는 $\pi_0$ weighting이 사라진다
> multi는 feasible set이 원래 같고(§2.3), 위처럼 맞추면 penalty dynamics까지 같아서 **E4 multi와 사실상 같은 실험**이다 (critic의 value loss 스케일만 다름). 즉 multi에서 $\pi_0$ weight는 "cost별 dual lr과 $\lambda_0$를 바꾸는 것"과 같다. $\pi_0$ weighting이 진짜로 다른 문제를 만드는 건 **aggregate**뿐이다 (λ가 하나라서 cost별로 맞출 수 없고, 합의 비중 자체가 바뀜).

선택지:
- **(a) $\eta = 0.02$ 그대로**: 단순함. 대신 multi의 결과가 E4와 달라도 그게 weight 때문이 아니라 cost별 λ 속도($k_i^2$배) 때문이다. seed 0 vase는 $5.6^2 \approx 31$배라 overshoot / over-constrained 위험이 큼.
- **(b) budget 모드와 맞춤**: multi는 $\eta_i = \eta/k_i^2$, $\lambda_{0,i} = \lambda_0/k_i$ (E4 재현이 되므로 multi는 E8에서 빼도 됨). aggregate는 λ가 하나라 정확히 맞출 수 없으니 평균 $\bar k$로 $\eta/\bar k^2$ 정도. **지금 코드는 dual lr이 scalar 하나라 cost별 $\eta_i$는 새 flag가 필요.**
- (c) 작은 sweep (예: 0.001 / 0.005 / 0.02)을 tuning seed로 돌려서 고르기. E6 hparam search 결과를 찾으면 그걸 참고.

### 그 외 선택
- E8에서 budget을 그대로 둘지($d_h = 31.9, d_v = 73.9$), $d_i = \alpha C_i^0$로 바꿀지.
- step 수 (2M로 E4와 맞출지, 5~10M로 늘릴지): E4가 2M에서 수렴하지 않았으므로 늘리는 게 맞지만, 그러면 E4와 직접 비교하려면 budget 모드도 다시 돌려야 함.
- 앞으로의 eval chunk: E4와 비교하려면 10 (E8 config는 10).

---

## 7. 용어

| 용어 | 뜻 |
|---|---|
| CMDP | Constrained MDP. reward를 최대화하되 cost 기댓값에 상한이 있는 문제 |
| budget $d_i$ | episode당 허용되는 cost $i$의 총량 |
| weight $w_i$ | cost를 더하기 전에 곱하는 단위 맞춤 계수. `budget` 모드 $1/d_i$, `pi0` 모드 $1/C_i^0$ |
| $x_i$ | 정규화 cost $= J_{c_i}/d_i$. 1이면 budget에 딱 맞음 (evaluation은 항상 이것) |
| $\pi_0$, $C_i^0$ | 학습 전(초기화 직후) policy와 그 평균 episode cost |
| violation | 제약을 넘은 양 ($x_i - 1 > 0$) |
| over-constrained | cost가 budget보다 한참 낮음 → 필요 이상으로 reward를 포기한 상태 |
| overshoot | $\lambda$가 violation 해소 시점에 이미 너무 커서 반대쪽으로 지나치는 현상 |
| native constraint | 그 method가 학습한 제약 (loose: 합 ≤ 2 등) |
| Lagrange multiplier $\lambda$ | violation에 대한 penalty 계수. violation이 있으면 커지고 만족하면 줄어듦 |
| primal / dual step | $\pi$ 업데이트 / $\lambda$ 업데이트 |
| joint pass rate | 모든 cost가 동시에 budget 이하인 episode의 비율 |
| held-out layouts | 학습 중에 한 번도 안 본 맵. 최종 평가에만 사용 |
| chunk (`--eval-num-envs`) | eval 때 한 번에 만드는 env 개수. layout을 바꾸므로 실험 안에서 고정 |
