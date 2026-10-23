---
tags: [research, safe-rl, multi-cost]
source: "github.com/c2-h1/multi-cost-rl @ spg-v2"
related: "[[260930]]"
updated: 2026-10-23
---

# Multi-cost Safe RL: SafetyPointGoal2-v0 (hazard + vase)
> [!NOTE]
> **TL;DR**
> - **질문**: safety cost를 하나로 합치지 않고 종류별(hazard, vase)로 나눠서 각각 제약을 걸면, 같은 safety 수준에서 return이 더 높아질까?
> - **방법**: PPO + Lagrangian(primal-dual). cost를 budget으로 나눠 단위를 맞춘 뒤 ($x_i = J_{c_i}/d_i$), 하나의 $\lambda$(aggregate)와 cost별 $\lambda_i$(multi)를 비교.
> - **지금까지 결과**: 2M step 안에 수렴하지 않음. aggregate는 overshoot 때문에 over-constrained, multi는 아직 budget violation.
> - **이번 세션 (10/22–23)**: repo 정리(experiment index, config, CLAUDE.md), branch1 archive. 교수님 제안의 해석 A(고정 weight, 합 ≤ 2)로 **E10 실행 중**: E4를 5M step, 더 큰 dual lr로 다시. 해석 B는 다음 랩미팅에서 상의 (§6).

---

## 0. 실험 ID (E1–E10)

이번 세션에서 모든 실험에 번호를 붙였다 (전체 목록: `docs/experiments/INDEX.md`, 설정: `configs/`). 미팅 노트의 Task 번호와의 관계: **Task 1 = E1, Task 2 = E2, Task 3 = E3 + E4**. 이 문서는 Task 3 이후(E3~E10)만 다룬다.

| ID | 무엇 | 상태 |
|---|---|---|
| E1, E2 | Task 1 (toy), Task 2 (논문 재현) | 완료 (이 문서에서는 생략) |
| **E3** | SafetyPointGoal2 budget calibration: unconstrained agent로 $d_h = 31.9$, $d_v = 73.9$를 정함 | 완료 |
| **E4** | SafetyPointGoal2 본 실험: unconstrained / aggregate_loose / multi × seed 3개, 2M step (§3.2의 표) | 완료 (수렴 안 함) |
| E5a | E4 follow-up: dual lr 0.04로 multi가 수렴하는지 | 종결: 수렴하지 않음 (파일 없음) |
| E5b | E4 follow-up: aggregate_conservative ($x_h + x_v \le 1$), multi와 feasible set을 맞춘 비교 대상 | 종결: 수렴하지 않음 (파일 없음) |
| E6 | Lagrangian hyperparameter search (successive halving) | 종결: 당분간 hyperparameter는 직접 고름 |
| E7 | HumanoidBench pilot | 실패 (생략) |
| E8 | (취소된 실험, 번호만 남김) | 취소 |
| E9 | hierarchical / priority safe RL (CSED499II 주제) | 계획 |
| **E10** | E4를 5M step으로 다시 (해석 A): aggregate_loose dual lr 0.02, multi 0.04, seed 3개 | **실행 중** (10/23 시작, 약 하루) |

---

## 1. 연구 질문

기존 Safety Gym 논문은 hazard와 vase를 **하나의 scalar cost**로 합친다.

$$
c_t = \mathbb{1}[\text{hazard 안에 있음} \ \lor\  \text{vase를 건드림}]
$$

이렇게 하면 $\lambda$가 하나라서 두 종류의 violation을 구분하지 못한다. 가설은 두 가지:

1. **Exploit 문제**: 합만 제약하면, vase는 전혀 안 건드리는 대신 hazard에 계속 들어가는 식으로 한쪽을 희생할 수 있다.
2. **Selective pressure**: cost별 $\lambda_i$를 쓰면 violation이 있는 cost에만 penalty가 커지므로, 같은 safety 수준에서 return이 더 높을 수 있다.

---

## 2. 수학적 설정

### 2.1 CMDP

finite horizon $T$ (Safety Gym: $T = 1000$), $\gamma = 1$.

$$
\max_\pi \  J_r(\pi) = \mathbb{E}_\pi\Big[\sum_{t=0}^{T-1} r_t\Big]
\quad \text{s.t.} \quad
J_{c_i}(\pi) = \mathbb{E}_\pi\Big[\sum_{t=0}^{T-1} c_{i,t}\Big] \le d_i, \quad i = 1,\dots,m
$$

- $c_{i,t}$: step $t$에서 cost $i$ (0 또는 1의 indicator)
- $d_i$: cost $i$의 episode당 budget (단위: "violation이 있었던 step 수 / episode")

### 2.2 정규화 (단위 맞추기)

hazard와 vase는 발생 빈도가 달라서 그대로 더하면 의미가 없다. 그래서 budget으로 나눈다 (weight $w_i = 1/d_i$).

$$
x_i(\pi) = \frac{J_{c_i}(\pi)}{d_i}
\qquad (x_i = 1 \iff \text{budget에 정확히 맞음})
$$

- $d_i$는 E3에서 unconstrained agent를 500K step 학습시킨 뒤 측정한 평균 cost의 절반 (§3.1).
- 코드에서는 **critic이 학습하는 cost 신호 자체를** $c_{i,t}/d_i$로 바꿔서, advantage도 처음부터 budget 단위가 된다 (`train.py`의 `cost / env.budgets`).

### 2.3 비교하는 방법 4가지 (`train.py --method`)

| 코드 이름 | 제약 | $\lambda$ 개수 | 슬라이드 이름 |
|---|---|---|---|
| `unconstrained` | 없음 ($\lambda = 0$) | 0 | Unconstrained |
| `aggregate_loose` | $x_h + x_v \le 2$ | 1 | Aggregate (sum ≤ 2) |
| `aggregate_conservative` | $x_h + x_v \le 1$ | 1 | (아직 실험 안 함) |
| `multi` | $x_h \le 1,\  x_v \le 1$ | 2 | Separate |

**Feasible set 포함 관계**:

$$
\underbrace{\lbrace x_h + x_v \le 1\rbrace}_{\text{conservative}}
\quad\subset\quad
\underbrace{\lbrace x_h \le 1,\  x_v \le 1\rbrace}_{\text{multi (정사각형)}}
\quad\subset\quad
\underbrace{\lbrace x_h + x_v \le 2\rbrace}_{\text{loose (삼각형)}}
$$

> [!NOTE]
> **직관**
> - loose의 삼각형은 multi의 정사각형을 포함하므로 **loose가 더 느슨한 제약**이다. 예: $(x_h, x_v) = (0.7, 1.1)$은 loose는 통과, multi는 violation.
> - "같은 safety"에서 비교하려면 feasible set을 맞춰야 한다. loose와 multi의 return을 그대로 비교하면 공정하지 않다.
> - **일반화**: 제약을 합치면 feasible set은 항상 넓어지거나 같다. $\forall i,\  x_i \le 1 \Rightarrow \sum_i x_i \le m$ 이지만 역은 성립하지 않는다.
> - conservative의 합 ≤ 1은 "모든 $x_i \le 1$을 보장하는 가장 큰 shared bound"다: $x_j \le \sum_i x_i \le 1$.

### 2.4 Lagrangian과 primal-dual

제약 문제를 $\lambda \ge 0$을 가진 min-max 문제로 바꾼다.

**multi**:

$$
\mathcal{L}(\pi, \boldsymbol\lambda) = J_r(\pi) - \sum_i \lambda_i \big(x_i(\pi) - 1\big)
$$

**aggregate (loose / conservative)**:

$$
\mathcal{L}(\pi, \lambda) = J_r(\pi) - \lambda \Big(\sum_i x_i(\pi) - k\Big), \qquad k = 2 \text{ (loose)},\  1 \text{ (conservative)}
$$

$$
\min_{\lambda \ge 0} \  \max_\pi \  \mathcal{L}(\pi, \lambda)
$$

번갈아 업데이트한다:
- **Primal step**: $\lambda$를 고정하고 PPO로 $\pi$를 업데이트 (reward는 올리고, $\lambda$만큼 cost는 내림)
- **Dual step**: $\pi$를 고정하고 $\lambda$를 업데이트 (violation이 있으면 $\lambda$가 커짐)

### 2.5 Primal step: PPO

**(a) GAE.** reward와 각 정규화 cost 신호 $s \in \lbrace r, c_h/d_h, c_v/d_v\rbrace$마다 따로 계산 ($\gamma = 1$, $\lambda_{\text{GAE}} = 0.95$, 마지막 $V = 0$):

$$
\delta_t^{s} = s_t + V^{s}(o_{t+1}) - V^{s}(o_t), \qquad
\hat A_t^{s} = \sum_{k \ge 0} (\lambda_{\text{GAE}})^k \  \delta_{t+k}^{s}
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
L^{\text{CLIP}}(\theta) = -\ \mathbb{E}_t\Big[\min\big(\rho_t \tilde A_t,\  \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\ \tilde A_t\big)\Big]
$$

total loss = $L^{\text{CLIP}} + 1.0 \cdot \tfrac12 \lVert V - \text{target} \rVert^2$ (3개 head 모두). 6 epoch, minibatch 512, approx KL > 0.03이면 early stop, grad norm clip 0.5.

### 2.6 Dual step: projected gradient ascent (`update_multipliers`)

PPO 업데이트가 끝난 뒤, 이번 batch(4 episodes)의 평균 정규화 cost $\hat x_i$로:

$$
\text{multi:}\quad \lambda_i \leftarrow \max\big(0,\  \lambda_i + \eta\ (\hat x_i - 1)\big)
\qquad
\text{aggregate:}\quad \lambda \leftarrow \max\Big(0,\  \lambda + \eta\ \big(\textstyle\sum_i \hat x_i - k\big)\Big)
$$

$\eta$ = `dual_lr`, $\lambda_0 = 0.1$. **어떤 dual lr이 어디서 쓰였는지** (이번 세션에서 정리; `train.py` 기본값 0.15는 SafetyPointGoal2에서는 안 씀):

| 실험 | dual lr | 근거 |
|---|---|---|
| SafetyPointGoal2 calibration, 본 실험 (E3/E4) | **0.02** | `run_pilot.py` FROZEN, slurm 스크립트, 미팅 노트 |
| follow-up E5a | 0.04 | branch1의 `run_followups.sh` |
| E10 | aggregate_loose 0.02, multi **0.04** | `configs/safety/sg-pilot-v2.json` |

> [!TIP]
> **직관: $\lambda$는 violation의 누적합이다 (integral controller)**
>
> $$\lambda_K = \lambda_0 + \eta \sum_{k=1}^{K} (\hat x_k - 1) \quad (\text{0 아래로 안 내려가는 것만 빼면})$$
>
> - violation이 계속되는 동안 $\lambda$는 **직선으로** 계속 오른다.
> - violation이 해소된 순간 $\lambda$는 이미 크다. 그래서 cost가 budget보다 한참 아래로 내려가고(**over-constrained**), $\lambda$가 다시 줄어드는 데 오래 걸린다 (**overshoot**).
> - $\lambda$의 반응 속도는 $\eta$ × (update 횟수) × (violation의 크기)로 정해진다.

---

## 3. SafetyPointGoal2-v0 설정

**환경**: Safety-Gymnasium `SafetyPointGoal2-v0`, episode 1000 step, Point robot이 goal에 도달하면 reward.

**cost 정의** (`safety_adapter.py`, 환경의 `cost_sum`은 쓰지 않음):
- $c_{h,t}$ = 1[`cost_hazards` > 0]
- $c_{v,t}$ = 1[`cost_vases_contact` > 0 or `cost_vases_velocity` > 0]
- 한 step이 두 cost에 동시에 들어갈 수 있음
- observation: 원래 60차원 + (경과 시간 비율, absorbed flag) = 62차원

**budget (E3)**: $d_h = 31.9$, $d_v = 73.9$. 구한 방법은 §3.1.

**본 실험 설정 (E4, `run_pilot.py` FROZEN)**: 4 env × 1000 step × 500 update = 2M step, dual lr 0.02, $\lambda_0 = 0.1$, 25 update마다 eval 10 episodes, 끝나면 final eval 50 episodes, held-out 평가는 `evaluate.py --environment-seed 900000 --action-seed 900001` (50 layouts).

> [!WARNING]
> **Evaluation layout은 env seed와 chunk 크기로 정해진다**
> `collect_chunked`가 chunk마다 `make_env(seed + start, size)`를 부르기 때문. branch1(E3/E4)은 chunk 10, main의 `run_pilot.py`는 `--eval-num-envs 2`. **E4와 비교하려면 `--eval-num-envs 10`** (E10은 학습 중 eval은 2, held-out 평가만 10). 한 실험 안에서는 절대 바꾸지 말 것.

### 3.1 Budget calibration (E3): $d_h = 31.9$, $d_v = 73.9$는 어떻게 나왔나

**목표**: "아무 제약 없이 학습한 agent가 내는 cost의 절반"을 budget으로 삼는다. 너무 느슨하면 제약이 inactive($\lambda = 0$)가 되고, 너무 빡빡하면 달성할 수 없으니 그 중간.

branch1의 `pipeline.py` stage 1이 Vast.ai에서 자동으로 실행했다 (2026-10-01). 순서:

**1) Unconstrained 학습** (`runs/dev/unconstrained-s100`)

```bash
python train.py --env safety --method unconstrained --seed 100 --budget-hazard 1 --budget-vase 1 \
  --num-envs 4 --updates 125 --dual-lr 0.02 --eval-every 25 --checkpoint-every 25 \
  --eval-episodes 10 --final-eval-episodes 50 --group dev --output runs/dev/unconstrained-s100
# + run_pilot.py FROZEN의 나머지 설정 (lr 3e-4, 6 epochs, minibatch 512, clip 0.2, GAE 0.95, target KL 0.03, ...)
```

- 4 env × 1000 step × 125 update = **500,000 step**.
- `--method unconstrained`라서 $\lambda$를 쓰지 않는다. budget (1, 1)은 로그의 정규화에만 쓰이고 학습에는 영향이 없다.
- seed 100은 개발용 seed. 본 실험 seed (0/1/2)와 분리.
- network, PPO 설정은 E4 본 실험과 같다 (§3 아래 "본 실험 설정").

**2) Calibration 평가**

```bash
python evaluate.py runs/dev/unconstrained-s100/checkpoint.pt --episodes 30 --environment-seed 500000 \
  --output runs/dev/unconstrained-s100/calibration.npz
```

- 마지막 checkpoint (125 update 후)를 **stochastic** policy로 실행 (`--deterministic` 없음).
- 30 episodes, 각각 1000 step 전부.
- action seed는 evaluate.py 기본값 $300000 + 100 = 300100$.
- **layout**: branch1의 evaluate.py는 chunk 10 고정. 그래서 env seed 500000, 500010, 500020으로 simulator 10개씩을 만든다. 각 simulator의 layout seed는 `np.random.default_rng(env_seed).integers(0, 2**31 - 1, 10)`에서 나온다.
- 학습 (seed 100), 학습 중 eval (10000+seed), held-out (900000)과 겹치지 않는 처음 보는 맵.

**3) Budget 계산** (`pipeline.py`의 `calibrate`)

$$
C_i^{(k)} = \sum_{t=0}^{999} c_{i,t}^{(k)}, \qquad
\bar C_i = \frac{1}{30}\sum_{k=1}^{30} C_i^{(k)}, \qquad
d_i = \max\big(1,\ \mathrm{round}(0.5\ \bar C_i,\ 1)\big)
$$

- $C_i^{(k)}$: $k$번째 episode에서 cost $i$가 켜진 step 수.
- $\max(1, \cdot)$: cost가 거의 0일 때 budget이 0이 되는 것 (0으로 나누기)을 막는다.
- 자동 경고: goals/episode < 1이면 "task를 못 배움". 평균 cost < 2이거나 episode의 80% 이상에서 cost가 0이면 "inactive constraint".
- 역산하면 $\bar C_h \approx 63.8$ (63.7–63.9), $\bar C_v \approx 147.8$ (147.7–147.9). 원본 `runs/pipeline/status.json`은 Vast.ai에만 있어서 정확한 값은 미확인.

**4) Sanity pilot 후 freeze**

- aggregate_loose와 multi를 seed 100으로 200K step (50 update) 짧게 돌려 확인만 했다.
- PLAN에는 "한 번의 수동 조정 허용" (budget을 0.75 × mean으로, 또는 dual lr 0.005)이 있었지만, **조정 없이 자동으로 freeze**했다.

**그래서 고정 weight는**

$$
w_h = \frac{1}{31.9} \approx 0.0313, \qquad w_v = \frac{1}{73.9} \approx 0.0135
$$

- vase 1 step의 무게는 hazard 1 step의 약 0.43배.
- 이 값이 E4와 E10 내내 고정 (해석 A). unconstrained agent 기준으로 보면 두 cost 모두 $x_i \approx 2$에서 출발한다는 뜻.

> [!WARNING]
> **한계**
> - 500K step의 unconstrained agent는 아직 덜 학습된 상태. E4의 unconstrained (2M step)는 $x_h = 1.76$, $x_v = 1.84$, 즉 $C_h \approx 56$, $C_v \approx 136$. 2M agent 기준으로는 budget이 평균의 약 57% / 54%.
> - seed 1개, 30 episodes라서 추정 노이즈가 있다 (calibration.npz가 없어서 표준오차는 모름).
> - 0.5 ("절반")는 관례적인 선택이고 이론적 근거는 없다 (미팅 노트: "적당히 절반으로").
> - E3의 결과 파일은 이 checkout에 없다. 근거는 branch1 코드 (`pipeline.py`, `evaluate.py`, `safety_adapter.py`), PLAN, 미팅 노트, `configs/safety/sg-calibration.json`.

### 3.2 E4 결과 (held-out 평가, 미팅 노트 260930)

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

> [!WARNING]
> **해석 시 주의**
> loose(삼각형)와 multi(정사각형)는 feasible set이 달라서, 이 표만으로 "어느 쪽이 return이 높다"고 결론 내리면 안 된다. 두 방법 모두 수렴하지 않았다. 결과를 쓸 때는 항상 **native constraint 만족 여부**와 **per-cost "둘 다 ≤ 1"**을 따로 보고할 것.

### 3.3 E10 실험 규모 (숫자로)

출처: `configs/safety/sg-pilot-v2.json`, `configs/safety/base.json`, `run_pilot.py`, `train.py`. 서버에서 확인하려면 `runs/sg-pilot-v2-*/manifest.json`, 각 run의 `config.json`.

**한 run (method 1개 × seed 1개)**

| 항목 | 값 | 계산 |
|---|---:|---|
| episode 길이 (trajectory) | 1000 step | 고정 horizon, 중간 reset 없음 |
| update 1번의 batch | **episode 4개 = 4,000 transition** | `num_envs` 4 × 1000 step. 4개 env가 서로 다른 random layout |
| update 횟수 | **1,250** | 5M / 4,000 |
| 학습 episode 총수 | **5,000** | 1,250 × 4 |
| 학습 transition 총수 | **5,000,000** | |
| PPO gradient step / update | 최대 48 | 6 epoch × 8 minibatch (4,000 / 512 = 7개 + 나머지 416개 1개). KL > 0.03이면 그 자리에서 중단 |
| PPO gradient step 총수 | 최대 60,000 | |
| $\lambda$ 업데이트 | 1,250번 | update마다 1번, 그 batch 4 episode의 평균 cost로 |
| 중간 eval | 51번 × 10 episode | 시작 시 1번 + 25 update(100K step)마다. 매번 **같은 10개 layout** (env seed 10000+seed, chunk 2) |
| final eval | 50 episode × 2 | stochastic, deterministic. env seed 200000+seed |
| held-out eval (끝난 뒤) | 50 episode | env seed 900000, chunk 10. **E4와 같은 50개 layout**. 최종 결론은 이것 |
| checkpoint | 25 update마다 | `checkpoint-{step}.pt` + 최신 `checkpoint.pt` |

**실험 전체**

| 항목 | 값 |
|---|---|
| method | `aggregate_loose` (dual lr 0.02), `multi` (dual lr 0.04) |
| seed | 0, 1, 2 |
| run 수 | 2 × 3 = 6 |
| 학습 transition 총합 | 30M (6 × 5M) |
| 실행 방식 | `run_pilot.py --device cpu --max-parallel 3`: loose 3개 동시 → 끝나면 multi 3개 동시. tmux `sg-pilot-v2` |
| 속도 (10/23 W&B 기준) | 3개 동시 실행에서 update 1번 약 23초 (eval 포함) → run 1개 약 8시간 |

**고정 설정 (E4와 동일)**: budget $d_h = 31.9$, $d_v = 73.9$, $\lambda_0 = 0.1$, PPO lr 3e-4, clip 0.2, GAE $\lambda$ 0.95, $\gamma = 1$, target KL 0.03, entropy 0, grad clip 0.5. E4와 다른 점은 **update 수 (500 → 1,250)**와 **multi의 dual lr (0.02 → 0.04)** 두 가지뿐.

**네트워크**: actor 62 → 64 → 64 → 2 (tanh), critic 62 → 64 → 64 → 3 (reward, hazard, vase). 파라미터 약 16.7K개 (actor 8.3K, critic 8.4K).

#### 왜 GPU를 안 쓰나

E10은 **`--device cpu`로 돌고 있어서 GPU를 전혀 쓰지 않는다.** GPU 온도가 안 오르는 게 정상이다. GPU로 돌려도 빨라지지 않는다 (PLAN §2에서 측정: CPU와 CUDA 속도 같음).

- **시간의 대부분은 MuJoCo 시뮬레이션**이다. CPU 코어 하나에서 Python loop로 4개 env를 차례로 step한다 (약 400 step/s/env). update 1번에 4,000 step.
- **네트워크가 너무 작다.** 16.7K 파라미터 MLP는 CPU에서도 순식간이다. rollout 중 추론은 매 step마다 batch 4개라서 GPU로 보내면 전송 비용이 계산보다 크다.
- PPO update는 update당 최대 48번의 작은 gradient step이라 몇 초 안 걸린다.
- 그래서 속도를 올리는 방법은 GPU가 아니라 **병렬 run 수 (CPU 코어)**다.

#### 그래프 읽을 때 알아둘 것

- **eval 점은 노이즈가 크다.** 10 episode 평균이고 100K step마다 하나라서, 이웃한 점끼리 정규화 cost 합이 ±0.5~1씩 튄다. 점 하나로 판단하지 말 것.
- **$\lambda$는 eval이 아니라 train batch로 움직인다.** $\lambda$의 움직임은 `train/normalized_hazard + train/normalized_vase` (update마다, 4 episode 평균)와 비교해야 맞다. W&B에서 smoothing을 켜면 추세가 보인다.
- **중간 관찰 (10/23, 약 2.3M step, aggregate_loose)**: 세 seed 모두 $\lambda$가 1M 근처에서 꺾임 (최고 약 4.5 / 5 / 6.6). seed 1은 eval cost 합이 1.7M 이후 대체로 2 아래이고 $\lambda$가 4.5 → 3.1로 내려가는 중 (약한 overshoot, dual update가 정상적으로 반응 중). 나머지 두 seed는 합이 2 근처에서 오르내리고 $\lambda$가 평평함. 결론은 5M과 held-out 결과를 보고 낸다.

---

## 4. 코드 구조 (`spg-v2` branch 기준)

### 4.1 SafetyPointGoal2-v0 실험에 쓰는 파일

| 파일 | 역할 |
|---|---|
| `train.py` | **유일한 학습 코드.** PPO + scalar/vector Lagrangian. 주요 함수: `collect`(rollout, 신호 = `[r, cost/d]`), `generalized_advantage`, `combine_advantages`, `update_multipliers`(dual step), `collect_chunked`(고정 layout eval), `train`. 출력: `config.json`, `metrics.jsonl`, checkpoint, `evaluation.npz`, `summary.json` |
| `safety_adapter.py` | Safety-Gymnasium을 감싸서 hazard, vase cost를 **따로** 꺼냄 |
| `run_pilot.py` | method × seed matrix launcher. Safety Gym 설정은 FROZEN, 이미 끝난 run은 건너뜀(resume 가능). **(신규)** `--extra "..."`로 train.py flag 추가 (`{seed}` 치환) |
| `evaluate.py` | 저장된 checkpoint를 처음 보는 layout(held-out)에서 평가. **(신규)** `--eval-num-envs`로 평가 chunk 지정 |
| `scripts/summarize_runs.py` | **(신규)** `runs/` 아래 결과로 표 재생성 (학습 안 함). status, return, $x_i$, native feasible vs 둘 다 ≤ 1, 마지막 $\lambda$, 마지막 1/4 동안의 $\lambda$ 변화(수렴 판단용) |
| `slurm_safety_calibrate_titanxp.sh`, `slurm_safety_multicost_titanxp.sh` | titanxp cluster에서 budget calibration / 본 실험 |
| `validate_safety.py` | cost 신호가 제대로 나오는지 검증 (영상 포함) |
| `render_policy.py` | **(신규, 10/23)** checkpoint의 episode 영상 (mp4). 맵은 `evaluate.py`와 같은 규칙 (기본: held-out layout 0, env seed 900000, chunk 10). 화면 위에 step, return, goals, 누적 hazard/vase cost와 $x_i$를 표시하고, cost가 생긴 step은 빨간 테두리. 결과는 `<run>/videos/`에 mp4 + json |
| `test_core.py` | unit test 12개. 이번 세션에 추가: native feasibility ≠ per-cost, config `extends`/CLI override, repo의 config 전부 load 가능 |

### 4.2 실험 기록 체계 (이번 세션에 만듦)

| 위치 | 내용 |
|---|---|
| `docs/experiments/INDEX.md` | **실험 목록 E1–E10**: 질문, 상태, config, 어느 코드(commit/branch)로 돌렸는지, 결과 위치, 근거(code / report / notes). 모르는 것은 unknown으로 둠 |
| `configs/safety/*.json` | 실험마다 JSON 하나. `base.json`(FROZEN 설정) + `extends`로 차이만. 확인 못 한 값은 `unknown`. SafetyPointGoal2 관련: `sg-calibration`(E3), `sg-pilot-v1`(E4), `sg-pilot-v1-duallr004`(E5a), `sg-pilot-v1-conservative`(E5b), `sg-pilot-v2`(E10) |
| `train.py --config X.json --experiment-id ...` | config의 `train_args`가 기본값이 되고 CLI flag가 덮어씀. 모르는 key가 있으면 에러 |
| run의 `config.json` | 이제 git commit/branch/dirty, 사용한 experiment config, evaluation protocol(eval seed, chunk)까지 저장 |
| `reports/sg-pilot-v1/` | 미팅 노트에서 옮긴 E4 결과 요약 |
| `docs/2026-09-30-lab-meeting.pdf` | 260930 미팅 노트 |
| `CLAUDE.md` | 앞으로 coding agent가 지킬 규칙 (기존 baseline을 몰래 바꾸지 않기, 새 기능은 flag 뒤에, chunk 크기 고정, 결과 삭제 금지, commit 전 pytest 등) |
| `README.md`, `PLAN.md` | 다시 씀 / 현재 상태 추가 |

### 4.3 한 update에서 일어나는 일 (`train.py`)

```text
for update in 1..1250:      # E10: 5M step (E4는 500 = 2M)
    # 1) rollout: 4개 env에서 1000 step씩 (= episode 4개)
    signals_t = [r_t, c_hazard_t / d_h, c_vase_t / d_v]

    # 2) GAE: 신호 3개 각각 advantage 계산
    A_r, A_h, A_v = GAE(signals, critic values)

    # 3) advantage 섞기 + 표준화
    A = A_r - λ_h A_h - λ_v A_v          # multi
    A = A_r - λ (A_h + A_v)              # aggregate
    A = (A - mean) / std

    # 4) PPO: 6 epoch, minibatch 512, KL > 0.03이면 중단
    # 5) dual update: λ += η * (batch 평균 x - target), 0 아래로 clip
    # 6) 25 update마다 eval 10 episodes (seed 10000+seed), checkpoint 저장
# 끝나면 final eval 50 episodes (seed 200000+seed), evaluation은 항상 C_i vs d_i
```

### 4.4 이번 세션 commit (branch `spg-v2`, 이전 이름 `pi0-weights`, GitHub에 push됨)

- **실험 기록**: `--config`, git 정보 기록, `summarize_runs.py`, `run_pilot.py --extra`, `evaluate.py --eval-num-envs`, INDEX/configs/README/PLAN/CLAUDE.md, 보고서와 미팅 노트, E10 등록
- **정리**: 잘못 생긴 파일 삭제, `.gitignore`, branch1 archive
- **π_0 weighting**: 세션 초반에 구현했다가 교수님 제안이 아니어서 10/23 제거 (코드는 tag `archive/pi0-weights`에 보관). 제거 후 budget 모드 결과가 이전과 bit-identical인 것을 toy run으로 확인.
- **branch1 archive**: tag `archive/branch1` (b240033, GitHub에도 push). GitHub의 `branch1`, `claude/fervent-wozniak-pamdhy` branch는 삭제. 옛 코드는 `git show archive/branch1:pipeline.py`로 볼 수 있음.

---

## 5. 남은 일 (구체적으로)

0. **E10 실행 중** (tmux `sg-pilot-v2`, `runs/sg-pilot-v2.sh`): aggregate_loose 3개 (약 12h) → multi 3개 (약 12h) → held-out 평가(E4와 같은 50 layout, `--eval-num-envs 10`) → `runs/sg-pilot-v2-summary.md`. 진행 확인: `tail -n1 runs/sg-pilot-v2-*/*.log | cut -c1-200`, `python scripts/summarize_runs.py --root runs/sg-pilot-v2-loose`. 끝나면 볼 것: $\lambda$가 평평해졌는지(`lambda_change_last_quarter`), aggregate가 여전히 over-constrained인지, multi가 둘 다 ≤ 1이 되는지.
   - **영상**: `nice -n 19 python render_policy.py runs/sg-pilot-v2-loose/*/` (각 run의 최신 `checkpoint.pt`, episode 1개에 약 1분). 학습 중에는 checkout의 HEAD를 바꾸지 않도록 별도 worktree에서 실행 (아직 시작 안 한 multi run이 기록하는 git commit이 바뀌지 않게).
   - **W&B**: tmux `wandb-tail`이 `metrics.jsonl`을 60초마다 업로드. project `multi-cost-rl`, group `spg-v2-aggregate-loose`, `spg-v2-multi` (W&B 쪽은 이미 새 이름).
   - [ ] **TODO (E10이 모두 끝난 뒤) 이름 변경 `sg-pilot-v2` → `spg-v2`**. 조건: tmux `sg-pilot-v2`에 "all done", tmux `wandb-tail`이 종료됨. 학습 중에는 절대 옮기지 말 것.
     1. `mv runs/sg-pilot-v2-loose runs/spg-v2-aggregate-loose`, `mv runs/sg-pilot-v2-multi runs/spg-v2-multi`
     2. `mv runs/sg-pilot-v2.sh runs/spg-v2.sh` (`.out`도), 표 다시 만들기: `python scripts/summarize_runs.py --root runs --output runs/spg-v2-summary.md`
     3. `git mv configs/safety/sg-pilot-v2.json configs/safety/spg-v2.json`, 안의 name/commands/output 수정. `docs/experiments/INDEX.md`, 이 문서도 수정 → commit
     4. 각 run의 `config.json`, `manifest.json` 안의 옛 group 이름은 기록이므로 **고치지 않음** (config에 "renamed from sg-pilot-v2-*" 메모만)

1. **E10 결과 정리**: 끝나면 held-out 결과로 `reports/spg-v2/README.md` 작성, INDEX 업데이트. E4와 같은 표(§3.2) 형식으로 비교.
2. **랩미팅**: 해석 B 상의 (§6).
3. **`spg-v2`를 `main`에 merge** (GitHub PR). 아직 안 함.
4. GitHub의 PR #1이 자동으로 닫혔는지 확인 (`gh`가 없어서 여기서는 확인 못 함).
5. 미팅 노트의 개선 항목:
   - [x] step 수 2M → 5M (E10, 실행 중)
   - [ ] seed 5개 이상 (tuning seed와 최종 보고 seed는 분리)
   - [ ] budget sweep ($d_i$를 여러 배율로)으로 **return vs. joint pass rate** 곡선 비교 (feasible set을 맞춘 공정한 비교)

---

## 6. 내가 결정해야 할 것

### 교수님 제안의 해석 → **일단 A로 진행 (10/23)**, B는 다음 랩미팅에서 상의
교수님 말: "hazard와 vase의 violation에 따라 weighting해서 cost 합이 2가 되도록 aggregation".

- **해석 A (고정 weight)**: $w_i = 1/d_i$로 $x_h + x_v \le 2$. **지금의 `aggregate_loose`(E4)와 같다.** 할 일은 E4를 수렴할 때까지 돌리는 것 (step 수↑, dual lr↑). → 진행 중.
- **해석 B (violation에 따라 바뀌는 weight)**: 아래.

#### 해석 B 정리 (랩미팅 상의용)

**식.** $w_h + w_v = 2$를 유지하면서, 매 update마다 violation이 큰 cost 쪽으로 weight를 옮긴다. 제약은 $\sum_i w_i x_i \le 2$, $\lambda$는 하나.

$$
w_i = 2\ \frac{\max(\hat x_i - 1, 0) + \epsilon}{\sum_j \left(\max(\hat x_j - 1, 0) + \epsilon\right)}
$$

- $x_i = J_{c_i}/d_i$ (§2.2의 정규화 cost, cost $i$가 budget의 몇 배인지). 학습 중에는 진짜 $J_{c_i}$를 모르므로 dual step(§2.6)과 같은 batch 추정치 $\hat x_i$ (이번 update의 4 episode 평균 cost $/ d_i$)를 쓴다.
- 둘 다 violation이 없으면 $w = (1, 1)$ → 지금의 aggregate. 예: $\hat x = (0.7, 1.1)$이면 violation $(0, 0.1)$ → $w \approx (0, 2)$, 제약은 $2\hat x_v \le 2$. 합만 맞추는 exploit(seed 0)을 막으려는 것.
- A와의 차이: A는 weight가 학습 전에 정해져서 고정, B는 $\lambda$처럼 weight도 feedback으로 움직임. (둘 다 on-policy. on/off-policy의 차이가 아님.)

**B의 사실상의 feasible set은 multi와 같다.** 한쪽이 넘으면 weight가 그쪽으로 몰려서 $x_i \le 1$이 되므로, 결국 정사각형 $\lbrace x_h \le 1, x_v \le 1\rbrace$. 즉 B는 "feasible set은 multi, $\lambda$는 하나를 같이 씀".

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

### 그 외 선택
- E10 결과를 보고: 5M에서도 수렴하지 않으면 10M으로 늘릴지, dual lr을 더 키울지, PID Lagrangian을 넣을지.
- 앞으로의 eval chunk: E4와 비교하려면 10.

---

## 7. 용어

| 용어 | 뜻 |
|---|---|
| CMDP | Constrained MDP. reward를 최대화하되 cost 기댓값에 상한이 있는 문제 |
| budget $d_i$ | episode당 허용되는 cost $i$의 총량 |
| weight $w_i$ | cost를 더하기 전에 곱하는 계수. 지금은 $1/d_i$ (해석 A). 해석 B는 violation에 따라 바뀜 |
| $x_i$ | 정규화 cost $= J_{c_i}/d_i$. 1이면 budget에 딱 맞음 (evaluation은 항상 이것) |
| violation | 제약을 넘은 양 ($x_i - 1 > 0$) |
| over-constrained | cost가 budget보다 한참 낮음 → 필요 이상으로 reward를 포기한 상태 |
| overshoot | $\lambda$가 violation 해소 시점에 이미 너무 커서 반대쪽으로 지나치는 현상 |
| native constraint | 그 method가 학습한 제약 (loose: 합 ≤ 2 등) |
| Lagrange multiplier $\lambda$ | violation에 대한 penalty 계수. violation이 있으면 커지고 만족하면 줄어듦 |
| primal / dual step | $\pi$ 업데이트 / $\lambda$ 업데이트 |
| joint pass rate | 모든 cost가 동시에 budget 이하인 episode의 비율 |
| held-out layouts | 학습 중에 한 번도 안 본 맵. 최종 평가에만 사용 |
| chunk (`--eval-num-envs`) | eval 때 한 번에 만드는 env 개수. layout을 바꾸므로 실험 안에서 고정 |
