# LQR for Multi-period Portfolio Optimiztion

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A **fast**, **scalable** Dynamic Programming solver for multi-period portfolio optimization problems using Linear-Quadratic Regulator (LQR) theory.

**Implementation based on Dynamic Programming principles from:**  
*The Elements of Statistical Learning: Data Mining, Inference, and Prediction* by Hastie, Tibshirani, and Friedman (2009).

---

## Why LQR for Portfolio Optimization?

Optimal portfolio execution is a **multi-period decision problem** where each trading action affects immediate costs (market impact, transaction costs) and future states (inventory, signals, execution risk).

Traditional convex optimization (e.g., CVXPY) becomes **computationally expensive** at scale and fails to exploit the problem's **temporal structure**. Dynamic Programming offers a superior approach through backward induction and closed-form solutions.

---

## Dynamic Programming Framework

### The Bellman Equation

DP recursively defines optimal value via the **Bellman equation**:

$$V(s) = \min_{u \in \mathcal{A}(s)} \left[ f(s, u) + V\big(T(s, u)\big) \right]$$

- $V(s)$: value function (minimum total cost-to-go from state $s$)
- $f(s, u)$: immediate stage cost
- $T(s, u)$: state transition, $s_{t+1} = T(s_t, u_t)$

**Key Insight:** Optimal cost-to-go = immediate stage cost + optimal continuation cost.

### Backward Induction

1. **Initialize** at terminal time $t = T$:

$$V_T(s) = g(s)$$

2. **Recurse backward** for $t = T-1, \ldots, 0$:

$$V_t(s_t) = \min_{u_t} \left[ c(s_t, u_t) + V_{t+1}\!\left(T(s_t, u_t)\right) \right]$$

3. **Extract optimal policy**:

$$\pi_t^*(s) = \arg\min_{u_t} \left[ c(s_t, u_t) + V_{t+1}\!\left(T(s_t, u_t)\right) \right]$$

---

## LQR Formulation (Special Case of DP)

### Condition Setup

**State dynamics (linear):**

$$s_{t+1} = A\, s_t + B\, u_t$$

| Symbol | Shape | Description |
|--------|-------|-------------|
| $s_t$  | $n \times 1$ | State vector (e.g., portfolio weights, alpha signals, cost-basis) |
| $u_t$  | $m \times 1$ | Control vector (trades / rebalancing) |
| $A$    | $n \times n$ | State transition matrix |
| $B$    | $n \times m$ | Control input matrix |

**Objective (quadratic cost):**

$$\min_{\{u_t\}_{t=0}^{T-1}} \underbrace{\sum_{t=0}^{T-1} \left[ s_t^\top Q\, s_t + u_t^\top R\, u_t + 2\, s_t^\top M\, u_t \right]}_{\text{running cost}} + \underbrace{s_T^\top P_T\, s_T}_{\text{terminal cost}}$$

| Symbol | Shape | Description |
|--------|-------|-------------|
| $Q$    | $n \times n$ | State cost (PSD): penalizes deviation of states |
| $R$    | $m \times m$ | Control cost (PD): penalizes large trades |
| $M$    | $n \times m$ | Cross-term cost: couples state and control costs |
| $P_T$  | $n \times n$ | Terminal cost (PSD): penalizes terminal state |

**Problem: Minimise $\displaystyle\sum_{t=0}^{T-1}\bigl[s_t^\top Q s_t + u_t^\top R u_t + 2s_t^\top M u_t\bigr]$ subject to $s_{t+1} = As_t + Bu_t$.**

---

## Full Derivation of $K_t$ and $P_t$

### Step 1 — Quadratic Ansatz
 
Assume the optimal cost-to-go is quadratic at every step:
 
$$V_t(s_t) = s_t^\top P_t\, s_t, \qquad P_T \text{ given (often } 0\text{)}$$
 
We verify this holds for all $t$ by induction working backward from $T$.
 
---
 
### Step 2 — Bellman Equation
 
Assume $V_{t+1}(s) = s^\top P_{t+1} s$. Substituting $s_{t+1} = As_t + Bu_t$:
 
$$V_t(s_t) = \min_{u_t}\Big[s_t^\top Q s_t + u_t^\top R u_t + 2s_t^\top M u_t + (As_t+Bu_t)^\top P_{t+1}(As_t+Bu_t)\Big]$$
 
Expand and collect by term type:
 
$$= \min_{u_t}\Big[\,s_t^\top(Q+A^\top P_{t+1}A)\,s_t + u_t^\top\underbrace{(R+B^\top P_{t+1}B)}_{\tilde{R}_t}u_t + 2\,s_t^\top\underbrace{(M+A^\top P_{t+1}B)}_{\tilde{M}_t}u_t\,\Big]$$
 
---
 
### Step 3 — Optimal Control
 
The expression is strictly convex in $u_t$ (since $\tilde{R}_t \succ 0$). Setting $\partial/\partial u_t = 0$:
 
$$2\,\tilde{R}_t\,u_t + 2\,\tilde{M}_t^\top s_t = 0$$
 
$$\boxed{u_t^* = -K_t s_t, \qquad K_t = \tilde{R}_t^{-1}\tilde{M}_t^\top = (R+B^\top P_{t+1}B)^{-1}(M+A^\top P_{t+1}B)^\top}$$
 
---
 
### Step 4 — Riccati Recursion
 
Substituting $u_t^* = -K_t s_t$ back into Step 2, cross terms cancel and we recover $V_t = s_t^\top P_t s_t$ with:
 
$$\boxed{P_t = Q + A^\top P_{t+1}A - K_t^\top\tilde{R}_t\,K_t}$$
 
---
 
### Step 5 — Algorithm
 
**Backward pass** — from $P_T$, for $t = T-1,\ldots,0$:
 
| | Formula | Gives |
|:--|:--|:--|
| 1 | $\tilde{R}_t = R + B^\top P_{t+1}B$ | effective control cost |
| 2 | $\tilde{M}_t = M + A^\top P_{t+1}B$ | effective cross cost |
| 3 | $K_t = \tilde{R}_t^{-1}\tilde{M}_t^\top$ | optimal feedback gain |
| 4 | $P_t = Q + A^\top P_{t+1}A - K_t^\top\tilde{R}_t K_t$ | cost-to-go matrix |
 
**Forward pass** — from $s_0$:
 
$$u_t^* = -K_t s_t \qquad s_{t+1} = As_t + Bu_t^*$$
 


> **Why LQR is Fast:** Linear dynamics + quadratic costs = closed-form solutions. No iterative optimization required.

| Feature | Benefit |
|---------|---------|
| **Speed** | ~20-50x faster than CVXPY |
| **Scalability** | Efficient for high-dimensional problems |
| **Interpretability** | Explicit gain matrices |
| **Memory** | Pure matrix operations |

---

## Installation

```bash
pip install .
```

---

## API Reference

### `solve_and_execute_lqr(T, A, B, Q, R, M, s0, P_terminal=None)`

Solve the LQR problem via backward induction and immediately execute the optimal policy.

**Parameters:**
- `T` (int): Time horizon (number of steps)
- `A` (ndarray): State transition matrix $(n, n)$
- `B` (ndarray): Control input matrix $(n, m)$
- `Q` (ndarray): State cost matrix $(n, n)$, must be PSD
- `R` (ndarray): Control cost matrix $(m, m)$, must be PD
- `M` (ndarray): Cross-term cost matrix $(n, m)$
- `s0` (ndarray): Initial state vector $(n,)$
- `P_terminal` (ndarray, optional): Terminal cost matrix $(n, n)$

**Returns** `LQRResult` with:
- `K_gains` (ndarray): Optimal feedback gains $(T, m, n)$
- `s_path` (ndarray): State trajectory $(T+1, n)$
- `u_path` (ndarray): Control trajectory $(T, m)$

>Note. the LQR implementation was validated against Merton's analytical solution, confirming the correctness of the numerical approach (shown in the section A of demo.ipynb).

---

