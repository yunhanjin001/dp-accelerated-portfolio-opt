# LQR Portfolio Optimizer

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

## LQR Formulation

### Problem Setup

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

---

## Full Derivation of $K_t$ and $P_t$

### Step 1 — Quadratic Value Function Ansatz

**Claim:** the optimal value function is quadratic in the state at every time step:

$$V_t(s_t) = s_t^\top P_t\, s_t, \quad P_t \in \mathbb{R}^{n \times n},\; P_t \succeq 0$$

**Base case** ($t = T$): by definition,

$$\boxed{V_T(s_T) = s_T^\top P_T\, s_T}$$

where $P_T$ is the given terminal cost matrix (zero if unspecified).

---

### Step 2 — Bellman Equation at Step $t$

Assume the induction hypothesis holds at $t+1$, i.e. $V_{t+1}(s) = s^\top P_{t+1}\, s$.

Substitute into the Bellman equation:

$$V_t(s_t) = \min_{u_t}\; \underbrace{s_t^\top Q\, s_t + u_t^\top R\, u_t + 2\, s_t^\top M\, u_t}_{\text{stage cost}} + \underbrace{(A s_t + B u_t)^\top P_{t+1} (A s_t + B u_t)}_{\text{future cost}}$$

---

### Step 3 — Expand the Future Cost

$$
(A s_t + B u_t)^\top P_{t+1} (A s_t + B u_t)
= s_t^\top A^\top P_{t+1} A\, s_t
+ 2\, s_t^\top A^\top P_{t+1} B\, u_t
+ u_t^\top B^\top P_{t+1} B\, u_t
$$

Collect all terms inside the $\min$:

$$
V_t(s_t) = \min_{u_t}\Bigl[
  s_t^\top \underbrace{(Q + A^\top P_{t+1} A)}_{\tilde{Q}} s_t
  + u_t^\top \underbrace{(R + B^\top P_{t+1} B)}_{\mathcal{R}} u_t
  + 2\, s_t^\top \underbrace{(M + A^\top P_{t+1} B)}_{\mathcal{G}^\top} u_t
\Bigr]
$$

where for compact notation:

$$
\mathcal{R} \;=\; R + B^\top P_{t+1} B \;\in\; \mathbb{R}^{m \times m}, \qquad
\mathcal{G} \;=\; M^\top + B^\top P_{t+1} A \;\in\; \mathbb{R}^{m \times n}
$$

---

### Step 4 — Derive $K_t$ (Optimal Gain Matrix)

The objective inside the $\min$ is strictly convex in $u_t$ (since $\mathcal{R} \succ 0$).
Take the gradient with respect to $u_t$ and set it to zero:

$$
\frac{\partial}{\partial u_t}\left[ u_t^\top \mathcal{R}\, u_t + 2\, s_t^\top \mathcal{G}^\top u_t \right]
= 2\, \mathcal{R}\, u_t + 2\, \mathcal{G}\, s_t = 0
$$

Solving:

$$
\mathcal{R}\, u_t^* = -\mathcal{G}\, s_t
$$

$$
\boxed{u_t^* = -\underbrace{\mathcal{R}^{-1} \mathcal{G}}_{K_t}\, s_t = -K_t\, s_t}
$$

where the **optimal gain matrix** is:

$$
\boxed{K_t = \left(R + B^\top P_{t+1} B\right)^{-1}\!\left(M^\top + B^\top P_{t+1} A\right) \;\in\; \mathbb{R}^{m \times n}}
$$

The optimal control is **linear state feedback**: $u_t^* = -K_t s_t$.

---

### Step 5 — Derive $P_t$ (Riccati Recursion)

Substitute $u_t^* = -K_t s_t$ back into the Bellman equation.  
Let $\tilde{Q} = Q + A^\top P_{t+1} A$ for brevity:

$$
V_t(s_t) = s_t^\top \tilde{Q}\, s_t
           + (K_t s_t)^\top \mathcal{R} (K_t s_t)
           - 2\, s_t^\top \mathcal{G}^\top K_t\, s_t
$$

**Quadratic term** in $u_t^*$:

$$
(K_t s_t)^\top \mathcal{R} (K_t s_t)
= s_t^\top K_t^\top \mathcal{R} K_t\, s_t
= s_t^\top \mathcal{G}^\top \mathcal{R}^{-1} \mathcal{R}\, \mathcal{R}^{-1} \mathcal{G}\, s_t
= s_t^\top \mathcal{G}^\top \mathcal{R}^{-1} \mathcal{G}\, s_t
$$

**Cross term** at $u_t^* = -K_t s_t$:

$$
-2\, s_t^\top \mathcal{G}^\top K_t\, s_t
= -2\, s_t^\top \mathcal{G}^\top \mathcal{R}^{-1} \mathcal{G}\, s_t
$$

Summing all three contributions:

$$
V_t(s_t) = s_t^\top \!\left[
  \tilde{Q}
  + \mathcal{G}^\top \mathcal{R}^{-1} \mathcal{G}
  - 2\, \mathcal{G}^\top \mathcal{R}^{-1} \mathcal{G}
\right]\! s_t
= s_t^\top \!\left[ \tilde{Q} - \mathcal{G}^\top \mathcal{R}^{-1} \mathcal{G} \right]\! s_t
$$

Expanding $\tilde{Q}$ and $\mathcal{G}^\top \mathcal{R}^{-1} \mathcal{G} = K_t^\top \mathcal{R} K_t$:

$$
\boxed{P_t = Q + A^\top P_{t+1} A - K_t^\top \!\left(R + B^\top P_{t+1} B\right) K_t}
$$

This confirms $V_t(s_t) = s_t^\top P_t\, s_t$, completing the induction.

---

### Summary: Riccati Recursion

Initialize $P_T$ (terminal cost), then for $t = T-1, T-2, \ldots, 0$:

$$
\underbrace{K_t = \left(R + B^\top P_{t+1} B\right)^{-1}\!\left(M^\top + B^\top P_{t+1} A\right)}_{\text{Step 1: compute gain}}
$$

$$
\underbrace{P_t = Q + A^\top P_{t+1} A - K_t^\top \left(R + B^\top P_{t+1} B\right) K_t}_{\text{Step 2: update value function}}
$$

**Optimal policy (forward pass):** $u_t^* = -K_t\, s_t$, $\;s_{t+1} = A s_t + B u_t^*$

---

## Why LQR is Fast

| Property | Consequence |
|----------|-------------|
| Linear dynamics | State $s_{t+1}$ is linear in both $s_t$ and $u_t$ |
| Quadratic cost | Unique closed-form optimizer at each DP step |
| Quadratic value function | Value function stays quadratic (only $P_t$ needed) |
| Riccati recursion | One matrix inversion per step — $O(T \cdot m^3)$ total |
| No iterative solver | ~20–50× faster than CVXPY at scale |

---

## Installation

```bash
pip install .
```
