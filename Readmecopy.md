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

## Full Derivation: How to Obtain $K_t$ and $P_t$

### Step 1 — Quadratic Value Function Ansatz

**Claim:** the optimal value function is quadratic in the state at every time step:

$$V_t(s_t) = s_t^\top P_t\, s_t, \quad P_t \in \mathbb{R}^{n \times n},\; P_t \succeq 0$$

**Base case** ($t = T$): given directly as the terminal cost,

$$\boxed{P_T \text{ is given (or set to } 0 \text{ if unspecified)}}$$

This is our starting point for the backward recursion.

---

### Step 2 — Write Out the Bellman Equation at Step $t$

Assume the induction hypothesis holds at $t+1$, i.e., $V_{t+1}(s) = s^\top P_{t+1}\, s$.

Substitute the stage cost and the future value into the Bellman equation:

$$V_t(s_t) = \min_{u_t} \Big[\; \underbrace{s_t^\top Q\, s_t + u_t^\top R\, u_t + 2\, s_t^\top M\, u_t}_{\text{stage cost}} \;+\; \underbrace{(A s_t + B u_t)^\top P_{t+1} (A s_t + B u_t)}_{\text{substitute } s_{t+1} = As_t + Bu_t} \;\Big]$$

---

### Step 3 — Expand the Future Cost Term

Expand $(A s_t + B u_t)^\top P_{t+1} (A s_t + B u_t)$ by distributing:

$$
(A s_t + B u_t)^\top P_{t+1} (A s_t + B u_t)
= s_t^\top \underbrace{A^\top P_{t+1} A}_{\text{ss term}}\, s_t
\;+\; 2\, s_t^\top \underbrace{A^\top P_{t+1} B}_{\text{su cross term}}\, u_t
\;+\; u_t^\top \underbrace{B^\top P_{t+1} B}_{\text{uu term}}\, u_t
$$

Now collect **all** terms involving $s_t^\top(\cdot)s_t$, $s_t^\top(\cdot)u_t$, and $u_t^\top(\cdot)u_t$:

$$
V_t(s_t) = \min_{u_t} \Big[\;
s_t^\top (Q + A^\top P_{t+1} A)\, s_t
\;+\; u_t^\top (R + B^\top P_{t+1} B)\, u_t
\;+\; 2\, s_t^\top (M + A^\top P_{t+1} B)\, u_t
\;\Big]
$$

Define shorthand to keep notation clean:

$$\tilde{R}_t \;=\; R + B^\top P_{t+1} B, \qquad \tilde{M}_t \;=\; M + A^\top P_{t+1} B$$

so the expression becomes:

$$V_t(s_t) = \min_{u_t} \Big[\; s_t^\top (Q + A^\top P_{t+1} A)\, s_t \;+\; u_t^\top \tilde{R}_t\, u_t \;+\; 2\, s_t^\top \tilde{M}_t\, u_t \;\Big]$$

---

### Step 4 — Take the First-Order Condition to Obtain $K_t$

The expression inside the $\min$ is **quadratic in $u_t$** and strictly convex (because $\tilde{R}_t \succ 0$). Differentiate with respect to $u_t$ and set to zero:

$$\frac{\partial}{\partial u_t}\Big[\ldots\Big] = 2\,\tilde{R}_t\, u_t + 2\,\tilde{M}_t^\top s_t = 0$$

Solve for the optimal control $u_t^*$:

$$\tilde{R}_t\, u_t^* = -\tilde{M}_t^\top s_t$$

$$\boxed{u_t^* = -\underbrace{\tilde{R}_t^{-1}\, \tilde{M}_t^\top}_{K_t}\, s_t \;=\; -K_t\, s_t}$$

where the **optimal feedback gain matrix** is:

$$\boxed{K_t \;=\; (R + B^\top P_{t+1} B)^{-1}\,(M + A^\top P_{t+1} B)^\top}$$

> **How to read this:** $K_t$ maps the current state $s_t$ to the optimal trade $u_t^*$.  
> - The $(R + B^\top P_{t+1} B)^{-1}$ part normalizes by the effective control cost (original $R$ plus the cost the next step will incur through $B$).  
> - The $(M + A^\top P_{t+1} B)^\top$ part encodes how much each unit of state creates cross-cost now ($M$) and future cost after propagating through $A$ and $B$.

---

### Step 5 — Substitute $u_t^* = -K_t s_t$ Back to Obtain $P_t$

Now substitute $u_t^* = -K_t s_t$ back into the value function expression from Step 3:

**Stage cost terms:**

$$
s_t^\top Q\, s_t + (-K_t s_t)^\top R\, (-K_t s_t) + 2\, s_t^\top M\, (-K_t s_t)
= s_t^\top \big(Q + K_t^\top R\, K_t - 2\, M K_t\big)\, s_t
$$

**Future cost term** (substitute $u_t^* = -K_t s_t$ into $\tilde{R}_t$ and $\tilde{M}_t$ terms):

$$
(-K_t s_t)^\top \tilde{R}_t (-K_t s_t) + 2\, s_t^\top \tilde{M}_t (-K_t s_t)
= s_t^\top \big(K_t^\top \tilde{R}_t K_t - 2\, \tilde{M}_t K_t\big)\, s_t
$$

Since $K_t = \tilde{R}_t^{-1} \tilde{M}_t^\top$, we have $\tilde{R}_t K_t = \tilde{M}_t^\top$ and $\tilde{M}_t K_t = \tilde{M}_t \tilde{R}_t^{-1} \tilde{M}_t^\top$. After simplification all cross terms cancel, leaving:

$$V_t(s_t) = s_t^\top P_t\, s_t$$

where the **discrete-time Riccati recursion** is:

$$\boxed{P_t \;=\; Q + A^\top P_{t+1} A \;-\; (M + A^\top P_{t+1} B)\,(R + B^\top P_{t+1} B)^{-1}\,(M + A^\top P_{t+1} B)^\top}$$

This can be written compactly using $K_t$:

$$P_t = Q + A^\top P_{t+1} A - K_t^\top (R + B^\top P_{t+1} B)\, K_t$$

> **Key insight:** Each $P_t$ is a positive semi-definite matrix that encodes the optimal cost-to-go starting from state $s_t$. It is computed purely by backward matrix algebra — no iterative solver required.

---

### Step 6 — Summary of the Backward Pass

Starting from $P_T$ (given), iterate for $t = T-1, T-2, \ldots, 0$:

| Step | Formula | What you get |
|------|---------|--------------|
| 1 | $\tilde{R}_t = R + B^\top P_{t+1} B$ | Effective control cost |
| 2 | $\tilde{M}_t = M + A^\top P_{t+1} B$ | Effective cross cost |
| 3 | $K_t = \tilde{R}_t^{-1}\, \tilde{M}_t^\top$ | Optimal feedback gain |
| 4 | $P_t = Q + A^\top P_{t+1} A - K_t^\top \tilde{R}_t\, K_t$ | Updated value matrix |

Then in the **forward pass**, starting from $s_0$:

$$u_t^* = -K_t\, s_t, \qquad s_{t+1} = A\, s_t + B\, u_t^*$$

---

## Why LQR is Fast

| Feature | Benefit |
|---------|---------|
| **Speed** | ~40–100x faster than CVXPY |
| **Scalability** | Efficient for high-dimensional problems |
| **Interpretability** | Explicit gain matrices $K_t$ at every step |
| **Memory** | Pure matrix operations, no solver overhead |

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
