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

$$V(s) = \max_{u \in A(s)} \left[ f(s, u) + V\big(T(s, u)\big) \right]$$

- $V(s)$: value function  
- $f(s, u)$: immediate reward  
- $T(s, u)$: state transition, $s_{t+1} = T(s_t, u_t)$

**Key Insight:** Optimal value = immediate reward + optimal continuation value

### Backward Induction

1. **Initialize** at $t=T$: $V_T(s) = g(s)$
2. **Recurse backward** for $t = T-1, \ldots, 0$:

   $$V_t(s) = \max_{u} \left[ f(s, u) + V_{t+1}\big(T(s, u)\big) \right]$$

3. **Extract policy**:

   $$\pi_t^*(s) = \arg\max_u \left[ f(s, u) + V_{t+1}\big(T(s, u)\big) \right]$$

---

## LQR Formulation

### Objective

Minimize quadratic cost subject to linear dynamics:

$$\min_{\{u_t\}} \sum_{t=0}^{T-1} \left[ s_t^\top Q s_t + u_t^\top R u_t + 2 s_t^\top M u_t \right]$$

$$s_{t+1} = A s_t + B u_t$$

### Closed-Form Solution

**Optimal policy:** $u_t^* = -K_t s_t$

**Gain matrices** using Riccati recursion:

$$K_t = (R + B^\top P_{t+1} B)^{-1} (M^\top + B^\top P_{t+1} A)$$

$$P_t = Q + A^\top P_{t+1} A - K_t^\top (R + B^\top P_{t+1} B) K_t$$

**Why LQR is Fast:** Linear dynamics + quadratic costs = closed-form solutions. No iterative optimization required.

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
