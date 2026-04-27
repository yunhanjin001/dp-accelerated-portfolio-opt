# LQR Portfolio Optimizer

[![PyPI version](https://badge.fury.io/py/lqr-portfolio-optimizer.svg)](https://badge.fury.io/py/lqr-portfolio-optimizer)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A **fast**, **scalable** Dynamic Programming solver for multi-period portfolio optimization problems using Linear-Quadratic Regulator (LQR) theory.

**Implementation based on Dynamic Programming principles from:**  
*The Elements of Statistical Learning: Data Mining, Inference, and Prediction* by Hastie, Tibshirani, and Friedman (2009).

---

## 🚀 Why LQR for Portfolio Optimization?


Optimal portfolio execution is a **multi-period decision problem** where each trading action affects immediate costs (market impact, transaction costs) and future states (inventory, signals, execution risk).

Traditional convex optimization (e.g., CVXPY) becomes **computationally expensive** at scale and fails to exploit the problem's **temporal structure**. Dynamic Programming offers a superior approach through backward induction and closed-form solutions.

---

## 🧮 Dynamic Programming Framework

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
3. **Extract policy**: $\pi_t^*(s) = \arg\max_u \left[ f(s, u) + V_{t+1}\big(T(s, u)\big) \right]$

---

## 📐 LQR Formulation

### Objective

Minimize quadratic cost subject to linear dynamics:

$$\min_{\{u_t\}} \sum_{t=0}^{T-1} \left[ s_t^\top Q s_t + u_t^\top R u_t + 2 s_t^\top M u_t \right]$$

$$s_{t+1} = A s_t + B u_t$$

### Closed-Form Solution

**Optimal policy:** $u_t^* = -K_t s_t$

**Gain matrices** (via Riccati recursion):

$$K_t = (R + B^\top P_{t+1} B)^{-1} (M^\top + B^\top P_{t+1} A)$$

$$P_t = Q + A^\top P_{t+1} A - K_t^\top (R + B^\top P_{t+1} B) K_t$$

**Why LQR is Fast:** Linear dynamics + quadratic costs = closed-form solutions. No iterative optimization required!

| Feature | Benefit |
|---------|---------|
| 🚀 **Speed** | ~20-50x faster than CVXPY |
| 📈 **Scalability** | Efficient for high-dimensional problems |
| 🧠 **Interpretability** | Explicit gain matrices |
| 💾 **Memory** | Pure matrix operations |

---

## 📦 Installation

```bash
pip install lqr-portfolio-optimizer
```

---

## 🎯 Quick Start

```python
import numpy as np
from lqr_portfolio_optimizer import solve_lqr, execute_lqr

# Problem parameters (optimal execution example)
T = 30  # Time horizon
gamma = 1.0  # Risk aversion
sigma_sq = 0.04  # Variance
eta = 0.1  # Execution cost coefficient
rho = 0.95  # Signal decay rate
beta = 0.8  # Impact decay rate
alpha0 = 1.0  # Initial signal

# State transition matrices
A = np.array([
    [1, 0, 0],
    [0, rho, 0],
    [0, 0, beta]
])

B = np.array([
    [1],
    [0],
    [beta * eta]
])

# Cost matrices
Q = np.diag([0.5 * gamma * sigma_sq, 0, 0])
R = np.array([[0.5 * gamma * sigma_sq + 0.5 * eta]])
M = np.array([
    [0.5 * gamma * sigma_sq],
    [-0.5],
    [0.5]
])

# Initial state [position, signal, impact]
s0 = np.array([0.0, alpha0, 0.0])

# Solve via Dynamic Programming
K_gains = solve_lqr(T, A, B, Q, R, M)
s_path, u_path = execute_lqr(T, A, B, K_gains, s0)

print(f"Optimal trading trajectory computed!")
print(f"Final position: {s_path[-1, 0]:.4f}")
```

---

## 🎨 Examples

See `notebooks/demo.ipynb` for complete implementations of both problems with visualizations and benchmarking.

### 1. Optimal Execution Problem

**Objective**: Execute a large order while balancing:
- **Alpha capture**: Trade in the signal direction while it lasts
- **Risk**: Large positions expose to volatility
- **Execution impact**: Past trades leave lingering market impact

**State**: $s_t = [w_t, \alpha_t, c_t]^\top$
- $w_t$: Current position
- $\alpha_t$: Predictive signal (decays as $\alpha_{t+1} = \rho \alpha_t$)
- $c_t$: Accumulated impact (decays as $c_{t+1} = \beta c_t + \beta \eta u_t$)

### 2. Multi-Asset Markowitz Portfolio

**Objective**: Optimize portfolio weights across $n$ assets considering:
- **Expected returns**: Maximize $\mu^\top w_t$
- **Risk**: Minimize variance $w_t^\top \Sigma w_t$
- **Transaction costs**: Penalize trading $\gamma_{\text{tc}} \|u_t\|^2$

**Augmented State**: $s_t = [w_t, 1]^\top$ where $w_t \in \mathbb{R}^n$

---

## 🛠️ API Reference

### `solve_lqr(T, A, B, Q, R, M)`

Compute optimal feedback gains via backward induction.

**Parameters:**
- `T` (int): Time horizon
- `A` (ndarray): State transition matrix (n, n)
- `B` (ndarray): Control input matrix (n, m)
- `Q` (ndarray): State cost matrix (n, n), must be PSD
- `R` (ndarray): Control cost matrix (m, m), must be PD
- `M` (ndarray): Cross-term cost matrix (n, m)

**Returns:**
- `K_gains` (ndarray): Optimal feedback gains (T, m, n)

---

### `execute_lqr(T, A, B, K_gains, s0)`

Forward simulation using pre-computed gains.

**Parameters:**
- `T` (int): Time horizon
- `A` (ndarray): State transition matrix (n, n)
- `B` (ndarray): Control input matrix (n, m)
- `K_gains` (ndarray): Pre-computed gains from `solve_lqr` (T, m, n)
- `s0` (ndarray): Initial state (n,)

**Returns:**
- `s_path` (ndarray): State trajectory (T+1, n)
- `u_path` (ndarray): Control trajectory (T, m)

---

## 📚 Use Cases

- **Algorithmic Trading**: Optimal execution of large orders
- **Portfolio Management**: Dynamic asset allocation with transaction costs
- **Market Making**: Inventory management with adverse selection
- **Quantitative Research**: Backtesting multi-period strategies
- **Academic Research**: Baseline for reinforcement learning approaches

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🔗 References

- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of 
Statistical Learning: Data Mining, Inference, and Prediction* (2nd ed.). 
Springer.
