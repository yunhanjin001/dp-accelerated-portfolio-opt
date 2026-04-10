# 2026_cu_numeth

# Introduction to Dynamic Programming (DP)

## Shortest Path Problem Motivation

Consider a shortest-path problem with a large number of nodes.

The naive approach is a **combinatorial search** over all paths ( Q ) from start node ( N_b ) to end node ( N_e ):

$$
\min_{Q, ; Q_0 = N_b,; Q_T = N_e} C(S)
$$

### Key Observations

*  **Overlapping subproblems**
  Different paths repeatedly visit the same states (e.g., arriving at Denver)

*  **Markov property**
  Future cost depends only on current state
  (e.g., cost from Denver → LA independent of earlier path)


### Recursive Form (Bellman Equation)

$$
V(s_0) = \max_{u_0 \in A(s_0)} \left$$ f(s_0, u_0) + V(T(s_0, u_0)) \right$$
$$

---

### Solution Method

* Use **Backward Induction**:

  1. Solve final step ( T )
  2. Move backward to ( T-1 )
  3. Repeat until ( t=0 )

---

#  DP for Portfolio Optimization

## Problem Setup

We consider a discretized utility:

$$
U(u) = \sum_{t=1}^{T} \left( f_t u_t - \frac{\lambda}{2} \sigma^2 (p_{t-1} + u_t)^2 - \left( \frac{\gamma}{2} u_t^2 + D_t u_t \right) \right)
$$

### Assumptions

* Equal time intervals
* Constant volatility
* Exponential decay dynamics:

$$
\beta = e^{-\rho \Delta t}
$$

* Initial and final positions = 0

---

## Model Dynamics

### Forecast

$$
f_t = \mathbb{E}$$r_{t \to T} | \mathcal{F}_0$$ = \rho^t f_0
$$

### Price Impact (OW Propagator)

$$
D_t = \beta (D_{t-1} + \gamma u_{t-1})
$$

---

## State Representation

Define state vector:

$$
S_t =
\begin{bmatrix}
p_{t-1} \
f_t \
D_t
\end{bmatrix}
$$

---

## Linear State Transition

$$
S_{t+1} = A S_t + B u_t
$$

Where:

$$
A =
\begin{bmatrix}
1 & 0 & 0 \
0 & \rho & 0 \
0 & 0 & \beta
\end{bmatrix}, \quad
B =
\begin{bmatrix}
1 \
0 \
\beta \gamma
\end{bmatrix}
$$

---

## Quadratic Cost Function

$$
C_t = S_t^\top Q S_t + u_t^\top R u_t + 2 S_t^\top M u_t
$$

Where:

$$
Q =
\begin{bmatrix}
\frac{\lambda \sigma^2}{2} & 0 & 0 \
0 & 0 & 0 \
0 & 0 & 0
\end{bmatrix}
$$

$$
R = \frac{\lambda}{2} \sigma^2 + \frac{\gamma}{2}
$$

$$
M =
\begin{bmatrix}
\frac{\lambda}{2} \sigma^2 \
-\frac{1}{2} \
-\frac{1}{2}
\end{bmatrix}
$$

---

# Bellman Equation (LQR Form)

Assume value function is quadratic:

$$
V_t(S_t) = S_t^\top \Phi_t S_t
$$

---

## Recursive Form

$$
V_t(S_t) = \min_{u_t} \left$$
S_t^\top Q S_t + u_t^\top R u_t + 2 S_t^\top M u_t

* S_{t+1}^\top \Phi_{t+1} S_{t+1}
  \right$$
  $$

Substitute:

$$
S_{t+1} = A S_t + B u_t
$$

---

## Optimal Control (FOC)

$$
u_t^* =

* (R + B^\top \Phi_{t+1} B)^{-1}
  (M^\top + B^\top \Phi_{t+1} A) S_t
  $$

---

## Gain Matrix

$$
K_t =
(R + B^\top \Phi_{t+1} B)^{-1}
(M^\top + B^\top \Phi_{t+1} A)
$$

---

## Final Policy

$$
u_t^* = -K_t S_t
$$

---

## Value Function Update

$$
\Phi_t =
Q + A^\top \Phi_{t+1} A

* K_t^\top (R + B^\top \Phi_{t+1} B) K_t
  $$

---

# Key Insight

> Under **linear dynamics + quadratic cost (LQR)**,
> multi-period optimization reduces to **matrix recursion**

---

#  Performance Comparison

| Method    | Execution Time | Speed |
| --------- | -------------- | ----- |
| CVXPY     | 15.45 ms       | 1x    |
| DP Solver | 0.65 ms        | ~23x  |

