# Dynamic Programming-Based Optimization for Multi-Period Portfolio Execution

## Motivation

Optimal portfolio execution is inherently a **multi-period decision problem**. Each trading action impacts not only immediate costs (e.g., market impact, transaction costs) but also future states such as remaining inventory, predictive signals, and execution risk. This creates a setting where decisions must be made **sequentially and strategically over time**, rather than in isolation.

A common approach is to formulate the problem as a large-scale convex optimization and solve it using tools like CVXPY. While flexible, these methods can become **computationally expensive** as the horizon or state dimension grows and fail to exploit the problem’s **temporal structure**.

Dynamic Programming (DP) provides a more natural and efficient framework. By leveraging the **principle of optimality**, DP decomposes the global optimization problem into a sequence of smaller subproblems. Solving the problem **backward in time** allows us to derive recursive solutions to the Bellman equation.

In our setting, the combination of **linear dynamics** and **quadratic costs** leads to a structured solution with:
- Closed-form **gain matrices**
- Explicit **optimal trading policies**
- Efficient **forward simulation** of the trading trajectory

This approach avoids repeatedly solving optimization problems and instead reduces computation to **matrix operations**, resulting in:

- 🚀 **Significant speed improvements**
- 📈 **Scalability to high-dimensional problems**
- 🧠 **Greater interpretability of trading strategies**

Overall, Dynamic Programming transforms the execution problem from a heavy numerical task into a **fast, structured, and analytically tractable solution framework**.

## Objective

Formally, the goal is to choose a sequence of actions that maximizes total expected reward over the trading horizon:

$$
\max_{\{u_t\}} \sum_{t=0}^{T-1} f(s_t, u_t) + g(s_T)
$$

## DP General Solution Method
**Bellman Equation (General)**

$$
V(s) = \max_{u \in A(s)} \left[ f(s, u) + V\big(T(s, u)\big) \right]
$$

- $V(s)$: value function (optimal future objective from state \( s \))  
- $f(s, u)$: immediate reward (or cost)  
- $T(s, u)$: state transition

$$
s_{t+1} = T(s_t, u_t)
$$

- $A(s)$: feasible action set  

> Optimal value = immediate reward + optimal continuation value

Then use **Backward Induction**:

  1. Solve final step ( $t=T$ )

  $$
  V_T(s) = g(s)
  $$
  
  $$
  \pi_T^*(s)\ \text{is trivial (no action at terminal time)}
  $$

  2. Move backward to ( $t = T-1$ )
  
  $$
  V_{T-1}(s) = \max_{u \in A(s)} \left[ f(s, u) + V_{T}\big(T(s, u)\big) \right]
  $$ 
  $$
  \pi_{T-1}^*(s) = \arg\max_{u \in A(s)} \left[ f(s, u) + V_{T}\big(T(s, u)\big) \right]
  $$
  
  3. Repeat until ( $t = T-1, \dots, 0$ )
  
  $$
  V_0(s)\ \text{: optimal value}
  $$
  
  $$
  \{\pi_t^*(s)\}_{t=0}^{T-1}\ \text{: optimal policy}
  $$

DP computes the optimal trading path via **backward recursion + forward simulation**, avoiding repeated large-scale optimization.

## Problem Setup

We consider a discretized utility:

$$
U(u) = \sum_{t=1}^{T} \left( f_t u_t - \frac{\lambda}{2} \sigma^2 (p_{t-1} + u_t)^2 - \left( \frac{\gamma}{2} u_t^2 + D_t u_t \right) \right)
$$

- $\lambda$: risk aversion parameter  
- $\sigma^2$: variance of asset returns  
- $\gamma$: temporary impact coefficient  
- $\beta$: decay rate of impact ($\beta = e^{-\rho \Delta t}$)  
- $\rho$: signal decay rate  

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
f_t = \mathbb{E}\big[r_{t \to T} | \mathcal{F}_0] = \rho^t f_0
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
p_{t-1} \\
f_t \\
D_t
\end{bmatrix}
$$

- $p_{t-1}$: inventory (Amount of asset remaining to trade at time $t$)  
- $f_t$: predictive signal (expected return / alpha)  
- $D_t$: market impact (temporary + decaying price impact from past trades)
---

## Linear State Transition

$$
S_{t+1} = A S_t + B u_t
$$

Where:

$$
A =
\begin{bmatrix}
1 & 0 & 0 \\
0 & \rho & 0 \\
0 & 0 & \beta
\end{bmatrix}, \quad
B =
\begin{bmatrix}
1 \\
0 \\
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
\frac{\lambda \sigma^2}{2} & 0 & 0 \\
0 & 0 & 0 \\
0 & 0 & 0
\end{bmatrix}
$$

$$
R = \frac{\lambda}{2} \sigma^2 + \frac{\gamma}{2}
$$

$$
M =
\begin{bmatrix}
\frac{\lambda}{2} \sigma^2 \\
-\frac{1}{2} \\
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
V_t(S_t) = \min_{u_t} \left[ S_t^\top Q S_t + u_t^\top R u_t + 2 S_t^\top M u_t + S_{t+1}^\top \Phi_{t+1} S_{t+1} \right]
$$


Substitute:

$$
S_{t+1} = A S_t + B u_t
$$

---

## Optimal Control (FOC)

$$
\begin{aligned}
u_t^* = - (R + B^\top \Phi_{t+1} B)^{-1}
(M^\top + B^\top \Phi_{t+1} A) S_t
\end{aligned}
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
\Phi_t = Q + A^\top \Phi_{t+1} A - K_t^\top (R + B^\top \Phi_{t+1} B) K_t
$$

---

## Key Insight

> Under **linear dynamics + quadratic cost (LQR)**,
> multi-period optimization reduces to **matrix recursion**

---

##  Performance Comparison

| Method    | Execution Time | Speed |
| --------- | -------------- | ----- |
| CVXPY     | 15.45 ms       | 1x    |
| DP Solver | 0.65 ms        | ~23x  |

