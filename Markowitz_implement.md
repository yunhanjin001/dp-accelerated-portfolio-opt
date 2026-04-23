# Multi-Asset Markowitz: DP/LQR vs CVXPY

## 1. Problem Setup

Consider a dynamic portfolio allocation problem with $n$ assets over $T$ periods. Let

- $w_t \in \mathbb{R}^n$: portfolio weights at time $t$
- $u_t \in \mathbb{R}^n$: trade (rebalancing) at time $t$
- $\mu \in \mathbb{R}^n$: expected return vector
- $\Sigma \in \mathbb{R}^{n\times n}$: covariance matrix (positive semidefinite)
- $\lambda > 0$: risk aversion coefficient
- $\gamma_{tc} > 0$: transaction cost coefficient

The dynamics are

$$
w_{t+1}=w_t+u_t,\quad t=0,\dots,T-1,\quad w_0\ \text{given}.
$$

The objective is to minimize total cost:

$$
\min_{\{u_t\}_{t=0}^{T-1}}
\sum_{t=0}^{T-1}
\left(
-\mu^\top w_{t+1}
+\frac{\lambda}{2}w_{t+1}^\top\Sigma w_{t+1}
+\frac{\gamma_{tc}}{2}u_t^\top u_t
\right).
$$

This matches the CVXPY objective used in `solve_cvxpy_markowitz(...)`.

---

## 2. CVXPY Approach

In `Markowitz_test.py`, the CVXPY formulation defines full-horizon variables:

- `W` with shape $(T+1,n)$ for $\{w_t\}$
- `U` with shape $(T,n)$ for $\{u_t\}$

Then it imposes:

1. Initial condition: $W_0=w_0$
2. Transition constraint: $W_{t+1}=W_t+U_t$

The stage cost is accumulated across all periods and solved by OSQP as a convex quadratic program.

---

## 3. DP/LQR Reformulation and Recursion

To apply dynamic programming, the linear return term is embedded into a quadratic form using an augmented state:

$$
S_t=\begin{bmatrix}w_{t-1}\\ 1\end{bmatrix},\quad
S_{t+1}=A_{\text{aug}}S_t+B_{\text{aug}}u_t,
$$

with

$$
A_{\text{aug}}=I_{n+1},\quad
B_{\text{aug}}=
\begin{bmatrix}
I_n\\
0
\end{bmatrix}.
$$

The stage cost is written in LQR form:

$$
\ell(S_t,u_t)=S_t^\top Q_{\text{aug}}S_t + u_t^\top R u_t + 2S_t^\top M u_t.
$$

In the code:

- $Q_{\text{aug}}$ captures quadratic risk and linear return terms in augmented form
- $R=\frac{\gamma_{tc}}{2}I+\frac{\lambda}{2}\Sigma$
- $M$ contains state-control cross terms (from $w^\top \Sigma u$ and $\mu^\top u$)

The terminal cost is set to $\Phi_T=0$ to match the finite-horizon sum objective.

### 3.1 Backward Riccati Recursion

For $t=T-1,\dots,0$:

$$
K_t=
\left(R+B^\top\Phi_{t+1}B\right)^{-1}
\left(M^\top+B^\top\Phi_{t+1}A\right),
$$

$$
\Phi_t=
Q+A^\top\Phi_{t+1}A
-K_t^\top\left(R+B^\top\Phi_{t+1}B\right)K_t.
$$

### 3.2 Forward Policy Execution

The optimal feedback law is

$$
u_t^*=-K_tS_t,\qquad w_{t+1}=w_t+u_t^*.
$$

This is implemented by:

- `solve_lqr_markowitz(...)`: backward pass to compute $\{K_t\}$
- `execute_lqr_markowitz(...)`: forward pass to generate $\{w_t\}$

---

## 4. Experiment Design

The script includes two experiments:

1. **Single-run comparison** (`run_single_comparison`)  
   Fix $T=30,n=5$ and compare:
   - Runtime: `time_dp` vs `time_cvx`
   - Objective value: `obj_dp` vs `obj_cvx`
   - Speedup: `speedup = time_cvx / time_dp`

2. **Scaling experiment** (`run_scaling_experiment`)  
   Fix $T=30$, vary $n\in\{2,3,5,7,10,15,20\}$, and run multiple random trials per $n$ to report mean and standard deviation.

`plot_all(...)` also produces four panels:

- Norm trajectory of portfolio weights (DP vs CVXPY)
- Per-asset weight trajectories under DP
- Runtime vs number of assets (with variance bands)
- Single-run runtime bar chart with objective summary

---

## 5. Mathematical Conclusion and Final Result

### 5.1 Mathematical Conclusion

This problem is a finite-horizon **linear dynamics + quadratic cost** control problem, so it can be exactly cast as an LQR-type DP problem.  
Under consistent modeling assumptions, DP/LQR and CVXPY solve the same convex optimization target; objective values should be nearly identical, while runtime can differ significantly.

### 5.2 Final Result (from project report)

Based on the existing project result in `README.md`:

| Method | Execution Time | Speed |
| --- | ---: | ---: |
| CVXPY | 15.45 ms | 1x |
| DP Solver | 0.65 ms | ~23x |

Under this setup, DP/LQR preserves solution consistency while providing a substantial speed advantage.

---

## 6. Practical Recommendations

- If the model remains linear-quadratic, DP/LQR is usually preferred for frequent re-optimization.
- If you need richer real-world constraints (leverage limits, box constraints, sector exposures), CVXPY is easier to extend.
- In practice, a hybrid workflow works well: DP/LQR as the fast core solver, CVXPY as a validation/extension layer.
