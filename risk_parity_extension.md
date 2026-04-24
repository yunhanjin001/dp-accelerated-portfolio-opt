# Risk Parity Extension: Dynamic Rebalancing via DP/LQR

## Motivation

In addition to the single-asset execution problem and the multi-asset Markowitz rebalancing problem, we also consider an extension based on **risk parity**.

Risk parity is a portfolio construction method where each asset contributes approximately equally to total portfolio risk. Unlike Markowitz optimization, which directly balances expected return and variance, risk parity focuses on distributing risk more evenly across assets.

A pure risk-parity optimization problem is generally **nonlinear**, because each asset's risk contribution depends on both its portfolio weight and its covariance with the full portfolio. Therefore, it does not directly fit the standard LQR framework.

However, we can still use DP/LQR in a natural way:

1. First, compute a static risk-parity target portfolio $w^{RP}$.
2. Then, solve a multi-period dynamic rebalancing problem that trades gradually toward $w^{RP}$ while penalizing transaction costs.

This transforms the risk-parity extension into a **linear-quadratic tracking problem**, which can be solved efficiently using dynamic programming.

---

## Static Risk Parity Target

Let $w \in \mathbb{R}^n$ be the portfolio weight vector, and let $\Sigma \in \mathbb{R}^{n \times n}$ be the covariance matrix of asset returns.

The total portfolio variance is:

$$\sigma_p^2=w^\top\Sigma w$$

The marginal risk contribution of asset $i$ is:

$$(\Sigma w)_i$$

The total risk contribution of asset $i$ is:

$$RC_i=w_i(\Sigma w)_i$$

Risk parity aims to make all risk contributions approximately equal:

$$RC_i\approx\frac{1}{n}\sum_{j=1}^nRC_j$$

Equivalently, the static risk-parity portfolio can be found by solving:

$$\min_w\sum_{i=1}^n\left(w_i(\Sigma w)_i-\frac{1}{n}w^\top\Sigma w\right)^2$$

subject to:

$$\sum_{i=1}^nw_i=1,\qquad w_i\geq0$$

This gives a target portfolio $w^{RP}$.

---

## Dynamic Rebalancing Toward the Risk Parity Target

Once the static risk-parity target $w^{RP}$ is obtained, we solve a dynamic rebalancing problem.

Let $w_t \in \mathbb{R}^n$ be the portfolio weights at time $t$, and let $u_t=w_{t+1}-w_t$ be the trade or rebalancing action.

The portfolio dynamics are linear:

$$w_{t+1}=w_t+u_t$$

We choose trades $u_t$ to gradually move the portfolio toward $w^{RP}$, while also penalizing transaction costs:

$$\min_{\{u_t\}_{t=0}^{T-1}}\sum_{t=0}^{T-1}\left[\frac{\kappa}{2}(w_{t+1}-w^{RP})^\top(w_{t+1}-w^{RP})+\frac{\gamma_{tc}}{2}u_t^\top u_t\right]$$

Here:

- $w^{RP}$: static risk-parity target portfolio
- $\kappa$: tracking penalty parameter
- $\gamma_{tc}$: transaction cost parameter
- $u_t$: trade at time $t$

The first term penalizes being far from the risk-parity target, while the second term penalizes excessive trading.

---

## DP/LQR Reformulation

To write the problem in LQR form, we augment the state with a constant:

$$S_t=\begin{bmatrix}w_t \\ 1\end{bmatrix}$$

The dynamics become:

$$S_{t+1}=A_{\text{aug}}S_t+B_{\text{aug}}u_t$$

where:

$$A_{\text{aug}}=I_{n+1},\qquad B_{\text{aug}}=\begin{bmatrix}I_n \\ 0\end{bmatrix}$$

The stage cost can be written as:

$$\ell(S_t,u_t)=S_t^\top Q S_t+u_t^\top R u_t+2S_t^\top M u_t$$

Because the target-tracking objective depends on $w_{t+1}=w_t+u_t$, we expand:

$$\frac{\kappa}{2}(w_t+u_t-w^{RP})^\top(w_t+u_t-w^{RP})+\frac{\gamma_{tc}}{2}u_t^\top u_t$$

This gives the LQR matrices:

$$Q=\begin{bmatrix}\frac{\kappa}{2}I_n & -\frac{\kappa}{2}w^{RP} \\ -\frac{\kappa}{2}(w^{RP})^\top & \frac{\kappa}{2}(w^{RP})^\top w^{RP}\end{bmatrix}$$

$$R=\frac{\kappa+\gamma_{tc}}{2}I_n$$

$$M=\begin{bmatrix}\frac{\kappa}{2}I_n \\ -\frac{\kappa}{2}(w^{RP})^\top\end{bmatrix}$$

Then the same finite-horizon Riccati recursion can be used.

---

## Backward Riccati Recursion

Assume the value function has quadratic form:

$$V_t(S_t)=S_t^\top\Phi_tS_t$$

Starting from terminal condition:

$$\Phi_T=0$$

we compute backward for $t=T-1,\dots,0$.

The optimal feedback gain is:

$$K_t=\left(R+B^\top\Phi_{t+1}B\right)^{-1}\left(M^\top+B^\top\Phi_{t+1}A\right)$$

The value matrix update is:

$$\Phi_t=Q+A^\top\Phi_{t+1}A-K_t^\top\left(R+B^\top\Phi_{t+1}B\right)K_t$$

The optimal trading policy is:

$$u_t^*=-K_tS_t$$

Then the portfolio path is generated forward using:

$$w_{t+1}=w_t+u_t^*$$

---

## Interpretation

This extension shows that although pure risk parity is not directly an LQR problem, dynamic rebalancing toward a fixed risk-parity target can be solved efficiently using DP/LQR.

The full workflow is:

1. Estimate or simulate the covariance matrix $\Sigma$.
2. Compute the static risk-parity target $w^{RP}$.
3. Use DP/LQR to trade gradually toward $w^{RP}$.
4. Compare the DP/LQR solution with a CVXPY benchmark.
5. Evaluate runtime, trajectory similarity, and objective consistency.

---

## Practical Conclusion

The risk-parity extension demonstrates that DP/LQR can also be used for target-based portfolio rebalancing problems.

Under the linear-quadratic tracking formulation, DP/LQR provides:

- fast backward recursion,
- explicit feedback trading rules,
- efficient forward simulation,
- strong scalability compared with generic convex optimization solvers.

This further supports the main conclusion of the project:

> When the portfolio execution or rebalancing problem has linear dynamics and quadratic costs, dynamic programming provides a fast, structured, and interpretable alternative to repeatedly solving large-scale convex optimization problems.
