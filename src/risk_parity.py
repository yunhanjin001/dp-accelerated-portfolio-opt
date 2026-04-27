"""
src/risk_parity.py  —  Risk-parity target tracking via DP/LQR

Problem
-------
Track a static risk-parity portfolio w^{RP} over T periods, minimising:

    sum_{t=1}^{T} [ (kappa/2) ||w_t - w^{RP}||^2  +  (gamma_tc/2) ||u_t||^2 ]

where u_t = Δw_t is the trade vector.

Risk-parity target: each asset i contributes equally to portfolio variance,
i.e.  w_i * (Sigma w)_i  =  w'Sigma w / n  for all i.

Augmented state (n+1 dim):  S_t = [w_t ; 1]

LQR matrices encoding the tracking objective:
  Q[:n,:n] = 0.5*kappa*I;   Q[n,:n] = Q[:n,n] = -0.5*kappa*w_RP
  Q[n, n]  = 0.5*kappa * w_RP'w_RP
  R        = 0.5*(kappa + gamma_tc)*I
  M[:n,:]  = 0.5*kappa*I;   M[n,:]  = -0.5*kappa*w_RP

Public API
----------
target(Sigma)                              -> w_RP   (n,)
build_matrices(w_RP, kappa, gamma_tc)      -> A, B, Q, R, M
solve(T, w_RP, kappa, gamma_tc)            -> K_gains  (T, n, n+1)
simulate(T, K_gains, w0)                   -> w_path   (T+1, n)
cost(w_path, w_RP, kappa, gamma_tc)        -> float
"""

import numpy as np
from scipy.optimize import minimize
from . import lqr


def target(Sigma: np.ndarray) -> np.ndarray:
    """
    Compute the static risk-parity portfolio.

    Each asset contributes equally to total portfolio variance:
        w_i * (Sigma w)_i  =  w'Sigma w / n  for all i.

    Parameters
    ----------
    Sigma : (n, n) covariance matrix (symmetric PD)

    Returns
    -------
    w_RP : ndarray, shape (n,)
        Risk-parity weights (sum to 1, non-negative).
    """
    n = Sigma.shape[0]

    def _obj(w):
        port_var = w @ Sigma @ w
        rc = w * (Sigma @ w)
        return np.sum((rc - port_var / n) ** 2)

    w0  = np.ones(n) / n
    res = minimize(
        _obj, w0,
        bounds=[(0.0, 1.0)] * n,
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        method="SLSQP",
    )
    return res.x


def build_matrices(
    w_RP: np.ndarray,
    kappa: float,
    gamma_tc: float,
) -> tuple:
    """
    Construct LQR matrices for the risk-parity tracking problem.

    Parameters
    ----------
    w_RP     : (n,) target risk-parity weights
    kappa    : tracking-penalty strength
    gamma_tc : transaction-cost coefficient

    Returns
    -------
    A, B, Q, R, M : ndarray
        Ready for lqr.solve().
    """
    n  = len(w_RP)
    na = n + 1

    A = np.eye(na)
    B = np.vstack([np.eye(n), np.zeros((1, n))])

    Q = np.zeros((na, na))
    Q[:n, :n] =  0.5 * kappa * np.eye(n)
    Q[n,  :n] = -0.5 * kappa * w_RP
    Q[:n,  n] = -0.5 * kappa * w_RP
    Q[n,   n] =  0.5 * kappa * np.dot(w_RP, w_RP)

    R = 0.5 * (kappa + gamma_tc) * np.eye(n)

    M = np.zeros((na, n))
    M[:n, :] =  0.5 * kappa * np.eye(n)
    M[n,  :] = -0.5 * kappa * w_RP

    return A, B, Q, R, M


def solve(
    T: int,
    w_RP: np.ndarray,
    kappa: float,
    gamma_tc: float,
) -> np.ndarray:
    """
    Compute feedback gains for the risk-parity tracking problem.

    Parameters
    ----------
    T        : planning horizon
    w_RP     : (n,) target risk-parity weights  (from target())
    kappa    : tracking-penalty strength
    gamma_tc : transaction-cost coefficient

    Returns
    -------
    K_gains : ndarray, shape (T, n, n+1)
        Feedback gains; trade at step t:  u_t = -K_gains[t] @ [w_t; 1]
    """
    A, B, Q, R, M = build_matrices(w_RP, kappa, gamma_tc)
    return lqr.solve(T, A, B, Q, R, M)


def simulate(
    T: int,
    K_gains: np.ndarray,
    w0: np.ndarray,
) -> np.ndarray:
    """
    Forward-simulate the risk-parity tracking policy.

    Parameters
    ----------
    T       : planning horizon
    K_gains : (T, n, n+1) gains from solve()
    w0      : (n,) initial portfolio weights

    Returns
    -------
    w_path : ndarray, shape (T+1, n)
        Portfolio weights at t = 0, 1, …, T.
    """
    n = len(w0)
    w = w0.copy().astype(float)
    w_path = np.zeros((T + 1, n))
    w_path[0] = w

    for t in range(T):
        S_aug    = np.append(w, 1.0)
        delta_w  = -(K_gains[t] @ S_aug)
        w        = w + delta_w
        w_path[t + 1] = w

    return w_path


def cost(
    w_path: np.ndarray,
    w_RP: np.ndarray,
    kappa: float,
    gamma_tc: float,
) -> float:
    """
    Evaluate the accumulated tracking cost along a weight trajectory (lower = better).

        sum_{t=1}^{T} [ (kappa/2)||w_t - w_RP||^2 + (gamma_tc/2)||Δw_t||^2 ]
    """
    total = 0.0
    for t in range(1, len(w_path)):
        dw     = w_path[t] - w_path[t - 1]
        err    = w_path[t] - w_RP
        total += 0.5 * kappa * np.dot(err, err) + 0.5 * gamma_tc * np.dot(dw, dw)
    return total
