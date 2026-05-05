"""
src/markowitz.py  —  Multi-asset Markowitz portfolio optimization via DP/LQR

Problem
-------
Over T periods, choose trades u_t = Δw_t to maximise cumulative:

    sum_{t=1}^{T} [ mu' w_t  -  (lam/2) w_t' Sigma w_t  -  (gamma_tc/2) ||u_t||^2 ]

Augmented state (n+1 dim):  S_t = [w_t ; 1]
Control:                     u_t = delta_w_t  (n-dim)

Dynamics (trivially):  S_{t+1} = A S_t + B u_t
  A = eye(n+1)
  B = vstack([eye(n), zeros(1,n)])

LQR matrices encoding the Markowitz objective:
  Q[:n,:n] = 0.5*lam*Sigma;  Q[n,:n] = Q[:n,n] = -0.5*mu
  R        = 0.5*gamma_tc*eye(n) + 0.5*lam*Sigma
  M[:n,:]  = 0.5*lam*Sigma;  M[n,:]  = -0.5*mu

Public API
----------
build_matrices(mu, Sigma, lam, gamma_tc)       -> A, B, Q, R, M
solve(T, mu, Sigma, lam, gamma_tc)             -> K_gains  (T, n, n+1)
simulate(T, K_gains, w0)                       -> w_path   (T+1, n)
objective(w_path, mu, Sigma, lam, gamma_tc)    -> float
"""

import numpy as np
from . import lqr


def build_matrices(
    mu: np.ndarray,
    Sigma: np.ndarray,
    lam: float,
    gamma_tc: float,
) -> tuple:
    """
    Construct LQR matrices for the multi-asset Markowitz problem.

    Parameters
    ----------
    mu       : (n,) expected returns vector
    Sigma    : (n, n) covariance matrix (symmetric PD)
    lam      : risk-aversion coefficient  (lambda)
    gamma_tc : transaction-cost coefficient

    Returns
    -------
    A, B, Q, R, M : ndarray
        Ready for lqr.solve().
    """
    n  = len(mu)
    na = n + 1

    A = np.eye(na)
    B = np.vstack([np.eye(n), np.zeros((1, n))])

    Q = np.zeros((na, na))
    Q[:n, :n] =  0.5 * lam * Sigma
    Q[n,  :n] = -0.5 * mu
    Q[:n,  n] = -0.5 * mu

    R = 0.5 * gamma_tc * np.eye(n) + 0.5 * lam * Sigma

    M = np.zeros((na, n))
    M[:n, :] =  0.5 * lam * Sigma
    M[n,  :] = -0.5 * mu

    return A, B, Q, R, M


def solve(
    T: int,
    mu: np.ndarray,
    Sigma: np.ndarray,
    lam: float,
    gamma_tc: float,
) -> np.ndarray:
    """
    Compute feedback gains for the Markowitz problem.

    Parameters
    ----------
    T        : planning horizon
    mu       : (n,) expected returns
    Sigma    : (n, n) covariance matrix
    lam      : risk-aversion coefficient
    gamma_tc : transaction-cost coefficient

    Returns
    -------
    K_gains : ndarray, shape (T, n, n+1)
        Feedback gains; trade at step t:  u_t = -K_gains[t] @ [w_t; 1]
    """
    A, B, Q, R, M = build_matrices(mu, Sigma, lam, gamma_tc)
    return lqr.solve(T, A, B, Q, R, M)


def simulate(
    T: int,
    K_gains: np.ndarray,
    w0: np.ndarray,
) -> np.ndarray:
    """
    Forward-simulate the Markowitz policy.

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


def objective(
    w_path: np.ndarray,
    mu: np.ndarray,
    Sigma: np.ndarray,
    lam: float,
    gamma_tc: float,
) -> float:
    """
    Evaluate the Markowitz objective along a weight trajectory.

    Returns total value (higher = better):
        sum_{t=1}^{T} [ mu'w_t - (lam/2) w_t'Sigma w_t - (gamma_tc/2)||Δw_t||^2 ]
    """
    total = 0.0
    for t in range(1, len(w_path)):
        dw     = w_path[t] - w_path[t - 1]
        total += (mu @ w_path[t]
                  - 0.5 * lam     * w_path[t] @ Sigma @ w_path[t]
                  - 0.5 * gamma_tc * dw @ dw)
    return total
