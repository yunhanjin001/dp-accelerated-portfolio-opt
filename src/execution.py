"""
src/execution.py  —  Single-asset optimal execution via DP/LQR

Problem
-------
Trade into a target position over T periods to balance:
  - Alpha capture  (expected return alpha_t drives trades)
  - Risk penalty   (gamma/2 * sigma^2 * w_t^2)
  - Execution cost (impact decays with factor beta, per-unit cost eta)

State (3-D):  s = [w,  alpha,  c]
  w      : current portfolio weight
  alpha  : expected return (mean-reverting, rho per step)
  c      : cumulative execution cost carry

Dynamics:
  A = [[1,   0,    0    ],    B = [[1       ],
       [0,   rho,  0    ],         [0       ],
       [0,   0,    beta ]]         [beta*eta]]

  s_{t+1} = A s_t + B u_t,   s_0 = [0, alpha_0, 0]

Public API
----------
build_matrices(gamma, sigma_sq, eta, rho, beta)      -> A, B, Q, R, M
solve(T, gamma, sigma_sq, eta, rho, beta)            -> K_gains (T, 1, 3)
simulate(T, K_gains, initial_alpha, eta, rho, beta)  -> w_path  (T,)
"""

import numpy as np
from . import lqr


def build_matrices(
    gamma: float,
    sigma_sq: float,
    eta: float,
    rho: float,
    beta: float,
) -> tuple:
    """
    Construct LQR matrices for the single-asset execution problem.

    Parameters
    ----------
    gamma    : risk-aversion coefficient
    sigma_sq : return variance (sigma^2)
    eta      : linear execution cost coefficient
    rho      : alpha mean-reversion rate  (0 < rho < 1)
    beta     : execution-cost decay rate  (0 < beta < 1)

    Returns
    -------
    A, B, Q, R, M : ndarray
        Dynamics and cost matrices ready for lqr.solve().
    """
    A = np.array([
        [1.0, 0.0, 0.0],
        [0.0, rho, 0.0],
        [0.0, 0.0, beta],
    ])
    B = np.array([[1.0], [0.0], [beta * eta]])

    Q = np.diag([0.5 * gamma * sigma_sq, 0.0, 0.0])
    R = np.array([[0.5 * gamma * sigma_sq + 0.5 * eta]])
    M = np.array([[0.5 * gamma * sigma_sq], [-0.5], [0.5]])

    return A, B, Q, R, M


def solve(
    T: int,
    gamma: float,
    sigma_sq: float,
    eta: float,
    rho: float,
    beta: float,
) -> np.ndarray:
    """
    Compute LQR feedback gains for the execution problem.

    Parameters
    ----------
    T        : planning horizon
    gamma    : risk-aversion coefficient
    sigma_sq : return variance
    eta      : execution cost coefficient
    rho      : alpha mean-reversion rate
    beta     : execution-cost decay rate

    Returns
    -------
    K_gains : ndarray, shape (T, 1, 3)
        Feedback gains;  u_t* = -K_gains[t] @ s_t
    """
    A, B, Q, R, M = build_matrices(gamma, sigma_sq, eta, rho, beta)
    return lqr.solve(T, A, B, Q, R, M)


def simulate(
    T: int,
    K_gains: np.ndarray,
    initial_alpha: float,
    eta: float,
    rho: float,
    beta: float,
) -> np.ndarray:
    """
    Forward-simulate the execution policy given pre-computed gains.

    Parameters
    ----------
    T             : planning horizon
    K_gains       : (T, 1, 3) array from solve()
    initial_alpha : starting expected return alpha_0
    eta           : execution cost coefficient (needed to build B)
    rho           : alpha mean-reversion rate  (needed to build A)
    beta          : execution-cost decay rate  (needed to build A, B)

    Returns
    -------
    w_path : ndarray, shape (T,)
        Portfolio weight w_t at each step t = 1, …, T
        (w_path[t] is the weight after trade t).
    """
    A = np.array([
        [1.0, 0.0, 0.0],
        [0.0, rho, 0.0],
        [0.0, 0.0, beta],
    ])
    B = np.array([[1.0], [0.0], [beta * eta]])
    s0 = np.array([0.0, initial_alpha, 0.0])

    s_path, _ = lqr.simulate(T, A, B, K_gains, s0)
    return s_path[1:, 0]   # weights at t = 1…T
