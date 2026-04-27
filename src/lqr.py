"""
src/lqr.py  —  Generic finite-horizon LQR via backward induction (DP)

Solves the quadratic objective:

    min  sum_{t=0}^{T-1} [ s_t' Q s_t  +  u_t' R u_t  +  2 s_t' M u_t ]

    s.t.  s_{t+1} = A s_t + B u_t,   s_0 given

Optimal control is linear feedback:  u_t* = -K_t @ s_t

Public API
----------
solve(T, A, B, Q, R, M)         -> K_gains  (T, m, n)
simulate(T, A, B, K_gains, s0)  -> s_path   (T+1, n),  u_path (T, m)
"""

import numpy as np


def solve(
    T: int,
    A: np.ndarray,   # (n, n)  state transition
    B: np.ndarray,   # (n, m)  control influence
    Q: np.ndarray,   # (n, n)  state cost (symmetric)
    R: np.ndarray,   # (m, m)  control cost (symmetric PD)
    M: np.ndarray,   # (n, m)  cross-term cost
) -> np.ndarray:
    """
    Finite-horizon LQR via backward induction (Bellman recursion).

    Parameters
    ----------
    T       : time horizon (number of steps)
    A, B    : linear dynamics   s_{t+1} = A s_t + B u_t
    Q, R, M : cost matrices for  s'Qs + u'Ru + 2s'Mu

    Returns
    -------
    K_gains : ndarray, shape (T, m, n)
        Time-varying feedback gains.  Optimal control: u_t* = -K_gains[t] @ s_t
    """
    n = A.shape[0]
    m = B.shape[1]

    P = np.zeros((n, n))
    K_gains = np.zeros((T, m, n))

    for t in range(T - 1, -1, -1):
        inner   = R + B.T @ P @ B                          # (m, m)
        K_t     = np.linalg.solve(inner, M.T + B.T @ P @ A)  # (m, n)
        K_gains[t] = K_t
        P       = Q + A.T @ P @ A - K_t.T @ inner @ K_t

    return K_gains


def simulate(
    T: int,
    A: np.ndarray,
    B: np.ndarray,
    K_gains: np.ndarray,
    s0: np.ndarray,
) -> tuple:
    """
    Forward simulation under pre-computed LQR feedback gains.

    Parameters
    ----------
    T       : time horizon
    A, B    : dynamics matrices
    K_gains : (T, m, n) array from solve()
    s0      : (n,) initial state

    Returns
    -------
    s_path : ndarray, shape (T+1, n)   state trajectory
    u_path : ndarray, shape (T,   m)   control trajectory
    """
    s = s0.copy().astype(float)
    n, m = len(s0), B.shape[1]
    s_path = np.zeros((T + 1, n))
    u_path = np.zeros((T, m))
    s_path[0] = s

    for t in range(T):
        u_t           = -(K_gains[t] @ s)
        u_path[t]     = u_t
        s             = A @ s + B @ u_t
        s_path[t + 1] = s

    return s_path, u_path
