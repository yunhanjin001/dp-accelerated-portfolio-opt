"""
Core LQR solver for portfolio optimization.

This module provides the main Dynamic Programming solver for finite-horizon
Linear-Quadratic Regulator (LQR) problems via backward induction.
"""

import numpy as np
from typing import Tuple


def solve_lqr(
    T: int,
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    M: np.ndarray,
) -> np.ndarray:
    """
    Solve finite-horizon LQR problem via backward induction.

    Minimizes the quadratic objective:
        sum_{t=0}^{T-1} [ s_t' Q s_t + u_t' R u_t + 2 s_t' M u_t ]
    
    subject to linear dynamics:
        s_{t+1} = A s_t + B u_t

    Parameters
    ----------
    T : int
        Time horizon (number of steps)
    A : np.ndarray, shape (n, n)
        State transition matrix
    B : np.ndarray, shape (n, m)
        Control input matrix
    Q : np.ndarray, shape (n, n)
        State cost matrix (positive semi-definite)
    R : np.ndarray, shape (m, m)
        Control cost matrix (positive definite)
    M : np.ndarray, shape (n, m)
        Cross-term cost matrix

    Returns
    -------
    K_gains : np.ndarray, shape (T, m, n)
        Optimal feedback gain matrices. The optimal control at time t is:
        u_t* = -K_gains[t] @ s_t

    Notes
    -----
    The algorithm uses the Riccati recursion:
        P_t = Q + A' P_{t+1} A - K_t' (R + B' P_{t+1} B) K_t
    where:
        K_t = (R + B' P_{t+1} B)^{-1} (M' + B' P_{t+1} A)

    Examples
    --------
    >>> import numpy as np
    >>> from lqr_portfolio_optimizer import solve_lqr
    >>> 
    >>> # Optimal execution problem
    >>> T = 30
    >>> gamma, sigma_sq, eta, rho, beta = 1.0, 0.04, 0.1, 0.95, 0.8
    >>> 
    >>> A = np.array([[1, 0, 0], [0, rho, 0], [0, 0, beta]])
    >>> B = np.array([[1], [0], [beta * eta]])
    >>> Q = np.diag([0.5 * gamma * sigma_sq, 0, 0])
    >>> R = np.array([[0.5 * gamma * sigma_sq + 0.5 * eta]])
    >>> M = np.array([[0.5 * gamma * sigma_sq], [-0.5], [0.5]])
    >>> 
    >>> K_gains = solve_lqr(T, A, B, Q, R, M)
    >>> print(f"Gain matrix shape: {K_gains.shape}")
    Gain matrix shape: (30, 1, 3)
    """
    n = A.shape[0]
    m = B.shape[1]

    # Initialize value function matrix and gain storage
    P = np.zeros((n, n))
    K_gains = np.zeros((T, m, n))

    # Backward induction
    for t in range(T - 1, -1, -1):
        # Compute inner term: R + B' P B
        inner = R + B.T @ P @ B  # shape (m, m)
        
        # Compute gain: K_t = (R + B' P B)^{-1} (M' + B' P A)
        K_t = np.linalg.solve(inner, M.T + B.T @ P @ A)  # shape (m, n)
        K_gains[t] = K_t
        
        # Update value function matrix via Riccati recursion
        P = Q + A.T @ P @ A - K_t.T @ inner @ K_t

    return K_gains


def execute_lqr(
    T: int,
    A: np.ndarray,
    B: np.ndarray,
    K_gains: np.ndarray,
    s0: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Execute LQR policy via forward simulation.

    Given pre-computed optimal gains, simulate the state and control
    trajectories forward in time.

    Parameters
    ----------
    T : int
        Time horizon
    A : np.ndarray, shape (n, n)
        State transition matrix
    B : np.ndarray, shape (n, m)
        Control input matrix
    K_gains : np.ndarray, shape (T, m, n)
        Optimal feedback gains from solve_lqr
    s0 : np.ndarray, shape (n,)
        Initial state

    Returns
    -------
    s_path : np.ndarray, shape (T+1, n)
        State trajectory including initial state
    u_path : np.ndarray, shape (T, m)
        Control trajectory

    Examples
    --------
    >>> import numpy as np
    >>> from lqr_portfolio_optimizer import solve_lqr, execute_lqr
    >>> 
    >>> # Setup (simplified)
    >>> T, n, m = 10, 3, 1
    >>> A = np.eye(n)
    >>> B = np.ones((n, m))
    >>> Q, R, M = np.eye(n) * 0.1, np.eye(m) * 0.5, np.zeros((n, m))
    >>> s0 = np.array([0.0, 1.0, 0.0])
    >>> 
    >>> K_gains = solve_lqr(T, A, B, Q, R, M)
    >>> s_path, u_path = execute_lqr(T, A, B, K_gains, s0)
    >>> 
    >>> print(f"State path shape: {s_path.shape}")
    >>> print(f"Control path shape: {u_path.shape}")
    State path shape: (11, 3)
    Control path shape: (10, 1)
    """
    s = s0.copy().astype(float)
    n = len(s0)
    m = B.shape[1]
    
    s_path = np.zeros((T + 1, n))
    u_path = np.zeros((T, m))
    s_path[0] = s

    # Forward simulation
    for t in range(T):
        # Optimal control: u_t = -K_t s_t
        u_t = -(K_gains[t] @ s)
        u_path[t] = u_t
        
        # State update: s_{t+1} = A s_t + B u_t
        s = A @ s + B @ u_t
        s_path[t + 1] = s

    return s_path, u_path
