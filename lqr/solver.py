"""
Improved LQR solver for portfolio optimization via Dynamic Programming.

This module provides a fast Dynamic Programming solver for finite-horizon
Linear-Quadratic Regulator (LQR) problems via backward induction (Riccati
recursion). The solver exploits the temporal structure of multi-period
portfolio optimization problems to achieve significant speedups over
traditional convex optimization approaches.

Reference:
    Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of
    Statistical Learning: Data Mining, Inference, and Prediction (2nd ed.).
    Springer.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class LQRResult:
    """Result from :func:`solve_lqr` and :func:`execute_lqr`."""

    K_gains: np.ndarray
    """Optimal feedback gain matrices, shape (T, m, n)."""

    s_path: np.ndarray
    """State trajectory, shape (T+1, n)."""

    u_path: np.ndarray
    """Control trajectory, shape (T, m)."""


def solve_lqr(
    T: int,
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    M: np.ndarray,
    P_terminal: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute optimal LQR feedback gains via backward induction.

    Solves the finite-horizon Linear-Quadratic Regulator problem by
    minimizing the quadratic objective::

        sum_{t=0}^{T-1} [ s_t' Q s_t + u_t' R u_t + 2 s_t' M u_t ]  +  s_T' P_T s_T

    where ``P_T`` is :paramref:`P_terminal` (default: zero matrix).

    Subject to linear dynamics::

        s_{t+1} = A s_t + B u_t

    using the Riccati recursion (backward induction in Dynamic Programming).
    The optimal control has the linear feedback form ``u_t* = -K_t s_t``.

    Parameters
    ----------
    T:
        Time horizon (number of steps).
    A:
        State transition matrix, shape (n, n).
    B:
        Control input matrix, shape (n, m).
    Q:
        State cost matrix (positive semi-definite), shape (n, n).
    R:
        Control cost matrix (positive definite), shape (m, m).
    M:
        Cross-term cost matrix, shape (n, m).
    P_terminal:
        Optional terminal state cost matrix, shape (n, n), positive
        semidefinite. The total objective includes
        ``s_T' P_terminal s_T`` in addition to the stage sum. The Riccati
        recursion is initialized with ``P_T`` (i.e. value function at the
        final time) equal to :paramref:`P_terminal`. If omitted, uses zero
        terminal cost (same as ``P_terminal = 0``).

    Returns
    -------
    np.ndarray
        Optimal feedback gain matrices, shape (T, m, n).
        The optimal control at time t is: ``u_t* = -K_gains[t] @ s_t``.

    Notes
    -----
    The algorithm uses the Riccati recursion::

        K_t = (R + B' P_{t+1} B)^{-1} (M' + B' P_{t+1} A)
        P_t = Q + A' P_{t+1} A - K_t' (R + B' P_{t+1} B) K_t

    Backward induction starts from ``P_T = P_terminal`` if given, else
    ``P_T = 0``.

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
    >>> K_gains.shape
    (30, 1, 3)
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    Q = np.asarray(Q, dtype=float)
    R = np.asarray(R, dtype=float)
    M = np.asarray(M, dtype=float)

    n = A.shape[0]
    m = B.shape[1]

    if A.shape != (n, n):
        raise ValueError("A must be a square matrix (n, n)")
    if B.shape != (n, m):
        raise ValueError(f"B must have shape (n, m) = ({n}, {m})")
    if Q.shape != (n, n):
        raise ValueError(f"Q must have shape (n, n) = ({n}, {n})")
    if R.shape != (m, m):
        raise ValueError(f"R must have shape (m, m) = ({m}, {m})")
    if M.shape != (n, m):
        raise ValueError(f"M must have shape (n, m) = ({n}, {m})")
    if P_terminal is not None:
        P_terminal = np.asarray(P_terminal, dtype=float)
        if P_terminal.shape != (n, n):
            raise ValueError(
                f"P_terminal must have shape (n, n) = ({n}, {n}), got {P_terminal.shape}"
            )

    # Initialize value function matrix and gain storage
    P = np.zeros((n, n)) if P_terminal is None else P_terminal.copy()
    K_gains = np.zeros((T, m, n))

    # Backward induction (Riccati recursion)
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
    """Execute LQR policy via forward simulation.

    Given pre-computed optimal feedback gains from :func:`solve_lqr`,
    simulate the state and control trajectories forward in time using
    the optimal policy ``u_t = -K_t s_t``.

    Parameters
    ----------
    T:
        Time horizon (number of steps).
    A:
        State transition matrix, shape (n, n).
    B:
        Control input matrix, shape (n, m).
    K_gains:
        Optimal feedback gains from :func:`solve_lqr`, shape (T, m, n).
    s0:
        Initial state vector, shape (n,).

    Returns
    -------
    s_path:
        State trajectory including initial state, shape (T+1, n).
    u_path:
        Control trajectory, shape (T, m).

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
    >>> s_path.shape
    (11, 3)
    >>> u_path.shape
    (10, 1)
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    K_gains = np.asarray(K_gains, dtype=float)
    s0 = np.asarray(s0, dtype=float)

    n = len(s0)
    m = B.shape[1]

    if A.shape != (n, n):
        raise ValueError(f"A must have shape (n, n) = ({n}, {n})")
    if B.shape != (n, m):
        raise ValueError(f"B must have shape (n, m) = ({n}, {m})")
    if K_gains.shape != (T, m, n):
        raise ValueError(f"K_gains must have shape (T, m, n) = ({T}, {m}, {n})")

    s = s0.copy()
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


def solve_and_execute_lqr(
    T: int,
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    M: np.ndarray,
    s0: np.ndarray,
    P_terminal: Optional[np.ndarray] = None,
) -> LQRResult:
    """Solve LQR problem and execute the optimal policy in one call.

    Convenience function that combines :func:`solve_lqr` and
    :func:`execute_lqr`.

    Parameters
    ----------
    T:
        Time horizon (number of steps).
    A:
        State transition matrix, shape (n, n).
    B:
        Control input matrix, shape (n, m).
    Q:
        State cost matrix (positive semi-definite), shape (n, n).
    R:
        Control cost matrix (positive definite), shape (m, m).
    M:
        Cross-term cost matrix, shape (n, m).
    s0:
        Initial state vector, shape (n,).
    P_terminal:
        Optional terminal cost (same meaning as in :func:`solve_lqr`).

    Returns
    -------
    LQRResult
        Dataclass with fields ``K_gains``, ``s_path``, and ``u_path``.

    Examples
    --------
    >>> import numpy as np
    >>> from lqr_portfolio_optimizer import solve_and_execute_lqr
    >>>
    >>> T, n, m = 10, 3, 1
    >>> A = np.eye(n)
    >>> B = np.ones((n, m))
    >>> Q, R, M = np.eye(n) * 0.1, np.eye(m) * 0.5, np.zeros((n, m))
    >>> s0 = np.array([0.0, 1.0, 0.0])
    >>>
    >>> result = solve_and_execute_lqr(T, A, B, Q, R, M, s0)
    >>> result.K_gains.shape
    (10, 1, 3)
    >>> result.s_path.shape
    (11, 3)
    """
    K_gains = solve_lqr(T, A, B, Q, R, M, P_terminal=P_terminal)
    s_path, u_path = execute_lqr(T, A, B, K_gains, s0)

    return LQRResult(K_gains=K_gains, s_path=s_path, u_path=u_path)
