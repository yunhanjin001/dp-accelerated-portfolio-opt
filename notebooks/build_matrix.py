"""LQR matrix builders for the demo notebook.

Each function returns the raw matrices ``(A, B, Q, R, M, s0[, P_T])`` consumed
by ``lqr.solve_and_execute_lqr``.

Sections covered (matching ``demo.ipynb``):

* :func:`build_execution_matrices`           - Part A, single-asset optimal execution
* :func:`build_markowitz_matrices`           - Part B, multi-asset Markowitz
* :func:`build_given_weight_single_matrices` - Part C, single-asset given-weight tracking
* :func:`build_given_weight_multi_matrices`  - Part C, multi-asset given-weight tracking
"""

import numpy as np


def build_execution_matrices(gamma, sigma_sq, eta, rho, beta, alpha0):
    """Optimal Execution (Part A) - 3-D augmented state ``[w, alpha, c]``.

    Parameters mirror the notebook block A1 exactly.
    """
    A = np.array([
        [1.0, 0.0, 0.0],
        [0.0, rho, 0.0],
        [0.0, 0.0, beta],
    ])
    B = np.array([
        [1.0],
        [0.0],
        [beta * eta],
    ])
    Q = np.diag([0.5 * gamma * sigma_sq, 0.0, 0.0])
    R = np.array([[0.5 * gamma * sigma_sq + 0.5 * eta]])
    M = np.array([
        [0.5 * gamma * sigma_sq],
        [-0.5],
        [0.5],
    ])
    s0 = np.array([0.0, alpha0, 0.0])
    return A, B, Q, R, M, s0


def build_markowitz_matrices(mu, Sigma, lam, gamma_tc, w0):
    """Multi-asset Markowitz Portfolio (Part B) - augmented state ``[w; 1]``.

    Per-step cost (LQR form): ``s'Qs + u'Ru + 2 s'Mu`` encodes
    ``-mu'w_{t+1} + 0.5*lam*w_{t+1}'Sigma w_{t+1} + 0.5*gamma_tc*||u||^2``.
    """
    mu = np.asarray(mu, dtype=float).ravel()
    Sigma = np.asarray(Sigma, dtype=float)
    w0 = np.asarray(w0, dtype=float).ravel()
    n = len(mu)
    na = n + 1

    A = np.eye(na)
    B = np.vstack([np.eye(n), np.zeros((1, n))])

    Q = np.zeros((na, na))
    Q[:n, :n] = 0.5 * lam * Sigma
    Q[n, :n] = -0.5 * mu
    Q[:n, n] = -0.5 * mu

    R = 0.5 * gamma_tc * np.eye(n) + 0.5 * lam * Sigma

    M = np.zeros((na, n))
    M[:n, :] = 0.5 * lam * Sigma
    M[n, :] = -0.5 * mu

    s0 = np.append(w0, 1.0)
    return A, B, Q, R, M, s0


def build_given_weight_single_matrices(w_given, kappa, gamma_tc, k_terminal, w0):
    """Single-asset Given-Weight tracking (Part C, scalar target).

    Cost form (per step): ``0.5*kappa*(w - w_given)^2 + 0.5*gamma_tc*u^2``,
    plus terminal penalty ``0.5*k_terminal*(w_T - w_given)^2``.
    """
    A = np.eye(2)
    B = np.array([[1.0], [0.0]])

    Q = np.array([
        [0.5 * kappa,                -0.5 * kappa * w_given],
        [-0.5 * kappa * w_given,      0.5 * kappa * (w_given ** 2)],
    ])
    R = np.array([[0.5 * (kappa + gamma_tc)]])
    M = np.array([
        [0.5 * kappa],
        [-0.5 * kappa * w_given],
    ])
    P_T = np.array([
        [0.5 * k_terminal,                -0.5 * k_terminal * w_given],
        [-0.5 * k_terminal * w_given,      0.5 * k_terminal * (w_given ** 2)],
    ])

    s0 = np.array([float(w0), 1.0])
    return A, B, Q, R, M, s0, P_T


def build_given_weight_multi_matrices(w_given, kappa, gamma_tc, k_terminal, w0):
    """Multi-asset Given-Weight tracking (Part C, vector target).

    Same cost shape as the single-asset version, applied component-wise to
    ``||w - w_given||^2`` with augmented state ``[w; 1]``.
    """
    w_given = np.asarray(w_given, dtype=float).ravel()
    w0 = np.asarray(w0, dtype=float).ravel()
    n = len(w_given)
    na = n + 1

    A = np.eye(na)
    B = np.vstack([np.eye(n), np.zeros((1, n))])

    wgwg = float(w_given @ w_given)

    Q = np.zeros((na, na))
    Q[:n, :n] = 0.5 * kappa * np.eye(n)
    Q[n, :n] = -0.5 * kappa * w_given
    Q[:n, n] = -0.5 * kappa * w_given
    Q[n, n] = 0.5 * kappa * wgwg

    R = 0.5 * (kappa + gamma_tc) * np.eye(n)

    M = np.zeros((na, n))
    M[:n, :] = 0.5 * kappa * np.eye(n)
    M[n, :] = -0.5 * kappa * w_given

    P_T = np.zeros((na, na))
    P_T[:n, :n] = 0.5 * k_terminal * np.eye(n)
    P_T[n, :n] = -0.5 * k_terminal * w_given
    P_T[:n, n] = -0.5 * k_terminal * w_given
    P_T[n, n] = 0.5 * k_terminal * wgwg

    s0 = np.append(w0, 1.0)
    return A, B, Q, R, M, s0, P_T
