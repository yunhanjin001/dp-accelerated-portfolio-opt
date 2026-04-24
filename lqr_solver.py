"""
lqr_solver.py
─────────────────────────────────────────────────────────────────────
Generic finite-horizon LQR solver via backward induction (DP).

Solves the quadratic objective:

    min  sum_{t=0}^{T-1} [ s_t' Q s_t  +  u_t' R u_t  +  2 s_t' M u_t ]

subject to:   s_{t+1} = A s_t + B u_t

Functions
─────────
solve_lqr(T, A, B, Q, R, M)         → K_gains  (T, m, n)
execute_lqr(T, A, B, K_gains, s0)   → s_path   (T+1, n),  u_path (T, m)

─────────────────────────────────────────────────────────────────────
Matrix construction cheat-sheet
─────────────────────────────────────────────────────────────────────
EXECUTION PROBLEM  (single asset, 3-D state  s=[w, alpha, c])
    A = [[1,   0,    0      ],      B = [[1         ],
         [0,   rho,  0      ],           [0         ],
         [0,   0,    beta   ]]           [beta*eta  ]]

    Q[0,0] = 0.5*gamma*sigma_sq
    R      = [[0.5*gamma*sigma_sq + 0.5*eta]]
    M      = [[0.5*gamma*sigma_sq], [-0.5], [0.5]]
    s0     = [0, initial_alpha, 0]

MARKOWITZ PROBLEM  (n assets, (n+1)-D augmented state  s=[w, 1])
    A = eye(n+1)
    B = vstack([eye(n), zeros(1,n)])

    Q[:n,:n] = 0.5*lam*Sigma;  Q[n,:n] = Q[:n,n] = -0.5*mu
    R        = 0.5*gamma_tc*eye(n) + 0.5*lam*Sigma
    M[:n,:]  = 0.5*lam*Sigma;  M[n,:] = -0.5*mu
    s0       = [w0, 1]
─────────────────────────────────────────────────────────────────────
"""

import numpy as np


def solve_lqr(
    T: int,
    A: np.ndarray,   # (n, n)
    B: np.ndarray,   # (n, m)
    Q: np.ndarray,   # (n, n)  state cost
    R: np.ndarray,   # (m, m)  control cost
    M: np.ndarray,   # (n, m)  cross-term cost
) -> np.ndarray:
    """
    Finite-horizon LQR via backward induction.

    Parameters
    ----------
    T       : horizon
    A, B    : dynamics  s_{t+1} = A s_t + B u_t
    Q, R, M : cost matrices

    Returns
    -------
    K_gains : (T, m, n) ndarray
        Optimal feedback gains.  u_t* = -K_gains[t] @ s_t
    """
    n = A.shape[0]
    m = B.shape[1]

    P       = np.zeros((n, n))
    K_gains = np.zeros((T, m, n))

    for t in range(T - 1, -1, -1):
        inner      = R + B.T @ P @ B            # (m, m)
        K_t        = np.linalg.solve(inner, M.T + B.T @ P @ A)  # (m, n)
        K_gains[t] = K_t
        P          = Q + A.T @ P @ A - K_t.T @ inner @ K_t

    return K_gains


def execute_lqr(
    T: int,
    A: np.ndarray,
    B: np.ndarray,
    K_gains: np.ndarray,
    s0: np.ndarray,
) -> tuple:
    """
    Forward simulation using pre-computed LQR gains.

    Parameters
    ----------
    T       : horizon
    A, B    : dynamics matrices
    K_gains : (T, m, n) gains from solve_lqr
    s0      : (n,) initial state

    Returns
    -------
    s_path : (T+1, n)  state trajectory
    u_path : (T,   m)  control trajectory
    """
    s      = s0.copy().astype(float)
    n, m   = len(s0), B.shape[1]
    s_path = np.zeros((T + 1, n))
    u_path = np.zeros((T, m))
    s_path[0] = s

    for t in range(T):
        u_t           = -(K_gains[t] @ s)
        u_path[t]     = u_t
        s             = A @ s + B @ u_t
        s_path[t + 1] = s

    return s_path, u_path


# ── Self-test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Execution-problem matrices
    T=20; gamma=1.0; sigma_sq=0.04; eta=0.1; rho=0.95; beta=0.8; alpha0=0.5
    A  = np.array([[1,0,0],[0,rho,0],[0,0,beta]], dtype=float)
    B  = np.array([[1],[0],[beta*eta]], dtype=float)
    Q  = np.diag([0.5*gamma*sigma_sq, 0, 0])
    R  = np.array([[0.5*gamma*sigma_sq + 0.5*eta]])
    M  = np.array([[0.5*gamma*sigma_sq],[-0.5],[0.5]])
    s0 = np.array([0.0, alpha0, 0.0])

    K = solve_lqr(T, A, B, Q, R, M)
    s_path, u_path = execute_lqr(T, A, B, K, s0)
    print("K shape:", K.shape)
    print("w path :", s_path[:, 0].round(4))
