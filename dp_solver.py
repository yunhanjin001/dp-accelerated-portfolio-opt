import numpy as np
import cvxpy as cp


def solve_lqr_execution(T, gamma, sigma_sq, eta, rho, beta):
    A = np.array([
        [1.0, 0.0, 0.0],
        [0.0, rho, 0.0],
        [0.0, 0.0, beta]
    ])

    B = np.array([
        [1.0],
        [0.0],
        [beta * eta]
    ])

    Q = np.array([
        [0.5 * gamma * sigma_sq, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
    ])

    R = np.array([
        [0.5 * gamma * sigma_sq + 0.5 * eta]
    ])

    M = np.array([
        [0.5 * gamma * sigma_sq],
        [-0.5],
        [0.5]
    ])

    P = np.zeros((3, 3))
    K_gains = np.zeros((T, 1, 3))

    for t in range(T - 1, -1, -1):
        inv_term = np.linalg.inv(R + B.T @ P @ B)
        K_t = inv_term @ (M.T + B.T @ P @ A)
        K_gains[t] = K_t
        P = Q + A.T @ P @ A - K_t.T @ (R + B.T @ P @ B) @ K_t

    return K_gains


def execute_lqr(T, K_gains, initial_alpha, eta, rho, beta):
    S = np.array([
        [0.0],
        [initial_alpha],
        [0.0]
    ])

    w_path = np.zeros(T)

    for t in range(T):
        u_t = -K_gains[t] @ S
        w_path[t] = S[0, 0] + u_t[0, 0]

        S = np.array([
            [w_path[t]],
            [rho * S[1, 0]],
            [beta * S[2, 0] + beta * eta * u_t[0, 0]]
        ])

    return w_path


def solve_cvxpy_execution(T, gamma, sigma_sq, eta, rho, beta, initial_alpha):
    alphas = initial_alpha * (rho ** np.arange(T))

    u = cp.Variable(T)
    w = cp.Variable(T)

    constraints = [w[0] == u[0]]
    for t in range(1, T):
        constraints.append(w[t] == w[t - 1] + u[t])

    i_idx, j_idx = np.indices((T, T))
    Q_ow = (eta / 2.0) * (beta ** np.abs(i_idx - j_idx))

    expected_pnl = alphas @ u
    risk_penalty = 0.5 * gamma * sigma_sq * cp.sum_squares(w)
    execution_cost = cp.quad_form(u, Q_ow)

    cost = -expected_pnl + risk_penalty + execution_cost

    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.OSQP)

    return w.value
