import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import time


plt.style.use('dark_background')
colors = {'cyan': '#00FFFF', 'gold': '#FFD700', 'pink': '#FF1493', 'white': '#FFFFFF', 'gray': '#555555'}

# --- 1. LQR Engine (From our previous math) ---
def solve_lqr_execution(T, gamma, sigma_sq, eta, rho, beta):
    # --- FIXED: The D dynamics now correctly apply the beta decay to the trade (beta * eta) ---
    A = np.array([[1.0, 0.0, 0.0], [0.0, rho, 0.0], [0.0, 0.0, beta]])
    B = np.array([[1.0], [0.0], [beta * eta]]) 
    
    Q = np.array([[0.5 * gamma * sigma_sq, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    R = np.array([[0.5 * gamma * sigma_sq + 0.5 * eta]])
    M = np.array([[0.5 * gamma * sigma_sq], [-0.5], [0.5]])

    P = np.zeros((3, 3))
    K_gains = np.zeros((T, 1, 3))
    for t in range(T - 1, -1, -1):
        inv_term = np.linalg.inv(R + B.T @ P @ B)
        K_t = inv_term @ (M.T + B.T @ P @ A)
        K_gains[t] = K_t
        P = Q + A.T @ P @ A - K_t.T @ (R + B.T @ P @ B) @ K_t
    return K_gains

def execute_lqr(T, K_gains, initial_alpha, eta, rho, beta):
    S = np.array([[0.0], [initial_alpha], [0.0]]) 
    w_path = np.zeros(T)
    for t in range(T):
        u_t = -K_gains[t] @ S
        w_path[t] = S[0, 0] + u_t[0, 0]
        # --- FIXED: The forward simulation state also applies the decay ---
        S = np.array([[w_path[t]], [rho * S[1, 0]], [beta * S[2, 0] + beta * eta * u_t[0, 0]]])
    return w_path

# --- 2. CVXPY Engine ---
def solve_cvxpy_execution(T, gamma, sigma_sq, eta, rho, beta, initial_alpha):
    alphas = initial_alpha * (rho ** np.arange(T))
    u = cp.Variable(T)
    w = cp.Variable(T)
    
    # We no longer need to track D[t] as a variable!
    constraints = [w[0] == u[0]]
    for t in range(1, T):
        constraints.append(w[t] == w[t-1] + u[t])
        
    # Build the exact OW Quadratic Form Matrix (Q)
    i_idx, j_idx = np.indices((T, T))
    Q_ow = (eta / 2.0) * (beta ** np.abs(i_idx - j_idx))
    
    # Vectorized objective function
    expected_pnl = alphas @ u
    risk_penalty = 0.5 * gamma * sigma_sq * cp.sum_squares(w)
    
    # DCP Compliant execution cost mapping D_t * u_t directly into the matrix
    execution_cost = cp.quad_form(u, Q_ow)
            
    cost = -expected_pnl + risk_penalty + execution_cost
    
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.OSQP)
    return w.value

# --- 3. The Horse Race ---
def run_cvx_dp_comparison():
    T = 50
    gamma, sigma_sq, eta = 0.1, 0.01, 0.05
    rho, beta, initial_alpha = 0.96, 0.6, 0.0050
    
    # Time LQR
    start_lqr = time.perf_counter()
    K_gains = solve_lqr_execution(T, gamma, sigma_sq, eta, rho, beta)
    w_lqr = execute_lqr(T, K_gains, initial_alpha, eta, rho, beta)
    time_lqr = (time.perf_counter() - start_lqr) * 1000 # ms
    
    # Time CVXPY
    start_cvx = time.perf_counter()
    w_cvxpy = solve_cvxpy_execution(T, gamma, sigma_sq, eta, rho, beta, initial_alpha)
    time_cvx = (time.perf_counter() - start_cvx) * 1000 # ms
    
    # --- 4. Output and Visualization ---
    md_table = (
        "| Optimizer Method | Execution Time (ms) | Speed Multiplier |\n"
        "| :--- | :--- | :--- |\n"
        f"| **CVXPY** | {time_cvx:.2f} ms | 1x |\n"
        f"| **DP Solver** | **{time_lqr:.2f} ms** | **~{int(time_cvx/time_lqr)}x faster** |"
    )
    
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot CVXPY as a thick pink line
    ax.plot(w_cvxpy, color=colors['pink'], linewidth=4, alpha=0.7, label='CVXPY Trajectory')
    
    # Plot LQR as cyan dots on top
    ax.plot(w_lqr, color=colors['cyan'], marker='o', markersize=5, linestyle='', label='DP Trajectory')
    
    ax.set_title("CVXPY vs DP Solved Trajectory", color='white', fontsize=14)
    ax.set_xlabel("Time Step", color='white')
    ax.set_ylabel("Portfolio Weight", color='white')
    ax.legend(frameon=False, fontsize=12)
    ax.grid(color='gray', linestyle=':', alpha=0.3)
    
    return md_table, fig