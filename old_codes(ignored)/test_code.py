import numpy as np
import matplotlib.pyplot as plt
import time
from dp_solver import solve_lqr_execution, execute_lqr, solve_cvxpy_execution


plt.style.use('dark_background')
colors = {'cyan': '#00FFFF', 'gold': '#FFD700', 'pink': '#FF1493', 'white': '#FFFFFF', 'gray': '#555555'}


# ---  The Horse Race ---
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

if __name__ == "__main__":
    md_table, fig = run_cvx_dp_comparison()
    print(md_table)
    plt.show()
