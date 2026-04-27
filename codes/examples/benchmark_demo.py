import sys
import os
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import time

# Ensure Python can find the 'src' folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.lqr_solver import MarkowitzLQR

# Visual Styling
plt.style.use('dark_background')
COLORS = {'dp': '#00BFFF', 'cvx': '#FF6347', 'white': '#FFFFFF', 'gray': '#555555'}
ASSET_PALETTE = ['#00BFFF','#FF6347','#FFD700','#DA70D6','#7FFF00',
                 '#FF8C00','#20B2AA','#FF1493','#ADFF2F','#BA55D3']

# --- Utility Functions from your original code ---

def random_params(n, seed=42):
    """Generates random market parameters."""
    rng = np.random.default_rng(seed)
    L = rng.standard_normal((n, n)) * 0.1
    Sigma = L @ L.T + np.diag(rng.uniform(0.01, 0.05, n))
    mu = rng.uniform(0.002, 0.010, n)
    w0 = rng.uniform(-0.05, 0.05, n)
    return mu, Sigma, w0

def solve_cvxpy_markowitz(T, mu, Sigma, lam, gamma_tc, w0):
    """CVXPY benchmark solver."""
    n = len(mu)
    W = cp.Variable((T + 1, n))
    U = cp.Variable((T, n))
    cost = 0
    constraints = [W[0] == w0]
    for t in range(T):
        constraints.append(W[t+1] == W[t] + U[t])
        cost += -mu @ W[t+1] + 0.5 * lam * cp.quad_form(W[t+1], Sigma) + 0.5 * gamma_tc * cp.sum_squares(U[t])
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.OSQP)
    return W.value

def compute_objective(w_path, mu, Sigma, lam, gamma_tc):
    """Calculates total objective value."""
    total = 0.0
    for t in range(1, len(w_path)):
        dw = w_path[t] - w_path[t - 1]
        total += (mu @ w_path[t] - 0.5 * lam * w_path[t] @ Sigma @ w_path[t] - 0.5 * gamma_tc * dw @ dw)
    return total

# --- Refactored Experiment Logic using the Black Box ---



def run_single_comparison(T=30, n=5, lam=2.0, gamma_tc=0.5, seed=42):
    """
    Runs a comparison between LQR Black Box and CVXPY, 
    measuring both accuracy and execution time.
    """
    mu, Sigma, w0 = random_params(n, seed)

    # --- 1. Measure DP/LQR (Your Black Box) ---
    solver = MarkowitzLQR(risk_aversion=lam, tc_coeff=gamma_tc)
    
    t0_dp = time.perf_counter()  # Start timer
    
    # Core LQR process: Solve once, then predict step-by-step
    solver.solve(T, mu, Sigma)
    w_dp = np.zeros((T + 1, n))
    w_dp[0] = w0
    for t in range(T):
        w_dp[t+1] = w_dp[t] + solver.predict(w_dp[t], t)
        
    t1_dp = time.perf_counter()  # End timer
    time_dp = (t1_dp - t0_dp) * 1000  # Convert to milliseconds

    # --- 2. Measure CVXPY (The standard solver) ---
    t0_cvx = time.perf_counter()
    
    w_cvx = solve_cvxpy_markowitz(T, mu, Sigma, lam, gamma_tc, w0)
    
    t1_cvx = time.perf_counter()
    time_cvx = (t1_cvx - t0_cvx) * 1000  # Convert to milliseconds

    # --- 3. Compute Metrics ---
    obj_dp  = compute_objective(w_dp,  mu, Sigma, lam, gamma_tc)
    obj_cvx = compute_objective(w_cvx, mu, Sigma, lam, gamma_tc)
    speedup = time_cvx / time_dp

    return {
        "T": T, "n": n, "mu": mu, "Sigma": Sigma,
        "w_dp": w_dp, "w_cvx": w_cvx,
        "time_dp": time_dp, 
        "time_cvx": time_cvx,
        "obj_dp": obj_dp, 
        "obj_cvx": obj_cvx,
        "speedup": speedup
    }




# --- Plotting function (Keep your original visual logic) ---

def plot_results(r):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    steps = np.arange(len(r['w_dp']))
    
    # Weights plot
    for i in range(r['n']):
        axes[0].plot(steps, r['w_dp'][:, i], label=f'Asset {i+1}', color=ASSET_PALETTE[i])
    axes[0].set_title("Portfolio Weights (DP)")
    axes[0].legend()
    
    # Execution time bar
    axes[1].bar(['DP/LQR', 'CVXPY'], [r['time_dp'], r['time_cvx']], color=[COLORS['dp'], COLORS['cvx']])
    axes[1].set_title(f"Speedup: {r['speedup']:.1f}x")
    axes[1].set_ylabel("Time (ms)")
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("=" * 40)
    print("  Markowitz Optimizer: DP vs CVXPY")
    print("=" * 40)
    
    
    results = run_single_comparison(n=5, T=30)
    
    
    print(f"\n[Performance Report]")
    print(f"LQR Solver (Black Box) : {results['time_dp']:.4f} ms")
    print(f"CVXPY Solver (Standard): {results['time_cvx']:.4f} ms")
    print(f"Speedup Factor         : {results['speedup']:.2f}x")
    
    
    print(f"\n[Accuracy Report]")
    print(f"DP Objective  : {results['obj_dp']:.6f}")
    print(f"CVX Objective : {results['obj_cvx']:.6f}")
    print("=" * 40)