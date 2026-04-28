import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import time
import sys
import os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__) if "__file__" in dir() else os.getcwd(), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from lqr import solve_and_execute_lqr


plt.style.use('dark_background')
COLORS = {
    'dp':    '#00BFFF',   # cyan-blue  -> DP/LQR
    'cvx':   '#FF6347',   # tomato     -> CVXPY
    'white': '#FFFFFF',
    'gray':  '#555555',
}

ASSET_PALETTE = ['#00BFFF','#FF6347','#FFD700','#DA70D6','#7FFF00',
                 '#FF8C00','#20B2AA','#FF1493','#ADFF2F','#BA55D3']


# ─────────────────────────────────────────────────────────
#  DP / LQR Multi-asset Markowitz
# ─────────────────────────────────────────────────────────

def execute_lqr_markowitz(T, mu, Sigma, w0, n, lam, gamma_tc):
    n = len(mu)
    na = n + 1 
    s0 = np.append(w0, 1.0)
    
    # Augmented dynamics: S_{t} = [w_{t-1}; 1], u_t = delta_w_t
    # S_{t+1} = A S_t + B u_t  =>  w_t = w_{t-1} + u_t
    A = np.eye(na)
    B = np.vstack([np.eye(n), np.zeros((1, n))])

    # --- Matrix Construction ---
    
    # Q matrix: terms involving only S_t (w_{t-1})
    Q = np.zeros((na, na))
    Q[:n, :n] = 0.5 * lam * Sigma
    Q[n, :n]  = -0.5 * mu
    Q[:n, n]  = -0.5 * mu

    # R matrix: terms involving only u_t
    # Includes transaction cost AND the quadratic risk contribution from u_t
    R = 0.5 * gamma_tc * np.eye(n) + 0.5 * lam * Sigma

    # M matrix: cross terms between S_t and u_t
    # Derived from expanding the quadratic form: lam * w_{t-1}' * Sigma * u_t
    M = np.zeros((na, n))
    M[:n, :] = 0.5 * lam * Sigma
    M[n, :]  = -0.5 * mu

    return solve_and_execute_lqr(T, A, B, Q, R, M, s0)


# ─────────────────────────────────────────────────────────
#  CVXPY Solver  (multi-asset Markowitz)
# ─────────────────────────────────────────────────────────
def solve_cvxpy_markowitz(T, mu, Sigma, lam, gamma_tc, w0):
    """
    Solve the same problem via CVXPY.
    """
    n = len(mu)
    W = cp.Variable((T + 1, n))
    U = cp.Variable((T, n))

    cost = 0
    constraints = [W[0] == w0]

    for t in range(T):
        # Transition: w_{t+1} = w_t + u_t
        constraints.append(W[t+1] == W[t] + U[t])
        
        # Objective: Return + Risk + Transaction Cost
        cost += -mu @ W[t+1] \
                + 0.5 * lam * cp.quad_form(W[t+1], Sigma) \
                + 0.5 * gamma_tc * cp.sum_squares(U[t])

    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.OSQP)
    return W.value

# ─────────────────────────────────────────────────────────
#  Utilities
# ─────────────────────────────────────────────────────────

def compute_objective(w_path, mu, Sigma, lam, gamma_tc):
    """Total objective value (higher = better)."""
    total = 0.0
    for t in range(1, len(w_path)):
        dw    = w_path[t] - w_path[t - 1]
        total += (mu @ w_path[t]
                  - 0.5 * lam    * w_path[t] @ Sigma @ w_path[t]
                  - 0.5 * gamma_tc * dw @ dw)
    return total


def random_params(n, seed=42):
    rng = np.random.default_rng(seed)
    L   = rng.standard_normal((n, n)) * 0.1
    Sigma = L @ L.T + np.diag(rng.uniform(0.01, 0.05, n))
    mu  = rng.uniform(0.002, 0.010, n)
    w0  = rng.uniform(-0.05, 0.05, n)
    return mu, Sigma, w0


# ─────────────────────────────────────────────────────────
#  Experiment 1 – single run comparison
# ─────────────────────────────────────────────────────────

def run_single_comparison(T=30, n=5, lam=2.0, gamma_tc=0.5, seed=42):
    mu, Sigma, w0 = random_params(n, seed)
    # DP
    t0 = time.perf_counter()
    result = execute_lqr_markowitz(T, mu, Sigma, w0, n, lam, gamma_tc)
    time_dp = (time.perf_counter() - t0) * 1000
    w_dp = result.s_path[:, :n]
    # CVXPY
    t0 = time.perf_counter()
    w_cvx = solve_cvxpy_markowitz(T, mu, Sigma, lam, gamma_tc, w0)
    time_cvx = (time.perf_counter() - t0) * 1000

    obj_dp  = compute_objective(w_dp,  mu, Sigma, lam, gamma_tc)
    obj_cvx = compute_objective(w_cvx, mu, Sigma, lam, gamma_tc)

    return dict(T=T, n=n, mu=mu, Sigma=Sigma, w0=w0,
                w_dp=w_dp, w_cvx=w_cvx,
                time_dp=time_dp, time_cvx=time_cvx,
                obj_dp=obj_dp, obj_cvx=obj_cvx,
                speedup=time_cvx / time_dp)


# ─────────────────────────────────────────────────────────
#  Experiment 2 – scaling over number of assets
# ─────────────────────────────────────────────────────────

def run_scaling_experiment(n_list=None, T=30, n_trials=5, lam=2.0, gamma_tc=0.5):
    if n_list is None:
        n_list = [2, 3, 5, 7, 10, 15, 20]

    rows = []
    for n in n_list:
        dp_times, cvx_times = [], []
        for seed in range(n_trials):
            mu, Sigma, w0 = random_params(n, seed)

            t0 = time.perf_counter()
            execute_lqr_markowitz(T, mu, Sigma, w0, n, lam, gamma_tc)
            dp_times.append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            solve_cvxpy_markowitz(T, mu, Sigma, lam, gamma_tc, w0)
            cvx_times.append((time.perf_counter() - t0) * 1000)

        rows.append(dict(n=n,
                         dp_mean=np.mean(dp_times),  dp_std=np.std(dp_times),
                         cvx_mean=np.mean(cvx_times), cvx_std=np.std(cvx_times),
                         speedup=np.mean(cvx_times) / np.mean(dp_times)))
    return rows


# ─────────────────────────────────────────────────────────
#  Visualisation
# ─────────────────────────────────────────────────────────

def plot_all(r, scale_rows):
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor('#111111')

    steps = np.arange(r['T'] + 1)
    norm_dp  = np.linalg.norm(r['w_dp'],  axis=1)
    norm_cvx = np.linalg.norm(r['w_cvx'], axis=1)
    
    # ── Panel 1: ‖w‖ trajectory ──────────────────────────
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_facecolor('#1a1a1a')
    ax1b = ax1.twinx()

    ax1.plot(steps, norm_dp,  color=COLORS['dp'],  lw=2.5, label='DP/LQR')
    ax1.plot(steps, norm_cvx, color=COLORS['cvx'], lw=2, ls='--', label='CVXPY')

    ax1.set_title('Portfolio norm ‖w‖₂ over time', color='white', fontsize=11)
    ax1.set_xlabel('Time step', color='white'); ax1.set_ylabel('‖w‖₂', color='white')
    ax1.tick_params(colors='white'); ax1b.tick_params(colors='white')
    lines1, lbs1 = ax1.get_legend_handles_labels()
    ax1.legend(lines1, lbs1, frameon=False, fontsize=9)
    ax1.grid(color='gray', ls=':', alpha=0.3)

    # ── Panel 2: individual asset weight paths (DP) ──────
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_facecolor('#1a1a1a')
    for i in range(r['n']):
        ax2.plot(steps, r['w_dp'][:, i],
                 color=ASSET_PALETTE[i % len(ASSET_PALETTE)],
                 lw=1.8, label=f'Asset {i+1} (μ={r["mu"][i]*100:.2f}%)')
    ax2.set_title('Asset weight paths (DP)', color='white', fontsize=11)
    ax2.set_xlabel('Time step', color='white'); ax2.set_ylabel('Weight', color='white')
    ax2.tick_params(colors='white')
    ax2.legend(frameon=False, fontsize=8, ncol=2)
    ax2.grid(color='gray', ls=':', alpha=0.3)
    ax2.axhline(0, color='white', lw=0.5, alpha=0.4)

    # ── Panel 3: scaling plot ─────────────────────────────
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_facecolor('#1a1a1a')
    ns      = [row['n']       for row in scale_rows]
    dp_m    = [row['dp_mean'] for row in scale_rows]
    dp_s    = [row['dp_std']  for row in scale_rows]
    cvx_m   = [row['cvx_mean'] for row in scale_rows]
    cvx_s   = [row['cvx_std']  for row in scale_rows]

    ax3.plot(ns, dp_m,  color=COLORS['dp'],  lw=2.5, marker='o', ms=5, label='DP/LQR')
    ax3.plot(ns, cvx_m, color=COLORS['cvx'], lw=2,   marker='s', ms=5, ls='--', label='CVXPY')
    ax3.fill_between(ns, np.array(dp_m)-dp_s,   np.array(dp_m)+dp_s,   color=COLORS['dp'],  alpha=0.15)
    ax3.fill_between(ns, np.array(cvx_m)-cvx_s, np.array(cvx_m)+cvx_s, color=COLORS['cvx'], alpha=0.15)

    ax3.set_title('Solve time vs number of assets', color='white', fontsize=11)
    ax3.set_xlabel('Number of assets n', color='white')
    ax3.set_ylabel('Time (ms)', color='white')
    ax3.tick_params(colors='white')
    ax3.legend(frameon=False, fontsize=9)
    ax3.grid(color='gray', ls=':', alpha=0.3)

    # ── Panel 4: timing bar + objective comparison ────────
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_facecolor('#1a1a1a')
    methods = ['DP/LQR', 'CVXPY']
    times   = [r['time_dp'], r['time_cvx']]
    bars    = ax4.bar(methods, times,
                      color=[COLORS['dp'], COLORS['cvx']],
                      width=0.4, alpha=0.85)
    for bar, t in zip(bars, times):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                 f'{t:.2f} ms', ha='center', va='bottom',
                 color='white', fontsize=11, fontweight='bold')

    ax4.set_title(
        f'Execution time  |  speedup: {r["speedup"]:.1f}×\n'
        f'Objective — DP: {r["obj_dp"]:.5f}  CVXPY: {r["obj_cvx"]:.5f}',
        color='white', fontsize=10)
    ax4.set_ylabel('Time (ms)', color='white')
    ax4.tick_params(colors='white')
    ax4.grid(axis='y', color='gray', ls=':', alpha=0.3)

    fig.suptitle(
        f'DP/LQR vs CVXPY  —  Multi-asset Markowitz  '
        f'(T={r["T"]}, n={r["n"]}, λ={2.0}, γ_tc={0.5})',
        color='white', fontsize=13, y=1.01)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Multi-asset Markowitz: DP/LQR vs CVXPY")
    print("=" * 60)

    # ── Single run (n=5 assets, T=30 steps) ──
    print("\n[1] Single-run comparison  (n=5, T=30)...")
    r = run_single_comparison(T=30, n=5, lam=2.0, gamma_tc=0.5, seed=42)

    print(f"\n  DP/LQR   : {r['time_dp']:.2f} ms   obj = {r['obj_dp']:.6f}")
    print(f"  CVXPY    : {r['time_cvx']:.2f} ms   obj = {r['obj_cvx']:.6f}")
    print(f"  Speedup  : {r['speedup']:.1f}×")

    md = (
        "\n| Method   | Time (ms) | Objective | Speedup |\n"
        "| :------- | --------: | --------: | ------: |\n"
        f"| **CVXPY**    | {r['time_cvx']:.2f} | {r['obj_cvx']:.6f} | 1× |\n"
        f"| **DP/LQR**   | {r['time_dp']:.2f} | {r['obj_dp']:.6f} | **~{r['speedup']:.0f}×** |"
    )
    print(md)

    # ── Scaling experiment ──
    print("\n[2] Scaling experiment  (n ∈ {2,3,5,7,10,15,20}, T=30)...")
    scale_rows = run_scaling_experiment(
        n_list=[2, 3, 5, 7, 10, 15, 20], T=30, n_trials=5)

    print("\n  n  |  DP (ms)  |  CVXPY (ms)  |  speedup")
    print("  ---+-----------+--------------+---------")
    for row in scale_rows:
        print(f"  {row['n']:2d} |  {row['dp_mean']:6.2f}    |  {row['cvx_mean']:9.2f}   |  {row['speedup']:.1f}×")

    # ── Plot ──
    print("\n[3] Generating plots...")
    fig = plot_all(r, scale_rows)
    plt.savefig('markowitz_dp_vs_cvxpy.png', dpi=150,
                bbox_inches='tight', facecolor='#111111')
    print("  Saved: markowitz_dp_vs_cvxpy.png")
    plt.show()

if __name__ == "__main__":
    main()