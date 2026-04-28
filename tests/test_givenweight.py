# Multi-Asset Risk Parity Tracking: DP/LQR vs CVXPY

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import time
from scipy.optimize import minimize


plt.style.use('dark_background')
COLORS = {
    'dp':    '#00BFFF',   # cyan-blue  -> DP/LQR
    'cvx':   '#FF6347',   # tomato     -> CVXPY
    'target':'#FFD700',   # gold       -> Target w_RP
    'white': '#FFFFFF',
    'gray':  '#555555',
}

ASSET_PALETTE = ['#00BFFF','#FF6347','#FFD700','#DA70D6','#7FFF00',
                 '#FF8C00','#20B2AA','#FF1493','#ADFF2F','#BA55D3']

# ─────────────────────────────────────────────────────────
#  CVXPY Solver  (Risk Parity Tracking)
# ─────────────────────────────────────────────────────────
def solve_cvxpy(T, w_given, kappa, gamma_tc, w0):
    """
    Solve the tracking problem via CVXPY.
    """
    n = len(w_given)
    W = cp.Variable((T + 1, n))
    U = cp.Variable((T, n))

    cost = 0
    constraints = [W[0] == w0]

    for t in range(T):
        # Transition: w_{t+1} = w_t + u_t
        constraints.append(W[t+1] == W[t] + U[t])
        
        # Objective: Tracking Penalty + Transaction Cost
        cost += 0.5 * kappa * cp.sum_squares(W[t+1] - w_RP) \
              + 0.5 * gamma_tc * cp.sum_squares(U[t])

    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.OSQP)
    return W.value

# ─────────────────────────────────────────────────────────
#  Visualisation
# ─────────────────────────────────────────────────────────

def plot_all(r, scale_rows):
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor('#111111')

    steps = np.arange(r['T'] + 1)
    
    # Calculate tracking errors: ||w_t - w_RP||_2
    err_dp  = np.linalg.norm(r['w_dp'] - r['w_RP'], axis=1)
    err_cvx = np.linalg.norm(r['w_cvx'] - r['w_RP'], axis=1)
    
    # ── Panel 1: Tracking Error trajectory ────────────────
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_facecolor('#1a1a1a')
    ax1b = ax1.twinx()

    ax1.plot(steps, err_dp,  color=COLORS['dp'],  lw=2.5, label='DP/LQR')
    ax1.plot(steps, err_cvx, color=COLORS['cvx'], lw=2, ls='--', label='CVXPY')

    ax1.set_title('Tracking Error ‖w - w^{RP}‖₂ over time', color='white', fontsize=11)
    ax1.set_xlabel('Time step', color='white'); ax1.set_ylabel('Tracking Error', color='white')
    ax1.tick_params(colors='white'); ax1b.tick_params(colors='white')
    lines1, lbs1 = ax1.get_legend_handles_labels()
    ax1.legend(lines1, lbs1, frameon=False, fontsize=9)
    ax1.grid(color='gray', ls=':', alpha=0.3)

    # ── Panel 2: individual asset weight paths (DP) ──────
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_facecolor('#1a1a1a')
    for i in range(r['n']):
        c = ASSET_PALETTE[i % len(ASSET_PALETTE)]
        ax2.plot(steps, r['w_dp'][:, i], color=c, lw=1.8, label=f'Asset {i+1}')
        # Plot target lines
        ax2.axhline(r['w_RP'][i], color=c, ls='--', lw=1, alpha=0.7)
        
    ax2.set_title('Asset weight paths (DP) vs Targets (dashed)', color='white', fontsize=11)
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
        f'Cost (lower=better) — DP: {r["cost_dp"]:.5f}  CVXPY: {r["cost_cvx"]:.5f}',
        color='white', fontsize=10)
    ax4.set_ylabel('Time (ms)', color='white')
    ax4.tick_params(colors='white')
    ax4.grid(axis='y', color='gray', ls=':', alpha=0.3)

    fig.suptitle(
        f'DP/LQR vs CVXPY  —  Risk Parity Target Tracking  '
        f'(T={r["T"]}, n={r["n"]}, κ={r["kappa"]}, γ_tc={r["gamma_tc"]})',
        color='white', fontsize=13, y=1.01)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Given Weight Extension: DP/LQR vs CVXPY")
    print("=" * 60)

    # ── Single run (n=5 assets, T=30 steps) ──
    print("\n[1] Single-run comparison  (n=5, T=30)...")
    r = run_single_comparison(T=30, n=5, kappa=2.0, gamma_tc=0.5, seed=42)

    print(f"\n  DP/LQR   : {r['time_dp']:.2f} ms   Cost = {r['cost_dp']:.6f}")
    print(f"  CVXPY    : {r['time_cvx']:.2f} ms   Cost = {r['cost_cvx']:.6f}")
    print(f"  Speedup  : {r['speedup']:.1f}×")

    md = (
        "\n| Method   | Time (ms) | Cost (lower=better) | Speedup |\n"
        "| :------- | --------: | ------------------: | ------: |\n"
        f"| **CVXPY** | {r['time_cvx']:.2f} | {r['cost_cvx']:.6f} | 1× |\n"
        f"| **DP/LQR** | {r['time_dp']:.2f} | {r['cost_dp']:.6f} | **~{r['speedup']:.0f}×** |"
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
    plt.savefig('risk_parity_dp_vs_cvxpy.png', dpi=150,
                bbox_inches='tight', facecolor='#111111')
    print("  Saved: risk_parity_dp_vs_cvxpy.png")
    plt.show()


if __name__ == "__main__":
    main()
