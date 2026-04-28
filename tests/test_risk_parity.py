"""Multi-Asset Risk Parity Tracking: DP/LQR (via lqr package) vs CVXPY."""

import os
import sys
import time

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from lqr import solve_and_execute_lqr

plt.style.use("dark_background")
COLORS = {
    "dp": "#00BFFF",
    "cvx": "#FF6347",
    "target": "#FFD700",
    "white": "#FFFFFF",
    "gray": "#555555",
}
ASSET_PALETTE = [
    "#00BFFF",
    "#FF6347",
    "#FFD700",
    "#DA70D6",
    "#7FFF00",
    "#FF8C00",
    "#20B2AA",
    "#FF1493",
    "#ADFF2F",
    "#BA55D3",
]


# ─────────────────────────────────────────────────────────
#  Static Risk-Parity target
# ─────────────────────────────────────────────────────────
def get_risk_parity_target(Sigma):
    """Solve static risk-parity target weights."""
    n = Sigma.shape[0]

    def objective(w):
        port_var = w.T @ Sigma @ w
        marginal_contrib = Sigma @ w
        risk_contrib = w * marginal_contrib
        target_contrib = port_var / n
        return np.sum((risk_contrib - target_contrib) ** 2)

    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bounds = tuple((0.0, 1.0) for _ in range(n))
    w0 = np.ones(n) / n
    res = minimize(objective, w0, bounds=bounds, constraints=constraints, method="SLSQP")
    return res.x


# ─────────────────────────────────────────────────────────
#  DP/LQR via lqr.solve_and_execute_lqr
# ─────────────────────────────────────────────────────────
def build_rp_tracking_lqr_matrices(w_rp, kappa, gamma_tc):
    """Build augmented-state LQR matrices (same objective as CVXPY block)."""
    n = len(w_rp)
    na = n + 1
    A_aug = np.eye(na)
    B_aug = np.vstack([np.eye(n), np.zeros((1, n))])

    Q_aug = np.zeros((na, na))
    Q_aug[:n, :n] = 0.5 * kappa * np.eye(n)
    Q_aug[n, :n] = -0.5 * kappa * w_rp
    Q_aug[:n, n] = -0.5 * kappa * w_rp
    Q_aug[n, n] = 0.5 * kappa * np.dot(w_rp, w_rp)

    R_mat = 0.5 * (kappa + gamma_tc) * np.eye(n)

    M_aug = np.zeros((na, n))
    M_aug[:n, :] = 0.5 * kappa * np.eye(n)
    M_aug[n, :] = -0.5 * kappa * w_rp
    return A_aug, B_aug, Q_aug, R_mat, M_aug


def solve_dp_risk_parity(T, w_rp, kappa, gamma_tc, w0):
    """Solve risk-parity tracking with package LQR solver."""
    n = len(w_rp)
    A, B, Q, R, M = build_rp_tracking_lqr_matrices(w_rp, kappa, gamma_tc)
    s0 = np.append(w0, 1.0)
    result = solve_and_execute_lqr(T, A, B, Q, R, M, s0)
    return result.s_path[:, :n]


# ─────────────────────────────────────────────────────────
#  CVXPY baseline
# ─────────────────────────────────────────────────────────
def solve_cvxpy_risk_parity(T, w_rp, kappa, gamma_tc, w0):
    """Solve the same tracking objective via CVXPY."""
    n = len(w_rp)
    W = cp.Variable((T + 1, n))
    U = cp.Variable((T, n))
    cost = 0
    constraints = []
    constraints.append(W[0] == w0)

    for t in range(T):
        constraints.append(W[t + 1] == W[t] + U[t])
        cost += 0.5 * kappa * cp.sum_squares(W[t + 1] - w_rp)
        cost += 0.5 * gamma_tc * cp.sum_squares(U[t])

    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.OSQP, verbose=False)
    return W.value


# ─────────────────────────────────────────────────────────
#  Utilities
# ─────────────────────────────────────────────────────────
def compute_tracking_cost(w_path, w_rp, kappa, gamma_tc):
    total = 0.0
    for t in range(1, len(w_path)):
        dw = w_path[t] - w_path[t - 1]
        err = w_path[t] - w_rp
        total += 0.5 * kappa * np.dot(err, err) + 0.5 * gamma_tc * np.dot(dw, dw)
    return float(total)


def random_params(n, seed=42):
    rng = np.random.default_rng(seed)
    L = rng.standard_normal((n, n)) * 0.1
    Sigma = L @ L.T + np.diag(rng.uniform(0.01, 0.05, n))
    w0 = rng.uniform(0.0, 0.4, n)
    w0 /= np.sum(w0)
    return Sigma, w0


# ─────────────────────────────────────────────────────────
#  Experiments
# ─────────────────────────────────────────────────────────
def run_single_comparison(T=30, n=5, kappa=2.0, gamma_tc=0.5, seed=42):
    Sigma, w0 = random_params(n, seed)
    w_rp = get_risk_parity_target(Sigma)

    t0 = time.perf_counter()
    w_dp = solve_dp_risk_parity(T, w_rp, kappa, gamma_tc, w0)
    time_dp = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    w_cvx = solve_cvxpy_risk_parity(T, w_rp, kappa, gamma_tc, w0)
    time_cvx = (time.perf_counter() - t0) * 1000

    cost_dp = compute_tracking_cost(w_dp, w_rp, kappa, gamma_tc)
    cost_cvx = compute_tracking_cost(w_cvx, w_rp, kappa, gamma_tc)

    return {
        "T": T,
        "n": n,
        "Sigma": Sigma,
        "w0": w0,
        "w_rp": w_rp,
        "kappa": kappa,
        "gamma_tc": gamma_tc,
        "w_dp": w_dp,
        "w_cvx": w_cvx,
        "time_dp": time_dp,
        "time_cvx": time_cvx,
        "cost_dp": cost_dp,
        "cost_cvx": cost_cvx,
        "speedup": time_cvx / time_dp if time_dp > 0 else np.nan,
        "max_abs_diff": float(np.max(np.abs(w_dp - w_cvx))),
    }


def run_scaling_experiment(n_list=None, T=30, n_trials=5, kappa=2.0, gamma_tc=0.5):
    if n_list is None:
        n_list = [2, 3, 5, 7, 10, 15, 20]

    rows = []
    for n in n_list:
        dp_times, cvx_times = [], []
        for seed in range(n_trials):
            Sigma, w0 = random_params(n, seed)
            w_rp = get_risk_parity_target(Sigma)

            t0 = time.perf_counter()
            solve_dp_risk_parity(T, w_rp, kappa, gamma_tc, w0)
            dp_times.append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            solve_cvxpy_risk_parity(T, w_rp, kappa, gamma_tc, w0)
            cvx_times.append((time.perf_counter() - t0) * 1000)

        rows.append(
            {
                "n": n,
                "dp_mean": float(np.mean(dp_times)),
                "dp_std": float(np.std(dp_times)),
                "cvx_mean": float(np.mean(cvx_times)),
                "cvx_std": float(np.std(cvx_times)),
                "speedup": float(np.mean(cvx_times) / np.mean(dp_times)),
            }
        )
    return rows


# ─────────────────────────────────────────────────────────
#  Visualisation
# ─────────────────────────────────────────────────────────
def plot_all(r, scale_rows):
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor("#111111")
    steps = np.arange(r["T"] + 1)
    err_dp = np.linalg.norm(r["w_dp"] - r["w_rp"], axis=1)
    err_cvx = np.linalg.norm(r["w_cvx"] - r["w_rp"], axis=1)

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_facecolor("#1a1a1a")
    ax1.plot(steps, err_dp, color=COLORS["dp"], lw=2.5, label="DP/LQR")
    ax1.plot(steps, err_cvx, color=COLORS["cvx"], lw=2, ls="--", label="CVXPY")
    ax1.set_title("Tracking Error ‖w - w_RP‖₂ over time", color="white", fontsize=11)
    ax1.set_xlabel("Time step", color="white")
    ax1.set_ylabel("Tracking Error", color="white")
    ax1.tick_params(colors="white")
    ax1.legend(frameon=False, fontsize=9)
    ax1.grid(color="gray", ls=":", alpha=0.3)

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_facecolor("#1a1a1a")
    for i in range(r["n"]):
        c = ASSET_PALETTE[i % len(ASSET_PALETTE)]
        ax2.plot(steps, r["w_dp"][:, i], color=c, lw=1.8, label=f"Asset {i+1}")
        ax2.axhline(r["w_rp"][i], color=c, ls="--", lw=1, alpha=0.7)
    ax2.set_title("Asset weight paths (DP) vs Targets (dashed)", color="white", fontsize=11)
    ax2.set_xlabel("Time step", color="white")
    ax2.set_ylabel("Weight", color="white")
    ax2.tick_params(colors="white")
    ax2.legend(frameon=False, fontsize=8, ncol=2)
    ax2.grid(color="gray", ls=":", alpha=0.3)
    ax2.axhline(0, color="white", lw=0.5, alpha=0.4)

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_facecolor("#1a1a1a")
    ns = [row["n"] for row in scale_rows]
    dp_m = [row["dp_mean"] for row in scale_rows]
    dp_s = [row["dp_std"] for row in scale_rows]
    cvx_m = [row["cvx_mean"] for row in scale_rows]
    cvx_s = [row["cvx_std"] for row in scale_rows]
    ax3.plot(ns, dp_m, color=COLORS["dp"], lw=2.5, marker="o", ms=5, label="DP/LQR")
    ax3.plot(ns, cvx_m, color=COLORS["cvx"], lw=2, marker="s", ms=5, ls="--", label="CVXPY")
    ax3.fill_between(ns, np.array(dp_m) - dp_s, np.array(dp_m) + dp_s, color=COLORS["dp"], alpha=0.15)
    ax3.fill_between(ns, np.array(cvx_m) - cvx_s, np.array(cvx_m) + cvx_s, color=COLORS["cvx"], alpha=0.15)
    ax3.set_title("Solve time vs number of assets", color="white", fontsize=11)
    ax3.set_xlabel("Number of assets n", color="white")
    ax3.set_ylabel("Time (ms)", color="white")
    ax3.tick_params(colors="white")
    ax3.legend(frameon=False, fontsize=9)
    ax3.grid(color="gray", ls=":", alpha=0.3)

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_facecolor("#1a1a1a")
    methods = ["DP/LQR", "CVXPY"]
    times = [r["time_dp"], r["time_cvx"]]
    bars = ax4.bar(methods, times, color=[COLORS["dp"], COLORS["cvx"]], width=0.4, alpha=0.85)
    for bar, t in zip(bars, times):
        ax4.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.2,
            f"{t:.2f} ms",
            ha="center",
            va="bottom",
            color="white",
            fontsize=11,
            fontweight="bold",
        )
    ax4.set_title(
        f'Execution time  |  speedup: {r["speedup"]:.1f}×\n'
        f'Cost (lower=better) — DP: {r["cost_dp"]:.5f}  CVXPY: {r["cost_cvx"]:.5f}',
        color="white",
        fontsize=10,
    )
    ax4.set_ylabel("Time (ms)", color="white")
    ax4.tick_params(colors="white")
    ax4.grid(axis="y", color="gray", ls=":", alpha=0.3)

    fig.suptitle(
        f'DP/LQR vs CVXPY  —  Risk Parity Target Tracking  '
        f'(T={r["T"]}, n={r["n"]}, κ={r["kappa"]}, γ_tc={r["gamma_tc"]})',
        color="white",
        fontsize=13,
        y=1.01,
    )
    plt.tight_layout()
    return fig


def main():
    print("=" * 60)
    print("  Risk Parity Extension: DP/LQR vs CVXPY")
    print("=" * 60)

    print("\n[1] Single-run comparison  (n=5, T=30)...")
    r = run_single_comparison(T=30, n=5, kappa=2.0, gamma_tc=0.5, seed=42)
    print(f"\n  DP/LQR   : {r['time_dp']:.2f} ms   Cost = {r['cost_dp']:.6f}")
    print(f"  CVXPY    : {r['time_cvx']:.2f} ms   Cost = {r['cost_cvx']:.6f}")
    print(f"  Speedup  : {r['speedup']:.1f}×")
    print(f"  max |W_DP - W_CVX| = {r['max_abs_diff']:.2e}")

    print("\n[2] Scaling experiment  (n ∈ {2,3,5,7,10,15,20}, T=30)...")
    scale_rows = run_scaling_experiment(n_list=[2, 3, 5, 7, 10, 15, 20], T=30, n_trials=5)
    print("\n  n  |  DP (ms)  |  CVXPY (ms)  |  speedup")
    print("  ---+-----------+--------------+---------")
    for row in scale_rows:
        print(f"  {row['n']:2d} |  {row['dp_mean']:6.2f}    |  {row['cvx_mean']:9.2f}   |  {row['speedup']:.1f}×")

    print("\n[3] Generating plots...")
    fig = plot_all(r, scale_rows)
    plt.savefig("risk_parity_dp_vs_cvxpy.png", dpi=150, bbox_inches="tight", facecolor="#111111")
    print("  Saved: risk_parity_dp_vs_cvxpy.png")
    plt.show()


if __name__ == "__main__":
    main()

