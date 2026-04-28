# Multi-Asset Given Weight Tracking: DP/LQR vs CVXPY

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import time

from lqr.solver import solve_lqr, execute_lqr


plt.style.use("dark_background")

COLORS = {
    "dp": "#00BFFF",
    "cvx": "#FF6347",
    "target": "#FFD700",
    "white": "#FFFFFF",
    "gray": "#555555",
}

ASSET_PALETTE = [
    "#00BFFF", "#FF6347", "#FFD700", "#DA70D6", "#7FFF00",
    "#FF8C00", "#20B2AA", "#FF1493", "#ADFF2F", "#BA55D3"
]


# ─────────────────────────────────────────────────────────
#  DP / LQR Solver Wrapper
# ─────────────────────────────────────────────────────────

def solve_tracking_dp(T, w_given, kappa, gamma_tc, w0):
    """
    Solve given-weight tracking using the generic LQR solver.
    """
    w_given = np.asarray(w_given, dtype=float)
    w0 = np.asarray(w0, dtype=float)

    n = len(w_given)
    na = n + 1

    # State: s_t = [w_t; 1]
    A_aug = np.eye(na)
    B_aug = np.vstack([np.eye(n), np.zeros((1, n))])

    # Cost:
    # (kappa/2)||w_t + u_t - w_given||^2
    # + (gamma_tc/2)||u_t||^2
    Q_aug = np.zeros((na, na))
    Q_aug[:n, :n] = 0.5 * kappa * np.eye(n)
    Q_aug[n, :n] = -0.5 * kappa * w_given
    Q_aug[:n, n] = -0.5 * kappa * w_given
    Q_aug[n, n] = 0.5 * kappa * np.dot(w_given, w_given)

    R_mat = 0.5 * (kappa + gamma_tc) * np.eye(n)

    M_aug = np.zeros((na, n))
    M_aug[:n, :] = 0.5 * kappa * np.eye(n)
    M_aug[n, :] = -0.5 * kappa * w_given

    K_gains = solve_lqr(T, A_aug, B_aug, Q_aug, R_mat, M_aug)

    s0 = np.append(w0, 1.0)
    s_path, u_path = execute_lqr(T, A_aug, B_aug, K_gains, s0)

    # Drop the augmented constant state.
    w_path = s_path[:, :n]

    return w_path


# ─────────────────────────────────────────────────────────
#  CVXPY Solver
# ─────────────────────────────────────────────────────────

def solve_cvxpy(T, w_given, kappa, gamma_tc, w0):
    """
    Solve the same tracking problem via CVXPY.
    """
    w_given = np.asarray(w_given, dtype=float)
    w0 = np.asarray(w0, dtype=float)

    n = len(w_given)
    W = cp.Variable((T + 1, n))
    U = cp.Variable((T, n))

    cost = 0
    constraints = [W[0] == w0]

    for t in range(T):
        constraints.append(W[t + 1] == W[t] + U[t])

        cost += (
            0.5 * kappa * cp.sum_squares(W[t + 1] - w_given)
            + 0.5 * gamma_tc * cp.sum_squares(U[t])
        )

    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.OSQP)

    if W.value is None:
        raise RuntimeError(f"CVXPY failed. Status: {prob.status}")

    return W.value


# ─────────────────────────────────────────────────────────
#  Utilities & Experiments
# ─────────────────────────────────────────────────────────

def compute_tracking_cost(w_path, w_given, kappa, gamma_tc):
    """
    Compute total tracking cost plus transaction cost.
    """
    w_path = np.asarray(w_path, dtype=float)
    w_given = np.asarray(w_given, dtype=float)

    total = 0.0

    for t in range(1, len(w_path)):
        dw = w_path[t] - w_path[t - 1]
        err = w_path[t] - w_given

        total += (
            0.5 * kappa * np.dot(err, err)
            + 0.5 * gamma_tc * np.dot(dw, dw)
        )

    return total


def make_example_weights(n, seed=42):
    """
    Generate random initial and target weights.
    """
    rng = np.random.default_rng(seed)

    w0 = rng.uniform(0.0, 0.4, n)
    w0 /= np.sum(w0)

    w_given = rng.uniform(0.0, 0.4, n)
    w_given /= np.sum(w_given)

    return w0, w_given


def run_single_comparison(T=30, n=5, kappa=2.0, gamma_tc=0.5, seed=42):
    w0, w_given = make_example_weights(n, seed)

    # DP / LQR
    t0 = time.perf_counter()
    w_dp = solve_tracking_dp(T, w_given, kappa, gamma_tc, w0)
    time_dp = (time.perf_counter() - t0) * 1000

    # CVXPY
    t0 = time.perf_counter()
    w_cvx = solve_cvxpy(T, w_given, kappa, gamma_tc, w0)
    time_cvx = (time.perf_counter() - t0) * 1000

    cost_dp = compute_tracking_cost(w_dp, w_given, kappa, gamma_tc)
    cost_cvx = compute_tracking_cost(w_cvx, w_given, kappa, gamma_tc)

    return {
        "T": T,
        "n": n,
        "w0": w0,
        "w_given": w_given,
        "kappa": kappa,
        "gamma_tc": gamma_tc,
        "w_dp": w_dp,
        "w_cvx": w_cvx,
        "time_dp": time_dp,
        "time_cvx": time_cvx,
        "cost_dp": cost_dp,
        "cost_cvx": cost_cvx,
        "speedup": time_cvx / time_dp,
    }


def run_scaling_experiment(n_list=None, T=30, n_trials=5, kappa=2.0, gamma_tc=0.5):
    if n_list is None:
        n_list = [2, 3, 5, 7, 10, 15, 20]

    rows = []

    for n in n_list:
        dp_times = []
        cvx_times = []

        for seed in range(n_trials):
            w0, w_given = make_example_weights(n, seed)

            t0 = time.perf_counter()
            solve_tracking_dp(T, w_given, kappa, gamma_tc, w0)
            dp_times.append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            solve_cvxpy(T, w_given, kappa, gamma_tc, w0)
            cvx_times.append((time.perf_counter() - t0) * 1000)

        rows.append({
            "n": n,
            "dp_mean": np.mean(dp_times),
            "dp_std": np.std(dp_times),
            "cvx_mean": np.mean(cvx_times),
            "cvx_std": np.std(cvx_times),
            "speedup": np.mean(cvx_times) / np.mean(dp_times),
        })

    return rows


# ─────────────────────────────────────────────────────────
#  Visualisation
# ─────────────────────────────────────────────────────────

def plot_all(r, scale_rows):
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor("#111111")

    steps = np.arange(r["T"] + 1)

    err_dp = np.linalg.norm(r["w_dp"] - r["w_given"], axis=1)
    err_cvx = np.linalg.norm(r["w_cvx"] - r["w_given"], axis=1)

    # Tracking error plot
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_facecolor("#1a1a1a")

    ax1.plot(steps, err_dp, color=COLORS["dp"], lw=2.5, label="DP/LQR")
    ax1.plot(steps, err_cvx, color=COLORS["cvx"], lw=2, ls="--", label="CVXPY")

    ax1.set_title("Tracking Error ‖w - w_target‖₂ over time", color="white", fontsize=11)
    ax1.set_xlabel("Time step", color="white")
    ax1.set_ylabel("Tracking Error", color="white")
    ax1.tick_params(colors="white")
    ax1.legend(frameon=False, fontsize=9)
    ax1.grid(color="gray", ls=":", alpha=0.3)

    # Asset weight paths
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_facecolor("#1a1a1a")

    for i in range(r["n"]):
        c = ASSET_PALETTE[i % len(ASSET_PALETTE)]
        ax2.plot(steps, r["w_dp"][:, i], color=c, lw=1.8, label=f"Asset {i + 1}")
        ax2.axhline(r["w_given"][i], color=c, ls="--", lw=1, alpha=0.7)

    ax2.set_title("Asset weight paths (DP) vs targets (dashed)", color="white", fontsize=11)
    ax2.set_xlabel("Time step", color="white")
    ax2.set_ylabel("Weight", color="white")
    ax2.tick_params(colors="white")
    ax2.legend(frameon=False, fontsize=8, ncol=2)
    ax2.grid(color="gray", ls=":", alpha=0.3)
    ax2.axhline(0, color="white", lw=0.5, alpha=0.4)

    # Scaling plot
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_facecolor("#1a1a1a")

    ns = [row["n"] for row in scale_rows]
    dp_m = [row["dp_mean"] for row in scale_rows]
    dp_s = [row["dp_std"] for row in scale_rows]
    cvx_m = [row["cvx_mean"] for row in scale_rows]
    cvx_s = [row["cvx_std"] for row in scale_rows]

    ax3.plot(ns, dp_m, color=COLORS["dp"], lw=2.5, marker="o", ms=5, label="DP/LQR")
    ax3.plot(ns, cvx_m, color=COLORS["cvx"], lw=2, marker="s", ms=5, ls="--", label="CVXPY")

    ax3.fill_between(
        ns,
        np.array(dp_m) - dp_s,
        np.array(dp_m) + dp_s,
        color=COLORS["dp"],
        alpha=0.15,
    )
    ax3.fill_between(
        ns,
        np.array(cvx_m) - cvx_s,
        np.array(cvx_m) + cvx_s,
        color=COLORS["cvx"],
        alpha=0.15,
    )

    ax3.set_title("Solve time vs number of assets", color="white", fontsize=11)
    ax3.set_xlabel("Number of assets n", color="white")
    ax3.set_ylabel("Time (ms)", color="white")
    ax3.tick_params(colors="white")
    ax3.legend(frameon=False, fontsize=9)
    ax3.grid(color="gray", ls=":", alpha=0.3)

    # Runtime bar plot
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_facecolor("#1a1a1a")

    methods = ["DP/LQR", "CVXPY"]
    times = [r["time_dp"], r["time_cvx"]]

    bars = ax4.bar(
        methods,
        times,
        color=[COLORS["dp"], COLORS["cvx"]],
        width=0.4,
        alpha=0.85,
    )

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
        f'Execution time | speedup: {r["speedup"]:.1f}×\n'
        f'Cost (lower=better) — DP: {r["cost_dp"]:.5f}  CVXPY: {r["cost_cvx"]:.5f}',
        color="white",
        fontsize=10,
    )
    ax4.set_ylabel("Time (ms)", color="white")
    ax4.tick_params(colors="white")
    ax4.grid(axis="y", color="gray", ls=":", alpha=0.3)

    fig.suptitle(
        f'DP/LQR vs CVXPY — Given Weight Tracking '
        f'(T={r["T"]}, n={r["n"]}, κ={r["kappa"]}, γ_tc={r["gamma_tc"]})',
        color="white",
        fontsize=13,
        y=1.01,
    )

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Given Weight Extension: DP/LQR vs CVXPY")
    print("=" * 60)

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

    print("\n[2] Scaling experiment  (n ∈ {2,3,5,7,10,15,20}, T=30)...")
    scale_rows = run_scaling_experiment(
        n_list=[2, 3, 5, 7, 10, 15, 20],
        T=30,
        n_trials=5,
    )

    print("\n  n  |  DP (ms)  |  CVXPY (ms)  |  speedup")
    print("  ---+-----------+--------------+---------")

    for row in scale_rows:
        print(
            f"  {row['n']:2d} |  {row['dp_mean']:6.2f}    |"
            f"  {row['cvx_mean']:9.2f}   |  {row['speedup']:.1f}×"
        )

    print("\n[3] Generating plots...")
    fig = plot_all(r, scale_rows)

    plt.savefig(
        "given_weight_dp_vs_cvxpy.png",
        dpi=150,
        bbox_inches="tight",
        facecolor="#111111",
    )

    print("  Saved: given_weight_dp_vs_cvxpy.png")
    plt.show()


if __name__ == "__main__":
    main()
