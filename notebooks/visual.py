"""Plotting helpers for the demo notebook.

All plots use a consistent dark-theme palette and follow the layout used in
``demo.ipynb``. Each ``plot_*`` function takes precomputed trajectories and
diagnostics, builds a figure, optionally saves it, and returns the figure
object so the caller can ``plt.show()`` it (or further customize).
"""

import os

import numpy as np
import matplotlib.pyplot as plt


COLORS = {
    'dp':     '#00BFFF',
    'cvx':    '#FF6347',
    'target': '#FFD700',
    'gold':   '#FFD700',
}

ASSET_PALETTE = [
    '#00BFFF', '#FF6347', '#FFD700', '#DA70D6', '#7FFF00',
    '#FF8C00', '#20B2AA', '#FF1493', '#ADFF2F', '#BA55D3',
]


def setup_dark_style():
    """Activate the matplotlib dark style used throughout the notebook."""
    plt.style.use('dark_background')


def style_ax(ax):
    """Apply the project's dark-theme styling to a single axes."""
    ax.set_facecolor('#1a1a1a')
    ax.tick_params(colors='white')
    ax.grid(color='gray', ls=':', alpha=0.3)
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')


def _maybe_save(fig, save_path):
    if save_path is None:
        return
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#111111')
    print(f'Saved: {save_path}')


def plot_execution_results(
    *,
    T, w_lqr, w_cvx, u_lqr, u_cvx, alphas, diff,
    time_lqr, time_cvx, speedup,
    gamma, sigma_sq, eta, rho, beta,
    save_path='result/execution_results.png',
):
    """Part A - 6-panel optimal-execution figure (positions, trades, alpha,
    cumulative PnL, residual, timing)."""
    steps = np.arange(T)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.patch.set_facecolor('#111111')
    axes = axes.flatten()
    for ax in axes:
        style_ax(ax)

    # 1 - Position paths
    axes[0].plot(steps, w_lqr, color=COLORS['dp'],  lw=2.5, marker='o', ms=3, label='LQR')
    axes[0].plot(steps, w_cvx, color=COLORS['cvx'], lw=2,   marker='s', ms=3, ls='--', label='CVXPY verify')
    axes[0].axhline(0, color='white', lw=0.5, alpha=0.4)
    axes[0].set_title('Optimal position path  w(t)', color='white', fontsize=11)
    axes[0].set_xlabel('Time step', color='white')
    axes[0].set_ylabel('w', color='white')
    axes[0].legend(frameon=False, fontsize=9)

    # 2 - Trade schedule
    axes[1].bar(steps - 0.2, u_lqr, width=0.35, color=COLORS['dp'],  alpha=0.85, label='LQR')
    axes[1].bar(steps + 0.2, u_cvx, width=0.35, color=COLORS['cvx'], alpha=0.85, label='CVXPY verify')
    axes[1].axhline(0, color='white', lw=0.5, alpha=0.4)
    axes[1].set_title('Trade schedule  u(t)', color='white', fontsize=11)
    axes[1].set_xlabel('Time step', color='white')
    axes[1].set_ylabel('Trade size', color='white')
    axes[1].legend(frameon=False, fontsize=9)

    # 3 - Alpha vs position
    ax3r = axes[2].twinx()
    axes[2].plot(steps, alphas, color=COLORS['gold'], lw=2, ls=':', label='Alpha')
    ax3r.plot(steps, w_lqr, color=COLORS['dp'],  lw=2.5, label='w (LQR)')
    ax3r.plot(steps, w_cvx, color=COLORS['cvx'], lw=1.5, ls='--', label='w (CVXPY)')
    axes[2].set_title('Alpha signal vs position', color='white', fontsize=11)
    axes[2].set_xlabel('Time step', color='white')
    axes[2].set_ylabel('Alpha', color=COLORS['gold'])
    axes[2].tick_params(axis='y', colors=COLORS['gold'])
    ax3r.set_ylabel('w', color='white')
    ax3r.tick_params(colors='white')
    l1, b1 = axes[2].get_legend_handles_labels()
    l2, b2 = ax3r.get_legend_handles_labels()
    axes[2].legend(l1 + l2, b1 + b2, frameon=False, fontsize=8)

    # 4 - Cumulative PnL
    cum_lqr = np.cumsum(alphas * u_lqr)
    cum_cvx = np.cumsum(alphas * u_cvx)
    axes[3].plot(steps, cum_lqr, color=COLORS['dp'],  lw=2.5, label='LQR')
    axes[3].plot(steps, cum_cvx, color=COLORS['cvx'], lw=2, ls='--', label='CVXPY verify')
    axes[3].fill_between(steps, cum_lqr, alpha=0.12, color=COLORS['dp'])
    axes[3].fill_between(steps, cum_cvx, alpha=0.10, color=COLORS['cvx'])
    axes[3].set_title('Cumulative PnL  (alpha * u)', color='white', fontsize=11)
    axes[3].set_xlabel('Time step', color='white')
    axes[3].set_ylabel('Cum PnL', color='white')
    axes[3].legend(frameon=False, fontsize=9)

    # 5 - Difference
    axes[4].bar(steps, diff, color=COLORS['gold'], alpha=0.85)
    axes[4].axhline(0, color='white', lw=0.7)
    axes[4].set_title(
        f'w difference  LQR - CVXPY   max|D|={np.abs(diff).max():.2e}',
        color='white', fontsize=10,
    )
    axes[4].set_xlabel('Time step', color='white')
    axes[4].set_ylabel('Dw', color='white')

    # 6 - Timing bar
    bars = axes[5].bar(
        ['LQR', 'CVXPY'], [time_lqr, time_cvx],
        color=[COLORS['dp'], COLORS['cvx']], width=0.4, alpha=0.85,
    )
    for bar, t in zip(bars, [time_lqr, time_cvx]):
        axes[5].text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
            f'{t:.2f} ms', ha='center', va='bottom',
            color='white', fontsize=12, fontweight='bold',
        )
    axes[5].set_title(f'Solve time  |  speedup: {speedup:.1f}x', color='white', fontsize=11)
    axes[5].set_ylabel('Time (ms)', color='white')

    fig.suptitle(
        f'Part A -- Optimal Execution  '
        f'(T={T}, gamma={gamma}, sigma_sq={sigma_sq}, eta={eta}, rho={rho}, beta={beta})',
        color='white', fontsize=12, y=1.01,
    )
    plt.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_markowitz_results(
    *,
    T, n_assets, w_lqr, w_cvx, mu,
    time_lqr, time_cvx, speedup, obj_lqr, obj_cvx,
    lam, gamma_tc,
    scaling=None,
    save_path='result/markowitz_results.png',
):
    """Part B - 4-panel Markowitz figure.

    ``scaling`` (optional) is a dict with keys
    ``{'n_list', 'lqr_med', 'lqr_iqr', 'cvx_med', 'cvx_iqr'}`` used for the
    bottom-left scaling panel; if omitted, that panel is left blank.
    """
    steps = np.arange(T + 1)
    norm_lqr = np.linalg.norm(w_lqr, axis=1)
    norm_cvx = np.linalg.norm(w_cvx, axis=1)

    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor('#111111')

    # 1 - Portfolio norm
    ax1 = fig.add_subplot(2, 2, 1)
    style_ax(ax1)
    ax1.plot(steps, norm_lqr, color=COLORS['dp'],  lw=2.5, label='LQR')
    ax1.plot(steps, norm_cvx, color=COLORS['cvx'], lw=2, ls='--', label='CVXPY verify')
    ax1.set_title('Portfolio norm  ||w||  over time', color='white', fontsize=11)
    ax1.set_xlabel('Time step', color='white')
    ax1.set_ylabel('||w||', color='white')
    ax1.legend(frameon=False, fontsize=9)

    # 2 - Asset weight paths (LQR)
    ax2 = fig.add_subplot(2, 2, 2)
    style_ax(ax2)
    for i in range(n_assets):
        ax2.plot(
            steps, w_lqr[:, i],
            color=ASSET_PALETTE[i % len(ASSET_PALETTE)],
            lw=1.8, label=f'Asset {i+1}  (mu={mu[i]*100:.2f}%)',
        )
    ax2.axhline(0, color='white', lw=0.5, alpha=0.4)
    ax2.set_title('Asset weight paths  (LQR)', color='white', fontsize=11)
    ax2.set_xlabel('Time step', color='white')
    ax2.set_ylabel('Weight', color='white')
    ax2.legend(frameon=False, fontsize=8, ncol=2)

    # 3 - Scaling
    ax3 = fig.add_subplot(2, 2, 3)
    style_ax(ax3)
    if scaling is not None:
        n_list  = scaling['n_list']
        lqr_med = np.asarray(scaling['lqr_med'])
        lqr_iqr = np.asarray(scaling['lqr_iqr'])
        cvx_med = np.asarray(scaling['cvx_med'])
        cvx_iqr = np.asarray(scaling['cvx_iqr'])
        ax3.plot(n_list, lqr_med, color=COLORS['dp'],  lw=2.5, marker='o', ms=5, label='LQR')
        ax3.plot(n_list, cvx_med, color=COLORS['cvx'], lw=2,   marker='s', ms=5, ls='--', label='CVXPY')
        ax3.fill_between(n_list, lqr_med - lqr_iqr, lqr_med + lqr_iqr, color=COLORS['dp'],  alpha=0.15)
        ax3.fill_between(n_list, cvx_med - cvx_iqr, cvx_med + cvx_iqr, color=COLORS['cvx'], alpha=0.15)
        ax3.set_xticks(n_list)
        ax3.set_xticklabels([str(n) for n in n_list])
        ax3.legend(frameon=False, fontsize=9)
    ax3.set_title(
        'Solve time vs number of assets\n(median ± IQR/2, 10 trials)',
        color='white', fontsize=11,
    )
    ax3.set_xlabel('Number of assets n', color='white')
    ax3.set_ylabel('Time (ms)', color='white')

    # 4 - Timing bar + objective
    ax4 = fig.add_subplot(2, 2, 4)
    style_ax(ax4)
    bars = ax4.bar(
        ['LQR', 'CVXPY'], [time_lqr, time_cvx],
        color=[COLORS['dp'], COLORS['cvx']], width=0.4, alpha=0.85,
    )
    for bar, t in zip(bars, [time_lqr, time_cvx]):
        ax4.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
            f'{t:.2f} ms', ha='center', va='bottom',
            color='white', fontsize=12, fontweight='bold',
        )
    ax4.set_title(
        f'Solve time  |  speedup: {speedup:.1f}x\n'
        f'Objective -- LQR: {obj_lqr:.5f}   CVXPY: {obj_cvx:.5f}',
        color='white', fontsize=10,
    )
    ax4.set_ylabel('Time (ms)', color='white')

    fig.suptitle(
        f'Part B -- Multi-Asset Markowitz  '
        f'(T={T}, n={n_assets}, lam={lam}, gamma_tc={gamma_tc})',
        color='white', fontsize=13, y=1.01,
    )
    plt.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_given_weight_single_results(
    *,
    T, w_lqr, w_cvx, u_lqr, u_cvx, w_given,
    time_lqr, time_cvx, speedup, obj_lqr, obj_cvx,
    w0, kappa, gamma_tc,
    save_path='result/givenweight_single_results.png',
):
    """Part C - 4-panel given-weight tracking figure (single asset)."""
    steps = np.arange(T + 1)

    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor('#111111')

    # 1 - Tracking error |w_t - w_given|
    ax1 = fig.add_subplot(2, 2, 1)
    style_ax(ax1)
    ax1.plot(steps, np.abs(w_lqr - w_given), color=COLORS['dp'],  lw=2.5, label='LQR')
    ax1.plot(steps, np.abs(w_cvx - w_given), color=COLORS['cvx'], lw=2, ls='--', label='CVXPY verify')
    ax1.set_title(r'Tracking error  $|w_t - w_{given}|$  over time',
                  color='white', fontsize=11)
    ax1.set_xlabel('Time step', color='white')
    ax1.set_ylabel('Abs error', color='white')
    ax1.legend(frameon=False, fontsize=9)

    # 2 - Weight path
    ax2 = fig.add_subplot(2, 2, 2)
    style_ax(ax2)
    ax2.plot(steps, w_lqr, color=COLORS['dp'],  lw=2.5, label='LQR path')
    ax2.plot(steps, w_cvx, color=COLORS['cvx'], lw=2, ls='--', label='CVXPY verify')
    ax2.axhline(w_given, color=COLORS['target'], lw=1.5, ls=':', label=r'$w_{given}$')
    ax2.set_title('Single-asset weight path', color='white', fontsize=11)
    ax2.set_xlabel('Time step', color='white')
    ax2.set_ylabel('Weight', color='white')
    ax2.legend(frameon=False, fontsize=9)

    # 3 - Trade path
    ax3 = fig.add_subplot(2, 2, 3)
    style_ax(ax3)
    ax3.plot(np.arange(T), u_lqr, color=COLORS['dp'],  lw=2.5, label='LQR $u_t$')
    ax3.plot(np.arange(T), u_cvx, color=COLORS['cvx'], lw=2, ls='--', label='CVXPY $u_t$')
    ax3.axhline(0, color='white', lw=0.5, alpha=0.4)
    ax3.set_title('Trade path  $u_t$', color='white', fontsize=11)
    ax3.set_xlabel('Time step', color='white')
    ax3.set_ylabel('Trade', color='white')
    ax3.legend(frameon=False, fontsize=9)

    # 4 - Timing + objective
    ax4 = fig.add_subplot(2, 2, 4)
    style_ax(ax4)
    bars = ax4.bar(
        ['LQR', 'CVXPY'], [time_lqr, time_cvx],
        color=[COLORS['dp'], COLORS['cvx']], width=0.4, alpha=0.85,
    )
    for bar, t in zip(bars, [time_lqr, time_cvx]):
        ax4.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
            f'{t:.2f} ms', ha='center', va='bottom',
            color='white', fontsize=12, fontweight='bold',
        )
    ax4.set_title(
        f'Solve time  |  speedup: {speedup:.1f}x\n'
        f'Objective -- LQR: {obj_lqr:.6f}   CVXPY: {obj_cvx:.6f}',
        color='white', fontsize=10,
    )
    ax4.set_ylabel('Time (ms)', color='white')

    fig.suptitle(
        f'Given Weight -- Single Asset  '
        f'(T={T}, w0={w0}, w_given={w_given}, kappa={kappa}, gamma_tc={gamma_tc})',
        color='white', fontsize=13, y=1.01,
    )
    plt.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_given_weight_multi_results(
    *,
    T, n_assets, W_lqr, W_cvx, w_given,
    time_lqr, time_cvx, speedup, obj_lqr, obj_cvx,
    kappa, gamma_tc,
    scaling=None,
    save_path='result/givenweight_results.png',
):
    """Part C - 4-panel given-weight tracking figure (multi-asset)."""
    steps = np.arange(T + 1)
    norm_lqr = np.linalg.norm(W_lqr, axis=1)
    norm_cvx = np.linalg.norm(W_cvx, axis=1)

    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor('#111111')

    # 1 - Portfolio norm
    ax1 = fig.add_subplot(2, 2, 1)
    style_ax(ax1)
    ax1.plot(steps, norm_lqr, color=COLORS['dp'],  lw=2.5, label='LQR')
    ax1.plot(steps, norm_cvx, color=COLORS['cvx'], lw=2, ls='--', label='CVXPY verify')
    ax1.set_title('Portfolio norm  ||w||  over time', color='white', fontsize=11)
    ax1.set_xlabel('Time step', color='white')
    ax1.set_ylabel('||w||', color='white')
    ax1.legend(frameon=False, fontsize=9)

    # 2 - Asset weight paths (LQR)
    ax2 = fig.add_subplot(2, 2, 2)
    style_ax(ax2)
    for i in range(n_assets):
        c = ASSET_PALETTE[i % len(ASSET_PALETTE)]
        ax2.plot(steps, W_lqr[:, i], color=c, lw=1.8,
                 label=f'Asset {i+1}  (target={w_given[i]:.3f})')
        ax2.axhline(w_given[i], color=c, ls='--', lw=1, alpha=0.7)
    ax2.axhline(0, color='white', lw=0.5, alpha=0.4)
    ax2.set_title('Asset weight paths  (LQR)', color='white', fontsize=11)
    ax2.set_xlabel('Time step', color='white')
    ax2.set_ylabel('Weight', color='white')
    ax2.legend(frameon=False, fontsize=8, ncol=2)

    # 3 - Scaling
    ax3 = fig.add_subplot(2, 2, 3)
    style_ax(ax3)
    if scaling is not None:
        n_list  = scaling['n_list']
        lqr_med = np.asarray(scaling['lqr_med'])
        lqr_iqr = np.asarray(scaling['lqr_iqr'])
        cvx_med = np.asarray(scaling['cvx_med'])
        cvx_iqr = np.asarray(scaling['cvx_iqr'])
        ax3.plot(n_list, lqr_med, color=COLORS['dp'],  lw=2.5, marker='o', ms=5, label='LQR')
        ax3.plot(n_list, cvx_med, color=COLORS['cvx'], lw=2,   marker='s', ms=5, ls='--', label='CVXPY')
        ax3.fill_between(n_list, lqr_med - lqr_iqr, lqr_med + lqr_iqr, color=COLORS['dp'],  alpha=0.15)
        ax3.fill_between(n_list, cvx_med - cvx_iqr, cvx_med + cvx_iqr, color=COLORS['cvx'], alpha=0.15)
        ax3.set_xticks(n_list)
        ax3.set_xticklabels([str(n) for n in n_list])
        ax3.legend(frameon=False, fontsize=9)
    ax3.set_title(
        'Solve time vs number of assets\n(median ± IQR/2, 10 trials)',
        color='white', fontsize=11,
    )
    ax3.set_xlabel('Number of assets n', color='white')
    ax3.set_ylabel('Time (ms)', color='white')

    # 4 - Timing + objective
    ax4 = fig.add_subplot(2, 2, 4)
    style_ax(ax4)
    bars = ax4.bar(
        ['LQR', 'CVXPY'], [time_lqr, time_cvx],
        color=[COLORS['dp'], COLORS['cvx']], width=0.4, alpha=0.85,
    )
    for bar, t in zip(bars, [time_lqr, time_cvx]):
        ax4.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
            f'{t:.2f} ms', ha='center', va='bottom',
            color='white', fontsize=12, fontweight='bold',
        )
    ax4.set_title(
        f'Solve time  |  speedup: {speedup:.1f}x\n'
        f'Objective -- LQR: {obj_lqr:.5f}   CVXPY: {obj_cvx:.5f}',
        color='white', fontsize=10,
    )
    ax4.set_ylabel('Time (ms)', color='white')

    fig.suptitle(
        f'Given Weight -- Multi-Asset  '
        f'(T={T}, n={n_assets}, kappa={kappa}, gamma_tc={gamma_tc})',
        color='white', fontsize=13, y=1.01,
    )
    plt.tight_layout()
    _maybe_save(fig, save_path)
    return fig
