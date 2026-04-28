"""Plotting helpers for the demo notebook.

Two top-level entry points cover every figure in ``demo.ipynb``:

* :func:`plot_single_asset_results` -- single-asset comparison
  (Part A optimal execution + Part C single-asset given-weight tracking).
* :func:`plot_multi_asset_results`  -- multi-asset comparison
  (Part B Markowitz + Part C multi-asset given-weight tracking).

Each function accepts precomputed LQR / CVXPY trajectories plus a few
optional inputs (``alphas`` / ``w_given`` / ``mu`` / ``scaling`` / ...) and
adapts its layout accordingly, so callers only need to forward the data
they actually have.
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


def _timing_panel(ax, time_lqr, time_cvx, speedup, obj_lqr=None, obj_cvx=None):
    """Render the standard 'LQR vs CVXPY' timing-bar panel onto ``ax``."""
    bars = ax.bar(
        ['LQR', 'CVXPY'], [time_lqr, time_cvx],
        color=[COLORS['dp'], COLORS['cvx']], width=0.4, alpha=0.85,
    )
    for bar, t in zip(bars, [time_lqr, time_cvx]):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
            f'{t:.2f} ms', ha='center', va='bottom',
            color='white', fontsize=12, fontweight='bold',
        )
    title = f'Solve time  |  speedup: {speedup:.1f}x'
    if obj_lqr is not None and obj_cvx is not None:
        title += f'\nObjective -- LQR: {obj_lqr:.5f}   CVXPY: {obj_cvx:.5f}'
        ax.set_title(title, color='white', fontsize=10)
    else:
        ax.set_title(title, color='white', fontsize=11)
    ax.set_ylabel('Time (ms)', color='white')


def _scaling_panel(ax, scaling):
    """Render the scaling timing panel (LQR vs CVXPY) onto ``ax``."""
    if scaling is not None:
        n_list  = scaling['n_list']
        lqr_med = np.asarray(scaling['lqr_med'])
        lqr_iqr = np.asarray(scaling['lqr_iqr'])
        cvx_med = np.asarray(scaling['cvx_med'])
        cvx_iqr = np.asarray(scaling['cvx_iqr'])
        ax.plot(n_list, lqr_med, color=COLORS['dp'],  lw=2.5, marker='o', ms=5, label='LQR')
        ax.plot(n_list, cvx_med, color=COLORS['cvx'], lw=2,   marker='s', ms=5, ls='--', label='CVXPY')
        ax.fill_between(n_list, lqr_med - lqr_iqr, lqr_med + lqr_iqr, color=COLORS['dp'],  alpha=0.15)
        ax.fill_between(n_list, cvx_med - cvx_iqr, cvx_med + cvx_iqr, color=COLORS['cvx'], alpha=0.15)
        ax.set_xticks(n_list)
        ax.set_xticklabels([str(n) for n in n_list])
        ax.legend(frameon=False, fontsize=9)
    ax.set_title(
        'Solve time vs number of assets\n(median ± IQR/2, 10 trials)',
        color='white', fontsize=11,
    )
    ax.set_xlabel('Number of assets n', color='white')
    ax.set_ylabel('Time (ms)', color='white')


def plot_single_asset_results(
    *,
    w_lqr, w_cvx,
    u_lqr, u_cvx,
    time_lqr, time_cvx, speedup,
    alphas=None,
    w_given=None,
    diff=None,
    obj_lqr=None, obj_cvx=None,
    suptitle=None,
    save_path=None,
):
    """Single-asset LQR vs CVXPY comparison.

    Two layouts are picked automatically:

    * **Execution mode** -- when ``alphas`` is provided. 6-panel 2x3 grid
      (position, trade bars, alpha+pos, cumulative PnL, residual, timing).
      Used for Part A.
    * **Tracking mode** -- otherwise. 4-panel 2x2 grid (tracking error or
      residual, weight path, trade lines, timing). Used for Part C single.

    Optional inputs:

    * ``w_given`` -- adds a target reference line on the weight panel and
      (in tracking mode) makes the first panel a tracking-error plot.
    * ``diff``    -- residual; auto-computed as ``w_lqr - w_cvx`` if missing.
    * ``obj_lqr`` / ``obj_cvx`` -- appended to the timing-panel title.
    * ``suptitle`` -- figure-level title.
    """
    w_lqr = np.asarray(w_lqr).ravel()
    w_cvx = np.asarray(w_cvx).ravel()
    u_lqr = np.asarray(u_lqr).ravel()
    u_cvx = np.asarray(u_cvx).ravel()
    steps_u = np.arange(len(u_lqr))
    steps_w = np.arange(len(w_lqr))
    is_execution = alphas is not None

    if is_execution:
        if diff is None:
            diff = w_lqr - w_cvx
        alphas = np.asarray(alphas).ravel()

        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        fig.patch.set_facecolor('#111111')
        axes = axes.flatten()
        for ax in axes:
            style_ax(ax)

        # 1 - Position path
        axes[0].plot(steps_w, w_lqr, color=COLORS['dp'],  lw=2.5, marker='o', ms=3, label='LQR')
        axes[0].plot(steps_w, w_cvx, color=COLORS['cvx'], lw=2,   marker='s', ms=3, ls='--', label='CVXPY verify')
        axes[0].axhline(0, color='white', lw=0.5, alpha=0.4)
        if w_given is not None:
            axes[0].axhline(w_given, color=COLORS['target'], lw=1.5, ls=':', label=r'$w_{given}$')
        axes[0].set_title('Optimal position path  w(t)', color='white', fontsize=11)
        axes[0].set_xlabel('Time step', color='white')
        axes[0].set_ylabel('w', color='white')
        axes[0].legend(frameon=False, fontsize=9)

        # 2 - Trade schedule (bars)
        axes[1].bar(steps_u - 0.2, u_lqr, width=0.35, color=COLORS['dp'],  alpha=0.85, label='LQR')
        axes[1].bar(steps_u + 0.2, u_cvx, width=0.35, color=COLORS['cvx'], alpha=0.85, label='CVXPY verify')
        axes[1].axhline(0, color='white', lw=0.5, alpha=0.4)
        axes[1].set_title('Trade schedule  u(t)', color='white', fontsize=11)
        axes[1].set_xlabel('Time step', color='white')
        axes[1].set_ylabel('Trade size', color='white')
        axes[1].legend(frameon=False, fontsize=9)

        # 3 - Alpha vs position
        ax3r = axes[2].twinx()
        axes[2].plot(steps_u, alphas, color=COLORS['gold'], lw=2, ls=':', label='Alpha')
        ax3r.plot(steps_w, w_lqr, color=COLORS['dp'],  lw=2.5, label='w (LQR)')
        ax3r.plot(steps_w, w_cvx, color=COLORS['cvx'], lw=1.5, ls='--', label='w (CVXPY)')
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
        axes[3].plot(steps_u, cum_lqr, color=COLORS['dp'],  lw=2.5, label='LQR')
        axes[3].plot(steps_u, cum_cvx, color=COLORS['cvx'], lw=2, ls='--', label='CVXPY verify')
        axes[3].fill_between(steps_u, cum_lqr, alpha=0.12, color=COLORS['dp'])
        axes[3].fill_between(steps_u, cum_cvx, alpha=0.10, color=COLORS['cvx'])
        axes[3].set_title('Cumulative PnL  (alpha * u)', color='white', fontsize=11)
        axes[3].set_xlabel('Time step', color='white')
        axes[3].set_ylabel('Cum PnL', color='white')
        axes[3].legend(frameon=False, fontsize=9)

        # 5 - Residual
        axes[4].bar(steps_w, diff, color=COLORS['gold'], alpha=0.85)
        axes[4].axhline(0, color='white', lw=0.7)
        axes[4].set_title(
            f'w difference  LQR - CVXPY   max|D|={np.abs(diff).max():.2e}',
            color='white', fontsize=10,
        )
        axes[4].set_xlabel('Time step', color='white')
        axes[4].set_ylabel('Dw', color='white')

        # 6 - Timing
        _timing_panel(axes[5], time_lqr, time_cvx, speedup, obj_lqr, obj_cvx)
    else:
        fig = plt.figure(figsize=(16, 12))
        fig.patch.set_facecolor('#111111')
        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 3)
        ax4 = fig.add_subplot(2, 2, 4)
        for ax in (ax1, ax2, ax3, ax4):
            style_ax(ax)

        # 1 - Tracking error (preferred) or residual
        if w_given is not None:
            ax1.plot(steps_w, np.abs(w_lqr - w_given), color=COLORS['dp'],  lw=2.5, label='LQR')
            ax1.plot(steps_w, np.abs(w_cvx - w_given), color=COLORS['cvx'], lw=2, ls='--', label='CVXPY verify')
            ax1.set_title(r'Tracking error  $|w_t - w_{given}|$  over time',
                          color='white', fontsize=11)
            ax1.set_ylabel('Abs error', color='white')
            ax1.legend(frameon=False, fontsize=9)
        else:
            d = diff if diff is not None else (w_lqr - w_cvx)
            ax1.bar(steps_w, d, color=COLORS['gold'], alpha=0.85)
            ax1.axhline(0, color='white', lw=0.7)
            ax1.set_title(
                f'w difference  LQR - CVXPY   max|D|={np.abs(d).max():.2e}',
                color='white', fontsize=10,
            )
            ax1.set_ylabel('Dw', color='white')
        ax1.set_xlabel('Time step', color='white')

        # 2 - Weight path (with optional target line)
        ax2.plot(steps_w, w_lqr, color=COLORS['dp'],  lw=2.5, label='LQR path')
        ax2.plot(steps_w, w_cvx, color=COLORS['cvx'], lw=2, ls='--', label='CVXPY verify')
        if w_given is not None:
            ax2.axhline(w_given, color=COLORS['target'], lw=1.5, ls=':', label=r'$w_{given}$')
        ax2.set_title('Single-asset weight path', color='white', fontsize=11)
        ax2.set_xlabel('Time step', color='white')
        ax2.set_ylabel('Weight', color='white')
        ax2.legend(frameon=False, fontsize=9)

        # 3 - Trade path (line)
        ax3.plot(steps_u, u_lqr, color=COLORS['dp'],  lw=2.5, label='LQR $u_t$')
        ax3.plot(steps_u, u_cvx, color=COLORS['cvx'], lw=2, ls='--', label='CVXPY $u_t$')
        ax3.axhline(0, color='white', lw=0.5, alpha=0.4)
        ax3.set_title('Trade path  $u_t$', color='white', fontsize=11)
        ax3.set_xlabel('Time step', color='white')
        ax3.set_ylabel('Trade', color='white')
        ax3.legend(frameon=False, fontsize=9)

        # 4 - Timing
        _timing_panel(ax4, time_lqr, time_cvx, speedup, obj_lqr, obj_cvx)

    if suptitle:
        fig.suptitle(suptitle, color='white', fontsize=12, y=1.01)
    plt.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_multi_asset_results(
    *,
    W_lqr, W_cvx,
    time_lqr, time_cvx, speedup,
    obj_lqr=None, obj_cvx=None,
    mu=None,
    w_given=None,
    scaling=None,
    suptitle=None,
    save_path=None,
):
    """Multi-asset LQR vs CVXPY 4-panel comparison.

    Always renders: portfolio norm, per-asset weight paths, scaling timing
    (if ``scaling`` provided), and a timing/objective bar.

    Optional inputs:

    * ``mu``      -- annotates each asset path with its mean return
      (Part B Markowitz).
    * ``w_given`` -- draws per-asset target reference lines and annotates
      the labels with the target weight (Part C multi-asset).
    * ``scaling`` -- dict ``{'n_list', 'lqr_med', 'lqr_iqr', 'cvx_med',
      'cvx_iqr'}`` for the bottom-left scaling panel.
    """
    W_lqr = np.asarray(W_lqr)
    W_cvx = np.asarray(W_cvx)
    n_assets = W_lqr.shape[1]
    steps = np.arange(W_lqr.shape[0])
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

    # 2 - Per-asset weight paths
    ax2 = fig.add_subplot(2, 2, 2)
    style_ax(ax2)
    for i in range(n_assets):
        c = ASSET_PALETTE[i % len(ASSET_PALETTE)]
        if mu is not None:
            label = f'Asset {i+1}  (mu={mu[i]*100:.2f}%)'
        elif w_given is not None:
            label = f'Asset {i+1}  (target={w_given[i]:.3f})'
        else:
            label = f'Asset {i+1}'
        ax2.plot(steps, W_lqr[:, i], color=c, lw=1.8, label=label)
        if w_given is not None:
            ax2.axhline(w_given[i], color=c, ls='--', lw=1, alpha=0.7)
    ax2.axhline(0, color='white', lw=0.5, alpha=0.4)
    ax2.set_title('Asset weight paths  (LQR)', color='white', fontsize=11)
    ax2.set_xlabel('Time step', color='white')
    ax2.set_ylabel('Weight', color='white')
    ax2.legend(frameon=False, fontsize=8, ncol=2)

    # 3 - Scaling
    ax3 = fig.add_subplot(2, 2, 3)
    style_ax(ax3)
    _scaling_panel(ax3, scaling)

    # 4 - Timing
    ax4 = fig.add_subplot(2, 2, 4)
    style_ax(ax4)
    _timing_panel(ax4, time_lqr, time_cvx, speedup, obj_lqr, obj_cvx)

    if suptitle:
        fig.suptitle(suptitle, color='white', fontsize=13, y=1.01)
    plt.tight_layout()
    _maybe_save(fig, save_path)
    return fig
