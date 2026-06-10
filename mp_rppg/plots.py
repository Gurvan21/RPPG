"""
Visualisations pour la comparaison des backends rPPG.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# Palette fixe par méthode pour cohérence entre figures
_COLORS = {
    'HC':    '#4C72B0',
    'Y5F':   '#DD8452',
    'MP':    '#55A868',
}
_METHOD_COLORS = {
    'CHROM': '#E64B35',
    'POS':   '#3182BD',
}


def _method_label(backend, region, method):
    if backend == 'MP':
        return f"MP-{region[:5]}/{method}"
    return f"{backend}/{method}"


# ── Figure 1 : barres MAE agrégées ───────────────────────────────────────────
def plot_mae_bars(summary, out_path):
    """
    summary : dict  {label -> {'MAE': float, 'RMSE': float, 'SNR_mean': float}}
    """
    labels = list(summary.keys())
    maes   = [summary[l]['MAE']  for l in labels]
    rmses  = [summary[l]['RMSE'] for l in labels]

    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.9), 5))
    bars_mae  = ax.bar(x - w/2, maes,  w, label='MAE',  color='#4C72B0', alpha=0.85)
    bars_rmse = ax.bar(x + w/2, rmses, w, label='RMSE', color='#DD8452', alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=8)
    ax.set_ylabel('Erreur HR (bpm)')
    ax.set_title('Comparaison MAE / RMSE — tous backends et régions')
    ax.legend()
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(axis='y', which='major', linestyle='--', alpha=0.4)

    for bar in bars_mae:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{bar.get_height():.1f}", ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Graphique MAE    → {out_path}")


# ── Figure 2 : scatter HR estimé vs HR GT ────────────────────────────────────
def plot_scatter(per_subject, out_path):
    """
    per_subject : liste de dict {label -> {'hr_pred': float, 'hr_gt': float}}
    """
    labels = list(per_subject[0].keys())
    n_cols = min(4, len(labels))
    n_rows = (len(labels) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.5 * n_cols, 4 * n_rows), squeeze=False)

    for idx, label in enumerate(labels):
        ax = axes[idx // n_cols][idx % n_cols]
        hr_gt   = [d[label]['hr_gt']   for d in per_subject if label in d]
        hr_pred = [d[label]['hr_pred'] for d in per_subject if label in d]

        ax.scatter(hr_gt, hr_pred, s=30, alpha=0.7, color='#4C72B0')
        lo = min(min(hr_gt), min(hr_pred)) - 5
        hi = max(max(hr_gt), max(hr_pred)) + 5
        ax.plot([lo, hi], [lo, hi], 'r--', lw=1, label='Idéal')
        mae = float(np.mean(np.abs(np.array(hr_pred) - np.array(hr_gt))))
        ax.set_title(f"{label}\nMAE={mae:.1f} bpm", fontsize=9)
        ax.set_xlabel('HR ground truth (bpm)')
        ax.set_ylabel('HR estimé (bpm)')
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    # Masquer les axes vides
    for idx in range(len(labels), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    plt.suptitle('HR estimé vs HR ground truth — par méthode', fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Scatter HR       → {out_path}")


# ── Figure 3 : SNR moyen par méthode ─────────────────────────────────────────
def plot_snr_bars(summary, out_path):
    labels = list(summary.keys())
    snrs   = [summary[l]['SNR_mean'] for l in labels]

    colors = ['#2ca02c' if s > 0 else '#d62728' for s in snrs]

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.9), 4))
    bars = ax.bar(labels, snrs, color=colors, alpha=0.85)
    ax.axhline(0, color='black', lw=0.8, linestyle='--')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=8)
    ax.set_ylabel('SNR moyen (dB)')
    ax.set_title('SNR moyen par méthode  (>0 dB = signal > bruit)')
    ax.grid(axis='y', which='major', linestyle='--', alpha=0.4)

    for bar, v in zip(bars, snrs):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + (0.1 if v >= 0 else -0.4),
                f"{v:.1f}", ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Graphique SNR    → {out_path}")


# ── Figure 4 : box plot erreurs absolues ─────────────────────────────────────
def plot_boxplot(errors_by_label, out_path):
    """
    errors_by_label : dict {label -> [abs_error, ...]}
    """
    labels = list(errors_by_label.keys())
    data   = [errors_by_label[l] for l in labels]

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.9), 5))
    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops=dict(color='black', lw=1.5))

    palette = plt.cm.tab10.colors
    for patch, color in zip(bp['boxes'], [palette[i % 10] for i in range(len(data))]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=8)
    ax.set_ylabel('Erreur absolue HR (bpm)')
    ax.set_title('Distribution des erreurs absolues par méthode')
    ax.grid(axis='y', which='major', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Box plot erreurs → {out_path}")


def save_all(summary, per_subject, errors_by_label, out_dir='result/mp_eval'):
    os.makedirs(out_dir, exist_ok=True)
    has_gt = any(
        not np.isnan(list(d.values())[0]['hr_gt'])
        for d in per_subject if d
    )
    plot_snr_bars(summary, os.path.join(out_dir, 'snr_bars.png'))
    if has_gt:
        plot_mae_bars(summary, os.path.join(out_dir, 'mae_bars.png'))
        plot_scatter(per_subject, os.path.join(out_dir, 'scatter_hr.png'))
        plot_boxplot(errors_by_label, os.path.join(out_dir, 'boxplot_errors.png'))
