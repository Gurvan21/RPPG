"""
Applique CHROM (De Haan), CHROM adaptatif et POS sur chaque ROI d'un fichier
CSV au format "rppg_rgb.csv" (colonnes <roi>_r, <roi>_g, <roi>_b par frame).

Usage :
    python scripts/analyze_csv_rois.py --csv results/personal_video/rppg_rgb.csv --fps 24.98
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import periodogram

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, ROOT)

from models.chrom_adaptive import load_coefficients, DEHAAN_COEFFICIENTS
from mp_rppg.methods import chrom, chrom_adaptive, pos
from mp_rppg.metrics import hr_from_fft, snr, _next_pow2

MODEL_PATH = os.path.join(ROOT, 'weights', 'chrom_adaptive_ubfc.pth')


def get_spectrum(bvp, fs):
    N = _next_pow2(len(bvp))
    f, pxx = periodogram(bvp, fs=fs, nfft=N, detrend=False)
    return f, pxx


def load_rois(csv_path):
    df = pd.read_csv(csv_path)
    rois = sorted({c.rsplit('_', 1)[0] for c in df.columns if c != 'frame'})
    result = {}
    for roi in rois:
        cols = [f"{roi}_r", f"{roi}_g", f"{roi}_b"]
        arr = df[cols].to_numpy(dtype=np.float64)
        # interpole les NaN (ex: première frame souvent vide)
        for c in range(3):
            col = arr[:, c]
            nans = np.isnan(col)
            if nans.all():
                col[:] = 0.0
            elif nans.any():
                idx = np.arange(len(col))
                col[nans] = np.interp(idx[nans], idx[~nans], col[~nans])
        result[roi] = arr
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default=os.path.join(ROOT, 'results', 'personal_video', 'rppg_rgb.csv'))
    parser.add_argument('--fps', type=float, default=24.98)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    out_dir = args.out or os.path.dirname(args.csv)
    coeffs = load_coefficients(MODEL_PATH)
    print("Coefficients adaptatifs :", coeffs)
    print("Coefficients De Haan    :", DEHAAN_COEFFICIENTS)

    rois = load_rois(args.csv)
    fps = args.fps
    print(f"\n{len(rois)} ROIs, {next(iter(rois.values())).shape[0]} frames @ {fps} fps")

    method_fns = {
        'CHROM (De Haan)': lambda rgb: chrom(rgb, fps),
        'CHROM (adaptatif)': lambda rgb: chrom_adaptive(rgb, fps, coeffs),
        'POS': lambda rgb: pos(rgb, fps),
    }
    colors = {'CHROM (De Haan)': 'steelblue', 'CHROM (adaptatif)': 'darkorange', 'POS': 'tomato'}

    # ── Calcul HR/SNR pour chaque ROI x méthode ────────────────────────────
    table = {}  # roi -> {method -> {'hr':, 'snr':, 'bvp':, 'f':, 'pxx':}}
    for roi, rgb in rois.items():
        table[roi] = {}
        print(f"\nROI: {roi}")
        for label, fn in method_fns.items():
            bvp = fn(rgb)
            hr = hr_from_fft(bvp, fps)
            s = snr(bvp, hr, fps)
            f, pxx = get_spectrum(bvp, fps)
            table[roi][label] = {'bvp': bvp, 'hr': hr, 'snr': s, 'f': f, 'pxx': pxx}
            print(f"  {label:<20} HR = {hr:6.1f} bpm   SNR = {s:6.2f} dB")

    # ── Figure 1 : grille HR par ROI x méthode ─────────────────────────────
    roi_names = list(rois.keys())
    method_names = list(method_fns.keys())

    fig, ax = plt.subplots(figsize=(max(8, len(roi_names) * 1.4), 4.5))
    x = np.arange(len(roi_names))
    width = 0.25
    for i, m in enumerate(method_names):
        hrs = [table[r][m]['hr'] for r in roi_names]
        ax.bar(x + (i - 1) * width, hrs, width, label=m, color=colors[m])
    ax.set_xticks(x)
    ax.set_xticklabels(roi_names, rotation=20, ha='right')
    ax.set_ylabel("HR (bpm)")
    ax.set_title("HR estimé par ROI et méthode")
    ax.legend()
    fig.tight_layout()
    out1 = os.path.join(out_dir, 'csv_rois_hr.png')
    fig.savefig(out1, dpi=120)
    plt.close(fig)
    print(f"\n[Graphique] {out1}")

    # ── Figure 2 : grille SNR par ROI x méthode ────────────────────────────
    fig, ax = plt.subplots(figsize=(max(8, len(roi_names) * 1.4), 4.5))
    for i, m in enumerate(method_names):
        snrs = [table[r][m]['snr'] for r in roi_names]
        ax.bar(x + (i - 1) * width, snrs, width, label=m, color=colors[m])
    ax.set_xticks(x)
    ax.set_xticklabels(roi_names, rotation=20, ha='right')
    ax.set_ylabel("SNR (dB)")
    ax.set_title("SNR par ROI et méthode")
    ax.legend()
    fig.tight_layout()
    out2 = os.path.join(out_dir, 'csv_rois_snr.png')
    fig.savefig(out2, dpi=120)
    plt.close(fig)
    print(f"[Graphique] {out2}")

    # ── Figure 3 : signaux + spectres pour la meilleure ROI (SNR max, CHROM De Haan) ──
    best_roi = max(roi_names, key=lambda r: table[r]['CHROM (De Haan)']['snr'])
    print(f"\nMeilleure ROI (SNR CHROM De Haan) : {best_roi}")

    fig, axes = plt.subplots(3, 2, figsize=(13, 10))
    for i, m in enumerate(method_names):
        r = table[best_roi][m]
        bvp = r['bvp']
        t = np.arange(len(bvp)) / fps
        ax = axes[i, 0]
        ax.plot(t, bvp, color=colors[m], lw=0.8)
        ax.set_title(f"{m} — {best_roi} (HR={r['hr']:.1f} bpm, SNR={r['snr']:.2f} dB)")
        ax.set_xlabel("Temps (s)")
        ax.set_ylabel("Amplitude")

        ax2 = axes[i, 1]
        f, pxx = r['f'], r['pxx']
        mask = (f >= 0.5) & (f <= 3.5)
        ax2.plot(f[mask] * 60, pxx[mask], color=colors[m], lw=1.2)
        ax2.axvline(r['hr'], color='black', lw=1.5, linestyle='--', label=f"Pic: {r['hr']:.1f} bpm")
        ax2.set_title(f"{m} — Spectre de puissance")
        ax2.set_xlabel("Fréquence (bpm)")
        ax2.set_ylabel("Puissance")
        ax2.set_xlim(30, 210)
        ax2.legend(fontsize=8)

    fig.suptitle(f"Analyse rPPG (CSV ROIs) — meilleure ROI : {best_roi}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out3 = os.path.join(out_dir, f'csv_rois_best_{best_roi}.png')
    fig.savefig(out3, dpi=120)
    plt.close(fig)
    print(f"[Graphique] {out3}")

    # ── Tableau récapitulatif ───────────────────────────────────────────────
    print(f"\n{'='*70}\n  Récapitulatif\n{'='*70}")
    print(f"  {'ROI':<20} {'Méthode':<20} {'HR (bpm)':>10} {'SNR (dB)':>10}")
    for roi in roi_names:
        for m in method_names:
            r = table[roi][m]
            print(f"  {roi:<20} {m:<20} {r['hr']:>10.1f} {r['snr']:>10.2f}")


if __name__ == '__main__':
    main()
