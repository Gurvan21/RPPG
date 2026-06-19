"""
Même analyse que analyze_personal_videos.py, appliquée à un sujet UBFC
(utilise le cache results/chrom_adaptive_cache/<subject>.npz si disponible,
sinon extrait le RGB depuis Data/<subject>/vid.avi).

Compare CHROM (De Haan), CHROM adaptatif, POS et le signal BVP de référence
(ground_truth.txt) : signaux temporels, spectres, SNR et erreur HR vs GT.

Usage :
    python scripts/analyze_ubfc_video.py --subject subject1
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import periodogram

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, ROOT)

from models.chrom_adaptive import load_coefficients, DEHAAN_COEFFICIENTS, bandpass_numpy
from mp_rppg.methods import chrom, chrom_adaptive, pos
from mp_rppg.metrics import hr_from_fft, snr, _next_pow2

CACHE_DIR = os.path.join(ROOT, 'results', 'chrom_adaptive_cache')
DATA_DIR = os.path.join(ROOT, 'Data')
OUT_DIR = os.path.join(ROOT, 'results', 'ubfc_sample')
MODEL_PATH = os.path.join(ROOT, 'weights', 'chrom_adaptive_ubfc.pth')


def get_spectrum(bvp, fs):
    N = _next_pow2(len(bvp))
    f, pxx = periodogram(bvp, fs=fs, nfft=N, detrend=False)
    return f, pxx


def load_subject(name):
    cache_path = os.path.join(CACHE_DIR, f"{name}.npz")
    if os.path.exists(cache_path):
        npz = np.load(cache_path)
        return npz['rgb'], float(npz['fps']), npz['bvp']

    import mediapipe  # noqa: F401
    import imageio.v3 as iio
    from mp_rppg.pipeline import extract_rgb

    vid = os.path.join(DATA_DIR, name, 'vid.avi')
    gt = os.path.join(DATA_DIR, name, 'ground_truth.txt')
    frames = np.stack(list(iio.imiter(vid, plugin="pyav")))
    fps = float(iio.immeta(vid, plugin="pyav")['fps'])
    rgb = extract_rgb(frames, verbose=True)['front']
    with open(gt) as f:
        bvp_gt = np.asarray([float(x) for x in f.read().strip().split('\n')[0].split()])
    n = min(len(rgb), len(bvp_gt))
    return rgb[:n], fps, bvp_gt[:n]


def main():
    parser = argparse.ArgumentParser(description="Analyse rPPG sur un sujet UBFC")
    parser.add_argument('--subject', default='subject1')
    args = parser.parse_args()
    name = args.subject

    coeffs = load_coefficients(MODEL_PATH)
    print("Coefficients adaptatifs utilisés :", coeffs)
    print("Coefficients De Haan             :", DEHAAN_COEFFICIENTS)

    rgb, fps, bvp_gt = load_subject(name)
    print(f"\n{'='*70}\n  {name}\n{'='*70}")
    print(f"  {len(rgb)} frames @ {fps:.2f} fps ({len(rgb)/fps:.1f} s)")

    hr_gt = hr_from_fft(bandpass_numpy(bvp_gt, fps), fps)
    print(f"  HR (référence GT) = {hr_gt:.1f} bpm")

    methods = {
        'GT (ground truth)': bandpass_numpy(bvp_gt, fps),
        'CHROM (De Haan)': chrom(rgb, fps),
        'CHROM (adaptatif)': chrom_adaptive(rgb, fps, coeffs),
        'POS': pos(rgb, fps),
    }

    results = {}
    for label, bvp in methods.items():
        hr = hr_from_fft(bvp, fps)
        s = snr(bvp, hr_gt, fps)
        f, pxx = get_spectrum(bvp, fps)
        results[label] = {'bvp': bvp, 'hr': hr, 'snr': s, 'f': f, 'pxx': pxx}
        err = abs(hr - hr_gt)
        print(f"  {label:<20} HR = {hr:6.1f} bpm   err = {err:5.1f} bpm   SNR = {s:6.2f} dB")

    # ── Figure : signaux temporels + spectres ─────────────────────────────
    colors = {'GT (ground truth)': 'gray', 'CHROM (De Haan)': 'steelblue',
              'CHROM (adaptatif)': 'darkorange', 'POS': 'tomato'}
    fig, axes = plt.subplots(4, 2, figsize=(13, 13))

    for i, (label, r) in enumerate(results.items()):
        bvp = r['bvp']
        t = np.arange(len(bvp)) / fps
        ax = axes[i, 0]
        ax.plot(t, bvp, color=colors[label], lw=0.8)
        ax.set_title(f"{label} — signal (HR={r['hr']:.1f} bpm, SNR={r['snr']:.2f} dB)")
        ax.set_xlabel("Temps (s)")
        ax.set_ylabel("Amplitude")

        ax2 = axes[i, 1]
        f, pxx = r['f'], r['pxx']
        mask = (f >= 0.5) & (f <= 3.5)
        ax2.plot(f[mask] * 60, pxx[mask], color=colors[label], lw=1.2)
        ax2.axvline(r['hr'], color='black', lw=1.5, linestyle='--',
                     label=f"Pic: {r['hr']:.1f} bpm")
        ax2.axvline(hr_gt, color='green', lw=1.5, linestyle=':', label=f"GT: {hr_gt:.1f} bpm")
        ax2.set_title(f"{label} — Spectre de puissance")
        ax2.set_xlabel("Fréquence (bpm)")
        ax2.set_ylabel("Puissance")
        ax2.set_xlim(30, 210)
        ax2.legend(fontsize=8)

    fig.suptitle(f"Analyse rPPG — UBFC {name}\n"
                  f"({len(rgb)} frames @ {fps:.2f} fps, {len(rgb)/fps:.1f} s, HR GT={hr_gt:.1f} bpm)",
                  fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"analysis_{name}.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  [Graphique] {out_path}")

    # ── Figure : comparaison SNR + erreur HR ───────────────────────────────
    fig2, axes2 = plt.subplots(1, 2, figsize=(9, 4))
    labels = [l for l in results if l != 'GT (ground truth)']

    snrs = [results[l]['snr'] for l in labels]
    bars = axes2[0].bar(labels, snrs, color=[colors[l] for l in labels])
    for bar, s in zip(bars, snrs):
        axes2[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                       f"{s:.2f}", ha='center', va='bottom' if s >= 0 else 'top')
    axes2[0].set_ylabel("SNR (dB)")
    axes2[0].set_title(f"Comparaison SNR — {name}")
    axes2[0].tick_params(axis='x', labelrotation=15)

    errs = [abs(results[l]['hr'] - hr_gt) for l in labels]
    bars2 = axes2[1].bar(labels, errs, color=[colors[l] for l in labels])
    for bar, e in zip(bars2, errs):
        axes2[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                       f"{e:.2f}", ha='center', va='bottom')
    axes2[1].set_ylabel("Erreur HR vs GT (bpm)")
    axes2[1].set_title(f"Erreur HR — {name}")
    axes2[1].tick_params(axis='x', labelrotation=15)

    fig2.tight_layout()
    out_path2 = os.path.join(OUT_DIR, f"snr_err_{name}.png")
    fig2.savefig(out_path2, dpi=120)
    plt.close(fig2)
    print(f"  [Graphique] {out_path2}")


if __name__ == '__main__':
    main()
