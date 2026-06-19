"""
Analyse rPPG avec le backend YOLO5Face sur un sujet UBFC, avec comparaison
au signal de référence (ground_truth.txt) — pour valider/invalider le
backend Y5F par rapport à un HR connu.

Usage :
    python scripts/analyze_y5f_ubfc.py --subject subject1
"""

import argparse
import os
import sys

import imageio.v3 as iio
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
from mp_rppg.backends import extract_rgb_y5f

TOOLBOX_ROOT = "/home/kemnhou/Bureau/old/rPPG-Toolbox"
sys.path.append(TOOLBOX_ROOT)

DATA_DIR = os.path.join(ROOT, 'Data')
OUT_DIR = os.path.join(ROOT, 'results', 'ubfc_sample')
MODEL_PATH = os.path.join(ROOT, 'weights', 'chrom_adaptive_ubfc.pth')


def read_video(path):
    frames = np.stack(list(iio.imiter(path, plugin="pyav")))
    meta = iio.immeta(path, plugin="pyav")
    return frames, float(meta['fps'])


def get_spectrum(bvp, fs):
    N = _next_pow2(len(bvp))
    f, pxx = periodogram(bvp, fs=fs, nfft=N, detrend=False)
    return f, pxx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', default='subject1')
    args = parser.parse_args()
    name = args.subject

    coeffs = load_coefficients(MODEL_PATH)

    vid = os.path.join(DATA_DIR, name, 'vid.avi')
    gt_path = os.path.join(DATA_DIR, name, 'ground_truth.txt')
    frames, fps = read_video(vid)
    with open(gt_path) as f:
        bvp_gt = np.asarray([float(x) for x in f.read().strip().split('\n')[0].split()])
    n = min(len(frames), len(bvp_gt))
    frames, bvp_gt = frames[:n], bvp_gt[:n]
    print(f"{name}: {len(frames)} frames @ {fps:.2f} fps ({len(frames)/fps:.1f} s)")

    hr_gt = hr_from_fft(bandpass_numpy(bvp_gt, fps), fps)
    print(f"HR (référence GT) = {hr_gt:.1f} bpm")

    print("Détection YOLO5Face...")
    rgb = extract_rgb_y5f(frames)['face']

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
        ax2.axvline(r['hr'], color='black', lw=1.5, linestyle='--', label=f"Pic: {r['hr']:.1f} bpm")
        ax2.axvline(hr_gt, color='green', lw=1.5, linestyle=':', label=f"GT: {hr_gt:.1f} bpm")
        ax2.set_title(f"{label} — Spectre de puissance")
        ax2.set_xlabel("Fréquence (bpm)")
        ax2.set_ylabel("Puissance")
        ax2.set_xlim(30, 210)
        ax2.legend(fontsize=8)

    fig.suptitle(f"Analyse rPPG (backend YOLO5Face) — UBFC {name}\n"
                  f"({len(frames)} frames @ {fps:.2f} fps, {len(frames)/fps:.1f} s, HR GT={hr_gt:.1f} bpm)",
                  fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"analysis_y5f_{name}.png")
    fig.savefig(out, dpi=120)
    print(f"[Graphique] {out}")


if __name__ == '__main__':
    main()
