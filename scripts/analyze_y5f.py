"""
Analyse rPPG sur une vidéo personnelle avec le backend YOLO5Face
(détection visage par réseau de neurones, crop 72x72, moyenne spatiale RGB).

Le modèle YOLO5Face n'est pas inclus dans ce repo : on le charge depuis
l'ancien clone rPPG-Toolbox (dataset/data_loader/face_detector/).

Usage :
    python scripts/analyze_y5f.py --video Nouvelle
"""

import argparse
import glob
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

from models.chrom_adaptive import load_coefficients, DEHAAN_COEFFICIENTS
from mp_rppg.methods import chrom, chrom_adaptive, pos
from mp_rppg.metrics import hr_from_fft, snr, _next_pow2
from mp_rppg.backends import extract_rgb_y5f

# Le module YOLO5Face vit dans l'ancien clone rPPG-Toolbox (ajouté après pour
# ne pas masquer le package mp_rppg/models de ce projet)
TOOLBOX_ROOT = "/home/kemnhou/Bureau/old/rPPG-Toolbox"
sys.path.append(TOOLBOX_ROOT)

VIDEO_DIR = os.path.join(ROOT, 'results', 'personal_video')
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
    parser.add_argument('--video', default='Nouvelle')
    args = parser.parse_args()

    candidates = sorted(glob.glob(os.path.join(VIDEO_DIR, '*.mp4')))
    matches = [v for v in candidates if args.video in os.path.basename(v)]
    if not matches:
        sys.exit(f"Aucune vidéo trouvée pour '{args.video}'")
    path = matches[0]
    name = os.path.splitext(os.path.basename(path))[0]

    coeffs = load_coefficients(MODEL_PATH)

    frames, fps = read_video(path)
    print(f"{name}: {len(frames)} frames @ {fps:.2f} fps ({len(frames)/fps:.1f} s)")

    print("Détection YOLO5Face...")
    rgb = extract_rgb_y5f(frames)['face']

    methods = {
        'CHROM (De Haan)': chrom(rgb, fps),
        'CHROM (adaptatif)': chrom_adaptive(rgb, fps, coeffs),
        'POS': pos(rgb, fps),
    }

    results = {}
    for label, bvp in methods.items():
        hr = hr_from_fft(bvp, fps)
        s = snr(bvp, hr, fps)
        f, pxx = get_spectrum(bvp, fps)
        results[label] = {'bvp': bvp, 'hr': hr, 'snr': s, 'f': f, 'pxx': pxx}
        print(f"  {label:<20} HR = {hr:6.1f} bpm   SNR = {s:6.2f} dB")

    colors = {'CHROM (De Haan)': 'steelblue', 'CHROM (adaptatif)': 'darkorange', 'POS': 'tomato'}
    fig, axes = plt.subplots(3, 2, figsize=(13, 10))
    for i, (label, r) in enumerate(results.items()):
        bvp = r['bvp']
        t = np.arange(len(bvp)) / fps
        ax = axes[i, 0]
        ax.plot(t, bvp, color=colors[label], lw=0.8)
        ax.set_title(f"{label} — signal BVP (HR={r['hr']:.1f} bpm, SNR={r['snr']:.2f} dB)")
        ax.set_xlabel("Temps (s)")
        ax.set_ylabel("Amplitude")

        ax2 = axes[i, 1]
        f, pxx = r['f'], r['pxx']
        mask = (f >= 0.5) & (f <= 3.5)
        ax2.plot(f[mask] * 60, pxx[mask], color=colors[label], lw=1.2)
        ax2.axvline(r['hr'], color='black', lw=1.5, linestyle='--', label=f"Pic: {r['hr']:.1f} bpm")
        ax2.set_title(f"{label} — Spectre de puissance")
        ax2.set_xlabel("Fréquence (bpm)")
        ax2.set_ylabel("Puissance")
        ax2.set_xlim(30, 210)
        ax2.legend(fontsize=8)

    fig.suptitle(f"Analyse rPPG (backend YOLO5Face) — {name}\n"
                  f"({len(frames)} frames @ {fps:.2f} fps, {len(frames)/fps:.1f} s)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(VIDEO_DIR, f"analysis_y5f_{name}.png")
    fig.savefig(out, dpi=120)
    print(f"[Graphique] {out}")


if __name__ == '__main__':
    main()
