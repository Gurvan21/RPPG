"""
Analyse des vidéos personnelles (results/personal_video/) :
  - Extraction RGB (front, MediaPipe FaceMesh)
  - CHROM (De Haan, coefficients fixes)
  - CHROM adaptatif (coefficients appris sur UBFC, weights/chrom_adaptive_ubfc.pth)
  - POS

Pour chaque méthode : HR (FFT), SNR, spectre de puissance.
Génère un graphique combiné par vidéo (signaux temporels, spectres, comparaison SNR).

Usage :
    python scripts/analyze_personal_videos.py
"""

import glob
import os
import sys

# cv2.VideoCapture entre en conflit avec mediapipe dans le même process
# (segfault) -> on importe mediapipe en premier et on décode avec imageio.
import mediapipe  # noqa: F401
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
from mp_rppg.pipeline import extract_rgb

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


def analyze_video(path, coeffs, skip_start=0.0):
    name = os.path.splitext(os.path.basename(path))[0]
    print(f"\n{'='*70}\n  {name}\n{'='*70}")

    frames, fps = read_video(path)
    if skip_start > 0:
        n_skip = int(round(skip_start * fps))
        frames = frames[n_skip:]
        print(f"  {n_skip} frames coupées au début ({skip_start:.1f} s)")
        name = f"{name}_skip{skip_start:g}s"
    print(f"  {len(frames)} frames @ {fps:.2f} fps ({len(frames)/fps:.1f} s)")

    rgb_regions = extract_rgb(frames, verbose=True)
    rgb = rgb_regions['front']

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

    # ── Figure : signaux temporels + spectres + barres SNR ────────────────
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
        ax2.axvline(r['hr'], color='black', lw=1.5, linestyle='--',
                     label=f"Pic: {r['hr']:.1f} bpm")
        ax2.set_title(f"{label} — Spectre de puissance")
        ax2.set_xlabel("Fréquence (bpm)")
        ax2.set_ylabel("Puissance")
        ax2.set_xlim(30, 210)
        ax2.legend(fontsize=8)

    fig.suptitle(f"Analyse rPPG — {name}\n"
                  f"({len(frames)} frames @ {fps:.2f} fps, {len(frames)/fps:.1f} s)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(VIDEO_DIR, f"analysis_{name}.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  [Graphique] {out_path}")

    # ── Figure : comparaison SNR ───────────────────────────────────────────
    fig2, ax = plt.subplots(figsize=(5, 4))
    labels = list(results.keys())
    snrs = [results[l]['snr'] for l in labels]
    bars = ax.bar(labels, snrs, color=[colors[l] for l in labels])
    for bar, s in zip(bars, snrs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{s:.2f}", ha='center', va='bottom' if s >= 0 else 'top')
    ax.set_ylabel("SNR (dB)")
    ax.set_title(f"Comparaison SNR — {name}")
    ax.tick_params(axis='x', labelrotation=15)
    fig2.tight_layout()
    out_path2 = os.path.join(VIDEO_DIR, f"snr_{name}.png")
    fig2.savefig(out_path2, dpi=120)
    plt.close(fig2)
    print(f"  [Graphique] {out_path2}")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', default=None, help="Nom du fichier (sous-chaîne) à analyser")
    parser.add_argument('--skip-start', type=float, default=0.0,
                         help="Nombre de secondes à couper au début de la vidéo")
    args = parser.parse_args()

    coeffs = load_coefficients(MODEL_PATH)
    print("Coefficients adaptatifs utilisés :", coeffs)
    print("Coefficients De Haan             :", DEHAAN_COEFFICIENTS)

    videos = sorted(glob.glob(os.path.join(VIDEO_DIR, '*.mp4')))
    if args.video:
        videos = [v for v in videos if args.video in os.path.basename(v)]
    if not videos:
        sys.exit(f"Aucune vidéo .mp4 trouvée dans {VIDEO_DIR}")

    summary = {}
    for v in videos:
        summary[os.path.basename(v)] = analyze_video(v, coeffs, skip_start=args.skip_start)

    # ── Tableau récapitulatif ───────────────────────────────────────────────
    print(f"\n{'='*70}\n  Récapitulatif\n{'='*70}")
    for vname, results in summary.items():
        print(f"\n{vname}")
        print(f"  {'Méthode':<20} {'HR (bpm)':>10} {'SNR (dB)':>10}")
        for label, r in results.items():
            print(f"  {label:<20} {r['hr']:>10.1f} {r['snr']:>10.2f}")


if __name__ == '__main__':
    main()
