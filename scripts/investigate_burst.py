"""
Analyse temps-fréquence (spectrogramme) du signal CHROM/POS pour vérifier
si le pic HR détecté provient d'un "faux signal" localisé en début de
vidéo (forte amplitude mais non représentatif du rythme cardiaque réel).

Usage :
    python scripts/investigate_burst.py --video Nouvelle
"""

import argparse
import glob
import os
import sys

import mediapipe  # noqa: F401
import imageio.v3 as iio
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, ROOT)

from models.chrom_adaptive import load_coefficients
from mp_rppg.methods import chrom, pos
from mp_rppg.metrics import hr_from_fft
from mp_rppg.pipeline import extract_rgb

VIDEO_DIR = os.path.join(ROOT, 'results', 'personal_video')
CACHE_DIR = os.path.join(ROOT, 'results', 'personal_video_cache')
MODEL_PATH = os.path.join(ROOT, 'weights', 'chrom_adaptive_ubfc.pth')


def read_video(path):
    frames = np.stack(list(iio.imiter(path, plugin="pyav")))
    meta = iio.immeta(path, plugin="pyav")
    return frames, float(meta['fps'])


def get_rgb(name, path):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{name}.npz")
    if os.path.exists(cache_path):
        npz = np.load(cache_path)
        return npz['rgb'], float(npz['fps'])
    frames, fps = read_video(path)
    rgb = extract_rgb(frames, verbose=True)['front']
    np.savez(cache_path, rgb=rgb, fps=fps)
    return rgb, fps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', default='Nouvelle')
    parser.add_argument('--win', type=float, default=5.0, help="Taille de fenêtre du spectrogramme (s)")
    args = parser.parse_args()

    candidates = sorted(glob.glob(os.path.join(VIDEO_DIR, '*.mp4')))
    matches = [v for v in candidates if args.video in os.path.basename(v)]
    if not matches:
        sys.exit(f"Aucune vidéo trouvée pour '{args.video}'")
    path = matches[0]
    name = os.path.splitext(os.path.basename(path))[0]

    rgb, fps = get_rgb(name, path)
    print(f"{name}: {len(rgb)} frames @ {fps:.2f} fps ({len(rgb)/fps:.1f} s)")

    bvp_chrom = chrom(rgb, fps)
    bvp_pos = pos(rgb, fps)

    hr_full = hr_from_fft(bvp_chrom, fps)
    print(f"HR (CHROM, signal entier)        : {hr_full:.1f} bpm")

    # HR sur la 1ère moitié vs la 2nde moitié du signal
    n = len(bvp_chrom)
    hr_first = hr_from_fft(bvp_chrom[:n // 2], fps)
    hr_second = hr_from_fft(bvp_chrom[n // 2:], fps)
    print(f"HR (CHROM, 1ère moitié, {n//2/fps:.1f}s)   : {hr_first:.1f} bpm")
    print(f"HR (CHROM, 2nde moitié, {(n-n//2)/fps:.1f}s)   : {hr_second:.1f} bpm")

    # Spectrogramme : où se situe la puissance dominante dans le temps ?
    nperseg = int(args.win * fps)
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for ax, bvp, label in [(axes[0], bvp_chrom, 'CHROM (De Haan)'),
                            (axes[1], bvp_pos, 'POS')]:
        f, t, Sxx = spectrogram(bvp, fs=fps, nperseg=nperseg,
                                 noverlap=nperseg - 1, scaling='density')
        mask = (f >= 0.5) & (f <= 3.5)
        pcm = ax.pcolormesh(t, f[mask] * 60, 10 * np.log10(Sxx[mask] + 1e-12), shading='gouraud', cmap='magma')
        ax.set_ylabel("Fréquence (bpm)")
        ax.set_title(f"{label} — Spectrogramme (puissance en dB)")
        fig.colorbar(pcm, ax=ax, label='dB')
        ax.set_ylim(40, 200)
    axes[1].set_xlabel("Temps (s)")
    fig.suptitle(f"{name} — où se situe la puissance dominante dans le temps ?")
    fig.tight_layout()
    out = os.path.join(VIDEO_DIR, f"spectrogram_{name}.png")
    fig.savefig(out, dpi=120)
    print(f"[Graphique] {out}")


if __name__ == '__main__':
    main()
