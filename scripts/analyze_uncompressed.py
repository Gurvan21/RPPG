"""
Analyse rPPG (CHROM De Haan, CHROM adaptatif, POS) sur une vidéo haute
résolution (ex: 1920x1080), traitée frame par frame pour éviter de charger
toute la vidéo en mémoire (np.stack de 1151 frames 1080p ≈ 7 Go).

Usage :
    python scripts/analyze_uncompressed.py --video uncompressed.mp4
"""

import argparse
import os
import sys

import mediapipe as mp  # noqa: F401  (importé avant cv2/imageio par convention du projet)
import cv2
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
from mp_rppg.pipeline import _poly_mask, _region_mean, _FRONT_IDX

VIDEO_DIR = os.path.join(ROOT, 'results', 'personal_video')
MODEL_PATH = os.path.join(ROOT, 'weights', 'chrom_adaptive_ubfc.pth')


def get_spectrum(bvp, fs):
    N = _next_pow2(len(bvp))
    f, pxx = periodogram(bvp, fs=fs, nfft=N, detrend=False)
    return f, pxx


def extract_front_rgb_streaming(path):
    """Extrait le RGB moyen de la région front, frame par frame (faible mémoire)."""
    meta = iio.immeta(path, plugin="pyav")
    fps = float(meta['fps'])

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    )

    values = []
    n_ok, n_total = 0, 0
    for frame in iio.imiter(path, plugin="pyav"):
        n_total += 1
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        res = face_mesh.process(frame)
        if not res.multi_face_landmarks:
            values.append(None)
            continue
        n_ok += 1
        lm = res.multi_face_landmarks[0].landmark
        m_front = _poly_mask(frame.shape, lm, _FRONT_IDX)
        values.append(_region_mean(bgr, m_front))

    face_mesh.close()
    print(f"  FaceMesh : {n_ok}/{n_total} frames détectées ({100*n_ok/n_total:.0f}%)")

    arr = np.full((len(values), 3), np.nan, dtype=np.float32)
    for i, v in enumerate(values):
        if v is not None:
            arr[i] = v
    for c in range(3):
        nans = np.isnan(arr[:, c])
        if nans.all():
            arr[:, c] = 0.0
        elif nans.any():
            idx = np.arange(len(arr))
            arr[:, c] = np.interp(idx, idx[~nans], arr[~nans, c])
    return arr, fps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', default='uncompressed.mp4')
    args = parser.parse_args()

    path = os.path.join(VIDEO_DIR, args.video)
    name = os.path.splitext(os.path.basename(path))[0]

    coeffs = load_coefficients(MODEL_PATH)
    print("Coefficients adaptatifs utilisés :", coeffs)
    print("Coefficients De Haan             :", DEHAAN_COEFFICIENTS)

    print(f"\n{'='*70}\n  {name}\n{'='*70}")
    rgb, fps = extract_front_rgb_streaming(path)
    print(f"  {len(rgb)} frames @ {fps:.2f} fps ({len(rgb)/fps:.1f} s)")

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

    fig.suptitle(f"Analyse rPPG — {name}\n"
                  f"({len(rgb)} frames @ {fps:.2f} fps, {len(rgb)/fps:.1f} s)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(VIDEO_DIR, f"analysis_{name}.png")
    fig.savefig(out, dpi=120)
    print(f"\n[Graphique] {out}")

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
    out2 = os.path.join(VIDEO_DIR, f"snr_{name}.png")
    fig2.savefig(out2, dpi=120)
    print(f"[Graphique] {out2}")


if __name__ == '__main__':
    main()
