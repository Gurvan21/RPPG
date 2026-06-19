"""
Inférence PhysNet (UBFC + SCAMPS) sur une vidéo haute résolution (1080p),
traitée frame par frame (crop 72x72 immédiat) pour éviter de charger toute
la vidéo en mémoire (np.stack de 1152 frames 1080p ≈ 7 Go -> OOM).

Usage :
    python scripts/run_physnet_uncompressed.py --video uncompressed.mp4
"""

import argparse
import os
import sys

import cv2
import imageio.v3 as iio
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import periodogram

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from models.physnet import PhysNet_padding_Encoder_Decoder_MAX
from infer_physnet import diff_normalize, run_physnet, hr_fft, WEIGHTS, CHUNK_LEN, BOX_COEF, RESIZE, HAAR_XML
from mp_rppg.metrics import snr, _next_pow2

VIDEO_DIR = os.path.join(ROOT, 'results', 'personal_video')


def get_spectrum(bvp, fs):
    N = _next_pow2(len(bvp))
    f, pxx = periodogram(bvp, fs=fs, nfft=N, detrend=False)
    return f, pxx


def stream_cropped_frames(path):
    """Décode la vidéo frame par frame, détecte le visage sur la 1ère frame
    (Haar Cascade), puis crop+resize 72x72 chaque frame -> (T,72,72,3)."""
    meta = iio.immeta(path, plugin="pyav")
    fps = float(meta['fps'])

    detector = cv2.CascadeClassifier(HAAR_XML) if os.path.exists(HAAR_XML) else None

    bbox = None
    buffer = []
    cropped = []

    for frame in iio.imiter(path, plugin="pyav"):
        if bbox is None:
            buffer.append(frame)
            if detector is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                zones = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
                if len(zones):
                    x, y, w, h = zones[np.argmax(zones[:, 2])]
                    cx, cy = x + w // 2, y + h // 2
                    hw, hh = int(w * BOX_COEF / 2), int(h * BOX_COEF / 2)
                    H, W = frame.shape[:2]
                    bbox = (max(0, cx-hw), max(0, cy-hh), min(W, cx+hw), min(H, cy+hh))
            if bbox is None and len(buffer) > 60:
                # pas de visage trouvé dans les 60 premières frames -> frame entière
                h, w = frame.shape[:2]
                bbox = (0, 0, w, h)
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                print(f"  Boîte visage HC : [{x1},{y1}]→[{x2},{y2}]")
                for f in buffer:
                    cropped.append(cv2.resize(f[y1:y2, x1:x2], (RESIZE, RESIZE),
                                                interpolation=cv2.INTER_AREA).astype(np.float32))
                buffer = []
            continue

        x1, y1, x2, y2 = bbox
        cropped.append(cv2.resize(frame[y1:y2, x1:x2], (RESIZE, RESIZE),
                                   interpolation=cv2.INTER_AREA).astype(np.float32))

    if bbox is None:  # vidéo entière < 60 frames, jamais flush
        h, w = buffer[0].shape[:2]
        bbox = (0, 0, w, h)
        x1, y1, x2, y2 = bbox
        for f in buffer:
            cropped.append(cv2.resize(f[y1:y2, x1:x2], (RESIZE, RESIZE),
                                       interpolation=cv2.INTER_AREA).astype(np.float32))

    return np.asarray(cropped), fps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', default='uncompressed.mp4')
    args = parser.parse_args()

    path = os.path.join(VIDEO_DIR, args.video)
    name = os.path.splitext(os.path.basename(path))[0]
    device = torch.device('cpu')

    print(f"Lecture + crop streaming : {path}")
    cropped, fps = stream_cropped_frames(path)
    print(f"  {len(cropped)} frames @ {fps:.2f} fps ({len(cropped)/fps:.1f} s)")

    frames_norm = diff_normalize(cropped)

    colors = {'UBFC': 'steelblue', 'SCAMPS': 'darkorange'}
    results = {}
    for wname, wpath in WEIGHTS.items():
        model = PhysNet_padding_Encoder_Decoder_MAX(frames=CHUNK_LEN).to(device)
        model.load_state_dict(torch.load(wpath, map_location=device, weights_only=False))
        model.eval()
        bvp = run_physnet(frames_norm, model, device)
        hr = hr_fft(bvp, fps)
        s = snr(bvp, hr, fps)
        f, pxx = get_spectrum(bvp, fps)
        results[wname] = {'bvp': bvp, 'hr': hr, 'snr': s, 'f': f, 'pxx': pxx}
        print(f"  PhysNet ({wname:<6}) HR = {hr:6.1f} bpm   SNR = {s:6.2f} dB")

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for i, (wname, r) in enumerate(results.items()):
        bvp, hr, s = r['bvp'], r['hr'], r['snr']
        t = np.arange(len(bvp)) / fps
        axes[i, 0].plot(t, bvp, color=colors[wname], lw=0.8)
        axes[i, 0].set_title(f"PhysNet ({wname}) — BVP (HR={hr:.1f} bpm, SNR={s:.2f} dB)")
        axes[i, 0].set_xlabel("Temps (s)")
        axes[i, 0].set_ylabel("Amplitude")

        f, pxx = r['f'], r['pxx']
        mask = (f >= 0.5) & (f <= 3.5)
        axes[i, 1].plot(f[mask] * 60, pxx[mask], color=colors[wname], lw=1.2)
        axes[i, 1].axvline(hr, color='black', lw=1.5, linestyle='--', label=f"Pic: {hr:.1f} bpm")
        axes[i, 1].set_title(f"PhysNet ({wname}) — Spectre de puissance")
        axes[i, 1].set_xlabel("Fréquence (bpm)")
        axes[i, 1].set_ylabel("Puissance")
        axes[i, 1].set_xlim(30, 210)
        axes[i, 1].legend(fontsize=8)

    fig.suptitle(f"PhysNet (UBFC vs SCAMPS) — {name}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(VIDEO_DIR, f"physnet_compare_{name}.png")
    fig.savefig(out, dpi=120)
    print(f"\n[Graphique] {out}")


if __name__ == '__main__':
    main()
