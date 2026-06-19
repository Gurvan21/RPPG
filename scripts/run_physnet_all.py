"""
Inférence PhysNet (poids UBFC et SCAMPS) sur toutes les vidéos personnelles
(results/personal_video/*.mp4). Génère un graphique BVP+spectre par
(vidéo, poids), et un tableau récapitulatif HR/SNR.

Usage :
    python scripts/run_physnet_all.py
"""

import glob
import os
import sys

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
from infer_physnet import (read_video, detect_and_crop, diff_normalize,
                            run_physnet, hr_fft, WEIGHTS, CHUNK_LEN)
from mp_rppg.metrics import snr, _next_pow2

VIDEO_DIR = os.path.join(ROOT, 'results', 'personal_video')


def get_spectrum(bvp, fs):
    N = _next_pow2(len(bvp))
    f, pxx = periodogram(bvp, fs=fs, nfft=N, detrend=False)
    return f, pxx


def main():
    device = torch.device('cpu')
    videos = sorted(glob.glob(os.path.join(VIDEO_DIR, '*.mp4')))
    print(f"{len(videos)} vidéos trouvées")

    # Charge les 2 modèles une seule fois
    models = {}
    for wname, wpath in WEIGHTS.items():
        m = PhysNet_padding_Encoder_Decoder_MAX(frames=CHUNK_LEN).to(device)
        m.load_state_dict(torch.load(wpath, map_location=device))
        m.eval()
        models[wname] = m

    results = {}  # video -> weights -> {'bvp','hr','snr','f','pxx','fps'}
    for vpath in videos:
        vname = os.path.splitext(os.path.basename(vpath))[0]
        print(f"\n{'='*60}\n  {vname}\n{'='*60}")
        frames, fps = read_video(vpath)
        if fps == 0:
            fps = 30.0
        print(f"  {len(frames)} frames @ {fps:.2f} fps ({len(frames)/fps:.1f} s)")

        cropped = detect_and_crop(frames)
        frames_norm = diff_normalize(cropped)

        results[vname] = {}
        for wname, model in models.items():
            bvp = run_physnet(frames_norm, model, device)
            hr = hr_fft(bvp, fps)
            s = snr(bvp, hr, fps)
            f, pxx = get_spectrum(bvp, fps)
            results[vname][wname] = {'bvp': bvp, 'hr': hr, 'snr': s, 'f': f, 'pxx': pxx, 'fps': fps}
            print(f"  PhysNet ({wname:<6}) HR = {hr:6.1f} bpm   SNR = {s:6.2f} dB")

    # ── Figure par vidéo : BVP + spectre pour chaque jeu de poids ──────────
    colors = {'UBFC': 'steelblue', 'SCAMPS': 'darkorange'}
    for vname, by_w in results.items():
        fig, axes = plt.subplots(2, 2, figsize=(12, 7))
        for i, (wname, r) in enumerate(by_w.items()):
            bvp, fps, hr, s = r['bvp'], r['fps'], r['hr'], r['snr']
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

        fig.suptitle(f"PhysNet (UBFC vs SCAMPS) — {vname}", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        out = os.path.join(VIDEO_DIR, f"physnet_compare_{vname}.png")
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"\n[Graphique] {out}")

    # ── Figure récap : HR et SNR par vidéo x poids ─────────────────────────
    vnames = list(results.keys())
    wnames = list(WEIGHTS.keys())
    x = np.arange(len(vnames))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(max(8, len(vnames) * 1.8), 4.5))
    for i, w in enumerate(wnames):
        hrs = [results[v][w]['hr'] for v in vnames]
        axes[0].bar(x + (i - 0.5) * width, hrs, width, label=w, color=colors[w])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(vnames, rotation=20, ha='right')
    axes[0].set_ylabel("HR (bpm)")
    axes[0].set_title("HR PhysNet par vidéo")
    axes[0].legend()

    for i, w in enumerate(wnames):
        snrs = [results[v][w]['snr'] for v in vnames]
        axes[1].bar(x + (i - 0.5) * width, snrs, width, label=w, color=colors[w])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(vnames, rotation=20, ha='right')
    axes[1].set_ylabel("SNR (dB)")
    axes[1].set_title("SNR PhysNet par vidéo")
    axes[1].legend()

    fig.tight_layout()
    out = os.path.join(VIDEO_DIR, "physnet_summary.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[Graphique] {out}")

    # ── Tableau récapitulatif ───────────────────────────────────────────────
    print(f"\n{'='*60}\n  Récapitulatif\n{'='*60}")
    print(f"  {'Vidéo':<35} {'Poids':<8} {'HR (bpm)':>10} {'SNR (dB)':>10}")
    for v in vnames:
        for w in wnames:
            r = results[v][w]
            print(f"  {v:<35} {w:<8} {r['hr']:>10.1f} {r['snr']:>10.2f}")


if __name__ == '__main__':
    main()
