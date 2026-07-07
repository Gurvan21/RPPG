#!/usr/bin/env python3
"""Trace le SPECTRE du signal CNN1D-visage pour une vidéo (chemin identique à
run_on_video). Usage: python scripts/plot_cnn_spectrum.py <video>"""
import os, sys
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
from pathlib import Path
import numpy as np, torch
from scipy.signal import periodogram
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from models.cnn1d_rppg import CNN1D_rPPG
from models.chrom_adaptive import bandpass_numpy
from scripts.train_cnn1d import _temporal_norm, CLIP_LEN
from scripts.extract_regions_bisenet import load_bisenet, extract_video, pick_device
from scripts.preextract_clips import load_video, resample_to_fps
from mp_rppg.metrics import hr_from_fft, snr, hr_candidates

video = sys.argv[1]
dev = pick_device()
frames, fps = load_video(video, max_dim=720)
if fps > 32.0:
    ft = np.arange(len(frames))*1000.0/fps; frames, _ = resample_to_fps(frames, ft, 30.0); fps = 30.0
print(f"{len(frames)} frames @ {fps:.1f} fps")
net = load_bisenet(dev)
x_reg, _, _ = extract_video(net, dev, frames, 4)
cnn = CNN1D_rPPG(in_channels=23*9).to(dev)
cnn.load_state_dict(torch.load(ROOT/'weights'/'cnn1d_rppg.pth', map_location=dev)); cnn.eval()
xn = _temporal_norm(x_reg); preds = []
for s in range(0, xn.shape[1]-CLIP_LEN+1, CLIP_LEN):
    with torch.no_grad():
        preds.append(cnn(torch.from_numpy(xn[:, s:s+CLIP_LEN]).unsqueeze(0).to(dev)).squeeze().cpu().numpy())
sig = bandpass_numpy(np.concatenate(preds), fps)

hr = hr_from_fft(sig, fps); sn = snr(sig, hr, fps); cands, ambig = hr_candidates(sig, fps)
n = 1
while n < len(sig): n *= 2
f, px = periodogram(sig, fs=fps, nfft=n, detrend='linear'); px /= px.max()
mk = (f*60 >= 40) & (f*60 <= 180)

print(f"\nCNN1D-visage : HR = {hr:.1f} bpm | SNR = {sn:.2f} dB | ambigu={ambig}")
print(f"candidats (bpm, force): {[(round(c[0],1), round(c[1],2)) for c in cands[:4]]}")

fig, ax = plt.subplots(2, 1, figsize=(11, 7))
tt = np.arange(len(sig))/fps
ax[0].plot(tt, sig, lw=0.7, color='crimson'); ax[0].set_title('Signal CNN1D-visage (temporel)')
ax[0].set_xlabel('temps (s)'); ax[0].set_ylabel('amp')
ax[1].plot(f[mk]*60, px[mk], color='crimson')
ax[1].axvline(hr, color='k', ls='--', lw=1, label=f'pic {hr:.0f} bpm')
for c in cands[1:3]:
    ax[1].axvline(c[0], color='gray', ls=':', lw=0.9)
ax[1].set_title(f'Spectre CNN1D-visage — SNR {sn:.2f} dB'); ax[1].set_xlabel('bpm'); ax[1].set_ylabel('puissance norm.')
ax[1].legend()
plt.tight_layout()
out = os.path.splitext(video)[0] + '_cnn_spectrum.png'
plt.savefig(out, dpi=110); print(f"Figure -> {out}")
