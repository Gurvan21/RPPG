#!/usr/bin/env python3
"""Trace l'onde rPPG (amplitude) produite par le CNN1D-main sur la paume + spectre."""
import os, sys
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
from pathlib import Path
import numpy as np, torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.preextract_clips import load_video, resample_to_fps
from scripts.extract_hand_regions import extract_video as extract_hand
from scripts.train_cnn1d import _temporal_norm, CLIP_LEN
from models.cnn1d_rppg import CNN1D_rPPG
from models.chrom_adaptive import bandpass_numpy
from mp_rppg.metrics import hr_from_fft, snr

dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
video = sys.argv[1] if len(sys.argv) > 1 else "DataVital/SubjecTestRonel/videoMainVisageIso800.mp4"

frames, fps = load_video(video, max_dim=720)
if fps > 32:
    ft = np.arange(len(frames))*1000.0/fps; frames, _ = resample_to_fps(frames, ft, 30.0); fps = 30.0
x, _, det = extract_hand(frames, 3)
m = CNN1D_rPPG(in_channels=18*9).to(dev)
m.load_state_dict(torch.load(ROOT/'weights'/'cnn1d_hand.pth', map_location=dev)); m.eval()
xn = _temporal_norm(x); T = xn.shape[1]; pr = []
for s in range(0, T-CLIP_LEN+1, CLIP_LEN):
    with torch.no_grad():
        pr.append(m(torch.from_numpy(xn[:, s:s+CLIP_LEN]).unsqueeze(0).to(dev)).squeeze().cpu().numpy())
raw = np.concatenate(pr)
sig = bandpass_numpy(raw, fps)
hr = hr_from_fft(sig, fps); sn = snr(sig, hr, fps)
t = np.arange(len(sig)) / fps

# spectre
n = len(sig); ps = np.abs(np.fft.rfft(sig*np.hanning(n)))**2
fr = np.fft.rfftfreq(n, 1/fps)*60  # bpm
band = (fr >= 40) & (fr <= 180)

fig, ax = plt.subplots(3, 1, figsize=(11, 8))
ax[0].plot(t, sig, lw=0.8, color='crimson'); ax[0].set_title(
    f"Onde rPPG CNN1D-main (paume) — {Path(video).name}  |  HR={hr:.0f} bpm, SNR={sn:+.1f} dB")
ax[0].set_xlabel("temps (s)"); ax[0].set_ylabel("amplitude (rel.)"); ax[0].grid(alpha=.3)
# zoom 8 s
z = (t >= 5) & (t <= 13)
ax[1].plot(t[z], sig[z], lw=1.3, color='crimson', marker='.', ms=2)
ax[1].set_title("Zoom 5-13 s (battements visibles)")
ax[1].set_xlabel("temps (s)"); ax[1].set_ylabel("amplitude (rel.)"); ax[1].grid(alpha=.3)
ax[2].plot(fr[band], ps[band]/ps[band].max(), color='navy')
ax[2].axvline(hr, color='crimson', ls='--', label=f"pic = {hr:.0f} bpm")
ax[2].set_title("Spectre"); ax[2].set_xlabel("fréquence (bpm)"); ax[2].set_ylabel("puissance (norm.)")
ax[2].legend(); ax[2].grid(alpha=.3)
plt.tight_layout()
out = ROOT / "scratch_cnn_wave_iso800.png"
plt.savefig(out, dpi=110); print(f"saved {out}")
print(f"amplitude: min={sig.min():.2f} max={sig.max():.2f} (pic-à-pic {sig.max()-sig.min():.2f}, rel.)")
print(f"HR={hr:.1f} bpm  SNR={sn:+.1f}  durée={len(sig)/fps:.0f}s  main={100*det:.0f}%")
