#!/usr/bin/env python3
"""Visualise l'effet des dégradations sur un vrai clip : image (propre/JPEG/mouvement/
dégradé complet) + pouls vert extrait + spectre (le pic FC survit-il ?)."""
import sys, glob
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.augment import frame_degrade_fixed, jpeg_compress, motion_jitter, _to_u8, _from_u8
from models.chrom_adaptive import bandpass_numpy
from mp_rppg.metrics import hr_from_fft, snr
from scipy.signal import periodogram, detrend
FPS = 30.0


def green_pulse(frames):
    g = frames[:, :, :, 1].mean(axis=(1, 2)).astype(np.float64)   # moyenne spatiale canal vert
    return bandpass_numpy(detrend(g), FPS)


def spectrum(sig):
    n = 1
    while n < len(sig): n *= 2
    f, px = periodogram(sig, fs=FPS, nfft=n, detrend='linear')
    m = (f*60 >= 40) & (f*60 <= 180)
    return f[m]*60, px[m]/(px[m].max()+1e-12)


# choisir un clip avec un pouls propre (bon SNR)
best = None
for fp in glob.glob(str(ROOT/'Data'/'clips_tscan'/'*'/'*.npz'))[:60]:
    d = np.load(fp); p = green_pulse(d['xr'].astype(np.float32))
    s = snr(p, hr_from_fft(p, FPS), FPS)
    if best is None or s > best[0]: best = (s, fp)
xr = np.load(best[1])['xr'].astype(np.float32)
print(f"clip choisi : {Path(best[1]).parent.name} (SNR propre {best[0]:.2f})")

rng = np.random.default_rng(1)
u8, lh = _to_u8(xr); t = 20
frame_clean = u8[t]
frame_jpeg = jpeg_compress(u8, 12)[t]
frame_motion = motion_jitter(u8, rng, 4, 6)[t]
deg = frame_degrade_fixed(xr); u8d, _ = _to_u8(deg); frame_deg = u8d[t]

p_clean = green_pulse(xr); p_deg = green_pulse(deg)
hr_c = hr_from_fft(p_clean, FPS); hr_d = hr_from_fft(p_deg, FPS)
snr_c = snr(p_clean, hr_c, FPS); snr_d = snr(p_deg, hr_d, FPS)
fc, pc = spectrum(p_clean); fd, pd = spectrum(p_deg)

fig = plt.figure(figsize=(13, 9))
titles = [f'PROPRE', 'JPEG q=12', 'Mouvement', 'DÉGRADÉ complet']
frames = [frame_clean, frame_jpeg, frame_motion, frame_deg]
for i, (im, ti) in enumerate(zip(frames, titles)):
    ax = fig.add_subplot(3, 4, i+1); ax.imshow(im); ax.set_title(ti, fontsize=10); ax.axis('off')
ax = fig.add_subplot(3, 1, 2)
tt = np.arange(len(p_clean))/FPS
ax.plot(tt, p_clean/np.std(p_clean), label=f'propre (SNR {snr_c:+.1f} dB)', lw=1.1)
ax.plot(tt, p_deg/np.std(p_deg)+6, label=f'dégradé (SNR {snr_d:+.1f} dB)', lw=1.1, color='crimson')
ax.set_title('Pouls extrait (canal vert, normalisé)'); ax.set_xlabel('temps (s)'); ax.legend(loc='upper right')
ax = fig.add_subplot(3, 1, 3)
ax.plot(fc, pc, label=f'propre — pic {hr_c:.0f} bpm', lw=1.3)
ax.plot(fd, pd, label=f'dégradé — pic {hr_d:.0f} bpm', lw=1.3, color='crimson')
ax.set_title('Spectre — le pic de FC survit-il ?'); ax.set_xlabel('bpm'); ax.set_ylabel('puissance norm.'); ax.legend()
plt.tight_layout()
out = ROOT/'degrade_demo.png'; plt.savefig(out, dpi=110)
print(f"HR propre {hr_c:.0f} (SNR {snr_c:+.1f}) | HR dégradé {hr_d:.0f} (SNR {snr_d:+.1f})")
print(f"Figure -> {out}")
