#!/usr/bin/env python3
import os, sys
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
from pathlib import Path
import numpy as np, torch
from scipy.signal import periodogram, find_peaks
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.preextract_clips import load_video, resample_to_fps
from scripts.extract_regions_bisenet import load_bisenet, extract_video as ef
from scripts.train_cnn1d import _temporal_norm, CLIP_LEN
from models.cnn1d_rppg import CNN1D_rPPG
from models.chrom_adaptive import bandpass_numpy
dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
v = "DataVital/SubjecTestRonel/Visage/VID_20260630_162042.mp4"
frames, fps = load_video(v, max_dim=720)
if fps > 32:
    ft = np.arange(len(frames))*1000.0/fps; frames, _ = resample_to_fps(frames, ft, 30.0); fps = 30.0
net = load_bisenet(dev); x_reg, _, _ = ef(net, dev, frames, 4)
xn = _temporal_norm(x_reg); m = CNN1D_rPPG(in_channels=23*9).to(dev)
m.load_state_dict(torch.load(ROOT/'weights'/'cnn1d_rppg.pth', map_location=dev)); m.eval()
pr = []
for s in range(0, xn.shape[1]-CLIP_LEN+1, CLIP_LEN):
    with torch.no_grad():
        pr.append(m(torch.from_numpy(xn[:, s:s+CLIP_LEN]).unsqueeze(0).to(dev)).squeeze().cpu().numpy())
sig = bandpass_numpy(np.concatenate(pr), fps)
print(f"fps={fps:.1f} durée={len(sig)/fps:.1f}s")


def npow2(n):
    p = 1
    while p < n: p *= 2
    return p


# (A) hr_from_fft : periodogram boxcar + zeropad
f, px = periodogram(sig, fs=fps, nfft=npow2(len(sig)), detrend=False)
mA = (f >= 0.7) & (f <= 2.5)
print(f"\n[A boxcar+zeropad, choix actuel] pic = {f[mA][np.argmax(px[mA])]*60:.1f} bpm")
# top pics
idx = np.argsort(px[mA])[::-1][:4]
for i in idx: print(f"    {f[mA][i]*60:5.1f} bpm  pow {px[mA][i]/px[mA].max():.2f}")

# (B) Hann sans zeropad (comme le graphe)
w = np.hanning(len(sig)); ps = np.abs(np.fft.rfft(sig*w))**2; fr = np.fft.rfftfreq(len(sig), 1/fps)
mB = (fr >= 0.7) & (fr <= 2.5)
print(f"\n[B Hann, comme le graphe] pic = {fr[mB][np.argmax(ps[mB])]*60:.1f} bpm")
idx = np.argsort(ps[mB])[::-1][:4]
for i in idx: print(f"    {fr[mB][i]*60:5.1f} bpm  pow {ps[mB][i]/ps[mB].max():.2f}")

# (C) Hann + zeropad (le meilleur des deux mondes)
NF = npow2(len(sig))*2
psz = np.abs(np.fft.rfft(sig*w, n=NF))**2; frz = np.fft.rfftfreq(NF, 1/fps)
mC = (frz >= 0.7) & (frz <= 2.5)
print(f"\n[C Hann+zeropad] pic = {frz[mC][np.argmax(psz[mC])]*60:.1f} bpm")

# (D) comptage de battements dans le temps
pk, _ = find_peaks(sig, distance=int(0.4*fps))
ibi = np.diff(pk)/fps
print(f"\n[D comptage battements] {len(pk)} pics, HR médian = {60/np.median(ibi):.1f} bpm  (IBI {np.median(ibi):.2f}s)")
