#!/usr/bin/env python3
"""Compare deux estimateurs de FC sur la sortie du CNN1D-main :
  (A) FFT (pic spectral dominant)  vs  (B) détection de pics temporelle (IBI médian).
Référence = FC de la PPG de contact (y). Sur tous les enregistrements hand_signals."""
import os, sys
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
from pathlib import Path
import numpy as np, torch
from scipy.signal import find_peaks
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from models.cnn1d_rppg import CNN1D_rPPG
from models.chrom_adaptive import bandpass_numpy
from mp_rppg.metrics import hr_from_fft
from scripts.train_cnn1d import _temporal_norm, CLIP_LEN
dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
m = CNN1D_rPPG(in_channels=18*9).to(dev); m.load_state_dict(torch.load(ROOT/'weights'/'cnn1d_hand.pth', map_location=dev)); m.eval()


def peak_hr(sig, fps):
    pk, _ = find_peaks(sig, distance=int(0.4*fps))       # min 0.4s entre pics (<150 bpm)
    if len(pk) < 4: return float('nan')
    ibi = np.diff(pk)/fps
    ibi = ibi[(ibi > 0.33) & (ibi < 1.5)]                # 40-180 bpm plausibles
    return 60.0/np.median(ibi) if len(ibi) else float('nan')


ef, ep, n = [], [], 0
for npz in sorted((ROOT/'Data'/'hand_signals').glob('*/*.npz')):
    d = np.load(str(npz), allow_pickle=True); fps = float(d['fps'])
    xn = _temporal_norm(d['x']); pr = []
    for s in range(0, xn.shape[1]-CLIP_LEN+1, CLIP_LEN):
        with torch.no_grad():
            pr.append(m(torch.from_numpy(xn[:, s:s+CLIP_LEN]).unsqueeze(0).to(dev)).squeeze().cpu().numpy())
    if not pr: continue
    sig = bandpass_numpy(np.concatenate(pr), fps)
    ref = hr_from_fft(bandpass_numpy(d['y'].astype(np.float32), fps), fps)
    hf = hr_from_fft(sig, fps); hp = peak_hr(sig, fps)
    if not np.isfinite(hp): continue
    ef.append(abs(hf-ref)); ep.append(abs(hp-ref)); n += 1
ef, ep = np.array(ef), np.array(ep)
print(f"n = {n} enregistrements (réf = FC PPG contact)\n")
print(f"  FFT (spectral)      : MAE = {ef.mean():.2f} bpm   %<5 = {100*(ef<5).mean():.0f}")
print(f"  PICS (temporel/IBI) : MAE = {ep.mean():.2f} bpm   %<5 = {100*(ep<5).mean():.0f}")
