#!/usr/bin/env python3
"""FFT seule vs FFT + ARBITRE par détection de pics : quand le spectre est AMBIGU
(deux pics candidats), on choisit le candidat le plus proche du comptage temporel
des battements. Held-out VISAGE (region_new), CNN1D-face."""
import os, sys
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
from pathlib import Path
import numpy as np, torch
from scipy.signal import find_peaks
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from models.cnn1d_rppg import CNN1D_rPPG
from models.chrom_adaptive import bandpass_numpy
from mp_rppg.metrics import hr_from_fft, hr_candidates
from scripts.train_cnn1d import _temporal_norm, CLIP_LEN
dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
m = CNN1D_rPPG(in_channels=23*9).to(dev); m.load_state_dict(torch.load(ROOT/'weights'/'cnn1d_rppg.pth', map_location=dev)); m.eval()


def peak_hr(sig, fps):
    pk, _ = find_peaks(sig, distance=int(0.4*fps))
    if len(pk) < 4: return float('nan')
    ibi = np.diff(pk)/fps; ibi = ibi[(ibi > 0.33) & (ibi < 1.5)]
    return 60.0/np.median(ibi) if len(ibi) else float('nan')


def arbiter(sig, fps):
    fft = hr_from_fft(sig, fps)
    cands, ambig = hr_candidates(sig, fps)
    if not ambig: return fft, False
    ph = peak_hr(sig, fps)
    if not np.isfinite(ph): return fft, False
    c = min(cands[:2], key=lambda z: abs(z[0]-ph))     # candidat le plus proche des pics
    return c[0], True


ef, ea, fired = [], [], []
for npz in sorted((ROOT/'Data'/'region_new').glob('*/*.npz')):
    d = np.load(str(npz), allow_pickle=True); fps = float(d['fps'])
    xn = _temporal_norm(d['x']); pr = []
    for s in range(0, xn.shape[1]-CLIP_LEN+1, CLIP_LEN):
        with torch.no_grad():
            pr.append(m(torch.from_numpy(xn[:, s:s+CLIP_LEN]).unsqueeze(0).to(dev)).squeeze().cpu().numpy())
    if not pr: continue
    sig = bandpass_numpy(np.concatenate(pr), fps)
    ref = hr_from_fft(bandpass_numpy(d['y'].astype(np.float32), fps), fps)
    hf = hr_from_fft(sig, fps); ha, fr = arbiter(sig, fps)
    ef.append(abs(hf-ref)); ea.append(abs(ha-ref)); fired.append(fr)
ef, ea, fired = np.array(ef), np.array(ea), np.array(fired)
print(f"n = {len(ef)} scénarios visage held-out  |  arbitre déclenché {fired.sum()}× (cas ambigus)\n")
print(f"  FFT seule           : MAE = {ef.mean():.2f} bpm   %<5 = {100*(ef<5).mean():.0f}")
print(f"  FFT + ARBITRE pics  : MAE = {ea.mean():.2f} bpm   %<5 = {100*(ea<5).mean():.0f}")
if fired.any():
    print(f"\n  Sur les {fired.sum()} cas AMBIGUS uniquement :")
    print(f"     FFT seule    : MAE = {ef[fired].mean():.2f} bpm")
    print(f"     avec arbitre : MAE = {ea[fired].mean():.2f} bpm")
