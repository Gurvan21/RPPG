#!/usr/bin/env python3
"""VRAI test anti-raccourci : on RETIRE le pouls de l'entrée (band-stop de la bande
cardiaque 0.6-4.5 Hz sur chaque canal) en gardant tout le reste (couleur statique,
dérives lentes, bruit HF), puis on demande la FC au CNN1D-main déjà entraîné.
  - FC survit  → le modèle lit la FC HORS du pouls = RACCOURCI (fuite).
  - FC s'effondre → il a besoin du pouls = pas de raccourci.
Comparé à CHROM (pouls-only, sans apprentissage) comme témoin."""
import os, sys, glob
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
from pathlib import Path
import numpy as np, torch
from scipy.signal import butter, filtfilt
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from models.cnn1d_rppg import CNN1D_rPPG
from models.chrom_adaptive import bandpass_numpy
from mp_rppg.metrics import hr_from_fft
from scripts.train_cnn1d import _temporal_norm, CLIP_LEN
dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
m = CNN1D_rPPG(in_channels=18*9).to(dev)
m.load_state_dict(torch.load(ROOT/'weights'/'cnn1d_hand.pth', map_location=dev)); m.eval()


def bandstop(xn, fps, lo=0.6, hi=4.5):
    """xn : (C, T). Retire la bande cardiaque de chaque canal (garde le reste)."""
    ny = fps/2.0
    b, a = butter(3, [lo/ny, hi/ny], btype='bandstop')
    return filtfilt(b, a, xn, axis=1).astype(np.float32)


def pred_hr(xn, fps):
    pr = []
    for s in range(0, xn.shape[1]-CLIP_LEN+1, CLIP_LEN):
        with torch.no_grad():
            pr.append(m(torch.from_numpy(xn[:, s:s+CLIP_LEN].copy()).unsqueeze(0).to(dev)).squeeze().cpu().numpy())
    if not pr: return None
    return hr_from_fft(bandpass_numpy(np.concatenate(pr), fps), fps)


e_norm, e_stop, ph_n, ph_s, gh = [], [], [], [], []
for p in sorted(glob.glob(str(ROOT/'Data'/'hand_signals'/'*'/'*.npz'))):
    d = np.load(p, allow_pickle=True); fps = float(d['fps'])
    xn = _temporal_norm(d['x']); y = d['y'].astype(np.float32)
    if xn.shape[1] < CLIP_LEN: continue
    g = hr_from_fft(bandpass_numpy(y, fps), fps)
    hn = pred_hr(xn, fps)                       # entrée normale
    hs = pred_hr(bandstop(xn, fps), fps)        # POULS retiré
    if hn is None or hs is None: continue
    e_norm.append(abs(hn-g)); e_stop.append(abs(hs-g)); ph_n.append(hn); ph_s.append(hs); gh.append(g)

print(f"n = {len(gh)} enregistrements\n")
print(f"Entrée NORMALE      : MAE = {np.mean(e_norm):.2f} bpm   r(FC,vérité) = {np.corrcoef(ph_n, gh)[0,1]:+.2f}")
print(f"POULS RETIRÉ (b-stop): MAE = {np.mean(e_stop):.2f} bpm   r(FC,vérité) = {np.corrcoef(ph_s, gh)[0,1]:+.2f}")
print("\n→ Si POULS RETIRÉ s'effondre (MAE haute, r~0) : le modèle a besoin du POULS = PAS de raccourci.")
print("→ Si POULS RETIRÉ reste bon : RACCOURCI (le modèle lit la FC hors du pouls).")
