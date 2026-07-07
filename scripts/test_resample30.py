#!/usr/bin/env python3
"""Compare CNN1D-main sur fps NATIF vs signaux rééchantillonnés à 30 fps (par
interpolation, pour matcher l'entraînement). Juge au SNR (pas de vérité-terrain)."""
import os, sys
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.preextract_clips import load_video, resample_to_fps
from scripts.extract_hand_regions import extract_video as eh
from scripts.train_cnn1d import _temporal_norm, CLIP_LEN
from models.cnn1d_rppg import CNN1D_rPPG
from models.chrom_adaptive import bandpass_numpy
from mp_rppg.metrics import hr_from_fft, snr
dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
_m = CNN1D_rPPG(in_channels=18*9).to(dev)
_m.load_state_dict(torch.load(ROOT/'weights'/'cnn1d_hand.pth', map_location=dev)); _m.eval()


def resample_x_30(x, fps):
    """x (T,R,C) au fps donné -> (T',R,C) à 30 fps par interpolation temporelle."""
    T, R, C = x.shape
    T30 = max(CLIP_LEN, int(round(T * 30.0 / fps)))
    old = np.arange(T) / fps; new = np.arange(T30) / 30.0
    flat = x.reshape(T, R * C); out = np.empty((T30, R * C), np.float32)
    for c in range(R * C):
        out[:, c] = np.interp(new, old, flat[:, c])
    return out.reshape(T30, R, C)


def infer(x, fps):
    xn = _temporal_norm(x); pr = []
    for s in range(0, xn.shape[1]-CLIP_LEN+1, CLIP_LEN):
        with torch.no_grad():
            pr.append(_m(torch.from_numpy(xn[:, s:s+CLIP_LEN]).unsqueeze(0).to(dev)).squeeze().cpu().numpy())
    if not pr: return None
    sig = bandpass_numpy(np.concatenate(pr), fps); h = hr_from_fft(sig, fps)
    return h, snr(sig, h, fps)


VIDS = [
    ("Paume/videoDeMain.mp4", 24.5),
    ("Paume/VID_20260630_161919.mp4", 24.5),
    ("PaumeVisage/VID_20260630_152419.mp4", 19.7),
    ("PaumeVisage/VID_20260630_161652.mp4", 29.6),
]
D = ROOT/"DataVital"/"SubjecTestRonel"
print(f"{'vidéo':<28}{'fps':>6}{'NATIF':>16}{'RESAMPLE 30':>16}")
for rel, _ in VIDS:
    p = D/rel
    if not p.exists(): print(f"{rel} introuvable"); continue
    frames, fps = load_video(str(p), max_dim=720)
    if fps > 32:
        ft = np.arange(len(frames))*1000.0/fps; frames, _ = resample_to_fps(frames, ft, 30.0); fps = 30.0
    x, _, det = eh(frames, 3)
    if det < 0.3: print(f"{rel}: main non détectée"); continue
    r_nat = infer(x, fps)
    r_res = infer(resample_x_30(x, fps), 30.0) if abs(fps-30) > 0.5 else r_nat
    def f(r): return f"{r[0]:.0f}bpm SNR{r[1]:+.1f}" if r else "—"
    print(f"{Path(rel).stem[:26]:<28}{fps:>6.1f}{f(r_nat):>16}{f(r_res):>16}")
