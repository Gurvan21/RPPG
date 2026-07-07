#!/usr/bin/env python3
"""Effet du fps sur le SNR, à CONTENU IDENTIQUE : on extrait le signal palmaire
une fois, puis on le décime (sous-échantillonnage entier = vrai bas-fps) et on
mesure HR + SNR à chaque cadence. Isole l'effet fps (même lumière, même prise)."""
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
m = CNN1D_rPPG(in_channels=18*9).to(dev)
m.load_state_dict(torch.load(ROOT/'weights'/'cnn1d_hand.pth', map_location=dev)); m.eval()


def infer(x, fps):
    xn = _temporal_norm(x)
    if xn.shape[1] < CLIP_LEN: return None
    pr = []
    for s in range(0, xn.shape[1]-CLIP_LEN+1, CLIP_LEN):
        with torch.no_grad():
            pr.append(m(torch.from_numpy(xn[:, s:s+CLIP_LEN]).unsqueeze(0).to(dev)).squeeze().cpu().numpy())
    sig = bandpass_numpy(np.concatenate(pr), fps); h = hr_from_fft(sig, fps)
    return h, snr(sig, h, fps)


for rel in ["PaumeVisage/VID_20260630_161652.mp4", "Paume/videoDeMain.mp4"]:
    p = ROOT/"DataVital"/"SubjecTestRonel"/rel
    if not p.exists(): continue
    frames, fps = load_video(str(p), max_dim=720)
    if fps > 32:
        ft = np.arange(len(frames))*1000.0/fps; frames, _ = resample_to_fps(frames, ft, 30.0); fps = 30.0
    x, _, det = eh(frames, 3)
    print(f"\n=== {Path(rel).stem}  (natif {fps:.1f} fps, main {100*det:.0f}%) ===")
    print(f"{'fps effectif':>14}{'HR':>10}{'SNR':>9}")
    for k in [1, 2, 3, 4]:
        xk = x[::k]; fk = fps / k
        r = infer(xk, fk)
        if r: print(f"{fk:>14.1f}{r[0]:>8.0f}bpm{r[1]:>+8.1f}")
        else: print(f"{fk:>14.1f}    (trop court)")
