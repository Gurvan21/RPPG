#!/usr/bin/env python3
"""Compare plusieurs CNN1D-main (jeux de canaux différents) sur VideoMainVisage.
Extrait la paume UNE fois (cache), puis évalue chaque modèle. Vérité ~76-78."""
import os, sys
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.preextract_clips import load_video, resample_to_fps
from scripts.extract_hand_regions import extract_video as extract_hand
from scripts.train_cnn1d import _temporal_norm, CLIP_LEN
from models.cnn1d_rppg import CNN1D_rPPG
from models.chrom_adaptive import bandpass_numpy
from mp_rppg.metrics import hr_from_fft, snr

dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
CACHE = ROOT / "scratch_mainvisage_hand.npz"
VID = ROOT / "DataVital" / "SubjecTestRonel" / "VideoMainVisage.mp4"
TRUTH = 77.0

if CACHE.exists():
    d = np.load(CACHE); x_reg = d['x']; fps = float(d['fps'])
    print(f"cache: x={x_reg.shape} @ {fps}fps")
else:
    frames, fps = load_video(str(VID), max_dim=720)
    if fps > 32:
        ft = np.arange(len(frames))*1000.0/fps; frames, _ = resample_to_fps(frames, ft, 30.0); fps = 30.0
    x_reg, _, det = extract_hand(frames, 3)
    np.savez(CACHE, x=x_reg, fps=fps); print(f"extrait x={x_reg.shape} main{100*det:.0f}% → cache")


def evalm(weights, color_idx, label):
    nc = 18 * (9 if color_idx is None else len(color_idx))
    m = CNN1D_rPPG(in_channels=nc).to(dev)
    m.load_state_dict(torch.load(weights, map_location=dev)); m.eval()
    xn = _temporal_norm(x_reg, color_idx); T = xn.shape[1]; pr = []
    for s in range(0, T-CLIP_LEN+1, CLIP_LEN):
        with torch.no_grad():
            pr.append(m(torch.from_numpy(xn[:, s:s+CLIP_LEN]).unsqueeze(0).to(dev)).squeeze().cpu().numpy())
    sig = bandpass_numpy(np.concatenate(pr), fps); h = hr_from_fft(sig, fps)
    print(f"  {label:22s}: {h:5.1f} bpm   SNR {snr(sig,h,fps):+.1f}   err {abs(h-TRUTH):.0f}")


print(f"\nVideoMainVisage — vérité ~{TRUTH:.0f} bpm")
evalm(ROOT/'weights'/'cnn1d_hand.pth', None, "162 features (RGB+YUV+Lab)")
rgb = ROOT/'weights'/'cnn1d_hand_rgb.pth'
if rgb.exists():
    evalm(rgb, [0,1,2], "54 features (RGB seul)")
