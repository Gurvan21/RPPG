#!/usr/bin/env python3
"""Peut-on récupérer une prise SOMBRE en l'éclaircissant en logiciel (gamma) ?
Compare SNR paume (CNN1D) : original vs gamma-corrigé. Si le gamma n'aide pas,
c'est que la lumière doit venir de la CAPTURE, pas du post-traitement."""
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
from mp_rppg.methods import chrom
from mp_rppg.metrics import hr_from_fft, snr
dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
m = CNN1D_rPPG(in_channels=18*9).to(dev); m.load_state_dict(torch.load(ROOT/'weights'/'cnn1d_hand.pth', map_location=dev)); m.eval()
_lut = {}


def gamma(frames, g):
    if g not in _lut:
        _lut[g] = (255*((np.arange(256)/255.0)**g)).astype(np.uint8)
    return _lut[g][frames]


def analyze(frames, fps):
    x, _, det = eh(frames, 3); xn = _temporal_norm(x); pr = []
    for s in range(0, xn.shape[1]-CLIP_LEN+1, CLIP_LEN):
        with torch.no_grad(): pr.append(m(torch.from_numpy(xn[:, s:s+CLIP_LEN]).unsqueeze(0).to(dev)).squeeze().cpu().numpy())
    sig = bandpass_numpy(np.concatenate(pr), fps); h = hr_from_fft(sig, fps)
    bright = x[:, 8, :3].mean()
    return h, snr(sig, h, fps), bright


VIDS = ["Paume/videoMainOpenCamera.mp4", "Paume/VID_20260702_155120.mp4", "Paume/videoDeMain.mp4"]
print(f"{'vidéo':<22}{'ORIGINAL':>22}{'GAMMA 0.5 (éclairci)':>24}")
for rel in VIDS:
    p = ROOT/"DataVital"/"SubjecTestRonel"/rel
    if not p.exists(): continue
    frames, fps = load_video(str(p), max_dim=720)
    if fps > 32:
        ft = np.arange(len(frames))*1000.0/fps; frames, _ = resample_to_fps(frames, ft, 30.0); fps = 30.0
    h0, s0, b0 = analyze(frames, fps)
    hg, sg, bg = analyze(gamma(frames, 0.5), fps)
    print(f"{Path(rel).stem[:20]:<22}{f'{h0:.0f}bpm SNR{s0:+.1f} (lum {b0:.0f})':>22}{f'{hg:.0f}bpm SNR{sg:+.1f} (lum {bg:.0f})':>24}")
