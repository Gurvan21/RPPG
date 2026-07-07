#!/usr/bin/env python3
"""Analyse paume (CNN1D-main + CHROM/POS) + visage sur UNE vidéo. Vérité en arg."""
import os, sys
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.preextract_clips import load_video, resample_to_fps
from scripts.extract_hand_regions import extract_video as extract_hand
from scripts.palm_rppg import extract_palm_rgb, interp_nan
from scripts.test_palm_poc import face_rgb
from scripts.train_cnn1d import _temporal_norm, CLIP_LEN
from models.cnn1d_rppg import CNN1D_rPPG
from models.chrom_adaptive import bandpass_numpy
from mp_rppg.methods import chrom, pos
from mp_rppg.metrics import hr_from_fft, snr

dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
video = sys.argv[1]; TRUTH = float(sys.argv[2]) if len(sys.argv) > 2 else None


def cnn_hand(frames, fps):
    x, _, det = extract_hand(frames, 3)
    if det < 0.3: return None, det
    m = CNN1D_rPPG(in_channels=18*9).to(dev)
    m.load_state_dict(torch.load(ROOT/'weights'/'cnn1d_hand.pth', map_location=dev)); m.eval()
    xn = _temporal_norm(x); T = xn.shape[1]; pr = []
    for s in range(0, T-CLIP_LEN+1, CLIP_LEN):
        with torch.no_grad():
            pr.append(m(torch.from_numpy(xn[:, s:s+CLIP_LEN]).unsqueeze(0).to(dev)).squeeze().cpu().numpy())
    sig = bandpass_numpy(np.concatenate(pr), fps); h = hr_from_fft(sig, fps)
    return (h, snr(sig, h, fps)), det


def classic(rgb_sig, fps):
    s = interp_nan(rgb_sig)
    if s is None: return {}
    return {nm: (hr_from_fft(bandpass_numpy(fn(s, fps), fps), fps),
                 snr(bandpass_numpy(fn(s, fps), fps), hr_from_fft(bandpass_numpy(fn(s, fps), fps), fps), fps))
            for nm, fn in (('CHROM', chrom), ('POS', pos))}


def line(label, h, sn):
    e = f"  err {abs(h-TRUTH):.0f}" if TRUTH else ""
    print(f"  {label:16s}: {h:5.1f} bpm  SNR {sn:+.1f}{e}")


frames, fps = load_video(video, max_dim=720)
if fps > 32:
    ft = np.arange(len(frames))*1000.0/fps; frames, _ = resample_to_fps(frames, ft, 30.0); fps = 30.0
print(f"{Path(video).name} — {len(frames)} frames @ {fps:.1f}fps" + (f"  (vérité ~{TRUTH:.0f})" if TRUTH else ""))
palm = extract_palm_rgb(frames)
print(f"── PAUME (main {100*palm[1]:.0f}%) ──")
ch, _ = cnn_hand(frames, fps)
if ch: line("CNN1D-main", *ch)
for k, v in classic(palm[0], fps).items(): line(f"paume {k}", *v)
print("── VISAGE ──")
for k, v in classic(face_rgb(frames), fps).items(): line(f"visage {k}", *v)
