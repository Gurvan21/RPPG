#!/usr/bin/env python3
"""Test des 2 nouvelles vidéos smartphone : CNN1D-main + CHROM/POS paume vs
visage. Pas de vérité-terrain → on juge au SNR et à l'accord paume/visage.
Test critique de domain-shift (caméra 60fps industrielle → smartphone)."""
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

D = ROOT / "DataVital" / "SubjecTestRonel"
dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


def cnn_hand(frames, fps):
    x, names, det = extract_hand(frames, 3)              # (T,18,9)
    if det < 0.3: return None
    cnn = CNN1D_rPPG(in_channels=18*9).to(dev)
    cnn.load_state_dict(torch.load(ROOT/'weights'/'cnn1d_hand.pth', map_location=dev)); cnn.eval()
    xn = _temporal_norm(x); T = xn.shape[1]; pr = []
    for s in range(0, T-CLIP_LEN+1, CLIP_LEN):
        with torch.no_grad():
            pr.append(cnn(torch.from_numpy(xn[:, s:s+CLIP_LEN]).unsqueeze(0).to(dev)).squeeze().cpu().numpy())
    if not pr: return None
    sig = bandpass_numpy(np.concatenate(pr), fps)
    h = hr_from_fft(sig, fps); return h, snr(sig, h, fps), det


def classic(rgb_sig, fps):
    s = interp_nan(rgb_sig)
    if s is None: return {}
    out = {}
    for nm, fn in (('CHROM', chrom), ('POS', pos)):
        sig = bandpass_numpy(fn(s, fps), fps); h = hr_from_fft(sig, fps)
        out[nm] = (h, snr(sig, h, fps))
    return out


def run(video, do_palm=True):
    print(f"\n{'='*58}\n{video}")
    frames, fps = load_video(str(D/video), max_dim=720)
    if fps > 32:
        ft = np.arange(len(frames))*1000.0/fps; frames, _ = resample_to_fps(frames, ft, 30.0); fps = 30.0
    print(f"  {len(frames)} frames @ {fps:.1f}fps")
    if do_palm:
        palm = extract_palm_rgb(frames)
        pc = classic(palm[0], fps) if palm[1] > 0.3 else {}
        ch = cnn_hand(frames, fps)
        print(f"  ── PAUME (main détectée {100*palm[1]:.0f}%) ──")
        if ch: print(f"     CNN1D-main : {ch[0]:.0f} bpm   SNR {ch[1]:+.1f}")
        for k, v in pc.items(): print(f"     {k:10s}: {v[0]:.0f} bpm   SNR {v[1]:+.1f}")
    fc = classic(face_rgb(frames), fps)
    print(f"  ── VISAGE ──")
    for k, v in fc.items(): print(f"     {k:10s}: {v[0]:.0f} bpm   SNR {v[1]:+.1f}")


run("VideoMainVisage.mp4", do_palm=True)
run("VideoVisage.mp4", do_palm=False)
