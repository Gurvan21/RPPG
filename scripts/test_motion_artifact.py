#!/usr/bin/env python3
"""Valide is_motion_artifact : pouls CNN1D-main + mouvement, sur plusieurs vidéos."""
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
from mp_rppg.motion import palm_motion, is_motion_artifact
dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
m = CNN1D_rPPG(in_channels=18*9).to(dev)
m.load_state_dict(torch.load(ROOT/'weights'/'cnn1d_hand.pth', map_location=dev)); m.eval()

B = ROOT/"DataVital"/"SubjecTestRonel"
VIDS = ["PaumeVisage/MainQuibouge.mov", "PaumeVisage/mainquibougeBeaucoupmaismoinsvite.mov",
        "Paume/VideoMainFaceArrière2.mp4", "PaumeVisage/VideLumièrenaturePaumeFace.mp4",
        "Paume/videoDeMain.mp4"]
print(f"{'vidéo':<26}{'HR':>5}{'SNR':>6}{'mot.HR':>8}{'in-band':>9}   verdict")
for rel in VIDS:
    p = B/rel
    if not p.exists(): print(f"{rel} introuvable"); continue
    frames, fps = load_video(str(p), max_dim=720)
    if fps > 32:
        ft = np.arange(len(frames))*1000.0/fps; frames, _ = resample_to_fps(frames, ft, 30.0); fps = 30.0
    x, _, det = eh(frames, 3)
    xn = _temporal_norm(x); pr = []
    for s in range(0, xn.shape[1]-CLIP_LEN+1, CLIP_LEN):
        with torch.no_grad():
            pr.append(m(torch.from_numpy(xn[:, s:s+CLIP_LEN]).unsqueeze(0).to(dev)).squeeze().cpu().numpy())
    sig = bandpass_numpy(np.concatenate(pr), fps); hr = hr_from_fft(sig, fps); sn = snr(sig, hr, fps)
    mhr, ib = palm_motion(frames, fps)
    art = is_motion_artifact(hr, sn, mhr, ib)
    print(f"{Path(rel).stem[:24]:<26}{hr:>5.0f}{sn:>+6.1f}{mhr:>8.0f}{ib:>9.3f}   "
          + ("❌ ARTEFACT MOUVEMENT" if art else "✅ pouls cru"))
