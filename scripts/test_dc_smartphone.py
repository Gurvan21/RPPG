#!/usr/bin/env python3
"""Compare CNN1D-main baseline vs CNN1D-main+DC sur des vidéos paume smartphone."""
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
RC = 18*9
dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
base = CNN1D_rPPG(in_channels=RC).to(dev); base.load_state_dict(torch.load(ROOT/'weights'/'cnn1d_hand.pth', map_location=dev)); base.eval()
dcm = CNN1D_rPPG(in_channels=2*RC).to(dev); dcm.load_state_dict(torch.load(ROOT/'weights'/'cnn1d_hand_dc.pth', map_location=dev)); dcm.eval()
st = np.load(ROOT/'weights'/'cnn1d_hand_dc_stats.npz'); mu = st['mu']; sd = st['sd']


def run(model, x2, fps):
    pr = []
    for s in range(0, x2.shape[1]-CLIP_LEN+1, CLIP_LEN):
        with torch.no_grad():
            pr.append(model(torch.from_numpy(x2[:, s:s+CLIP_LEN].copy()).unsqueeze(0).to(dev)).squeeze().cpu().numpy())
    sig = bandpass_numpy(np.concatenate(pr), fps); h = hr_from_fft(sig, fps); return h, snr(sig, h, fps)


VIDS = ["Paume/videoMainOpenCamera.mp4", "Paume/VID_20260702_155038.mp4", "Paume/VID_20260702_155120.mp4",
        "Paume/videoDeMain.mp4", "PaumeVisage/VideLumièrenaturePaumeFace.mp4"]
print(f"{'vidéo':<24}{'BASELINE':>16}{'AVEC DC':>16}")
for rel in VIDS:
    p = ROOT/"DataVital"/"SubjecTestRonel"/rel
    if not p.exists(): print(f"{rel} introuvable"); continue
    frames, fps = load_video(str(p), max_dim=720)
    if fps > 32:
        ft = np.arange(len(frames))*1000.0/fps; frames, _ = resample_to_fps(frames, ft, 30.0); fps = 30.0
    x, _, det = eh(frames, 3); T, R, C = x.shape
    flat = x.reshape(T, R*C).astype(np.float32); mean = flat.mean(0)
    normed = (flat/(mean+1e-8) - 1.0).T                         # (RC,T)
    hb, sb = run(base, normed, fps)
    dc = np.repeat(((mean-mu)/sd).astype(np.float32)[:, None], T, axis=1)
    aug = np.concatenate([normed, dc], axis=0)
    hd, sd_ = run(dcm, aug, fps)
    print(f"{Path(rel).stem[:22]:<24}{f'{hb:.0f}bpm SNR{sb:+.1f}':>16}{f'{hd:.0f}bpm SNR{sd_:+.1f}':>16}")
