#!/usr/bin/env python3
"""Onde + spectre de 3 signaux sur une vidéo : CNN1D-main (paume), rBCG, CNN1D-face."""
import os, sys
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
from pathlib import Path
import numpy as np, torch
from scipy.signal import periodogram
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.preextract_clips import load_video, resample_to_fps, track_face_bboxes
from scripts.extract_hand_regions import extract_video as eh
from scripts.extract_regions_bisenet import load_bisenet, extract_video as ef, pick_device
from scripts.train_cnn1d import _temporal_norm, CLIP_LEN
from models.cnn1d_rppg import CNN1D_rPPG
from models.chrom_adaptive import bandpass_numpy
from mp_rppg.bcg import bcg_hr
from mp_rppg.metrics import hr_from_fft, snr, hr_candidates
dev = pick_device()
video = sys.argv[1]


def cnn_sig(x, weights, in_ch, fps):
    m = CNN1D_rPPG(in_channels=in_ch).to(dev); m.load_state_dict(torch.load(weights, map_location=dev)); m.eval()
    xn = _temporal_norm(x); pr = []
    for s in range(0, xn.shape[1]-CLIP_LEN+1, CLIP_LEN):
        with torch.no_grad():
            pr.append(m(torch.from_numpy(xn[:, s:s+CLIP_LEN]).unsqueeze(0).to(dev)).squeeze().cpu().numpy())
    return bandpass_numpy(np.concatenate(pr), fps) if pr else None


frames, fps = load_video(video, max_dim=720)
if fps > 32:
    ft = np.arange(len(frames))*1000.0/fps; frames, _ = resample_to_fps(frames, ft, 30.0); fps = 30.0
net = load_bisenet(dev)
xh, _, _ = eh(frames, 3); xf, _, _ = ef(net, dev, frames, 4)
palm = cnn_sig(xh, ROOT/'weights'/'cnn1d_hand.pth', 18*9, fps)
face = cnn_sig(xf, ROOT/'weights'/'cnn1d_rppg.pth', 23*9, fps)
bb = track_face_bboxes(frames); bbox = tuple(np.median(np.array(bb), axis=0).astype(int)) if len(bb) else None
_, _, bcg = bcg_hr(frames, fps, bbox=bbox)

sigs = [("CNN1D-main (paume)", palm, 'crimson'), ("rBCG (mouvement tête)", bcg, 'teal'),
        ("CNN1D-visage", face, 'darkorange')]
fig, ax = plt.subplots(3, 2, figsize=(13, 9))
for r, (name, sig, col) in enumerate(sigs):
    if sig is None: ax[r, 0].set_title(f"{name} : —"); continue
    t = np.arange(len(sig))/fps; hr = hr_from_fft(sig, fps); sn = snr(sig, hr, fps)
    cands, amb = hr_candidates(sig, fps)
    nf = 1
    while nf < len(sig): nf *= 2
    f, px = periodogram(sig, fs=fps, nfft=nf, detrend=False); fr = f*60; b = (fr >= 40) & (fr <= 180)
    z = (t >= 4) & (t <= 12)
    ax[r, 0].plot(t[z], sig[z], color=col, lw=1.2); ax[r, 0].set_title(f"{name} — onde (4-12s)")
    ax[r, 0].set_xlabel("temps (s)"); ax[r, 0].grid(alpha=.3)
    ax[r, 1].plot(fr[b], px[b]/px[b].max(), color='navy')
    ax[r, 1].axvline(hr, color='crimson', ls='--', label=f"{hr:.0f} bpm")
    if amb: ax[r, 1].axvline(cands[1][0], color='orange', ls=':', lw=2, label=f"rival {cands[1][0]:.0f}")
    ax[r, 1].set_title(f"{name} — spectre | HR={hr:.0f} SNR={sn:+.1f}" + ("  ⚠AMBIGU" if amb else ""))
    ax[r, 1].set_xlabel("fréquence (bpm)"); ax[r, 1].legend(); ax[r, 1].grid(alpha=.3)
    print(f"{name}: HR={hr:.1f} SNR={sn:+.1f}" + (f"  AMBIGU {cands[0][0]:.0f}/{cands[1][0]:.0f}" if amb else ""))
plt.tight_layout(); out = ROOT/f"scratch_3spectra_{Path(video).stem}.png"; plt.savefig(out, dpi=100); plt.close()
print(f"→ {out.name}")
