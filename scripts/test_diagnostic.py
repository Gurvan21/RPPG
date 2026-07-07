#!/usr/bin/env python3
"""DIAGNOSTIC COMPLET d'une prise paume : tous les indices + décomposition spectrale.
Usage : python scripts/test_diagnostic.py <video>"""
import os, sys
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
from pathlib import Path
import numpy as np, cv2, torch, mediapipe as mp
from scipy.signal import periodogram
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.preextract_clips import load_video, resample_to_fps
from scripts.extract_hand_regions import extract_video as eh, HULL_LM
from scripts.train_cnn1d import _temporal_norm, CLIP_LEN
from models.cnn1d_rppg import CNN1D_rPPG
from models.chrom_adaptive import bandpass_numpy
from mp_rppg.methods import chrom, pos
from mp_rppg.metrics import hr_from_fft, snr, hr_candidates
from mp_rppg.motion import is_motion_artifact
dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
video = sys.argv[1]
m = CNN1D_rPPG(in_channels=18*9).to(dev); m.load_state_dict(torch.load(ROOT/'weights'/'cnn1d_hand.pth', map_location=dev)); m.eval()

frames, fps = load_video(video, max_dim=720)
if fps > 32:
    ft = np.arange(len(frames))*1000.0/fps; frames, _ = resample_to_fps(frames, ft, 30.0); fps = 30.0
H, W = frames.shape[1:3]

# 1 passe Hands : region signals via extract_video ; centroïde/aire/lumino via une passe légère
x, _, det = eh(frames, 3)
palm_rgb = x[:, 8, :3].astype(np.float32)
brightness = float(palm_rgb.mean())

hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1,
                                 min_detection_confidence=0.5, min_tracking_confidence=0.5)
T = len(frames); cx = np.full(T, np.nan); cy = np.full(T, np.nan); sc = np.full(T, np.nan); areas = []
for t, f in enumerate(frames):
    r = hands.process(f)
    if not r.multi_hand_landmarks: continue
    lm = r.multi_hand_landmarks[0].landmark
    pts = np.array([[lm[k].x*W, lm[k].y*H] for k in HULL_LM], np.float32)
    cx[t], cy[t] = pts.mean(0); sc[t] = np.linalg.norm([lm[0].x*W-lm[9].x*W, lm[0].y*H-lm[9].y*H])
    hull = cv2.convexHull(pts.astype(np.int32)); mk = np.zeros((H, W), np.uint8); cv2.fillConvexPoly(mk, hull, 1)
    areas.append(100*mk.mean())
hands.close()
for a in (cx, cy, sc):
    g = np.isfinite(a); a[~g] = np.interp(np.where(~g)[0], np.where(g)[0], a[g]) if g.sum() > 1 else 0
scale = np.median(sc)+1e-6
mdx = bandpass_numpy((cx-cx.mean())/scale, fps); mdy = bandpass_numpy((cy-cy.mean())/scale, fps)
motion_sig = mdx + mdy; inband = float(np.sqrt(np.mean(mdx**2)+np.mean(mdy**2))); motion_hr = hr_from_fft(motion_sig, fps)

# signaux pouls
xn = _temporal_norm(x); pr = []
for s in range(0, xn.shape[1]-CLIP_LEN+1, CLIP_LEN):
    with torch.no_grad(): pr.append(m(torch.from_numpy(xn[:, s:s+CLIP_LEN]).unsqueeze(0).to(dev)).squeeze().cpu().numpy())
sig_cnn = bandpass_numpy(np.concatenate(pr), fps)
sig_chrom = bandpass_numpy(chrom(palm_rgb, fps), fps); sig_pos = bandpass_numpy(pos(palm_rgb, fps), fps)

methods = [("CNN1D-main", sig_cnn, 'crimson'), ("CHROM", sig_chrom, 'green'),
           ("POS", sig_pos, 'purple'), ("Mouvement", motion_sig, 'teal')]
print(f"\n===== DIAGNOSTIC : {Path(video).name} =====")
print(f"{len(frames)} frames @ {fps:.1f}fps  |  détection {100*det:.0f}%  aire {np.median(areas):.1f}%  "
      f"luminosité {brightness:.0f}  |  mouvement {motion_hr:.0f}bpm in-band {inband:.3f}")
hrs = []
for nm, s, _ in methods[:3]:
    h = hr_from_fft(s, fps); sn = snr(s, h, fps); cands, amb = hr_candidates(s, fps); hrs.append(h)
    print(f"  {nm:12s}: HR={h:5.0f}  SNR={sn:+5.1f}  " + (f"AMBIGU {cands[0][0]:.0f}/{cands[1][0]:.0f}" if amb else "pic net"))
hr0 = hr_from_fft(sig_cnn, fps); sn0 = snr(sig_cnn, hr0, fps)
art = is_motion_artifact(hr0, sn0, motion_hr, inband)
print(f"  accord (écart-type) = {np.std(hrs):.1f} bpm  |  artefact mouvement : "
      + ("❌ OUI" if art else "✅ non"))

# figure : onde + spectre des 4
fig, ax = plt.subplots(4, 2, figsize=(13, 11))
for r, (nm, s, col) in enumerate(methods):
    t = np.arange(len(s))/fps; h = hr_from_fft(s, fps); sn = snr(s, h, fps)
    nf = 1
    while nf < len(s): nf *= 2
    f, px = periodogram(s, fs=fps, nfft=nf, detrend=False); fr = f*60; b = (fr >= 40) & (fr <= 180)
    z = (t >= 4) & (t <= 12)
    ax[r, 0].plot(t[z], s[z], color=col, lw=1.1); ax[r, 0].set_title(f"{nm} — onde (4-12s)"); ax[r, 0].grid(alpha=.3)
    ax[r, 1].plot(fr[b], px[b]/px[b].max(), color='navy')
    ax[r, 1].axvline(h, color=col, ls='--', label=f"pic {h:.0f}")
    ax[r, 1].axvline(motion_hr, color='teal', ls=':', lw=1.5, label=f"mouvement {motion_hr:.0f}")
    ax[r, 1].set_title(f"{nm} — spectre (SNR {sn:+.1f})"); ax[r, 1].set_xlabel("bpm"); ax[r, 1].legend(fontsize=8); ax[r, 1].grid(alpha=.3)
plt.tight_layout(); out = ROOT/f"scratch_diag_{Path(video).stem}.png"; plt.savefig(out, dpi=95); plt.close()
print(f"  → figure : {out.name}")
