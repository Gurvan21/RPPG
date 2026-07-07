#!/usr/bin/env python3
"""Garde-fou MOUVEMENT : mesure l'oscillation de la main DANS la bande cardiaque
(celle qui crée un faux pouls) via la trajectoire des landmarks, normalisée par la
taille de la paume (invariante à la distance). Refuse si trop fort.
Usage : python scripts/test_motion_gate.py <video1> <video2> ..."""
import os, sys
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
from pathlib import Path
import numpy as np, mediapipe as mp
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.preextract_clips import load_video, resample_to_fps
from scripts.extract_hand_regions import HULL_LM
from mp_rppg.metrics import hr_from_fft
from models.chrom_adaptive import bandpass_numpy

JITTER_REJECT = 0.060        # jitter global (fraction de la paume) → refuse au-dessus
INBAND_REJECT = 0.025        # + refuse si oscillation in-band vraiment forte


def motion_metrics(video):
    frames, fps = load_video(video, max_dim=720)
    if fps > 32:
        ft = np.arange(len(frames))*1000.0/fps; frames, _ = resample_to_fps(frames, ft, 30.0); fps = 30.0
    H, W = frames.shape[1:3]
    hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1,
                                     min_detection_confidence=0.5, min_tracking_confidence=0.5)
    T = len(frames); cx = np.full(T, np.nan); cy = np.full(T, np.nan); sc = np.full(T, np.nan)
    for t, f in enumerate(frames):
        r = hands.process(f)
        if not r.multi_hand_landmarks: continue
        lm = r.multi_hand_landmarks[0].landmark
        pts = np.array([[lm[k].x*W, lm[k].y*H] for k in HULL_LM], np.float32)
        cx[t], cy[t] = pts.mean(0)
        sc[t] = np.linalg.norm([lm[0].x*W-lm[9].x*W, lm[0].y*H-lm[9].y*H])
    hands.close()
    for a in (cx, cy, sc):
        g = np.isfinite(a); a[~g] = np.interp(np.where(~g)[0], np.where(g)[0], a[g]) if g.sum() > 1 else 0
    scale = np.median(sc) + 1e-6
    dx = (cx - cx.mean())/scale; dy = (cy - cy.mean())/scale       # déplacement en unités de paume
    total = np.sqrt(np.std(dx)**2 + np.std(dy)**2)                  # jitter global
    ibx = bandpass_numpy(dx, fps); iby = bandpass_numpy(dy, fps)    # mouvement DANS la bande cardiaque
    inband = np.sqrt(np.mean(ibx**2) + np.mean(iby**2))
    mot_hr = hr_from_fft(ibx + iby, fps)
    return total, inband, mot_hr, fps


print(f"{'vidéo':<26}{'jitter':>9}{'in-band':>9}{'mot.HR':>8}   verdict")
for v in sys.argv[1:]:
    p = Path(v)
    if not p.exists(): print(f"{p.name} introuvable"); continue
    tot, ib, mhr, fps = motion_metrics(v)
    rej = tot > JITTER_REJECT or ib > INBAND_REJECT
    print(f"{p.stem[:24]:<26}{tot:>9.3f}{ib:>9.3f}{mhr:>7.0f}   "
          + ("❌ REFUSE (main bouge trop)" if rej else "✅ OK"))
print(f"\nSeuils : jitter > {JITTER_REJECT} ou in-band > {INBAND_REJECT} (fraction de paume)")
