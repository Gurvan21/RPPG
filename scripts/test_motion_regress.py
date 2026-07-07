#!/usr/bin/env python3
"""Régression du mouvement : on retire du signal couleur de la paume la composante
expliquée par la GÉOMÉTRIE de la main (centre + échelle + vitesses), indépendante
du pouls. But : supprimer la fausse périodicité de mouvement et révéler le pouls."""
import os, sys
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
from pathlib import Path
import numpy as np, cv2, mediapipe as mp
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.preextract_clips import load_video, resample_to_fps
from scripts.palm_rppg import interp_nan
from scripts.extract_hand_regions import HULL_LM
from mp_rppg.methods import chrom, pos
from mp_rppg.metrics import hr_from_fft, snr
from models.chrom_adaptive import bandpass_numpy

video = sys.argv[1]
frames, fps = load_video(video, max_dim=720)
if fps > 32:
    ft = np.arange(len(frames))*1000.0/fps; frames, _ = resample_to_fps(frames, ft, 30.0); fps = 30.0
H, W = frames.shape[1:3]
hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1,
                                 min_detection_confidence=0.5, min_tracking_confidence=0.5)
T = len(frames)
rgb = np.full((T, 3), np.nan, np.float32)      # couleur paume
cx = np.full(T, np.nan); cy = np.full(T, np.nan); sc = np.full(T, np.nan)  # géométrie
for t, f in enumerate(frames):
    r = hands.process(f)
    if not r.multi_hand_landmarks: continue
    lm = r.multi_hand_landmarks[0].landmark
    pts = np.array([[lm[k].x*W, lm[k].y*H] for k in HULL_LM], np.float32)
    cx[t], cy[t] = pts.mean(0)
    sc[t] = np.linalg.norm([lm[0].x*W-lm[9].x*W, lm[0].y*H-lm[9].y*H])
    hull = cv2.convexHull(pts.astype(np.int32)); mask = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(mask, hull, 1); mask = cv2.erode(mask, np.ones((9, 9), np.uint8))
    sel = mask.astype(bool); px = f[sel].astype(np.float32)
    lum = px.mean(1); keep = (lum > 30) & (lum < 245)
    if keep.sum() > 100: rgb[t] = px[keep].mean(0)
hands.close()

rgb = interp_nan(rgb)
for a in (cx, cy, sc):
    g = np.isfinite(a); a[~g] = np.interp(np.where(~g)[0], np.where(g)[0], a[g]) if g.sum() > 1 else 0

# régresseurs de mouvement : géométrie + vitesses (standardisés)
def z(v): return (v - v.mean())/(v.std()+1e-8)
M = np.column_stack([np.ones(T), z(cx), z(cy), z(sc),
                     z(np.gradient(cx)), z(np.gradient(cy)), z(np.gradient(sc))])
# résidu couleur = couleur - part expliquée par le mouvement
beta, *_ = np.linalg.lstsq(M, rgb, rcond=None)
rgb_clean = rgb - M @ beta

# fréquence propre du MOUVEMENT (confirme d'où vient le faux pic)
mot = z(cx) + z(cy) + z(sc)
print(f"Mouvement dominant : {hr_from_fft(bandpass_numpy(mot, fps), fps):.0f} bpm\n")

for label, sig_rgb in [("AVANT (brut)", rgb), ("APRÈS (mouvement régressé)", rgb_clean)]:
    for nm, fn in (('CHROM', chrom), ('POS', pos)):
        s = bandpass_numpy(fn(sig_rgb, fps), fps); h = hr_from_fft(s, fps)
        print(f"  {label:28s} {nm}: {h:.0f} bpm  SNR {snr(s, h, fps):+.1f}")
print("\n(vérité attendue ~61 bpm, donnée par le visage immobile ; faux ~95 = mouvement)")
