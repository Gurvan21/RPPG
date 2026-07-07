#!/usr/bin/env python3
"""Métriques OBJECTIVES de capture (indépendantes du modèle) : détection main,
taille de la paume (% du cadre), luminosité, mouvement in-band. Pour comparer
bonnes vs mauvaises prises et localiser le problème."""
import os, sys
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
from pathlib import Path
import numpy as np, cv2, mediapipe as mp
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.preextract_clips import load_video, resample_to_fps
from scripts.extract_hand_regions import HULL_LM
from mp_rppg.motion import palm_motion

B = ROOT/"DataVital"/"SubjecTestRonel"
VIDS = [("Paume/videoDeMain.mp4", "BON +2.9"),
        ("PaumeVisage/VideLumièrenaturePaumeFace.mp4", "BON récent +5.8"),
        ("Paume/VideoMainFaceArrière1.mp4", "raté récent -1.3"),
        ("Paume/videoMainOpenCamera.mp4", "raté récent -3.4")]

print(f"{'vidéo':<22}{'note':<18}{'détect':>7}{'aire%':>7}{'lumin.':>8}{'mvt in-band':>12}")
for rel, note in VIDS:
    p = B/rel
    if not p.exists(): print(f"{rel} introuvable"); continue
    frames, fps = load_video(str(p), max_dim=720)
    if fps > 32:
        ft = np.arange(len(frames))*1000.0/fps; frames, _ = resample_to_fps(frames, ft, 30.0); fps = 30.0
    H, W = frames.shape[1:3]
    hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1,
                                     min_detection_confidence=0.5, min_tracking_confidence=0.5)
    det = 0; areas = []; lums = []
    try:
        for f in frames:
            r = hands.process(f)
            if not r.multi_hand_landmarks: continue
            det += 1; lm = r.multi_hand_landmarks[0].landmark
            pts = np.array([[lm[k].x*W, lm[k].y*H] for k in HULL_LM], np.int32)
            hull = cv2.convexHull(pts); mask = np.zeros((H, W), np.uint8)
            cv2.fillConvexPoly(mask, hull, 1)
            areas.append(100*mask.mean())
            sel = mask.astype(bool)
            if sel.sum() > 50: lums.append(f[sel].mean())
    finally:
        hands.close()
    mhr, ib = palm_motion(frames, fps)
    dr = det/len(frames)
    print(f"{Path(rel).stem[:20]:<22}{note:<18}{100*dr:>6.0f}%"
          f"{np.median(areas) if areas else 0:>7.1f}{np.mean(lums) if lums else 0:>8.0f}{ib:>12.3f}")
