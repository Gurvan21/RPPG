#!/usr/bin/env python3
"""
rPPG sur la PAUME via MediaPipe Hands.

Hypothèse d'équité : la paume est la peau la MOINS pigmentée du corps, même chez
les sujets Fitzpatrick 5-6 → moins de mélanine absorbe le signal → un rPPG
potentiellement MEILLEUR que sur le visage pour les carnations foncées.

On extrait le signal RGB moyen de la région palmaire (polygone des landmarks
0,1,5,9,13,17 = poignet + base des doigts), légèrement érodé pour éviter le fond,
puis CHROM/POS → HR + SNR. À comparer au visage sur la MÊME vidéo facepalm.
"""
import cv2
import numpy as np
import mediapipe as mp

# landmarks délimitant la paume (poignet + bases des doigts)
PALM_LM = [0, 1, 2, 5, 9, 13, 17]


def extract_palm_rgb(frames, min_det=0.5):
    """frames : (T,H,W,3) RGB. Retourne (rgb (T,3) avec NaN si pas de main,
    det_rate, mask_area_frac_moyenne)."""
    hands = mp.solutions.hands.Hands(
        static_image_mode=False, max_num_hands=1,
        min_detection_confidence=min_det, min_tracking_confidence=0.5)
    H, W = frames.shape[1:3]
    out = np.full((len(frames), 3), np.nan, np.float32)
    areas = []
    try:
        for i, f in enumerate(frames):
            res = hands.process(f)  # mediapipe veut du RGB
            if not res.multi_hand_landmarks:
                continue
            lm = res.multi_hand_landmarks[0].landmark
            pts = np.array([[lm[k].x * W, lm[k].y * H] for k in PALM_LM], np.float32)
            hull = cv2.convexHull(pts.astype(np.int32))
            mask = np.zeros((H, W), np.uint8)
            cv2.fillConvexPoly(mask, hull, 1)
            # éroder pour rester sur la peau (pas les bords/fond)
            k = max(3, int(0.02 * min(H, W)) | 1)
            mask = cv2.erode(mask, np.ones((k, k), np.uint8))
            if mask.sum() < 200:
                continue
            sel = mask.astype(bool)
            px = f[sel].astype(np.float32)
            # exclure pixels trop sombres/clairs (ombres, spéculaire)
            lum = px.mean(1)
            keep = (lum > 30) & (lum < 245)
            if keep.sum() < 100:
                continue
            out[i] = px[keep].mean(0)
            areas.append(sel.mean())
    finally:
        hands.close()
    det = np.isfinite(out[:, 0]).mean()
    return out, float(det), (float(np.mean(areas)) if areas else 0.0)


def interp_nan(rgb):
    """Interpole linéairement les frames sans main détectée (par canal)."""
    T = len(rgb)
    idx = np.arange(T)
    out = rgb.copy()
    for c in range(3):
        good = np.isfinite(rgb[:, c])
        if good.sum() < 2:
            return None
        out[:, c] = np.interp(idx, idx[good], rgb[good, c])
    return out
