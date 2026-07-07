#!/usr/bin/env python3
"""
Validation de la normalisation d'éclairage par sclère (blanc de l'œil) pour
le calcul de l'ITA (Individual Typology Angle).

Idée : le blanc de l'œil est ~blanc quelle que soit la carnation. Sa couleur
observée = couleur de l'éclairage local. En divisant le RGB de la peau par le
RGB de la sclère, on retire la dominante couleur ET la luminosité de la lumière
→ ITA plus représentatif de la vraie pigmentation, robuste à l'éclairage.

Ce script compare, sur quelques sujets DataVital :
  - ITA brut          (depuis RGB peau moyen)
  - ITA normalisé-sclère (depuis RGB peau / RGB sclère)
et reporte la fiabilité de l'estimation de sclère (nb de frames où détectée).

Usage :
    python scripts/test_sclera_ita.py --range 1 8
"""

import argparse
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.chrom_adaptive import compute_ita
from scripts.preextract_clips import _find_vitalvideos_json, load_video

# Contours des yeux (FaceMesh) — la sclère est à l'intérieur, hors iris
RIGHT_EYE = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
LEFT_EYE  = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]
# Joues / front pour le RGB peau (réutilise les régions validées)
SKIN_FRONT = [103, 67, 109, 10, 338, 297, 332, 333, 168, 104]


def poly_mask(shape, lm, idx):
    h, w = shape[:2]
    pts = np.array([[int(lm[i].x * w), int(lm[i].y * h)] for i in idx], np.int32)
    m = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(m, pts, 255)
    return m


def sclera_rgb(bgr, lm):
    """RGB de la sclère = pixels DÉSATURÉS et clairs (blanc) dans le contour
    des yeux. Le blanc se distingue de la peau par sa faible saturation (HSV),
    pas juste sa luminance. Retourne (R,G,B) ou None."""
    mask = cv2.bitwise_or(poly_mask(bgr.shape, lm, RIGHT_EYE),
                          poly_mask(bgr.shape, lm, LEFT_EYE))
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    inside = mask > 0
    if inside.sum() < 30:
        return None
    H = hsv[..., 0]; S = hsv[..., 1]; V = hsv[..., 2]
    v_in = V[inside]
    v_thr = max(60, np.percentile(v_in, 60))      # assez clair (relatif à l'œil)
    # sclère = dans l'œil ET clair ET peu saturé (blanc/gris)
    sel = inside & (V >= v_thr) & (S <= 60)
    if sel.sum() < 8:
        return None
    px = bgr[sel].astype(np.float32)
    b, g, r = px[:, 0].mean(), px[:, 1].mean(), px[:, 2].mean()
    return np.array([r, g, b], np.float32)


def skin_rgb(bgr, lm):
    px = bgr[poly_mask(bgr.shape, lm, SKIN_FRONT) > 0].astype(np.float32)
    if len(px) < 30:
        return None
    b, g, r = px[:, 0].mean(), px[:, 1].mean(), px[:, 2].mean()
    return np.array([r, g, b], np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--range', nargs=2, type=int, default=[1, 8])
    ap.add_argument('--step', type=int, default=15, help="1 frame sur N (échantillonnage)")
    args = ap.parse_args()

    fm = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)

    print(f"{'Sujet':<11}{'Fitz':>5}{'skinRGB':>16}{'scleraRGB':>17}"
          f"{'ITA_brut':>9}{'ITA_norm':>9}{'%sclère':>9}")
    print('-' * 76)

    import json
    for i in range(args.range[0], args.range[1] + 1):
        name = f"Subject {i}"
        sd = ROOT / 'DataVital' / name
        meta = _find_vitalvideos_json(sd)
        if meta is None:
            continue
        fz = json.load(open([j for j in sd.glob('*.json') if j.name != 'metadata.json'][0]))['participant']['fitzpatrick']
        rgb_meta = meta['scenarios'][0]['recordings'].get('RGB', {})
        vid = sd / rgb_meta.get('filename', '')
        if not vid.exists():
            continue
        frames, _ = load_video(vid)

        skins, scleras, n_scl = [], [], 0
        for f in frames[::args.step]:
            bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
            res = fm.process(f)
            if not res.multi_face_landmarks:
                continue
            lm = res.multi_face_landmarks[0].landmark
            sk = skin_rgb(bgr, lm); sc = sclera_rgb(bgr, lm)
            if sk is not None:
                skins.append(sk)
            if sc is not None:
                scleras.append(sc); n_scl += 1
        if not skins:
            continue
        skin = np.mean(skins, axis=0)
        ita_raw = compute_ita(skin)
        if scleras:
            sclera = np.mean(scleras, axis=0)
            # normalisation : peau / sclère, rescale sur [0,255] (sclère ~= blanc 255)
            skin_norm = np.clip(skin / (sclera + 1e-6) * 255.0, 0, 255)
            ita_norm = compute_ita(skin_norm)
            scl_str = f"({sclera[0]:.0f},{sclera[1]:.0f},{sclera[2]:.0f})"
            pct = 100 * n_scl / max(1, len(frames[::args.step]))
        else:
            ita_norm, scl_str, pct = float('nan'), "—", 0
        print(f"{name:<11}{fz:>5}  ({skin[0]:.0f},{skin[1]:.0f},{skin[2]:.0f})"
              f"{scl_str:>17}{ita_raw:>9.0f}{ita_norm:>9.0f}{pct:>8.0f}%")
    fm.close()


if __name__ == '__main__':
    main()
