"""
ITA (Individual Typology Angle) normalisé par la SCLÈRE — robuste à l'éclairage.

Problème : l'ITA calculé sur la peau dépend autant de la lumière que de la
mélanine (même personne sous lumière chaude vs neutre → ITA très différent).

Solution (color constancy) : la sclère (blanc de l'œil) est ~neutre quelle que
soit la carnation ; sa couleur observée ≈ couleur de l'éclairage local. En
corrigeant la peau par la sclère (peau ÷ sclère × K), on retire la dominante
couleur ET la luminosité de la lumière → ITA représentatif de la pigmentation.

K calibré sur DataVital (Fitz 6 → ITA médian ≈ -30, conforme au seuil standard).
Repli automatique sur l'ITA brut si la sclère n'est pas assez détectée.
"""
import cv2
import numpy as np

from models.chrom_adaptive import compute_ita

K_DEFAULT = 120.0          # constante calibrée (réflectance sclère ~ K/255)
MIN_SCLERA_PCT = 0.5       # repli sur ITA brut si < 50% des images ont une sclère

# Contours des yeux (MediaPipe FaceMesh) — la sclère est à l'intérieur, hors iris
RIGHT_EYE = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
LEFT_EYE = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]
SKIN_FRONT = [103, 67, 109, 10, 338, 297, 332, 333, 168, 104]


def _poly_mask(shape, lm, idx):
    h, w = shape[:2]
    pts = np.array([[int(lm[i].x * w), int(lm[i].y * h)] for i in idx], np.int32)
    m = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(m, pts, 255)
    return m


def _sclera_rgb(bgr, lm):
    """Sclère = pixels clairs ET peu saturés (blanc) dans le contour des yeux."""
    mask = cv2.bitwise_or(_poly_mask(bgr.shape, lm, RIGHT_EYE),
                          _poly_mask(bgr.shape, lm, LEFT_EYE))
    inside = mask > 0
    if inside.sum() < 30:
        return None
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    S, V = hsv[..., 1], hsv[..., 2]
    v_thr = max(60, np.percentile(V[inside], 60))
    sel = inside & (V >= v_thr) & (S <= 60)
    if sel.sum() < 8:
        return None
    px = bgr[sel].astype(np.float32)
    return np.array([px[:, 2].mean(), px[:, 1].mean(), px[:, 0].mean()], np.float32)  # RGB


def _skin_rgb(bgr, lm):
    px = bgr[_poly_mask(bgr.shape, lm, SKIN_FRONT) > 0].astype(np.float32)
    if len(px) < 30:
        return None
    return np.array([px[:, 2].mean(), px[:, 1].mean(), px[:, 0].mean()], np.float32)  # RGB


def sclera_corrected_ita(frames, fm=None, skin_rgb_fallback=None, K=K_DEFAULT, step=15):
    """
    frames : (T,H,W,3) RGB uint8.
    fm     : instance MediaPipe FaceMesh (refine_landmarks=True). Créée si None.
    skin_rgb_fallback : RGB peau déjà calculé (ex. région BiSeNet) pour l'ITA brut
                        si la sclère échoue. Sinon on utilise la peau FaceMesh.
    Retourne dict : {ita, ita_raw, pct_sclera, used} (used = 'sclera' | 'raw').
    """
    import mediapipe as mp
    own_fm = fm is None
    if own_fm:
        fm = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
            refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    skins, scleras, n_seen = [], [], 0
    try:
        for f in frames[::step]:
            n_seen += 1
            res = fm.process(f)
            if not res.multi_face_landmarks:
                continue
            lm = res.multi_face_landmarks[0].landmark
            bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
            s = _skin_rgb(bgr, lm); c = _sclera_rgb(bgr, lm)
            if s is not None:
                skins.append(s)
            if c is not None:
                scleras.append(c)
    finally:
        if own_fm:
            fm.close()

    skin = np.mean(skins, 0) if skins else skin_rgb_fallback
    ita_raw = float(compute_ita(skin)) if skin is not None else float('nan')
    pct = len(scleras) / max(1, n_seen)
    if scleras and pct >= MIN_SCLERA_PCT and skin is not None:
        sclera = np.mean(scleras, 0)
        skin_norm = np.clip(skin / (sclera + 1e-6) * K, 0, 255)
        return {"ita": float(compute_ita(skin_norm)), "ita_raw": ita_raw,
                "pct_sclera": pct, "used": "sclera"}
    return {"ita": ita_raw, "ita_raw": ita_raw, "pct_sclera": pct, "used": "raw"}
