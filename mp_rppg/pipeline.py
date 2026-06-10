"""
Extraction du signal RGB par régions faciales via MediaPipe FaceMesh.
Retourne un dict de tableaux (T, 3) pour chaque région.
"""

import cv2
import numpy as np

# Indices landmarks FaceMesh pour chaque région
_FRONT_IDX = [103, 67, 109, 10, 338, 297, 332, 333, 9, 104]
_LEFT_IDX  = [187, 206, 203, 101, 118, 117, 50]
_RIGHT_IDX = [411, 426, 423, 330, 347, 346, 280]

_KERNEL = np.ones((5, 5), np.uint8)


def _skin_mask(bgr):
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    return cv2.inRange(ycrcb,
                       np.array([0,   110,  60], np.uint8),
                       np.array([255, 190, 145], np.uint8))


def _poly_mask(shape, lm, indices):
    h, w = shape[:2]
    pts = np.array([[int(lm[i].x * w), int(lm[i].y * h)]
                    for i in indices], dtype=np.int32)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(mask, pts, 255)
    return mask


def _region_mean(bgr, mask):
    """RGB moyen de la région (polygone ∩ peau). None si < 50 px."""
    final = cv2.bitwise_and(mask, _skin_mask(bgr))
    px = bgr[final > 0]
    if len(px) < 50:
        return None
    b, g, r = np.mean(px[:, 0]), np.mean(px[:, 1]), np.mean(px[:, 2])
    return np.array([r, g, b], dtype=np.float32)


def extract_rgb(frames_rgb, verbose=True):
    """
    Extrait le signal RGB moyen sur front, joue gauche, joue droite et leur moyenne.

    Args:
        frames_rgb : (T, H, W, 3) uint8 RGB
        verbose    : affiche le % de détection

    Returns:
        dict avec clés 'front', 'left', 'right', 'mean' → chacun (T, 3) float32
    """
    import mediapipe as mp

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    buf = {k: [] for k in ('front', 'left', 'right')}
    n_ok = 0

    for frame in frames_rgb:
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        res = face_mesh.process(frame)

        if not res.multi_face_landmarks:
            for lst in buf.values():
                lst.append(None)
            continue

        n_ok += 1
        lm = res.multi_face_landmarks[0].landmark
        sh = frame.shape

        m_front = _poly_mask(sh, lm, _FRONT_IDX)
        m_left  = cv2.erode(_poly_mask(sh, lm, _LEFT_IDX),  _KERNEL, iterations=1)
        m_right = cv2.erode(_poly_mask(sh, lm, _RIGHT_IDX), _KERNEL, iterations=1)

        buf['front'].append(_region_mean(bgr, m_front))
        buf['left'].append(_region_mean(bgr, m_left))
        buf['right'].append(_region_mean(bgr, m_right))

    face_mesh.close()

    if verbose:
        pct = 100 * n_ok / max(len(frames_rgb), 1)
        print(f"    FaceMesh : {n_ok}/{len(frames_rgb)} frames détectées ({pct:.0f}%)")

    # Interpolation linéaire des frames manquantes
    result = {}
    for key, lst in buf.items():
        arr = np.full((len(lst), 3), np.nan, dtype=np.float32)
        for i, v in enumerate(lst):
            if v is not None:
                arr[i] = v
        for c in range(3):
            nans = np.isnan(arr[:, c])
            if nans.all():
                arr[:, c] = 0.0
            elif nans.any():
                idx = np.arange(len(arr))
                arr[:, c] = np.interp(idx, idx[~nans], arr[~nans, c])
        result[key] = arr

    result['mean'] = np.mean([result['front'], result['left'], result['right']], axis=0)
    return result
