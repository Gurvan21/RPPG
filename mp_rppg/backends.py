"""
Backends HC et Y5F : détection visage → crop → moyenne spatiale RGB (T,3).
Même logique que BaseLoader du toolbox, factorisée ici pour comparaison.
"""

import os
import sys
import cv2
import numpy as np

_ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, _ROOT)

HAAR_XML = os.path.join(_ROOT, "assets/haarcascade_frontalface_default.xml")
RESIZE    = 72
BOX_COEF  = 1.5


def _expand_box(x, y, w, h, frame_hw, coef=BOX_COEF):
    H, W = frame_hw
    cx, cy = x + w // 2, y + h // 2
    hw, hh = int(w * coef / 2), int(h * coef / 2)
    return (max(0, cx - hw), max(0, cy - hh),
            min(W, cx + hw), min(H, cy + hh))


# ── Haar Cascade ──────────────────────────────────────────────────────────────
def _detect_hc(gray, detector):
    zones = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if not len(zones):
        return None
    x, y, w, h = zones[np.argmax(zones[:, 2])]
    return _expand_box(x, y, w, h, gray.shape)


def extract_rgb_hc(frames_rgb):
    """
    Haar Cascade : détection sur la première frame disponible,
    crop statique sur toute la vidéo, retourne dict {'face': (T,3)}.
    """
    if not os.path.exists(HAAR_XML):
        raise FileNotFoundError(f"Haar XML introuvable : {HAAR_XML}")

    detector = cv2.CascadeClassifier(HAAR_XML)
    bbox = None
    for frame in frames_rgb:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        bbox = _detect_hc(gray, detector)
        if bbox is not None:
            break

    if bbox is None:
        print("    [HC] Aucun visage détecté — frame entière utilisée")
        h, w = frames_rgb[0].shape[:2]
        bbox = (0, 0, w, h)

    x1, y1, x2, y2 = bbox
    print(f"    [HC] Boîte : [{x1},{y1}]→[{x2},{y2}]")

    rgb_list = []
    for frame in frames_rgb:
        crop = cv2.resize(frame[y1:y2, x1:x2], (RESIZE, RESIZE),
                          interpolation=cv2.INTER_AREA).astype(np.float32)
        rgb_list.append(crop.mean(axis=(0, 1)))   # moyenne spatiale
    return {'face': np.array(rgb_list)}


# ── YOLO5Face ─────────────────────────────────────────────────────────────────
def _detect_y5f(frame_bgr, model):
    res = model.detect_face(frame_bgr)
    if res is None:
        return None
    x1, y1, x2, y2 = res
    w, h = x2 - x1, y2 - y1
    return _expand_box(x1, y1, w, h, frame_bgr.shape[:2])


def extract_rgb_y5f(frames_rgb):
    """
    YOLO5Face : idem HC mais avec réseau de neurones.
    Retourne dict {'face': (T,3)}.
    """
    try:
        from dataset.data_loader.face_detector.YOLO5Face import YOLO5Face
    except ImportError as e:
        raise ImportError(f"YOLO5Face non disponible : {e}")

    model = YOLO5Face(backend="Y5F")
    bbox = None
    for frame in frames_rgb:
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        bbox = _detect_y5f(bgr, model)
        if bbox is not None:
            break

    if bbox is None:
        print("    [Y5F] Aucun visage détecté — frame entière utilisée")
        h, w = frames_rgb[0].shape[:2]
        bbox = (0, 0, w, h)

    x1, y1, x2, y2 = bbox
    print(f"    [Y5F] Boîte : [{x1},{y1}]→[{x2},{y2}]")

    rgb_list = []
    for frame in frames_rgb:
        crop = cv2.resize(frame[y1:y2, x1:x2], (RESIZE, RESIZE),
                          interpolation=cv2.INTER_AREA).astype(np.float32)
        rgb_list.append(crop.mean(axis=(0, 1)))
    return {'face': np.array(rgb_list)}
