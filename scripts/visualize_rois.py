"""
Visualise les régions ROI MediaPipe (front, joue gauche, joue droite)
sur plusieurs frames de la vidéo et les sauvegarde en PNG.

Usage :
  python scripts/visualize_rois.py --video ma_video.avi
"""

import argparse
import cv2
import numpy as np
import mediapipe as mp
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "results/personal_video")
FRAMES_TO_SHOW = [50, 200, 400, 600, 800]   # frames à visualiser

# ── Indices landmarks (identiques à analyze_video.py) ────────────────────────
FRONT = [103, 67, 109, 10, 338, 297, 332, 333, 9, 104]
LEFT  = [187, 206, 203, 101, 118, 117, 50]
RIGHT = [411, 426, 423, 330, 347, 346, 280]


def skin_mask_ycrcb(bgr):
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    return cv2.inRange(ycrcb,
                       np.array([0,  100,  40], np.uint8),
                       np.array([255, 200, 155], np.uint8))


def polygon_mask(shape, lm, indices):
    h, w = shape[:2]
    pts = np.array([[int(lm[i].x * w), int(lm[i].y * h)]
                    for i in indices], dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, pts, 255)
    return mask


def draw_rois_on_frame(bgr_frame, face_mesh):
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)

    overlay = bgr_frame.copy()
    info = {}

    if not res.multi_face_landmarks:
        cv2.putText(overlay, "AUCUN VISAGE DETECTE", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return overlay, info

    lm = res.multi_face_landmarks[0].landmark
    kernel = np.ones((5, 5), np.uint8)

    m_front = polygon_mask(bgr_frame.shape, lm, FRONT)
    m_left  = cv2.erode(polygon_mask(bgr_frame.shape, lm, LEFT),  kernel, 1)
    m_right = cv2.erode(polygon_mask(bgr_frame.shape, lm, RIGHT), kernel, 1)
    skin    = skin_mask_ycrcb(bgr_frame)

    # Couleurs : front=vert, joue gauche=bleu, joue droite=rouge
    # Zone peau seule (intersection avec masque peau)
    for mask, color, label in [
        (m_front,                                  (0,  200,  0),  "Front"),
        (cv2.bitwise_and(m_left,  skin),           (255,  80, 80), "Joue G"),
        (cv2.bitwise_and(m_right, skin),           (80,  80, 255), "Joue D"),
    ]:
        colored = np.zeros_like(bgr_frame)
        colored[mask > 0] = color
        cv2.addWeighted(colored, 0.45, overlay, 1.0, 0, overlay)

        # Contour polygone
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 2)

        # Compter pixels utilisables
        n = int(np.sum(mask > 0))
        info[label] = n

    # Légende
    legend = [
        ("Front (vert)",       (0, 200, 0),   info.get("Front",  0)),
        ("Joue gauche (bleu)", (255, 80, 80),  info.get("Joue G", 0)),
        ("Joue droite (rouge)",(80,  80, 255), info.get("Joue D", 0)),
    ]
    for i, (txt, col, n) in enumerate(legend):
        cv2.rectangle(overlay, (10, 10 + i*28), (26, 26 + i*28), col, -1)
        cv2.putText(overlay, f"{txt}  ({n} px)", (32, 24 + i*28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)

    return overlay, info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True, help="Chemin vers la vidéo")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[ERREUR] Impossible d'ouvrir {args.video}")
        return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    print(f"Vidéo : {total} frames @ {fps:.1f} FPS")

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    frames_drawn = []
    target_set = set(FRAMES_TO_SHOW)
    frame_idx = 0

    while frame_idx <= max(FRAMES_TO_SHOW):
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in target_set:
            t = frame_idx / fps
            annotated, info = draw_rois_on_frame(frame, face_mesh)
            # Titre en haut
            cv2.putText(annotated, f"Frame {frame_idx}  ({t:.1f}s)",
                        (frame.shape[1]//2 - 100, frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
            frames_drawn.append(annotated)
            print(f"  Frame {frame_idx:4d} ({t:.1f}s) — {info}")
        frame_idx += 1

    cap.release()
    face_mesh.close()

    if not frames_drawn:
        print("Aucune frame traitée.")
        return

    # Assembler en grille (2 colonnes)
    cols = min(len(frames_drawn), 3)
    rows = (len(frames_drawn) + cols - 1) // cols
    h, w = frames_drawn[0].shape[:2]
    grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)

    for i, img in enumerate(frames_drawn):
        r, c = divmod(i, cols)
        grid[r*h:(r+1)*h, c*w:(c+1)*w] = img

    # Redimensionner si trop grand
    max_dim = 1400
    if grid.shape[1] > max_dim:
        scale = max_dim / grid.shape[1]
        grid = cv2.resize(grid, (max_dim, int(grid.shape[0] * scale)))

    out_path = os.path.join(OUT_DIR, "rois_mediapipe.png")
    cv2.imwrite(out_path, grid)
    print(f"\nImage sauvegardée : {out_path}")


if __name__ == "__main__":
    main()
