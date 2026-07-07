"""
Détection d'ARTEFACT DE MOUVEMENT pour le rPPG palmaire.

Leçon des tests (2026-07-02) : un seuil de jitter global NE marche PAS (une main
qui bouge lentement/amplement peut garder un bon pouls → faux rejet). Le bon
critère : ne suspecter le mouvement QUE quand le pouls est BORDERLINE (SNR
faible-mais-positif) ET que la FC coïncide avec la fréquence du mouvement (ou son
harmonique). Un pouls fort (SNR élevé) est cru quel que soit le mouvement.
"""
import numpy as np
import cv2
import mediapipe as mp

from models.chrom_adaptive import bandpass_numpy
from mp_rppg.metrics import hr_from_fft

HULL_LM = [0, 1, 2, 5, 9, 13, 17]


def palm_motion(frames_rgb, fps):
    """Trajectoire de la paume (centre, normalisée par la taille) → fréquence de
    mouvement dominante (bpm) + amplitude dans la bande cardiaque (fraction de
    paume). Retourne (motion_hr, inband_amp) ; (nan, 0) si pas de main."""
    H, W = frames_rgb.shape[1:3]
    hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1,
                                     min_detection_confidence=0.5, min_tracking_confidence=0.5)
    T = len(frames_rgb); cx = np.full(T, np.nan); cy = np.full(T, np.nan); sc = np.full(T, np.nan)
    try:
        for t, f in enumerate(frames_rgb):
            r = hands.process(f)
            if not r.multi_hand_landmarks:
                continue
            lm = r.multi_hand_landmarks[0].landmark
            pts = np.array([[lm[k].x*W, lm[k].y*H] for k in HULL_LM], np.float32)
            cx[t], cy[t] = pts.mean(0)
            sc[t] = np.linalg.norm([lm[0].x*W-lm[9].x*W, lm[0].y*H-lm[9].y*H])
    finally:
        hands.close()
    if np.isfinite(cx).sum() < 10:
        return float('nan'), 0.0
    for a in (cx, cy, sc):
        g = np.isfinite(a)
        a[~g] = np.interp(np.where(~g)[0], np.where(g)[0], a[g]) if g.sum() > 1 else 0
    scale = np.median(sc) + 1e-6
    dx = (cx - cx.mean())/scale; dy = (cy - cy.mean())/scale
    ibx = bandpass_numpy(dx, fps); iby = bandpass_numpy(dy, fps)
    inband = float(np.sqrt(np.mean(ibx**2) + np.mean(iby**2)))
    motion_hr = float(hr_from_fft(ibx + iby, fps))
    return motion_hr, inband


def is_motion_artifact(hr, snr_val, motion_hr, inband_amp,
                       snr_strong=1.5, tol_bpm=10.0, inband_min=0.006):
    """True si la FC est probablement un ARTEFACT de mouvement.
    - pouls fort (snr>=snr_strong) → jamais suspect (on le croit),
    - mouvement in-band négligeable → jamais suspect,
    - sinon suspect si FC ≈ fréquence de mouvement (ou 2×, ou ×½)."""
    if not np.isfinite(motion_hr) or snr_val >= snr_strong or inband_amp < inband_min:
        return False
    near = min(abs(hr - motion_hr), abs(hr - 2*motion_hr), abs(2*hr - motion_hr)) < tol_bpm
    return bool(near)
