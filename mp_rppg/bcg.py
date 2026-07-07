"""
rBCG — remote BallistoCardioGraphy (Balakrishnan et al., CVPR 2013).

Mesure la FC via les MICRO-MOUVEMENTS de la tête provoqués par l'éjection du
sang à chaque battement (recul newtonien). Signal MÉCANIQUE → INDÉPENDANT de la
couleur de peau → complémentaire du rPPG (marche là où l'optique échoue sur peau
foncée). Pas d'apprentissage : pur traitement du signal.

Pipeline :
  1. Détecter des points stables (front + nez) dans la 1re image.
  2. Les suivre image par image (flux optique Lucas-Kanade) → trajectoires Y.
  3. Filtrer dans la bande cardiaque (0.7–2.5 Hz) chaque trajectoire.
  4. PCA → garder la composante la plus PÉRIODIQUE (SNR spectral max).
  5. FFT → FC ; + SNR aveugle.

Limite : exige une tête immobile et une caméra stable (assis/trépied). Le
mouvement volontaire (parole, rotation) domine le micro-mouvement cardiaque.
"""
import cv2
import numpy as np
from scipy.signal import butter, filtfilt

from mp_rppg.metrics import hr_from_fft, snr


def _bp(x, fs, lo, hi, order=4):
    nyq = fs / 2.0
    b, a = butter(order, [max(lo / nyq, 1e-3), min(hi / nyq, 0.999)], btype='band')
    return filtfilt(b, a, x, axis=0)


def bcg_hr(frames, fps, bbox=None, lo=0.7, hi=2.5, max_pts=120):
    """
    frames : (T,H,W,3) RGB uint8.
    bbox   : (x1,y1,x2,y2) du visage (optionnel) ; sinon ROI centrale-haute.
    Retourne (hr_bpm, snr_db, signal) — (nan,-99,None) si échec.
    """
    T = len(frames)
    if T < int(3 * fps):
        return float('nan'), -99.0, None
    gray = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    H, W = gray[0].shape

    # ROI rigide = front + arête du nez (haut-centre du visage), hors yeux/bouche
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        fw, fh = x2 - x1, y2 - y1
        rx0, rx1 = int(x1 + 0.25 * fw), int(x1 + 0.75 * fw)
        ry0, ry1 = int(y1 + 0.05 * fh), int(y1 + 0.45 * fh)
    else:
        rx0, rx1 = int(W * 0.30), int(W * 0.70)
        ry0, ry1 = int(H * 0.12), int(H * 0.50)
    mask = np.zeros((H, W), np.uint8); mask[ry0:ry1, rx0:rx1] = 255

    p0 = cv2.goodFeaturesToTrack(gray[0], maxCorners=max_pts, qualityLevel=0.01,
                                 minDistance=7, mask=mask)
    if p0 is None or len(p0) < 5:
        return float('nan'), -99.0, None

    # suivi Lucas-Kanade sur toute la séquence
    traj = np.zeros((T, len(p0), 2), np.float32); traj[0] = p0.reshape(-1, 2)
    valid = np.ones(len(p0), bool); pts = p0
    lk = dict(winSize=(15, 15), maxLevel=2,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
    for t in range(1, T):
        pts2, st, _ = cv2.calcOpticalFlowPyrLK(gray[t - 1], gray[t], pts, None, **lk)
        st = st.reshape(-1).astype(bool)
        valid &= st
        traj[t] = pts2.reshape(-1, 2); pts = pts2

    Y = traj[:, valid, 1]                       # positions verticales (T, N)
    if Y.shape[1] < 5:
        return float('nan'), -99.0, None

    # rejeter les points trop agités (mouvement non-cardiaque) : garder la moitié
    # la plus stable (faible amplitude max après détrend)
    Yc = Y - Y.mean(0)
    amp = np.abs(Yc).max(0)
    keep = amp <= np.median(amp)               # points stables
    Yc = Yc[:, keep] if keep.sum() >= 5 else Yc

    Yf = _bp(Yc, fps, lo, hi)                   # bande cardiaque par trajectoire
    # PCA (SVD) sur les trajectoires filtrées
    Yf = Yf - Yf.mean(0)
    U, S, Vt = np.linalg.svd(Yf, full_matrices=False)

    # composante la plus périodique (SNR aveugle max) parmi les 5 premières
    best_sig, best_snr = None, -np.inf
    for k in range(min(5, U.shape[1])):
        comp = U[:, k]
        h = hr_from_fft(comp, fps, low=lo, high=hi)
        s = snr(comp, h, fps, low=lo, high=hi)
        if s > best_snr:
            best_snr, best_sig = s, comp
    hr = hr_from_fft(best_sig, fps, low=lo, high=hi)
    return float(hr), float(best_snr), best_sig
