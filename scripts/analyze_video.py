"""
Script d'analyse rPPG sur une vidéo personnelle.
Applique CHROM et POS pour estimer la fréquence cardiaque.

Backends de détection :
  HC   : Haar Cascade OpenCV (rapide, CPU)
  Y5F  : YOLO5Face (réseau de neurones, GPU)
  MP   : MediaPipe FaceMesh (landmarks précis + masque peau, recommandé)

Usage :
  python scripts/analyze_video.py \\
      --video ma_video.avi --backend MP --debug --ref-hr 77
"""

import argparse
import math
import sys
import os
import numpy as np
import cv2
from scipy import signal, sparse

# ── Paramètres par défaut ─────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HAAR_XML = os.path.join(ROOT, "assets/haarcascade_frontalface_default.xml")
FACE_BOX_COEF = 1.5   # agrandissement boîte visage
RESIZE = 72           # crop redimensionné en RESIZE×RESIZE
LPF   = 0.7           # Hz  — fréquence cardiaque min (~42 bpm)
HPF   = 2.5           # Hz  — fréquence cardiaque max (~150 bpm)


# ── Lecture vidéo ─────────────────────────────────────────────────────────────
def read_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"[ERREUR] Impossible d'ouvrir : {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.asarray(frames, dtype=np.uint8), fps


# ── Détection visage — Haar Cascade ──────────────────────────────────────────
def detect_face_hc(frame, detector):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    zones = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(zones) == 0:
        return None
    idx = np.argmax(zones[:, 2])
    x, y, w, h = zones[idx]
    cx, cy = x + w // 2, y + h // 2
    half_w = int(w * FACE_BOX_COEF / 2)
    half_h = int(h * FACE_BOX_COEF / 2)
    H, W = frame.shape[:2]
    return (max(0, cx - half_w), max(0, cy - half_h),
            min(W, cx + half_w), min(H, cy + half_h))


# ── Détection visage — YOLO5Face ──────────────────────────────────────────────
def detect_face_y5f(frame, y5f_model):
    """Utilise YOLO5Face (réseau de neurones, plus précis que HC)."""
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    res = y5f_model.detect_face(bgr)   # retourne [x1,y1,x2,y2] ou None
    if res is None:
        return None
    x1, y1, x2, y2 = res
    # agrandissement centré identique à HC
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    w, h = x2 - x1, y2 - y1
    half_w = int(w * FACE_BOX_COEF / 2)
    half_h = int(h * FACE_BOX_COEF / 2)
    H, W = frame.shape[:2]
    return (max(0, cx - half_w), max(0, cy - half_h),
            min(W, cx + half_w), min(H, cy + half_h))


def crop_and_resize(frames, detector, backend="HC"):
    print(f"[1/4] Détection du visage (backend={backend})...")
    bbox = None
    for frame in frames:
        if backend == "Y5F":
            bbox = detect_face_y5f(frame, detector)
        else:
            bbox = detect_face_hc(frame, detector)
        if bbox is not None:
            break
    if bbox is None:
        print("  Aucun visage détecté — utilisation de la frame entière.")
        bbox = (0, 0, frames.shape[2], frames.shape[1])

    x1, y1, x2, y2 = bbox
    print(f"  Boîte visage : [{x1},{y1}] → [{x2},{y2}]")
    cropped = []
    for frame in frames:
        roi = frame[y1:y2, x1:x2]
        roi = cv2.resize(roi, (RESIZE, RESIZE), interpolation=cv2.INTER_AREA)
        cropped.append(roi)
    return np.asarray(cropped, dtype=np.float32)


# ── Moyenne spatiale simple (HC / Y5F) ───────────────────────────────────────
def process_video(frames):
    """(T, H, W, 3) → (T, 3) : RGB moyen par frame."""
    RGB = []
    for frame in frames:
        s = np.sum(np.sum(frame, axis=0), axis=0)
        RGB.append(s / (frame.shape[0] * frame.shape[1]))
    return np.asarray(RGB)


# ── MediaPipe FaceMesh — extraction RGB par régions ──────────────────────────
# Indices des landmarks FaceMesh pour chaque région
_FRONT_IDX  = [103, 67, 109, 10, 338, 297, 332, 333, 9, 104]
_LEFT_IDX   = [187, 206, 203, 101, 118, 117, 50]
_RIGHT_IDX  = [411, 426, 423, 330, 347, 346, 280]


def _skin_mask_ycrcb(bgr_frame):
    """Masque peau dans l'espace YCrCb — exclut cheveux, sourcils, fond."""
    ycrcb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2YCrCb)
    lower = np.array([0,   110,  60], dtype=np.uint8)
    upper = np.array([255, 190, 145], dtype=np.uint8)
    return cv2.inRange(ycrcb, lower, upper)


def _polygon_mask(shape, landmarks, indices):
    """Crée un masque binaire à partir d'un polygone de landmarks."""
    h, w = shape[:2]
    pts = np.array([[int(landmarks[i].x * w), int(landmarks[i].y * h)]
                    for i in indices], dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, pts, 255)
    return mask


def _region_mean_rgb(bgr_frame, poly_mask):
    """RGB moyen de la région (intersection polygone × masque peau)."""
    skin   = _skin_mask_ycrcb(bgr_frame)
    final  = cv2.bitwise_and(poly_mask, skin)
    pixels = bgr_frame[final > 0]          # shape (N_pixels, 3) en BGR
    if len(pixels) < 50:
        return None
    b = float(np.mean(pixels[:, 0]))
    g = float(np.mean(pixels[:, 1]))
    r = float(np.mean(pixels[:, 2]))
    return np.array([r, g, b])


def process_video_mediapipe(frames_rgb):
    """
    Extrait le signal RGB moyen sur 3 régions (front, joue gauche, joue droite)
    pour chaque frame via MediaPipe FaceMesh + masque peau YCrCb.

    Retourne un dict :
      'front'  : (T, 3) ou None si région manquante
      'left'   : (T, 3)
      'right'  : (T, 3)
      'mean'   : (T, 3)  moyenne des 3 régions disponibles
    """
    import mediapipe as mp
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    kernel = np.ones((5, 5), np.uint8)
    regions = {'front': [], 'left': [], 'right': []}
    n_detected = 0

    for frame_rgb in frames_rgb:
        bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        results = face_mesh.process(frame_rgb)   # MediaPipe attend du RGB

        if not results.multi_face_landmarks:
            for k in regions:
                regions[k].append(None)
            continue

        n_detected += 1
        lm = results.multi_face_landmarks[0].landmark
        sh = frame_rgb.shape

        # Polygone front
        m_front = _polygon_mask(sh, lm, _FRONT_IDX)
        # Polygones joues (érodés pour éviter les bords)
        m_left  = cv2.erode(_polygon_mask(sh, lm, _LEFT_IDX),  kernel, iterations=1)
        m_right = cv2.erode(_polygon_mask(sh, lm, _RIGHT_IDX), kernel, iterations=1)


        regions['front'].append(_region_mean_rgb(bgr, m_front))
        regions['left'].append(_region_mean_rgb(bgr, m_left))
        regions['right'].append(_region_mean_rgb(bgr, m_right))

    face_mesh.close()
    print(f"  FaceMesh détecté sur {n_detected}/{len(frames_rgb)} frames")

    # Interpole les frames sans détection (None → interpolation linéaire)
    result = {}
    for key, lst in regions.items():
        arr = np.full((len(lst), 3), np.nan)
        for i, v in enumerate(lst):
            if v is not None:
                arr[i] = v
        # Interpolation colonne par colonne
        for c in range(3):
            nans = np.isnan(arr[:, c])
            if nans.all():
                arr[:, c] = 0.0
            elif nans.any():
                idx = np.arange(len(arr))
                arr[:, c] = np.interp(idx, idx[~nans], arr[~nans, c])
        result[key] = arr

    # Moyenne des 3 régions
    result['mean'] = np.nanmean([result['front'], result['left'], result['right']], axis=0)
    return result


# ── Détrend de Whittaker ──────────────────────────────────────────────────────
def detrend(sig, lam=100):
    n = sig.shape[0]
    H = np.identity(n)
    ones = np.ones(n)
    D = sparse.spdiags(np.array([ones, -2 * ones, ones]), [0, 1, 2],
                       n - 2, n).toarray()
    return np.dot((H - np.linalg.inv(H + lam ** 2 * D.T @ D)), sig)


# ── CHROM ─────────────────────────────────────────────────────────────────────
def CHROM(frames, fs, RGB=None):
    """De Haan & Jeanne, IEEE TBME 2013.
    Si RGB est fourni (T,3), skip process_video (cas MediaPipe).
    """
    if RGB is None:
        RGB = process_video(frames)
    FN = RGB.shape[0]
    NyquistF = fs / 2
    B, A = signal.butter(3, [LPF / NyquistF, HPF / NyquistF], 'bandpass')
    WinSec = 1.6
    WinL = math.ceil(WinSec * fs)
    if WinL % 2:
        WinL += 1
    NWin = math.floor((FN - WinL // 2) / (WinL // 2))
    totallen = (WinL // 2) * (NWin + 1)
    S = np.zeros(totallen)
    WinS, WinM, WinE = 0, WinL // 2, WinL

    for _ in range(NWin):
        RGBBase = np.mean(RGB[WinS:WinE], axis=0)
        RGBNorm = RGB[WinS:WinE] / RGBBase
        Xs = 3 * RGBNorm[:, 0] - 2 * RGBNorm[:, 1]
        Ys = 1.5 * RGBNorm[:, 0] + RGBNorm[:, 1] - 1.5 * RGBNorm[:, 2]
        Xf = signal.filtfilt(B, A, Xs)
        Yf = signal.filtfilt(B, A, Ys)
        alpha = np.std(Xf) / np.std(Yf)
        SWin = (Xf - alpha * Yf) * signal.windows.hann(WinL)
        S[WinS:WinM] += SWin[:WinL // 2]
        S[WinM:WinE]  = SWin[WinL // 2:]
        WinS = WinM
        WinM = WinS + WinL // 2
        WinE = WinS + WinL
    return S


# ── POS ───────────────────────────────────────────────────────────────────────
def POS(frames, fs, RGB=None):
    """Wang et al., IEEE TBME 2017.
    Si RGB est fourni (T,3), skip process_video (cas MediaPipe).
    """
    if RGB is None:
        RGB = process_video(frames)
    N = RGB.shape[0]
    H = np.zeros(N)
    l = math.ceil(1.6 * fs)
    P = np.array([[0, 1, -1], [-2, 1, 1]])

    for n in range(N):
        m = n - l
        if m < 0:
            continue
        Cn = RGB[m:n] / np.mean(RGB[m:n], axis=0)
        S = P @ Cn.T                           # (2, l)
        h = S[0] + (np.std(S[0]) / np.std(S[1])) * S[1]
        H[m:n] += h - np.mean(h)

    H = detrend(H, 100)
    b, a = signal.butter(1, [0.75 / fs * 2, 3 / fs * 2], btype='bandpass')
    return signal.filtfilt(b, a, H.astype(np.float64))


# ── Estimation HR ─────────────────────────────────────────────────────────────
def estimate_hr_fft(bvp, fs, low=0.7, high=2.5):
    N = 1 if bvp.shape[0] == 0 else 2 ** (bvp.shape[0] - 1).bit_length()
    f, pxx = signal.periodogram(bvp, fs=fs, nfft=N, detrend=False)
    mask = (f >= low) & (f <= high)
    hr_hz = f[mask][np.argmax(pxx[mask])]
    return hr_hz * 60


def estimate_hr_peaks(bvp, fs):
    peaks, _ = signal.find_peaks(bvp, distance=int(fs * 0.4))  # min 0.4s entre pics (~150 bpm max)
    if len(peaks) < 2:
        return float('nan')
    return 60 / (np.mean(np.diff(peaks)) / fs)


def compute_snr(bvp, fs, hr_bpm, low=0.7, high=2.5):
    """SNR autour du fondamental + 2ème harmonique vs reste de la bande."""
    N = 2 ** (bvp.shape[0] - 1).bit_length()
    f, pxx = signal.periodogram(bvp, fs=fs, nfft=N, detrend=False)
    dev = 6 / 60  # ±6 bpm autour de chaque harmonique
    h1, h2 = hr_bpm / 60, 2 * hr_bpm / 60
    sig_mask = (((f >= h1 - dev) & (f <= h1 + dev)) |
                ((f >= h2 - dev) & (f <= h2 + dev)))
    noise_mask = ((f >= low) & (f <= high)) & ~sig_mask
    pxx = np.squeeze(pxx)
    sig_power = np.sum(pxx[sig_mask])
    noise_power = np.sum(pxx[noise_mask])
    if noise_power == 0:
        return 0.0
    return 10 * np.log10(sig_power / noise_power)


def get_spectrum(bvp, fs):
    N = 2 ** (bvp.shape[0] - 1).bit_length()
    f, pxx = signal.periodogram(bvp, fs=fs, nfft=N, detrend=False)
    return f, np.squeeze(pxx)


# ── Affichage des résultats ───────────────────────────────────────────────────
def print_results(name, bvp, fs, ref_hr=None):
    hr_fft   = estimate_hr_fft(bvp, fs)
    hr_peaks = estimate_hr_peaks(bvp, fs)
    snr      = compute_snr(bvp, fs, hr_fft)
    print(f"\n{'─'*50}")
    print(f"  Méthode : {name}")
    print(f"  HR  (FFT)       : {hr_fft:.1f} bpm")
    print(f"  HR  (pic-à-pic) : {hr_peaks:.1f} bpm")
    print(f"  SNR             : {snr:.2f} dB  {'✓ bon' if snr > 3 else '✗ bruité'}")
    if ref_hr:
        err = abs(hr_fft - ref_hr)
        print(f"  Erreur vs {ref_hr} bpm : {err:.1f} bpm")
    print(f"{'─'*50}")


# ── Debug : visualisation du crop et du spectre ───────────────────────────────
def save_debug(frames, bbox, bvp_chrom, bvp_pos, fps, backend, ref_hr=None):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    out_dir = os.path.join(ROOT, "results/personal_video")
    os.makedirs(out_dir, exist_ok=True)
    x1, y1, x2, y2 = bbox

    # ── Figure 1 : frame avec boîte détectée ──────────────────────────────────
    mid_frame = frames[len(frames) // 2]
    _, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.imshow(mid_frame)
    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                              linewidth=2, edgecolor='lime', facecolor='none')
    ax.add_patch(rect)
    ax.set_title(f"Détection {backend} — frame {len(frames)//2}\n"
                 f"Boîte : [{x1},{y1}]→[{x2},{y2}]  "
                 f"({x2-x1}×{y2-y1} px)")
    ax.axis('off')
    out_face = os.path.join(out_dir, f"debug_face_{backend}.png")
    plt.tight_layout()
    plt.savefig(out_face, dpi=150)
    plt.close()
    print(f"  Frame crop sauvegardée : {out_face}")

    # ── Figure 2 : signaux BVP temporels ──────────────────────────────────────
    t_c = np.arange(len(bvp_chrom)) / fps
    t_p = np.arange(len(bvp_pos))   / fps
    _, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=False)
    axes[0].plot(t_c, bvp_chrom, color='steelblue', lw=0.8)
    axes[0].set_title(f"CHROM BVP — HR≈{estimate_hr_fft(bvp_chrom, fps):.1f} bpm  "
                      f"SNR={compute_snr(bvp_chrom, fps, estimate_hr_fft(bvp_chrom, fps)):.1f} dB")
    axes[0].set_ylabel("Amplitude")
    axes[1].plot(t_p, bvp_pos, color='tomato', lw=0.8)
    axes[1].set_title(f"POS BVP — HR≈{estimate_hr_fft(bvp_pos, fps):.1f} bpm  "
                      f"SNR={compute_snr(bvp_pos, fps, estimate_hr_fft(bvp_pos, fps)):.1f} dB")
    axes[1].set_ylabel("Amplitude")
    axes[1].set_xlabel("Temps (s)")
    plt.tight_layout()
    out_bvp = os.path.join(out_dir, f"debug_bvp_{backend}.png")
    plt.savefig(out_bvp, dpi=150)
    plt.close()
    print(f"  Signaux BVP sauvegardés : {out_bvp}")

    # ── Figure 3 : spectres de fréquence ──────────────────────────────────────
    f_c, pxx_c = get_spectrum(bvp_chrom, fps)
    f_p, pxx_p = get_spectrum(bvp_pos,   fps)
    _, axes = plt.subplots(2, 1, figsize=(10, 6))

    for ax, f, pxx, bvp, name, color in [
        (axes[0], f_c, pxx_c, bvp_chrom, "CHROM", "steelblue"),
        (axes[1], f_p, pxx_p, bvp_pos,   "POS",   "tomato"),
    ]:
        mask = (f >= 0.5) & (f <= 3.5)
        ax.plot(f[mask] * 60, pxx[mask], color=color, lw=1.2)  # axe en bpm
        hr = estimate_hr_fft(bvp, fps)
        ax.axvline(hr, color='black', lw=1.5, linestyle='--', label=f"Pic détecté: {hr:.1f} bpm")
        if ref_hr:
            ax.axvline(ref_hr, color='green', lw=1.5, linestyle=':', label=f"Réf: {ref_hr} bpm")
        ax.set_title(f"{name} — Spectre de puissance")
        ax.set_xlabel("Fréquence (bpm)")
        ax.set_ylabel("Puissance")
        ax.legend(fontsize=8)
        ax.set_xlim(30, 210)

    plt.tight_layout()
    out_spec = os.path.join(out_dir, f"debug_spectrum_{backend}.png")
    plt.savefig(out_spec, dpi=150)
    plt.close()
    print(f"  Spectres sauvegardés    : {out_spec}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Analyse rPPG CHROM/POS")
    parser.add_argument("--video",   required=True, help="Chemin vers la vidéo")
    parser.add_argument("--fps",     type=float, default=None, help="FPS forcé")
    parser.add_argument("--backend", default="HC", choices=["HC", "Y5F", "MP"],
                        help="HC=Haar Cascade, Y5F=YOLO5Face, MP=MediaPipe FaceMesh")
    parser.add_argument("--plot",    action="store_true", help="Afficher les graphiques")
    parser.add_argument("--debug",   action="store_true",
                        help="Sauvegarde crop visage + spectres + BVP pour diagnostic")
    parser.add_argument("--ref-hr",  type=float, default=None,
                        help="Fréquence cardiaque de référence en bpm (ex: 77)")
    args = parser.parse_args()

    # 1. Lecture
    print(f"[0/4] Lecture de : {args.video}")
    frames, fps_detected = read_video(args.video)
    fps = args.fps if args.fps else fps_detected
    print(f"  {len(frames)} frames @ {fps:.1f} FPS  ({len(frames)/fps:.1f}s)")

    # ── Backend MediaPipe ─────────────────────────────────────────────────────
    if args.backend == "MP":
        print("[1/4] MediaPipe FaceMesh — extraction par régions (front, joues)...")
        mp_rgb = process_video_mediapipe(frames)

        regions = {
            'Front seul'       : mp_rgb['front'],
            'Joue gauche seule': mp_rgb['left'],
            'Joue droite seule': mp_rgb['right'],
            'Moyenne 3 régions': mp_rgb['mean'],
        }
        print("[2-3/4] CHROM + POS sur chaque région...")
        all_results = {}
        for name, rgb in regions.items():
            bvp_c = CHROM(None, fps, RGB=rgb)
            bvp_p = POS(None,   fps, RGB=rgb)
            all_results[name] = (bvp_c, bvp_p)

        print("[4/4] Résultats :")
        best_chrom_bvp, best_pos_bvp = None, None
        best_chrom_snr, best_pos_snr = -999, -999

        for name, (bvp_c, bvp_p) in all_results.items():
            hr_c  = estimate_hr_fft(bvp_c, fps)
            hr_p  = estimate_hr_fft(bvp_p, fps)
            snr_c = compute_snr(bvp_c, fps, hr_c)
            snr_p = compute_snr(bvp_p, fps, hr_p)
            err_c = f"  erreur={abs(hr_c-args.ref_hr):.1f} bpm" if args.ref_hr else ""
            err_p = f"  erreur={abs(hr_p-args.ref_hr):.1f} bpm" if args.ref_hr else ""
            print(f"\n  ── {name}")
            print(f"     CHROM : {hr_c:.1f} bpm  SNR={snr_c:.2f} dB{err_c}")
            print(f"     POS   : {hr_p:.1f} bpm  SNR={snr_p:.2f} dB{err_p}")
            if snr_c > best_chrom_snr:
                best_chrom_snr, best_chrom_bvp = snr_c, bvp_c
            if snr_p > best_pos_snr:
                best_pos_snr, best_pos_bvp = snr_p, bvp_p

        bvp_chrom, bvp_pos = best_chrom_bvp, best_pos_bvp
        bbox = (0, 0, frames.shape[2], frames.shape[1])   # frame entière pour debug

    # ── Backend HC / Y5F ─────────────────────────────────────────────────────
    else:
        if args.backend == "Y5F":
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from dataset.data_loader.face_detector.YOLO5Face import YOLO5Face
            detector = YOLO5Face(backend="Y5F")
            print("  YOLO5Face chargé (Y5sF_WFRGB.pth)")
        else:
            if not os.path.exists(HAAR_XML):
                sys.exit(f"[ERREUR] Haar Cascade introuvable : {HAAR_XML}")
            detector = cv2.CascadeClassifier(HAAR_XML)

        bbox = None
        for frame in frames:
            bbox = detect_face_y5f(frame, detector) if args.backend == "Y5F" \
                   else detect_face_hc(frame, detector)
            if bbox is not None:
                break
        if bbox is None:
            bbox = (0, 0, frames.shape[2], frames.shape[1])

        cropped = crop_and_resize(frames, detector, backend=args.backend)

        print("[2/4] CHROM...")
        bvp_chrom = CHROM(cropped, fps)
        print("[3/4] POS...")
        bvp_pos = POS(cropped, fps)

        print("[4/4] Résultats :")
        print_results("CHROM (De Haan 2013)", bvp_chrom, fps, args.ref_hr)
        print_results("POS   (Wang 2017)",    bvp_pos,   fps, args.ref_hr)

    # 5. Debug
    if args.debug or args.plot:
        try:
            save_debug(frames, bbox, bvp_chrom, bvp_pos, fps, args.backend, args.ref_hr)
        except ImportError:
            print("  matplotlib non disponible.")


if __name__ == "__main__":
    main()
