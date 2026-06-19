"""
Évaluation de la méthode BKF (Bounded Kalman Filter) sur UBFC-rPPG
et vidéos personnelles.

Adapté du code original (Kalman_filter2.py + Main Program.py) pour
traitement batch offline :
  - MediaPipe FaceMesh à la place de dlib
  - geompreds.orient2d remplacé par un produit vectoriel numpy
  - cv2.KalmanFilter conservé (même logique bounded-Kalman)
  - Extraction du canal H de l'espace HSV sur 3 ROIs (front, joue G/D)
  - Signal processing : detrend + butterworth bandpass + FFT → HR

Usage :
    python scripts/eval_bkf_ubfc.py
    python scripts/eval_bkf_ubfc.py --subjects 1 10 11 13 32
    python scripts/eval_bkf_ubfc.py --personal-video results/personal_video/ROnel.mp4
"""

import argparse
import os
import sys

import mediapipe as mp   # avant cv2
import cv2
import imageio.v3 as iio
import numpy as np
from scipy.signal import butter, lfilter, detrend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mp_rppg.pipeline import _poly_mask, _FRONT_IDX, _LEFT_IDX, _RIGHT_IDX
from mp_rppg.metrics import snr, _next_pow2, hr_from_fft
from mp_rppg.methods import chrom, pos
from models.chrom_adaptive import bandpass_numpy, load_coefficients, DEHAAN_COEFFICIENTS
from scipy.signal import periodogram

WEIGHTS_PATH = os.path.join(ROOT, 'weights', 'chrom_adaptive_ubfc.pth')

UBFC_DIR = os.path.join(ROOT, 'Data')
OUT_DIR  = os.path.join(ROOT, 'results', 'ubfc_sample')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Remplacement de geompreds.orient2d ────────────────────────────────────────
def orient2d(a, b, c):
    """Produit vectoriel (b-a)×(c-a) : >0 si CCW, <0 si CW, 0 si colinéaires."""
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])

def ccw(a, b, c):
    return orient2d(a, b, c) > 0

def segments_intersect(a, b, c, d):
    return ccw(a,c,d) != ccw(b,c,d) and ccw(a,b,c) != ccw(a,b,d)

# ── Filtre de Kalman borné (version simplifiée 3 points) ──────────────────────
MEAS   = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
TRANS  = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
NOISE  = np.eye(4, dtype=np.float32) * 0.03

class BoundedKalman:
    """Kalman (4D state: x,y,vx,vy) avec prédiction bornée au bbox visage."""
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix  = MEAS.copy()
        self.kf.transitionMatrix   = TRANS.copy()
        self.kf.processNoiseCov    = NOISE.copy()
        self._initialized = False

    def update(self, pt):
        m = np.array([[np.float32(pt[0])], [np.float32(pt[1])]])
        self.kf.correct(m)
        if not self._initialized:
            self._initialized = True

    def predict(self, bbox):
        """Prédit la prochaine position, clampée au bbox (x1,y1,x2,y2)."""
        if not self._initialized:
            return None
        tp = self.kf.predict()
        x, y = int(tp[0]), int(tp[1])
        x1, y1, x2, y2 = bbox
        x = max(x1, min(x2, x))
        y = max(y1, min(y2, y))
        return (x, y)


# ── ROI : centroïde du polygone MediaPipe ─────────────────────────────────────
def poly_centroid(lm, indices, H, W):
    """Retourne (cx, cy) en pixels du centroïde du polygone."""
    pts = [(int(lm[i].x * W), int(lm[i].y * H)) for i in indices]
    arr = np.array(pts, dtype=np.float32)
    return int(arr[:,0].mean()), int(arr[:,1].mean())

def face_bbox(lm, H, W, pad=0.1):
    """Bounding box des landmarks, avec un padding relatif."""
    xs = [lm[i].x for i in range(len(lm))]
    ys = [lm[i].y for i in range(len(lm))]
    x1 = max(0, int((min(xs) - pad) * W))
    y1 = max(0, int((min(ys) - pad) * H))
    x2 = min(W, int((max(xs) + pad) * W))
    y2 = min(H, int((max(ys) + pad) * H))
    return x1, y1, x2, y2

def extract_h_patch(hsv_frame, cx, cy, half):
    """Moyenne circulaire du canal H (évite le wrap-around 179→0)."""
    H, W = hsv_frame.shape[:2]
    x1, x2 = max(0, cx-half), min(W, cx+half)
    y1, y2 = max(0, cy-half), min(H, cy+half)
    patch = hsv_frame[y1:y2, x1:x2, 0].astype(np.float32)
    if patch.size == 0:
        return 0.0
    angle = patch * (np.pi / 90.0)
    mean_angle = np.arctan2(np.mean(np.sin(angle)), np.mean(np.cos(angle)))
    if mean_angle < 0:
        mean_angle += 2 * np.pi
    return float(mean_angle * (90.0 / np.pi))

def extract_bgr_patch(bgr_frame, cx, cy, half):
    """Moyenne BGR sur un carré centré en (cx,cy) → retourné en ordre RGB."""
    H, W = bgr_frame.shape[:2]
    x1, x2 = max(0, cx-half), min(W, cx+half)
    y1, y2 = max(0, cy-half), min(H, cy+half)
    patch = bgr_frame[y1:y2, x1:x2].astype(np.float64)
    if patch.size == 0:
        return np.array([0., 0., 0.])
    return patch.mean(axis=(0, 1))[::-1]  # BGR → RGB


# ── Traitement du signal ───────────────────────────────────────────────────────
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return b, a

def bandpass_filter(signal, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return lfilter(b, a, signal)

def hr_from_bkf_signal(sig, fs, low_hz=0.75, high_hz=2.5):
    """FFT sur le signal filtré → pic dans [low_hz, high_hz]."""
    N = _next_pow2(len(sig))
    f, pxx = periodogram(sig, fs=fs, nfft=N, detrend=False)
    mask = (f >= low_hz) & (f <= high_hz)
    peak_freq = f[mask][np.argmax(pxx[mask])]
    return peak_freq * 60.0, f, pxx


# ── Pipeline BKF offline ───────────────────────────────────────────────────────
def run_bkf(video_path, mode='CHROM'):
    """
    mode='H'     : signal = canal H de HSV (méthode originale BKF)
    mode='CHROM' : signal = CHROM sur RGB extrait aux positions Kalman (hybride)
    mode='POS'   : signal = POS sur RGB extrait aux positions Kalman (hybride)
    """
    meta = iio.immeta(video_path, plugin='pyav')
    fps  = float(meta.get('fps', 30.0))

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    trackers = [BoundedKalman() for _ in range(3)]
    h_signal  = []   # pour mode H
    rgb_signal = []  # pour mode CHROM/POS
    n_ok, n_total = 0, 0

    for frame_rgb in iio.imiter(video_path, plugin='pyav'):
        n_total += 1
        H, W = frame_rgb.shape[:2]
        bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        res = face_mesh.process(frame_rgb)

        if res.multi_face_landmarks:
            n_ok += 1
            lm   = res.multi_face_landmarks[0].landmark
            bbox = face_bbox(lm, H, W)
            centers = [
                poly_centroid(lm, _FRONT_IDX, H, W),
                poly_centroid(lm, _LEFT_IDX,  H, W),
                poly_centroid(lm, _RIGHT_IDX, H, W),
            ]
            for tr, c in zip(trackers, centers):
                tr.update(c)
            extract_pts = centers
        else:
            bbox = (0, 0, W, H)
            preds = [tr.predict(bbox) for tr in trackers]
            extract_pts = [p if p else (W//2, H//2) for p in preds]

        x1b, _, x2b, _ = bbox
        half = max(5, (x2b - x1b) // 8)

        if mode == 'H':
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            h_vals = [extract_h_patch(hsv, cx, cy, half) for cx, cy in extract_pts]
            h_signal.append(np.mean(h_vals))
        else:
            rgb_vals = [extract_bgr_patch(bgr, cx, cy, half) for cx, cy in extract_pts]
            rgb_signal.append(np.mean(rgb_vals, axis=0))  # moyenne des 3 ROIs → (3,)

    face_mesh.close()
    print(f"  FaceMesh : {n_ok}/{n_total} frames détectées ({100*n_ok/n_total:.0f}%)")

    if mode == 'H':
        sig = np.array(h_signal, dtype=np.float64)
        sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-9)
        sig = detrend(sig)
        sig = bandpass_filter(sig, 0.75, 2.5, fps, order=5)
        return sig, fps
    else:
        rgb_arr = np.array(rgb_signal, dtype=np.float64)  # (T, 3)
        if mode == 'POS':
            bvp = pos(rgb_arr, fps)
        else:  # CHROM (défaut)
            bvp = chrom(rgb_arr, fps)
        return bvp, fps


# ── Chargement GT UBFC ────────────────────────────────────────────────────────
def load_gt_hr(subject_dir, fps):
    """Le fichier ground_truth.txt est un vecteur BVP sur une ligne.
    Le HR GT se calcule par FFT sur le signal bandpassé."""
    gt_file = os.path.join(subject_dir, 'ground_truth.txt')
    with open(gt_file) as f:
        bvp_gt = np.array([float(x) for x in f.read().strip().split('\n')[0].split()])
    hr_gt = hr_from_fft(bandpass_numpy(bvp_gt, fps), fps)
    return hr_gt


# ── Analyse multi-modes + graphique comparatif ────────────────────────────────
MODES   = ['H', 'CHROM', 'POS']
COLORS  = {'H': 'tomato', 'CHROM': 'steelblue', 'POS': 'darkorange'}

def analyze_and_plot(video_path, label, hr_gt=None, out_prefix=None):
    """Exécute BKF en mode H, CHROM et POS et génère un graphique comparatif."""
    results_by_mode = {}

    # Run les 3 modes (un seul passage vidéo par mode)
    for mode in MODES:
        print(f"  [{mode}] ", end='', flush=True)
        sig, fps = run_bkf(video_path, mode=mode)
        hr, f, pxx = hr_from_bkf_signal(sig, fps)
        s = snr(sig, hr, fps)
        err = abs(hr - hr_gt) if hr_gt is not None else None
        results_by_mode[mode] = {'sig': sig, 'fps': fps, 'hr': hr,
                                  'snr': s, 'err': err, 'f': f, 'pxx': pxx}
        line = f"HR={hr:.1f} bpm  SNR={s:.2f} dB"
        if err is not None:
            line += f"  Err={err:.1f} bpm"
        print(line)

    # Graphique 3×2 : signal temporel + spectre par mode
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    for i, mode in enumerate(MODES):
        r   = results_by_mode[mode]
        sig = r['sig']
        t   = np.arange(len(sig)) / r['fps']
        col = COLORS[mode]

        axes[i, 0].plot(t, sig, color=col, lw=0.8)
        axes[i, 0].set_title(f"BKF-{mode} — signal BVP "
                              f"(HR={r['hr']:.1f} bpm, SNR={r['snr']:.2f} dB)")
        axes[i, 0].set_xlabel("Temps (s)")
        axes[i, 0].set_ylabel("Amplitude")

        f_, pxx_ = r['f'], r['pxx']
        mask = (f_ >= 0.5) & (f_ <= 3.5)
        axes[i, 1].plot(f_[mask]*60, pxx_[mask], color=col, lw=1.2)
        axes[i, 1].axvline(r['hr'], color='black', lw=1.5, linestyle='--',
                           label=f"BKF-{mode} : {r['hr']:.1f} bpm")
        if hr_gt is not None:
            axes[i, 1].axvline(hr_gt, color='green', lw=1.5, linestyle=':',
                               label=f"GT : {hr_gt:.1f} bpm")
        axes[i, 1].set_title(f"BKF-{mode} — Spectre")
        axes[i, 1].set_xlabel("Fréquence (bpm)")
        axes[i, 1].set_ylabel("Puissance")
        axes[i, 1].set_xlim(30, 210)
        axes[i, 1].legend(fontsize=8)

    title = f"BKF (H vs CHROM vs POS) — {label}"
    if hr_gt:
        title += f"  [GT={hr_gt:.1f} bpm]"
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    slug = out_prefix or label.replace(' ', '_')
    out  = os.path.join(OUT_DIR, f"bkf_{slug}.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"  [Graphique] {out}")

    return results_by_mode


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subjects', nargs='*', default=['1', '10', '11', '13', '32'],
                        help='Numéros de sujets UBFC à tester')
    parser.add_argument('--personal-video', default=None,
                        help='Chemin vers une vidéo personnelle (sans GT)')
    args = parser.parse_args()

    print("=" * 70)
    print("  BKF — Bounded Kalman Filter — évaluation UBFC-rPPG")
    print("=" * 70)

    all_results = {}  # name → {mode → {hr, snr, err, ...}}

    # ── Sujets UBFC ───────────────────────────────────────────────────────────
    for sid in args.subjects:
        subj_dir   = os.path.join(UBFC_DIR, f'subject{sid}')
        video_path = os.path.join(subj_dir, 'vid.avi')
        if not os.path.exists(video_path):
            print(f"\n[SKIP] {video_path} introuvable")
            continue

        print(f"\n{'─'*50}\n  subject{sid}  |  {video_path}")
        meta  = iio.immeta(video_path, plugin='pyav')
        v_fps = float(meta.get('fps', 30.0))
        hr_gt = load_gt_hr(subj_dir, v_fps)
        print(f"  HR GT  : {hr_gt:.1f} bpm")

        r = analyze_and_plot(video_path, f"subject{sid}", hr_gt,
                             out_prefix=f"subject{sid}")
        all_results[f"subject{sid}"] = (hr_gt, r)

    # ── Vidéo personnelle (optionnelle) ───────────────────────────────────────
    if args.personal_video:
        vpath = args.personal_video
        if not os.path.exists(vpath):
            print(f"\n[SKIP] {vpath} introuvable")
        else:
            name = os.path.splitext(os.path.basename(vpath))[0]
            print(f"\n{'─'*50}\n  {name}  |  {vpath}  (pas de GT)")
            r = analyze_and_plot(vpath, name, hr_gt=None, out_prefix=name)
            all_results[name] = (None, r)

    # ── Tableau récap ─────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    header = f"  {'Vidéo':<16} {'GT':>6}"
    for m in MODES:
        header += f"  {'HR-'+m:>9} {'Err-'+m:>7} {'SNR-'+m:>7}"
    print(header)
    print(f"  {'-'*75}")
    for name, (hr_gt, r) in all_results.items():
        gt_s = f"{hr_gt:.1f}" if hr_gt else "—"
        row  = f"  {name:<16} {gt_s:>6}"
        for m in MODES:
            rm  = r[m]
            err_s = f"{rm['err']:.1f}" if rm['err'] is not None else "—"
            row += f"  {rm['hr']:>9.1f} {err_s:>7} {rm['snr']:>7.2f}"
        print(row)


if __name__ == '__main__':
    main()
