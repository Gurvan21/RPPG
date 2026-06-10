"""
Inférence PhysNet sur une vidéo personnelle.
Utilise les poids pré-entraînés du toolbox (SCAMPS, UBFC, PURE...).

Entrée PhysNet : (Batch, 3, 128, 72, 72) DiffNormalized — format NCDHW
Sortie         : BVP signal (T,) → estimation HR par FFT

Usage :
  python scripts/infer_physnet.py --video ma_video.avi --weights SCAMPS
  python scripts/infer_physnet.py --video ma_video.avi --weights UBFC --ref-hr 77
"""

import argparse
import os
import sys
import cv2
import numpy as np
from scipy.signal import periodogram

# ── PyTorch ────────────────────────────────────────────────────────────────────
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from models.physnet import PhysNet_padding_Encoder_Decoder_MAX

# ── Paramètres ─────────────────────────────────────────────────────────────────
HAAR_XML   = os.path.join(ROOT, "assets/haarcascade_frontalface_default.xml")
RESIZE     = 72
BOX_COEF   = 1.5
CHUNK_LEN  = 128   # PhysNet traite 128 frames à la fois
STRIDE     = 64    # chevauchement 50% pour signal plus lisse

WEIGHTS = {
    'SCAMPS': os.path.join(ROOT, 'weights/SCAMPS_PhysNet_DiffNormalized.pth'),
    'UBFC':   os.path.join(ROOT, 'weights/UBFC-rPPG_PhysNet_DiffNormalized.pth'),
}


# ── Lecture vidéo ──────────────────────────────────────────────────────────────
def read_video(path):
    """Lit la vidéo via imageio/pyav (plus robuste qu'OpenCV pour les AVI UBFC)."""
    try:
        import imageio.v3 as iio
        meta = iio.immeta(path, plugin='pyav')
        fps  = float(meta.get('fps', 30.0))
        frames = []
        for frame in iio.imiter(path, plugin='pyav'):
            frames.append(frame)          # déjà RGB avec pyav
        return np.asarray(frames, dtype=np.uint8), fps
    except Exception:
        # Fallback OpenCV
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return np.asarray(frames, dtype=np.uint8), float(fps)


# ── Détection et crop visage (Haar Cascade) ────────────────────────────────────
def detect_and_crop(frames):
    if not os.path.exists(HAAR_XML):
        print("  Haar XML introuvable — frame entière utilisée")
        h, w = frames[0].shape[:2]
        bbox = (0, 0, w, h)
    else:
        detector = cv2.CascadeClassifier(HAAR_XML)
        bbox = None
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            zones = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            if len(zones):
                x, y, w, h = zones[np.argmax(zones[:, 2])]
                cx, cy = x + w // 2, y + h // 2
                hw = int(w * BOX_COEF / 2)
                hh = int(h * BOX_COEF / 2)
                H, W = frame.shape[:2]
                bbox = (max(0, cx-hw), max(0, cy-hh),
                        min(W, cx+hw), min(H, cy+hh))
                break
        if bbox is None:
            h, w = frames[0].shape[:2]
            bbox = (0, 0, w, h)

    x1, y1, x2, y2 = bbox
    print(f"  Boîte visage HC : [{x1},{y1}]→[{x2},{y2}]")

    cropped = []
    for frame in frames:
        roi = cv2.resize(frame[y1:y2, x1:x2], (RESIZE, RESIZE),
                         interpolation=cv2.INTER_AREA).astype(np.float32)
        cropped.append(roi)
    return np.asarray(cropped)   # (T, 72, 72, 3)


# ── DiffNormalized ─────────────────────────────────────────────────────────────
def diff_normalize(data):
    """
    (T, H, W, 3) → (T, H, W, 3) DiffNormalized.
    Formule : (frame_{t+1} - frame_t) / (frame_{t+1} + frame_t + 1e-7)
    puis division par std globale.
    """
    T = len(data)
    out = np.zeros_like(data, dtype=np.float32)
    for t in range(T - 1):
        out[t] = (data[t+1] - data[t]) / (data[t+1] + data[t] + 1e-7)
    # dernière frame = 0 (padding)
    std = np.std(out[:-1])
    if std > 0:
        out = out / std
    out[np.isnan(out)] = 0
    return out


# ── Inférence PhysNet par chunks ───────────────────────────────────────────────
def run_physnet(frames_norm, model, device):
    """
    frames_norm : (T, H, W, 3) DiffNormalized float32
    Retourne    : (T,) BVP signal

    Traitement par fenêtres glissantes de CHUNK_LEN frames
    avec chevauchement STRIDE → overlap-add avec pondération Hann.
    """
    T = len(frames_norm)
    bvp_sum = np.zeros(T, dtype=np.float64)
    weight  = np.zeros(T, dtype=np.float64)
    hann    = np.hanning(CHUNK_LEN)

    starts = list(range(0, T - CHUNK_LEN + 1, STRIDE))
    if not starts:
        starts = [0]

    model.eval()
    with torch.no_grad():
        for s in starts:
            chunk = frames_norm[s:s+CHUNK_LEN]          # (128, 72, 72, 3)
            if len(chunk) < CHUNK_LEN:
                # pad si dernier chunk trop court
                pad = np.zeros((CHUNK_LEN - len(chunk),) + chunk.shape[1:],
                               dtype=np.float32)
                chunk = np.concatenate([chunk, pad], axis=0)

            # (128,72,72,3) → (1,3,128,72,72) NCDHW
            x = torch.from_numpy(chunk).permute(3, 0, 1, 2).unsqueeze(0).to(device)

            rppg, _, _, _ = model(x)
            pred = rppg.squeeze().cpu().numpy()          # (128,)

            end = min(s + CHUNK_LEN, T)
            length = end - s
            bvp_sum[s:end] += pred[:length] * hann[:length]
            weight[s:end]  += hann[:length]

    # Normalisation overlap-add
    mask = weight > 0
    bvp_sum[mask] /= weight[mask]
    return bvp_sum


# ── HR par FFT ─────────────────────────────────────────────────────────────────
def hr_fft(bvp, fs, low=0.7, high=2.5):
    N = 1 if len(bvp) == 0 else 2 ** (len(bvp) - 1).bit_length()
    f, pxx = periodogram(bvp, fs=fs, nfft=N, detrend=False)
    mask = (f >= low) & (f <= high)
    return float(f[mask][np.argmax(pxx[mask])] * 60)


# ── Sauvegarde graphiques ──────────────────────────────────────────────────────
def save_plots(bvp, fps, weights_name, ref_hr=None):
    import matplotlib.pyplot as plt
    from scipy.signal import periodogram

    out_dir = os.path.join(ROOT, "results/personal_video")
    os.makedirs(out_dir, exist_ok=True)
    hr = hr_fft(bvp, fps)
    t  = np.arange(len(bvp)) / fps

    fig, axes = plt.subplots(2, 1, figsize=(12, 6))

    # Signal temporel
    axes[0].plot(t, bvp, color='steelblue', lw=0.8)
    axes[0].set_title(f"PhysNet ({weights_name}) — BVP  |  HR≈{hr:.1f} bpm")
    axes[0].set_xlabel("Temps (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(alpha=0.3)

    # Spectre
    N = 2 ** (len(bvp) - 1).bit_length()
    f, pxx = periodogram(bvp, fs=fps, nfft=N, detrend=False)
    mask = (f >= 0.5) & (f <= 3.5)
    axes[1].plot(f[mask] * 60, pxx[mask], color='tomato', lw=1.2)
    axes[1].axvline(hr, color='black', lw=1.5, linestyle='--',
                    label=f"Détecté : {hr:.1f} bpm")
    if ref_hr:
        axes[1].axvline(ref_hr, color='green', lw=1.5, linestyle=':',
                        label=f"Réf : {ref_hr} bpm")
    axes[1].set_title("Spectre de puissance")
    axes[1].set_xlabel("Fréquence (bpm)")
    axes[1].set_ylabel("Puissance")
    axes[1].set_xlim(30, 180)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(out_dir, f"physnet_{weights_name}_bvp.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Graphique sauvegardé : {out}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Inférence PhysNet")
    parser.add_argument('--video',   required=True)
    parser.add_argument('--weights', default='SCAMPS',
                        choices=list(WEIGHTS.keys()),
                        help="Dataset d'entraînement des poids")
    parser.add_argument('--ref-hr',  type=float, default=None,
                        help="HR de référence en bpm")
    parser.add_argument('--cpu',     action='store_true',
                        help="Forcer CPU même si GPU disponible")
    args = parser.parse_args()

    device = torch.device('cpu') if args.cpu or not torch.cuda.is_available() \
             else torch.device('cuda')
    print(f"Device : {device}")

    # 1. Vidéo
    print(f"[1/5] Lecture : {args.video}")
    frames, fps = read_video(args.video)
    if fps == 0:
        fps = 30.0
    print(f"  {len(frames)} frames @ {fps:.1f} FPS  ({len(frames)/fps:.1f}s)")

    # 2. Crop visage
    print("[2/5] Détection et crop visage (HC)...")
    cropped = detect_and_crop(frames)     # (T, 72, 72, 3)

    # 3. DiffNormalized
    print("[3/5] DiffNormalized...")
    frames_norm = diff_normalize(cropped)

    # 4. Chargement modèle
    weights_path = WEIGHTS[args.weights]
    print(f"[4/5] Chargement PhysNet ({args.weights}) depuis {weights_path}...")
    model = PhysNet_padding_Encoder_Decoder_MAX(frames=CHUNK_LEN).to(device)
    ckpt  = torch.load(weights_path, map_location=device)
    model.load_state_dict(ckpt)
    print(f"  Poids chargés ({os.path.getsize(weights_path)//1024} Ko)")

    # 5. Inférence
    print(f"[5/5] Inférence PhysNet ({len(frames)} frames, chunks={CHUNK_LEN}, stride={STRIDE})...")
    bvp = run_physnet(frames_norm, model, device)

    hr = hr_fft(bvp, fps)
    print(f"\n{'─'*50}")
    print(f"  PhysNet ({args.weights})")
    print(f"  HR estimé (FFT) : {hr:.1f} bpm")
    if args.ref_hr:
        print(f"  HR référence    : {args.ref_hr} bpm")
        print(f"  Erreur          : {abs(hr - args.ref_hr):.1f} bpm")
    print(f"{'─'*50}")

    save_plots(bvp, fps, args.weights, args.ref_hr)


if __name__ == '__main__':
    main()
