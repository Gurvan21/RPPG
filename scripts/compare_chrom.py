"""
Compare CHROM De Haan (coefficients fixes) et CHROM adaptatif (coefficients
appris) sur une vidéo : extraction RGB via MediaPipe, HR par FFT pour les
deux variantes, et affichage des coefficients utilisés.

Usage :
    python scripts/compare_chrom.py --video ma_video.mp4

    # avec un modèle entraîné, sélectionné automatiquement par phototype
    python scripts/compare_chrom.py --video ma_video.mp4 \\
        --model weights/ --skin-type 5

    # région du visage utilisée (front, left, right, mean)
    python scripts/compare_chrom.py --video ma_video.mp4 --region mean
"""

import argparse
import os
import sys

import cv2
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, ROOT)

from models.chrom_adaptive import load_coefficients
from mp_rppg.methods import chrom, chrom_adaptive
from mp_rppg.metrics import hr_from_fft
from mp_rppg.pipeline import extract_rgb


def read_video(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.asarray(frames, dtype=np.uint8), float(fps)


def main():
    parser = argparse.ArgumentParser(
        description="Compare CHROM De Haan vs CHROM adaptatif sur une vidéo")
    parser.add_argument('--video', required=True, help="Chemin de la vidéo")
    parser.add_argument('--region', default='front',
                        choices=['front', 'left', 'right', 'mean'],
                        help="Région faciale MediaPipe utilisée (défaut : front)")
    parser.add_argument('--model', default=None,
                        help="Checkpoint .pth ou dossier de checkpoints CHROM "
                             "adaptatif (défaut : coefficients De Haan)")
    parser.add_argument('--skin-type', type=int, choices=range(1, 7), default=None,
                        help="Phototype (1-6) pour sélectionner le checkpoint "
                             "si --model est un dossier")
    parser.add_argument('--config', default=None,
                        help="Config YAML (MODEL.SAVE_DIR / MODEL.SKIN_TYPE) — "
                             "sert de valeur par défaut pour --model/--skin-type")
    args = parser.parse_args()

    if args.config:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        model_cfg = cfg.get('MODEL', {})
        if args.model is None and 'SAVE_DIR' in model_cfg:
            args.model = os.path.join(ROOT, model_cfg['SAVE_DIR'])
        if args.skin_type is None and 'SKIN_TYPE' in model_cfg:
            args.skin_type = model_cfg['SKIN_TYPE']

    if not os.path.exists(args.video):
        sys.exit(f"[ERREUR] Vidéo introuvable : {args.video}")

    print(f"Lecture de la vidéo : {args.video}")
    frames, fps = read_video(args.video)
    print(f"  {len(frames)} frames @ {fps:.1f} FPS ({len(frames) / fps:.1f}s)")

    print("\nExtraction du signal RGB (MediaPipe)...")
    rgb_regions = extract_rgb(frames, verbose=True)
    rgb = rgb_regions[args.region]
    print(f"  Région utilisée : {args.region}")

    # ── CHROM De Haan (coefficients fixes) ──────────────────────────────
    bvp_dehaan = chrom(rgb, fps)
    hr_dehaan = hr_from_fft(bvp_dehaan, fps)

    # ── CHROM adaptatif (coefficients appris, ou De Haan si pas de modèle)
    coeffs = load_coefficients(args.model, args.skin_type)
    bvp_adaptive = chrom_adaptive(rgb, fps, coeffs)
    hr_adaptive = hr_from_fft(bvp_adaptive, fps)

    print("\n" + "═" * 56)
    print("  Résultats")
    print("═" * 56)
    print(f"  CHROM De Haan     : {hr_dehaan:6.1f} bpm")
    print(f"  CHROM adaptatif   : {hr_adaptive:6.1f} bpm")
    if args.model:
        print(f"\n  Modèle utilisé    : {args.model}"
              + (f" (phototype {args.skin_type})" if args.skin_type else ""))
    else:
        print("\n  (Aucun modèle fourni — coefficients De Haan utilisés)")
    print(f"  Coefficients      : a1={coeffs['a1']:.3f}  a2={coeffs['a2']:.3f}  "
          f"a3={coeffs['a3']:.3f}  a4={coeffs['a4']:.3f}  a5={coeffs['a5']:.3f}")
    print("═" * 56)


if __name__ == '__main__':
    main()
