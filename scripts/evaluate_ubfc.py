"""
Évaluation CHROM + POS sur UBFC-rPPG avec comparaison des 3 backends :
  - HC  : Haar Cascade → crop 72×72 → moyenne spatiale
  - Y5F : YOLO5Face    → crop 72×72 → moyenne spatiale
  - MP  : MediaPipe FaceMesh → polygones (front, joue G, joue D, moyenne)

Note : le backend Y5F (YOLO5Face) n'est pas inclus dans ce repo — il est
désactivé par défaut (--no-y5f). HC et MediaPipe fonctionnent de base.

Usage :
    # UBFC complet (avec ground truth)
    python scripts/evaluate_ubfc.py --data /path/to/UBFC-rPPG

    # Test rapide sur 5 sujets
    python scripts/evaluate_ubfc.py --data /path/to/UBFC-rPPG --subjects 5

    # Test sur une vidéo personnelle (sans GT)
    python scripts/evaluate_ubfc.py --data /path/to/dossier --no-gt
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, ROOT)

from mp_rppg.pipeline  import extract_rgb   as extract_rgb_mp
from mp_rppg.backends  import extract_rgb_hc, extract_rgb_y5f
from mp_rppg.methods   import chrom, pos
from mp_rppg.metrics   import hr_from_fft, snr, aggregate
from mp_rppg.plots     import save_all


# ── I/O ───────────────────────────────────────────────────────────────────────
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


def read_gt(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        content = f.read().strip()
    if not content:
        return None
    return np.asarray([float(x) for x in content.split()], dtype=np.float64)


# ── Évaluation d'une paire (RGB, fs, hr_gt) ───────────────────────────────────
def eval_rgb(label, rgb, fps, hr_gt):
    """
    Applique CHROM + POS sur rgb (T,3), retourne un dict de résultats.
    """
    bvp_c = chrom(rgb, fps)
    bvp_p = pos(rgb,   fps)

    hr_c  = hr_from_fft(bvp_c, fps)
    hr_p  = hr_from_fft(bvp_p, fps)

    ref_c = hr_gt if hr_gt is not None else hr_c
    ref_p = hr_gt if hr_gt is not None else hr_p

    snr_c = snr(bvp_c, ref_c, fps)
    snr_p = snr(bvp_p, ref_p, fps)

    err_c = abs(hr_c - hr_gt) if hr_gt is not None else float('nan')
    err_p = abs(hr_p - hr_gt) if hr_gt is not None else float('nan')

    gt_str = f"  GT={hr_gt:.1f}" if hr_gt else ""
    print(f"     [{label:12s}]  "
          f"CHROM={hr_c:5.1f} bpm (err={err_c:.1f}, SNR={snr_c:.1f} dB)  "
          f"POS={hr_p:5.1f} bpm (err={err_p:.1f}, SNR={snr_p:.1f} dB)"
          f"{gt_str}")

    return {
        f"{label}/CHROM": {'hr_pred': hr_c, 'hr_gt': hr_gt or float('nan'),
                           'err': err_c, 'snr': snr_c},
        f"{label}/POS":   {'hr_pred': hr_p, 'hr_gt': hr_gt or float('nan'),
                           'err': err_p, 'snr': snr_p},
    }


# ── Évaluation d'un sujet ──────────────────────────────────────────────────────
def evaluate_subject(subject_path, use_gt, use_y5f):
    vid_path = os.path.join(subject_path, 'vid.avi')
    gt_path  = os.path.join(subject_path, 'ground_truth.txt')

    if not os.path.exists(vid_path):
        return None

    print(f"\n  ── {os.path.basename(subject_path)}")
    frames, fps = read_video(vid_path)
    print(f"     {len(frames)} frames @ {fps:.1f} FPS  ({len(frames)/fps:.1f}s)")

    hr_gt = None
    if use_gt:
        gt_bvp = read_gt(gt_path)
        if gt_bvp is not None:
            hr_gt = hr_from_fft(gt_bvp, fps)
            print(f"     HR ground truth : {hr_gt:.1f} bpm")

    subject_results = {}

    # ── Backend HC ────────────────────────────────────────────────────────────
    print("     ·· HC")
    try:
        hc_rgb = extract_rgb_hc(frames)
        subject_results.update(eval_rgb('HC face', hc_rgb['face'], fps, hr_gt))
    except Exception as e:
        print(f"     HC ERREUR : {e}")

    # ── Backend Y5F ───────────────────────────────────────────────────────────
    if use_y5f:
        print("     ·· Y5F")
        try:
            y5f_rgb = extract_rgb_y5f(frames)
            subject_results.update(eval_rgb('Y5F face', y5f_rgb['face'], fps, hr_gt))
        except Exception as e:
            print(f"     Y5F ERREUR : {e}")

    # ── Backend MediaPipe ─────────────────────────────────────────────────────
    print("     ·· MediaPipe")
    try:
        mp_rgb = extract_rgb_mp(frames, verbose=True)
        for region in ('front', 'left', 'right', 'mean'):
            subject_results.update(
                eval_rgb(f"MP {region}", mp_rgb[region], fps, hr_gt))
    except Exception as e:
        print(f"     MP ERREUR : {e}")

    return subject_results


# ── Agrégation et résumé ───────────────────────────────────────────────────────
def build_summary(all_results):
    """
    all_results : liste de dicts {label -> {hr_pred, hr_gt, err, snr}}

    Retourne :
      summary         : {label -> MAE/RMSE/SNR_mean}
      per_subject     : list de dicts {label -> {hr_pred, hr_gt}}
      errors_by_label : {label -> [abs_errors]}
    """
    if not all_results:
        return {}, [], {}

    labels = list(all_results[0].keys())

    summary = {}
    errors_by_label = {}
    per_subject = []

    for label in labels:
        errs = [r[label]['err'] for r in all_results
                if label in r and not np.isnan(r[label]['err'])]
        snrs = [r[label]['snr'] for r in all_results if label in r]
        summary[label] = aggregate(errs, snrs) if errs else \
                         {'MAE': float('nan'), 'RMSE': float('nan'), 'SNR_mean': np.mean(snrs)}
        errors_by_label[label] = errs

    for r in all_results:
        per_subject.append({
            label: {'hr_pred': r[label]['hr_pred'],
                    'hr_gt':   r[label]['hr_gt']}
            for label in labels if label in r
        })

    return summary, per_subject, errors_by_label


def print_summary(summary, n_subjects):
    print("\n" + "═" * 72)
    print("  RÉSUMÉ — Métriques agrégées")
    print("═" * 72)
    print(f"  {'Méthode':<18} {'MAE':>7} {'RMSE':>7} {'SNR moy':>10}")
    print("  " + "─" * 46)
    for label, m in summary.items():
        mae  = f"{m['MAE']:.2f}"  if not np.isnan(m['MAE'])  else " N/A "
        rmse = f"{m['RMSE']:.2f}" if not np.isnan(m['RMSE']) else " N/A "
        print(f"  {label:<18} {mae:>7} {rmse:>7} {m['SNR_mean']:>9.2f} dB")
    print("═" * 72)
    print(f"  Sujets évalués : {n_subjects}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Évaluation CHROM+POS (HC vs Y5F vs MediaPipe) sur UBFC-rPPG")
    parser.add_argument('--data', required=True,
                        help="Dossier racine contenant les dossiers subject*/")
    parser.add_argument('--no-gt',    action='store_true',
                        help="Pas de ground truth (test sans vérité terrain)")
    parser.add_argument('--y5f',      action='store_true',
                        help="Activer le backend Y5F (non fourni dans ce repo)")
    parser.add_argument('--subjects', type=int, default=None,
                        help="Limiter à N sujets")
    parser.add_argument('--out',      default=os.path.join(ROOT, 'results/mp_eval'),
                        help="Dossier de sortie pour les graphiques")
    args = parser.parse_args()

    subject_dirs = sorted(glob.glob(os.path.join(args.data, 'subject*')))
    if not subject_dirs:
        sys.exit(f"[ERREUR] Aucun dossier subject* trouvé dans : {args.data}")
    if args.subjects:
        subject_dirs = subject_dirs[:args.subjects]

    print(f"Dataset : {args.data}")
    print(f"Sujets  : {len(subject_dirs)}")
    print(f"GT      : {'non' if args.no_gt else 'oui'}")
    print(f"Y5F     : {'oui' if args.y5f else 'non'}")

    all_results = []
    for sdir in subject_dirs:
        r = evaluate_subject(sdir,
                             use_gt=not args.no_gt,
                             use_y5f=args.y5f)
        if r is not None:
            all_results.append(r)

    summary, per_subject, errors_by_label = build_summary(all_results)
    print_summary(summary, len(all_results))

    print(f"\n[Graphiques] Génération dans {args.out} ...")
    try:
        save_all(summary, per_subject, errors_by_label, out_dir=args.out)
    except Exception as e:
        print(f"  Erreur graphiques : {e}")


if __name__ == '__main__':
    main()
