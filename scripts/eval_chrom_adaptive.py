"""
Entraîne et évalue CHROM adaptatif sur un dossier UBFC-rPPG
(sujets avec vid.avi + ground_truth.txt).

Étapes :
  1. Extraction RGB (front, MediaPipe) pour chaque sujet, mise en cache (.npz)
  2. Split train/test des sujets
  3. Entraînement de CHROMAdaptatif (Pearson loss vs BVP de référence)
  4. Comparaison CHROM De Haan vs CHROM adaptatif sur le set de test
     (HR par FFT, erreur vs GT, SNR) + graphique de comparaison SNR

Usage :
    python scripts/eval_chrom_adaptive.py --data Data --epochs 100 --test-frac 0.3
"""

import argparse
import glob
import os
import sys

# imageio (pyav) doit décoder la vidéo — cv2.VideoCapture entre en conflit
# avec mediapipe une fois celui-ci importé dans le même process et fait
# segfaulter le process. On importe donc mediapipe avant tout, et on décode
# avec imageio.
import mediapipe  # noqa: F401
import imageio.v3 as iio
import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, ROOT)

from models.chrom_adaptive import (CHROMAdaptatif, DEHAAN_COEFFICIENTS,
                                    bandpass_numpy, bandpass_straight_through,
                                    loss_pearson)
from mp_rppg.metrics import hr_from_fft, snr
from mp_rppg.methods import chrom, chrom_adaptive
from mp_rppg.pipeline import extract_rgb


def read_video(path):
    frames = np.stack(list(iio.imiter(path, plugin="pyav")))
    meta = iio.immeta(path, plugin="pyav")
    return frames, float(meta['fps'])


def read_gt_bvp(path):
    """ground_truth.txt UBFC : ligne 1 = signal BVP, ligne 2 = HR, ligne 3 = temps."""
    with open(path) as f:
        lines = f.read().strip().split('\n')
    return np.asarray([float(x) for x in lines[0].split()], dtype=np.float64)


def extract_features(data_dir, cache_dir, region):
    """Extrait (rgb, fps, bvp_gt) pour chaque sujet avec vid.avi + ground_truth.txt,
    avec mise en cache sur disque."""
    os.makedirs(cache_dir, exist_ok=True)
    subjects = []
    for d in sorted(glob.glob(os.path.join(data_dir, 'subject*'))):
        vid = os.path.join(d, 'vid.avi')
        gt = os.path.join(d, 'ground_truth.txt')
        if os.path.exists(vid) and os.path.exists(gt):
            subjects.append(d)

    print(f"{len(subjects)} sujets avec vid.avi + ground_truth.txt")

    features = {}
    for d in subjects:
        name = os.path.basename(d)
        cache_path = os.path.join(cache_dir, f"{name}.npz")

        if os.path.exists(cache_path):
            npz = np.load(cache_path)
            features[name] = (npz['rgb'], float(npz['fps']), npz['bvp'])
            print(f"  {name} : cache")
            continue

        print(f"  {name} : extraction...")
        frames, fps = read_video(os.path.join(d, 'vid.avi'))
        rgb_regions = extract_rgb(frames, verbose=False)
        rgb = rgb_regions[region]
        bvp = read_gt_bvp(os.path.join(d, 'ground_truth.txt'))

        n = min(len(rgb), len(bvp))
        rgb, bvp = rgb[:n], bvp[:n]

        np.savez(cache_path, rgb=rgb, fps=fps, bvp=bvp)
        features[name] = (rgb, fps, bvp)

    return features


def train_chrom_adaptive(features, train_subjects, epochs, lr):
    model = CHROMAdaptatif()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    print(f"\nCoefficients initiaux (De Haan) : {model.get_coefficients()}\n")

    for epoch in range(epochs):
        total_loss, n_valid = 0.0, 0
        for name in train_subjects:
            rgb, fps, bvp = features[name]

            R = torch.tensor((rgb[:, 0] / (rgb[:, 0].mean() + 1e-8)).astype(np.float32))
            G = torch.tensor((rgb[:, 1] / (rgb[:, 1].mean() + 1e-8)).astype(np.float32))
            B = torch.tensor((rgb[:, 2] / (rgb[:, 2].mean() + 1e-8)).astype(np.float32))
            label = torch.tensor(bandpass_numpy(bvp, fps).astype(np.float32))

            sig = model(R, G, B)
            sig_t = bandpass_straight_through(sig, fps)

            loss = loss_pearson(sig_t, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_valid += 1

        scheduler.step()
        if epoch % 10 == 0:
            c = model.get_coefficients()
            print(f"Epoch {epoch:3d} | Loss: {total_loss / n_valid:.4f} | "
                  f"[{c['a1']:.3f}, {c['a2']:.3f}, {c['a3']:.3f}, {c['a4']:.3f}, {c['a5']:.3f}]")

    return model


def evaluate(features, subjects, coeffs, label):
    """Retourne dict subject -> {hr_pred, hr_gt, err, snr}."""
    results = {}
    for name in subjects:
        rgb, fps, bvp = features[name]
        if label == 'dehaan':
            bvp_pred = chrom(rgb, fps)
        else:
            bvp_pred = chrom_adaptive(rgb, fps, coeffs)

        hr_gt = hr_from_fft(bvp, fps)
        hr_pred = hr_from_fft(bvp_pred, fps)
        results[name] = {
            'hr_pred': hr_pred,
            'hr_gt': hr_gt,
            'err': abs(hr_pred - hr_gt),
            'snr': snr(bvp_pred, hr_gt, fps),
        }
    return results


def main():
    parser = argparse.ArgumentParser(description="Évaluation CHROM adaptatif sur UBFC")
    parser.add_argument('--data', default=os.path.join(ROOT, 'Data'))
    parser.add_argument('--region', default='front', choices=['front', 'left', 'right', 'mean'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--test-frac', type=float, default=0.3)
    parser.add_argument('--cache', default=os.path.join(ROOT, 'results', 'chrom_adaptive_cache'))
    parser.add_argument('--out', default=os.path.join(ROOT, 'results', 'chrom_adaptive'))
    parser.add_argument('--save', default=os.path.join(ROOT, 'weights', 'chrom_adaptive_ubfc.pth'))
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    features = extract_features(args.data, args.cache, args.region)
    names = sorted(features.keys())

    rng = np.random.RandomState(args.seed)
    rng.shuffle(names)
    n_test = max(1, int(len(names) * args.test_frac))
    test_subjects = sorted(names[:n_test])
    train_subjects = sorted(names[n_test:])

    print(f"\nTrain ({len(train_subjects)}) : {train_subjects}")
    print(f"Test  ({len(test_subjects)}) : {test_subjects}")

    model = train_chrom_adaptive(features, train_subjects, args.epochs, args.lr)
    coeffs = model.get_coefficients()

    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    torch.save({'model_state_dict': model.state_dict(),
                'coefficients': coeffs, 'fps': None}, args.save)

    print("\n" + "═" * 60)
    print("  Coefficients appris vs De Haan")
    print("═" * 60)
    for k in ('a1', 'a2', 'a3', 'a4', 'a5'):
        print(f"  {k} : {coeffs[k]:7.4f}   (De Haan = {DEHAAN_COEFFICIENTS[k]})")

    res_dehaan = evaluate(features, test_subjects, None, 'dehaan')
    res_adapt  = evaluate(features, test_subjects, coeffs, 'adaptive')

    print("\n" + "═" * 60)
    print("  Résultats sur le set de test")
    print("═" * 60)
    print(f"  {'Sujet':<12} {'GT':>6} {'DeHaan':>8} {'err':>6} {'SNR':>7} | "
          f"{'Adapt':>8} {'err':>6} {'SNR':>7}")
    for name in test_subjects:
        d, a = res_dehaan[name], res_adapt[name]
        print(f"  {name:<12} {d['hr_gt']:6.1f} {d['hr_pred']:8.1f} {d['err']:6.1f} {d['snr']:7.2f} | "
              f"{a['hr_pred']:8.1f} {a['err']:6.1f} {a['snr']:7.2f}")

    mae_dehaan = np.mean([r['err'] for r in res_dehaan.values()])
    mae_adapt  = np.mean([r['err'] for r in res_adapt.values()])
    snr_dehaan = np.mean([r['snr'] for r in res_dehaan.values()])
    snr_adapt  = np.mean([r['snr'] for r in res_adapt.values()])
    print("  " + "─" * 56)
    print(f"  {'MOYENNE':<12} {'':>6} {'':>8} {mae_dehaan:6.1f} {snr_dehaan:7.2f} | "
          f"{'':>8} {mae_adapt:6.1f} {snr_adapt:7.2f}")
    print("═" * 60)

    # ── Plot comparaison SNR ─────────────────────────────────────────────
    os.makedirs(args.out, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    x = np.arange(len(test_subjects))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(6, len(test_subjects) * 0.6), 4))
    ax.bar(x - width / 2, [res_dehaan[n]['snr'] for n in test_subjects], width, label='CHROM De Haan')
    ax.bar(x + width / 2, [res_adapt[n]['snr'] for n in test_subjects], width, label='CHROM adaptatif')
    ax.set_xticks(x)
    ax.set_xticklabels(test_subjects, rotation=45, ha='right')
    ax.set_ylabel('SNR (dB)')
    ax.set_title('SNR par sujet (set de test) — CHROM De Haan vs adaptatif')
    ax.legend()
    fig.tight_layout()
    out_path = os.path.join(args.out, 'snr_comparison.png')
    fig.savefig(out_path, dpi=120)
    print(f"\n[Graphique] {out_path}")

    # Résumé moyennes
    fig2, ax2 = plt.subplots(figsize=(4, 4))
    ax2.bar(['De Haan', 'Adaptatif'], [snr_dehaan, snr_adapt], color=['#888888', '#22aa77'])
    ax2.set_ylabel('SNR moyen (dB)')
    ax2.set_title('SNR moyen — set de test')
    fig2.tight_layout()
    out_path2 = os.path.join(args.out, 'snr_mean_comparison.png')
    fig2.savefig(out_path2, dpi=120)
    print(f"[Graphique] {out_path2}")


if __name__ == '__main__':
    main()
