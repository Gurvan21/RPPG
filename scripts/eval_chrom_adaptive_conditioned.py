"""
Variante de eval_chrom_adaptive_datavital.py : les 5 coefficients CHROM sont
produits par un petit MLP conditionné sur l'ITA (Individual Typology Angle,
descripteur continu de carnation calculé depuis la vidéo elle-même), au lieu
d'un jeu de coefficients global fixe ou discret par catégorie Fitzpatrick.

Compare 3 variantes sur le set de test :
  - CHROM De Haan (coefficients fixes d'origine)
  - CHROM adaptatif global (un seul jeu de coefficients, voir
    eval_chrom_adaptive_datavital.py)
  - CHROM conditionné ITA (coefficients = f(carnation), continu)

Usage :
    python scripts/eval_chrom_adaptive_conditioned.py --data DataVital --epochs 100
"""

import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, ROOT)

from models.chrom_adaptive import (CHROMAdaptiveConditioned, DEHAAN_COEFFICIENTS,
                                    bandpass_numpy, bandpass_straight_through,
                                    compute_ita, loss_pearson)
from mp_rppg.metrics import hr_from_fft, snr
from mp_rppg.methods import chrom, chrom_adaptive
from scripts.eval_chrom_adaptive_datavital import extract_features, _flatten


def train(features, epochs, lr):
    model = CHROMAdaptiveConditioned()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    # ITA par échantillon, calculé une fois (moyenne temporelle du RGB extrait)
    itas = {name: compute_ita(rgb.mean(axis=0)) for name, (rgb, fps, bvp) in features.items()}
    print(f"\nITA : min={min(itas.values()):.1f}  max={max(itas.values()):.1f}  "
          f"moyenne={np.mean(list(itas.values())):.1f}")

    for epoch in range(epochs):
        total_loss, n_valid = 0.0, 0
        for name, (rgb, fps, bvp) in features.items():
            R = torch.tensor((rgb[:, 0] / (rgb[:, 0].mean() + 1e-8)).astype(np.float32))
            G = torch.tensor((rgb[:, 1] / (rgb[:, 1].mean() + 1e-8)).astype(np.float32))
            B = torch.tensor((rgb[:, 2] / (rgb[:, 2].mean() + 1e-8)).astype(np.float32))
            label = torch.tensor(bandpass_numpy(bvp, fps).astype(np.float32))

            sig = model(R, G, B, itas[name])
            sig_t = bandpass_straight_through(sig, fps)

            loss = loss_pearson(sig_t, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_valid += 1

        scheduler.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Loss: {total_loss / n_valid:.4f}")

    return model, itas


def evaluate(features, names, mode, model=None, itas=None):
    results = {}
    for name in names:
        rgb, fps, bvp = features[name]
        if mode == 'dehaan':
            bvp_pred = chrom(rgb, fps)
        elif mode == 'conditioned':
            coeffs = model.get_coefficients(itas[name])
            bvp_pred = chrom_adaptive(rgb, fps, coeffs)
        else:
            raise ValueError(mode)

        hr_gt = hr_from_fft(bvp, fps)
        hr_pred = hr_from_fft(bvp_pred, fps)
        results[name] = {
            'hr_pred': hr_pred, 'hr_gt': hr_gt,
            'err': abs(hr_pred - hr_gt), 'snr': snr(bvp_pred, hr_gt, fps),
        }
    return results


def main():
    parser = argparse.ArgumentParser(description="CHROM conditionné ITA sur DataVital")
    parser.add_argument('--data', default=os.path.join(ROOT, 'DataVital'))
    parser.add_argument('--region', default='front', choices=['front', 'left', 'right', 'mean'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--test-frac', type=float, default=0.2)
    parser.add_argument('--cache', default=os.path.join(ROOT, 'results', 'chrom_adaptive_datavital_cache'))
    parser.add_argument('--save', default=os.path.join(ROOT, 'weights', 'chrom_adaptive_conditioned.pth'))
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    by_subject = extract_features(args.data, args.cache, args.region)
    subjects = sorted(by_subject.keys())

    rng = np.random.RandomState(args.seed)
    rng.shuffle(subjects)
    n_test = max(1, int(len(subjects) * args.test_frac))
    test_subjects = sorted(subjects[:n_test])
    train_subjects = sorted(subjects[n_test:])

    train_features = _flatten(by_subject, train_subjects)
    test_features  = _flatten(by_subject, test_subjects)
    print(f"Train : {len(train_subjects)} sujets / {len(train_features)} scénarios")
    print(f"Test  : {len(test_subjects)} sujets / {len(test_features)} scénarios")

    model, itas_train = train(train_features, args.epochs, args.lr)

    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    torch.save({'model_state_dict': model.state_dict(), 'fps': None}, args.save)

    itas_test = {name: compute_ita(rgb.mean(axis=0)) for name, (rgb, fps, bvp) in test_features.items()}
    test_names = sorted(test_features.keys())

    res_dehaan = evaluate(test_features, test_names, 'dehaan')
    res_cond   = evaluate(test_features, test_names, 'conditioned', model=model, itas=itas_test)

    print("\n" + "═" * 70)
    print("  Résultats sur le set de test")
    print("═" * 70)
    print(f"  {'Scénario':<16}{'ITA':>7}{'GT':>7}{'DeHaan':>8}{'err':>6}{'SNR':>7} | "
          f"{'Cond':>8}{'err':>6}{'SNR':>7}")
    for name in test_names:
        d, c = res_dehaan[name], res_cond[name]
        print(f"  {name:<16}{itas_test[name]:7.1f}{d['hr_gt']:7.1f}{d['hr_pred']:8.1f}"
              f"{d['err']:6.1f}{d['snr']:7.2f} | {c['hr_pred']:8.1f}{c['err']:6.1f}{c['snr']:7.2f}")

    mae_dehaan = np.mean([r['err'] for r in res_dehaan.values()])
    mae_cond   = np.mean([r['err'] for r in res_cond.values()])
    snr_dehaan = np.mean([r['snr'] for r in res_dehaan.values()])
    snr_cond   = np.mean([r['snr'] for r in res_cond.values()])
    print("  " + "─" * 66)
    print(f"  {'MOYENNE':<16}{'':>7}{'':>7}{'':>8}{mae_dehaan:6.1f}{snr_dehaan:7.2f} | "
          f"{'':>8}{mae_cond:6.1f}{snr_cond:7.2f}")
    print("═" * 70)
    print(f"\nModèle sauvegardé : {args.save}")


if __name__ == '__main__':
    main()
