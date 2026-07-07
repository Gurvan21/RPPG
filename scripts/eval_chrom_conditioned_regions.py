#!/usr/bin/env python3
"""
Teste le MLP CHROM conditionné ITA (models.chrom_adaptive.CHROMAdaptiveConditioned)
sur les signaux déjà extraits par BiSeNet (Data/region_signals), au niveau
SCÉNARIO (signal complet ~20s), sans re-extraction lente.

Les coefficients CHROM (a1..a5) sont produits par un petit MLP conditionné sur
l'ITA (carnation continue, calculé depuis le RGB moyen du visage). Comparé à
CHROM De Haan (coefficients fixes) sur le même set de test, ventilé par Fitzpatrick.

Usage :
    python scripts/eval_chrom_conditioned_regions.py --epochs 150
"""

import argparse
import json
import os
import sys
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.chrom_adaptive import (CHROMAdaptiveConditioned, DEHAAN_COEFFICIENTS,
                                   bandpass_numpy, bandpass_straight_through,
                                   compute_ita, loss_pearson)
from mp_rppg.metrics import hr_from_fft, snr
from mp_rppg.methods import chrom_adaptive

REGION_FRONT = 0            # 'front' dans region_names
RGB_IDX = [0, 1, 2]        # R,G,B dans color_names


def fitz_of(name):
    js = [j for j in (ROOT / 'DataVital' / name).glob('*.json') if j.name != 'metadata.json']
    return json.load(open(js[0]))['participant']['fitzpatrick'] if js else '?'


def load_scenarios(subject_dirs, region=REGION_FRONT):
    """Retourne liste de dicts {rgb (T,3), gt (T,), fps, ita, subj, fitz, sc}."""
    out = []
    for d in subject_dirs:
        fz = fitz_of(d.name)
        for npz in sorted(Path(d).glob('*.npz')):
            data = np.load(str(npz), allow_pickle=True)
            rgb = data['x'][:, region, :][:, RGB_IDX].astype(np.float32)   # (T,3)
            gt = data['y'].astype(np.float32)
            fps = float(data['fps'])
            ita = compute_ita(rgb.mean(axis=0))
            out.append({'rgb': rgb, 'gt': gt, 'fps': fps, 'ita': ita,
                        'subj': d.name, 'fitz': fz, 'sc': npz.stem})
    return out


def norm_rgb(rgb):
    R = rgb[:, 0] / (rgb[:, 0].mean() + 1e-8)
    G = rgb[:, 1] / (rgb[:, 1].mean() + 1e-8)
    B = rgb[:, 2] / (rgb[:, 2].mean() + 1e-8)
    return (torch.tensor(R.astype(np.float32)),
            torch.tensor(G.astype(np.float32)),
            torch.tensor(B.astype(np.float32)))


def train(model, scenarios, epochs, lr):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=40, gamma=0.5)
    for ep in range(epochs):
        tot = 0.0
        random.shuffle(scenarios)
        for s in scenarios:
            Rn, Gn, Bn = norm_rgb(s['rgb'])
            sig = model(Rn, Gn, Bn, s['ita'])
            sig_t = bandpass_straight_through(sig, s['fps'])
            label = torch.tensor(bandpass_numpy(s['gt'], s['fps']).astype(np.float32))
            loss = loss_pearson(sig_t, label)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        sched.step()
        if ep % 20 == 0 or ep == epochs - 1:
            print(f"  epoch {ep:3d}  loss={tot/len(scenarios):.4f}")


def evaluate(scenarios, coeffs_fn):
    """coeffs_fn(scenario) -> dict a1..a5.  Retourne err par scénario + infos."""
    rows = []
    for s in scenarios:
        bvp = chrom_adaptive(s['rgb'], s['fps'], coeffs_fn(s))
        m = min(len(bvp), len(s['gt']))
        bvp = bandpass_numpy(bvp[:m], s['fps'])
        gt = bandpass_numpy(s['gt'][:m], s['fps'])
        hg, hp = hr_from_fft(gt, s['fps']), hr_from_fft(bvp, s['fps'])
        rows.append({**s, 'err': abs(hp - hg), 'hg': hg, 'hp': hp,
                     'snr': snr(bvp, hg, s['fps'])})
    return rows


def summarize(rows, title):
    errs = np.array([r['err'] for r in rows])
    print(f"\n{title} : MAE={errs.mean():.2f}  médiane={np.median(errs):.2f}  "
          f"RMSE={np.sqrt((errs**2).mean()):.2f}  %<5bpm={100*(errs<5).mean():.0f}%")
    by = defaultdict(list)
    for r in rows:
        by[r['fitz']].append(r['err'])
    for f in sorted(by):
        e = np.array(by[f])
        print(f"   Fitz {f}: n={len(e):2d}  MAE={e.mean():6.2f}  médiane={np.median(e):5.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=str(ROOT / 'Data' / 'region_signals'))
    ap.add_argument('--epochs', type=int, default=150)
    ap.add_argument('--lr', type=float, default=0.02)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--val-split', type=float, default=0.1)
    ap.add_argument('--test-split', type=float, default=0.1)
    ap.add_argument('--save', default=str(ROOT / 'weights' / 'chrom_conditioned_regions.pth'))
    args = ap.parse_args()

    subs = sorted(d for d in Path(args.data).iterdir() if d.is_dir())
    random.seed(args.seed); random.shuffle(subs)
    n_val = max(1, int(len(subs) * args.val_split))
    n_test = max(1, int(len(subs) * args.test_split))
    test_d, train_d = subs[:n_test], subs[n_test + n_val:]
    print(f"{len(subs)} sujets — {len(train_d)} train / {len(test_d)} test")
    print(f"Sujets test : {[d.name for d in test_d]}")

    train_sc = load_scenarios(train_d)
    test_sc = load_scenarios(test_d)
    itas = [s['ita'] for s in train_sc + test_sc]
    print(f"{len(train_sc)} scénarios train / {len(test_sc)} test  |  "
          f"ITA min={min(itas):.0f} max={max(itas):.0f}\n")

    print("Entraînement MLP conditionné ITA...")
    model = CHROMAdaptiveConditioned()
    train(model, train_sc, args.epochs, args.lr)
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    torch.save({'model_state_dict': model.state_dict()}, args.save)

    # éval : De Haan fixe vs MLP conditionné
    rows_dh = evaluate(test_sc, lambda s: dict(DEHAAN_COEFFICIENTS))
    rows_cd = evaluate(test_sc, lambda s: model.get_coefficients(s['ita']))

    print("\n" + "=" * 60)
    summarize(rows_dh, "CHROM De Haan (fixe)        ")
    summarize(rows_cd, "CHROM conditionné ITA (MLP) ")
    print("=" * 60)

    # détail par scénario
    print(f"\n{'Sujet':<12}{'Fitz':>5}{'ITA':>6}{'HRgt':>7}{'DeHaan':>8}{'Cond':>8}")
    for rd, rc in zip(rows_dh, rows_cd):
        print(f"{rd['subj']:<12}{rd['fitz']:>5}{rd['ita']:>6.0f}{rd['hg']:>7.1f}"
              f"{rd['err']:>8.1f}{rc['err']:>8.1f}")


if __name__ == '__main__':
    main()
