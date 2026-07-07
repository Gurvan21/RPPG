#!/usr/bin/env python3
"""
Test de fusion multi-méthodes sur sujets externes (ex. 70-75), niveau scénario.

4 méthodes indépendantes (toutes depuis Data/region_signals, pas de re-extraction) :
  - CNN 1D (régions+couleurs)
  - CHROM conditionné ITA (région front)
  - CHROM classique (région peau complète)
  - POS classique (région peau complète)

Pour chaque méthode on calcule un SNR AVEUGLE : SNR autour du pic que la méthode
détecte elle-même (pas la vérité-terrain) — c'est la seule confiance utilisable
en déploiement réel.

On compare 4 stratégies de combinaison :
  - sélection max-SNR  : on prend la HR de la méthode au meilleur SNR aveugle
  - consensus pondéré  : vote des HR (tolérance ±5 bpm) pondéré par le SNR
  - fusion de signaux  : somme des signaux z-normés pondérés par le SNR (10^(snr/10))
                          puis une seule HR
  - oracle (référence) : meilleure méthode connaissant la vérité (borne sup, triche)

Usage :
    python scripts/test_fusion.py --range 70 75
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.cnn1d_rppg import CNN1D_rPPG
from models.chrom_adaptive import CHROMAdaptiveConditioned, compute_ita, bandpass_numpy
from mp_rppg.metrics import hr_from_fft, snr
from mp_rppg.methods import chrom, pos, chrom_adaptive
from scripts.train_cnn1d import _temporal_norm, CLIP_LEN

FRONT, FULLSKIN = 0, 6      # indices régions
RGB_IDX = [0, 1, 2]


def fitz_of(name):
    js = [j for j in (ROOT / 'DataVital' / name).glob('*.json') if j.name != 'metadata.json']
    return json.load(open(js[0]))['participant']['fitzpatrick'] if js else '?'


def blind_snr(sig, fps):
    """SNR autour du pic détecté par la méthode elle-même (pas la vérité)."""
    hr = hr_from_fft(sig, fps)
    return hr, snr(sig, hr, fps)


def znorm(s):
    return (s - s.mean()) / (s.std() + 1e-8)


def weighted_consensus(hrs, snrs, tol=5.0):
    """HR = celle qui maximise la somme des SNR des méthodes d'accord (±tol)."""
    best_hr, best_score = hrs[0], -1e9
    for i, cand in enumerate(hrs):
        score = sum(max(s, 0.01) for h, s in zip(hrs, snrs) if abs(h - cand) <= tol)
        if score > best_score:
            best_hr, best_score = cand, score
    return best_hr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=str(ROOT / 'Data' / 'region_signals'))
    ap.add_argument('--range', nargs=2, type=int, default=None)
    ap.add_argument('--subjects', nargs='*', default=None)
    ap.add_argument('--cnn', default=str(ROOT / 'weights' / 'cnn1d_rppg.pth'))
    ap.add_argument('--chrom', default=str(ROOT / 'weights' / 'chrom_conditioned_regions.pth'))
    args = ap.parse_args()

    names = ([f"Subject {i}" for i in range(args.range[0], args.range[1] + 1)]
             if args.range else args.subjects)

    dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    cnn = CNN1D_rPPG(in_channels=207).to(dev)
    cnn.load_state_dict(torch.load(args.cnn, map_location=dev)); cnn.eval()
    cmlp = CHROMAdaptiveConditioned()
    cmlp.load_state_dict(torch.load(args.chrom, map_location='cpu')['model_state_dict']); cmlp.eval()

    METHODS = ['CNN1D', 'CHROM-ITA', 'CHROM', 'POS']
    err = defaultdict(list)   # stratégie/méthode -> liste d'erreurs

    hdr = f"{'Sujet':<11}{'Fz':>3}{'HRgt':>6}"
    for m in METHODS:
        hdr += f"{m[:7]:>9}{'snr':>6}"
    hdr += f"{'SEL':>7}{'CONS':>7}{'FUS':>7}"
    print(hdr); print('-' * len(hdr))

    for name in names:
        d = Path(args.data) / name
        if not d.exists():
            continue
        fz = fitz_of(name)
        for npz in sorted(d.glob('*.npz')):
            data = np.load(str(npz), allow_pickle=True)
            fps = float(data['fps'])
            gt = bandpass_numpy(data['y'].astype(np.float32), fps)
            hg = hr_from_fft(gt, fps)

            front = data['x'][:, FRONT, :][:, RGB_IDX].astype(np.float32)
            skin = data['x'][:, FULLSKIN, :][:, RGB_IDX].astype(np.float32)

            # --- signaux des 4 méthodes ---
            x = _temporal_norm(data['x']); T = x.shape[1]
            preds = []
            for s in range(0, T - CLIP_LEN + 1, CLIP_LEN):
                xw = torch.from_numpy(x[:, s:s + CLIP_LEN]).unsqueeze(0).to(dev)
                with torch.no_grad():
                    preds.append(cnn(xw).squeeze().cpu().numpy())
            sigs = {
                'CNN1D':     bandpass_numpy(np.concatenate(preds), fps),
                'CHROM-ITA': bandpass_numpy(chrom_adaptive(front, fps, cmlp.get_coefficients(compute_ita(front.mean(0)))), fps),
                'CHROM':     bandpass_numpy(chrom(skin, fps), fps),
                'POS':       bandpass_numpy(pos(skin, fps), fps),
            }

            hrs, snrs, blinds = [], [], {}
            for m in METHODS:
                h, sn = blind_snr(sigs[m], fps)
                hrs.append(h); snrs.append(sn); blinds[m] = (h, sn)
                err[m].append(abs(h - hg))

            # stratégie 1 : max SNR aveugle
            sel_hr = hrs[int(np.argmax(snrs))]
            err['SEL'].append(abs(sel_hr - hg))
            # stratégie 2 : consensus pondéré
            cons_hr = weighted_consensus(hrs, snrs)
            err['CONS'].append(abs(cons_hr - hg))
            # stratégie 3 : fusion de signaux (poids = puissance 10^(snr/10))
            L = min(len(s) for s in sigs.values())
            w = {m: 10 ** (blinds[m][1] / 10) for m in METHODS}
            fused = sum(w[m] * znorm(sigs[m][:L]) for m in METHODS)
            fus_hr = hr_from_fft(bandpass_numpy(fused, fps), fps)
            err['FUS'].append(abs(fus_hr - hg))
            # oracle (borne sup, triche avec la vérité)
            err['ORACLE'].append(min(abs(h - hg) for h in hrs))

            row = f"{name:<11}{fz:>3}{hg:>6.1f}"
            for m in METHODS:
                row += f"{blinds[m][0]:>9.1f}{blinds[m][1]:>6.1f}"
            row += f"{abs(sel_hr-hg):>7.1f}{abs(cons_hr-hg):>7.1f}{abs(fus_hr-hg):>7.1f}"
            print(row)

    print('-' * len(hdr))
    print("\n=== MAE par stratégie ===")
    for k in METHODS + ['SEL', 'CONS', 'FUS', 'ORACLE']:
        e = np.array(err[k])
        tag = {'SEL': 'max-SNR', 'CONS': 'consensus', 'FUS': 'fusion',
               'ORACLE': 'oracle (triche)'}.get(k, k)
        print(f"  {tag:<18} MAE={e.mean():6.2f}  médiane={np.median(e):5.2f}  %<5bpm={100*(e<5).mean():3.0f}%")


if __name__ == '__main__':
    main()
