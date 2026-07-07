#!/usr/bin/env python3
"""
Évaluation PhysNet AU NIVEAU SCÉNARIO (et non par fenêtre de 128 frames).

Pour chaque scénario test, on reconstruit le signal BVP complet (~20s) en
concaténant les prédictions des fenêtres consécutives, puis on calcule UNE
seule HR par FFT sur ce signal long — bien plus robuste que la HR par fenêtre
de 2s (trop courte pour une FFT stable).

Réplique le split sujet de finetune_physnet.py (même seed) pour retrouver le
set de test, et ventile les résultats par phototype Fitzpatrick.

Usage :
    python scripts/eval_physnet_scenario.py \
        --clips-dir Data/dataVital_clips_v3 \
        --weights weights/finetune_v3_A_pure/physnet_africa1_best.pth
"""

import argparse
import os
import re
import sys
import random
import json
from pathlib import Path
from collections import defaultdict

os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.physnet import PhysNet_padding_Encoder_Decoder_MAX
from mp_rppg.metrics import hr_from_fft, snr
from scripts.preextract_clips import bandpass

DATAVITAL = ROOT / "DataVital"


def replicate_test_split(clips_dir, seed, val_split, test_split):
    all_dirs = sorted([d for d in Path(clips_dir).iterdir() if d.is_dir()])
    random.seed(seed)
    random.shuffle(all_dirs)
    n_val = max(1, int(len(all_dirs) * val_split))
    n_test = max(1, int(len(all_dirs) * test_split))
    return all_dirs[:n_test]


def fitzpatrick_of(subject_name):
    src = DATAVITAL / subject_name
    js = [j for j in src.glob('*.json') if j.name != 'metadata.json']
    if not js:
        return '?'
    try:
        return json.load(open(js[0])).get('participant', {}).get('fitzpatrick', '?')
    except Exception:
        return '?'


def clip_start(p):
    return int(re.search(r'clip_(\d+)', p.name).group(1))


def eval_subject(model, device, subj_dir):
    """Retourne liste de (sc_name, hr_gt, hr_pred, err, snr_pred, pearson)."""
    by_sc = defaultdict(list)
    for npz in subj_dir.glob('*.npz'):
        by_sc[npz.name.split('_clip_')[0]].append(npz)

    rows = []
    for sc, files in sorted(by_sc.items()):
        files = sorted(files, key=clip_start)
        preds, gts, fps = [], [], None
        for npz in files:
            data = np.load(str(npz))
            fps = float(data['fps'])
            x = torch.from_numpy(data['x'].astype(np.float32)).permute(3, 0, 1, 2).unsqueeze(0).to(device)
            with torch.no_grad():
                preds.append(model(x)[0].squeeze().cpu().numpy())
            gts.append(data['y'].astype(np.float32))
        fp = bandpass(np.concatenate(preds), fps)
        fg = bandpass(np.concatenate(gts), fps)
        hg, hp = hr_from_fft(fg, fps), hr_from_fft(fp, fps)
        rows.append((sc, hg, hp, abs(hp - hg), snr(fp, hg, fps),
                     float(np.corrcoef(fp, fg)[0, 1])))
    return rows


def main():
    ap = argparse.ArgumentParser(description="Éval PhysNet au niveau scénario")
    ap.add_argument('--clips-dir', required=True)
    ap.add_argument('--weights', required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--val-split', type=float, default=0.1)
    ap.add_argument('--test-split', type=float, default=0.1)
    ap.add_argument('--cpu', action='store_true')
    args = ap.parse_args()

    device = torch.device('cpu') if args.cpu or not (torch.cuda.is_available() or torch.backends.mps.is_available()) \
        else (torch.device('cuda') if torch.cuda.is_available() else torch.device('mps'))
    print(f"Device : {device}")

    model = PhysNet_padding_Encoder_Decoder_MAX(frames=128).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    dirs_test = replicate_test_split(args.clips_dir, args.seed, args.val_split, args.test_split)
    print(f"Sujets test ({len(dirs_test)}) : {[d.name for d in dirs_test]}\n")

    all_rows, by_fitz = [], defaultdict(list)
    print(f"{'Sujet':<12}{'Fitz':>5}{'Scénario':>14}{'HRgt':>7}{'HRpred':>8}{'Err':>7}{'SNR':>8}{'Pear':>7}")
    print('-' * 68)
    for d in dirs_test:
        fitz = fitzpatrick_of(d.name)
        for sc, hg, hp, err, s, pear in eval_subject(model, device, d):
            flag = '  <<<' if err > 15 else ''
            print(f"{d.name:<12}{fitz:>5}{sc:>14}{hg:>7.1f}{hp:>8.1f}{err:>7.1f}{s:>8.2f}{pear:>7.2f}{flag}")
            all_rows.append(err)
            by_fitz[fitz].append(err)

    errs = np.array(all_rows)
    print('-' * 68)
    print(f"\n=== GLOBAL ({len(errs)} scénarios) ===")
    print(f"  MAE    : {errs.mean():.2f} bpm")
    print(f"  RMSE   : {np.sqrt((errs**2).mean()):.2f} bpm")
    print(f"  Médiane: {np.median(errs):.2f} bpm")
    print(f"  % < 5 bpm : {100*(errs<5).mean():.0f}%")

    print(f"\n=== PAR FITZPATRICK ===")
    for f in sorted(by_fitz):
        e = np.array(by_fitz[f])
        print(f"  Fitz {f} : n={len(e):2d}  MAE={e.mean():6.2f}  médiane={np.median(e):5.2f}  %<5bpm={100*(e<5).mean():3.0f}%")


if __name__ == '__main__':
    main()
