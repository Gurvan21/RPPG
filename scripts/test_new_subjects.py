#!/usr/bin/env python3
"""
Test externe des deux meilleures méthodes (CNN 1D régions + CHROM conditionné
ITA) sur des sujets jamais vus (ex. Subject 70-75), au niveau scénario.
Donne le SNR du signal prédit pour chaque cas, + SNR du PPG de référence.

Usage :
    python scripts/test_new_subjects.py --subjects "Subject 70" "Subject 71" ...
    python scripts/test_new_subjects.py --range 70 75
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.cnn1d_rppg import CNN1D_rPPG
from models.chrom_adaptive import CHROMAdaptiveConditioned, compute_ita, bandpass_numpy
from mp_rppg.metrics import hr_from_fft, snr
from mp_rppg.methods import chrom_adaptive
from scripts.train_cnn1d import _temporal_norm, CLIP_LEN

REGION_FRONT = 0
RGB_IDX = [0, 1, 2]


def fitz_of(name):
    js = [j for j in (ROOT / 'DataVital' / name).glob('*.json') if j.name != 'metadata.json']
    return json.load(open(js[0]))['participant']['fitzpatrick'] if js else '?'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=str(ROOT / 'Data' / 'region_signals'))
    ap.add_argument('--subjects', nargs='*', default=None)
    ap.add_argument('--range', nargs=2, type=int, default=None)
    ap.add_argument('--cnn', default=str(ROOT / 'weights' / 'cnn1d_rppg.pth'))
    ap.add_argument('--chrom', default=str(ROOT / 'weights' / 'chrom_conditioned_regions.pth'))
    ap.add_argument('--colors', default='all', help="all|rgb|<indices> (doit matcher l'entraînement du CNN)")
    args = ap.parse_args()

    if args.colors == 'all':
        color_idx, n_col = None, 9
    elif args.colors == 'rgb':
        color_idx, n_col = [0, 1, 2], 3
    else:
        color_idx = [int(c) for c in args.colors.split(',')]; n_col = len(color_idx)
    in_ch = 23 * n_col

    if args.range:
        names = [f"Subject {i}" for i in range(args.range[0], args.range[1] + 1)]
    elif args.subjects:
        names = args.subjects
    else:
        ap.error("--subjects ou --range requis")

    dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    cnn = CNN1D_rPPG(in_channels=in_ch).to(dev)
    cnn.load_state_dict(torch.load(args.cnn, map_location=dev)); cnn.eval()
    chrom_mlp = CHROMAdaptiveConditioned()
    chrom_mlp.load_state_dict(torch.load(args.chrom, map_location='cpu')['model_state_dict']); chrom_mlp.eval()

    print(f"{'Sujet':<12}{'Fitz':>5}{'sc':>5}{'HRgt':>7}"
          f"{'CNN_HR':>8}{'CNN_err':>8}{'CNN_SNR':>9}"
          f"{'CHR_HR':>8}{'CHR_err':>8}{'CHR_SNR':>9}{'ref_SNR':>9}")
    print('-' * 95)

    agg = {'cnn': [], 'chrom': []}
    for name in names:
        d = Path(args.data) / name
        if not d.exists():
            print(f"{name:<12}  [pas de signaux extraits]")
            continue
        fz = fitz_of(name)
        for npz in sorted(d.glob('*.npz')):
            data = np.load(str(npz), allow_pickle=True)
            fps = float(data['fps'])
            gt = bandpass_numpy(data['y'].astype(np.float32), fps)
            hg = hr_from_fft(gt, fps)
            ref_snr = snr(gt, hg, fps)

            # CNN 1D : reconstruction signal complet par fenêtres
            x = _temporal_norm(data['x'], color_idx); T = x.shape[1]
            preds = []
            for s in range(0, T - CLIP_LEN + 1, CLIP_LEN):
                xw = torch.from_numpy(x[:, s:s + CLIP_LEN]).unsqueeze(0).to(dev)
                with torch.no_grad():
                    preds.append(cnn(xw).squeeze().cpu().numpy())
            cnn_sig = bandpass_numpy(np.concatenate(preds), fps)
            cnn_hr = hr_from_fft(cnn_sig, fps)
            cnn_err = abs(cnn_hr - hg); cnn_snr = snr(cnn_sig, hg, fps)

            # CHROM conditionné ITA : sur région front, signal complet
            rgb = data['x'][:, REGION_FRONT, :][:, RGB_IDX].astype(np.float32)
            ita = compute_ita(rgb.mean(axis=0))
            coeffs = chrom_mlp.get_coefficients(ita)
            chr_sig = bandpass_numpy(chrom_adaptive(rgb, fps, coeffs), fps)
            chr_hr = hr_from_fft(chr_sig, fps)
            chr_err = abs(chr_hr - hg); chr_snr = snr(chr_sig, hg, fps)

            agg['cnn'].append(cnn_err); agg['chrom'].append(chr_err)
            print(f"{name:<12}{fz:>5}{npz.stem:>5}{hg:>7.1f}"
                  f"{cnn_hr:>8.1f}{cnn_err:>8.1f}{cnn_snr:>9.2f}"
                  f"{chr_hr:>8.1f}{chr_err:>8.1f}{chr_snr:>9.2f}{ref_snr:>9.2f}")

    print('-' * 95)
    if agg['cnn']:
        c = np.array(agg['cnn']); h = np.array(agg['chrom'])
        print(f"{'CNN 1D':<12} MAE={c.mean():.2f}  médiane={np.median(c):.2f}  %<5bpm={100*(c<5).mean():.0f}%")
        print(f"{'CHROM-ITA':<12} MAE={h.mean():.2f}  médiane={np.median(h):.2f}  %<5bpm={100*(h<5).mean():.0f}%")


if __name__ == '__main__':
    main()
