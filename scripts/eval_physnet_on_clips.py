#!/usr/bin/env python3
"""
Évalue un checkpoint PhysNet (SCAMPS, UBFC, ou autre) sur des clips déjà
pré-extraits (.npz issus de preextract_clips.py), sans aucun fine-tuning —
sert de baseline pour savoir si un mauvais score vient du fine-tuning ou
déjà des poids pré-entraînés bruts.

Usage :
  python scripts/eval_physnet_on_clips.py --clips-dir Data/test_dataVital_clips --weights UBFC
  python scripts/eval_physnet_on_clips.py --clips-dir Data/test_dataVital_clips --weights SCAMPS
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')   # avant import torch (MaxPool3d sur MPS)

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from models.physnet import PhysNet_padding_Encoder_Decoder_MAX
from mp_rppg.metrics import hr_from_fft, snr, aggregate

WEIGHTS = {
    'SCAMPS':  os.path.join(ROOT, 'weights/SCAMPS_PhysNet_DiffNormalized.pth'),
    'UBFC':    os.path.join(ROOT, 'weights/UBFC-rPPG_PhysNet_DiffNormalized.pth'),
    'PURE':    os.path.join(ROOT, 'weights/PURE_PhysNet_DiffNormalized.pth'),
    'MA-UBFC': os.path.join(ROOT, 'weights/MA-UBFC_physnet.pth'),
    'BP4D':    os.path.join(ROOT, 'weights/BP4D_PseudoLabel_PhysNet_DiffNormalized.pth'),
}


def pick_device(force_cpu: bool) -> torch.device:
    if force_cpu:
        return torch.device('cpu')
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def eval_subject(model, device, npz_files):
    errors, snrs, pearsons = [], [], []
    for npz_path in npz_files:
        data = np.load(str(npz_path))
        fps  = float(data['fps'])
        g_np = data['y'].astype(np.float32)

        x = torch.from_numpy(data['x'].astype(np.float32)).permute(3, 0, 1, 2).unsqueeze(0).to(device)

        with torch.no_grad():
            pred, _, _, _ = model(x)
        p_np = pred.squeeze().cpu().numpy()

        hr_g = hr_from_fft(g_np, fps)
        hr_p = hr_from_fft(p_np, fps)
        errors.append(abs(hr_p - hr_g))
        snrs.append(snr(p_np, hr_g, fps))
        pearsons.append(float(np.corrcoef(p_np, g_np)[0, 1]))
    return errors, snrs, pearsons


def main():
    parser = argparse.ArgumentParser(description="Baseline PhysNet (sans fine-tuning) sur des clips pré-extraits")
    parser.add_argument('--clips-dir', required=True)
    parser.add_argument('--weights',   default='UBFC', choices=list(WEIGHTS.keys()))
    parser.add_argument('--cpu',       action='store_true')
    args = parser.parse_args()

    device = pick_device(args.cpu)
    print(f"Device  : {device}")
    print(f"Poids   : {args.weights} ({WEIGHTS[args.weights]})")

    model = PhysNet_padding_Encoder_Decoder_MAX(frames=128).to(device)
    ckpt  = torch.load(WEIGHTS[args.weights], map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    clips_dir = Path(args.clips_dir)
    subjects  = sorted(d for d in clips_dir.iterdir() if d.is_dir())
    if not subjects:
        raise FileNotFoundError(f"Aucun sous-dossier dans {clips_dir}")

    all_errors, all_snrs, all_pearsons = [], [], []
    print(f"\n{'Sujet':<12}{'n':>4}{'MAE':>9}{'RMSE':>9}{'SNR':>9}{'Pearson':>10}")
    print('-' * 53)
    for subj_dir in subjects:
        npz_files = sorted(subj_dir.glob('*.npz'))
        if not npz_files:
            continue
        errors, snrs, pearsons = eval_subject(model, device, npz_files)
        m = aggregate(errors, snrs)
        print(f"{subj_dir.name:<12}{len(errors):>4}{m['MAE']:>9.2f}{m['RMSE']:>9.2f}"
              f"{m['SNR_mean']:>9.2f}{np.mean(pearsons):>+10.3f}")
        all_errors.extend(errors)
        all_snrs.extend(snrs)
        all_pearsons.extend(pearsons)

    m = aggregate(all_errors, all_snrs)
    print('-' * 53)
    print(f"{'TOTAL':<12}{len(all_errors):>4}{m['MAE']:>9.2f}{m['RMSE']:>9.2f}"
          f"{m['SNR_mean']:>9.2f}{np.mean(all_pearsons):>+10.3f}")


if __name__ == '__main__':
    main()
