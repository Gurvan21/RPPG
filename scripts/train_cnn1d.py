#!/usr/bin/env python3
"""
Entraîne le CNN 1D rPPG (models/cnn1d_rppg) sur les signaux multi-régions
extraits par scripts/extract_regions_bisenet.py.

  - Découpe chaque scénario en fenêtres de CLIP_LEN frames
  - Normalisation temporelle par canal (CHROM-like : x / mean - 1)
  - Loss : corrélation de Pearson négative (même loss que PhysNet/CHROM adaptatif)
  - Split au niveau SUJET (pas de fuite)
  - Éval test : MAE/RMSE HR, SNR, Pearson

Usage :
    python scripts/train_cnn1d.py --data Data/region_signals --epochs 60
"""

import argparse
import os
import sys
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.cnn1d_rppg import CNN1D_rPPG
from mp_rppg.metrics import hr_from_fft, snr
from scripts.preextract_clips import bandpass

CLIP_LEN = 128


def pearson_loss(pred, target):
    pred = pred - pred.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    num = (pred * target).sum(dim=-1)
    den = pred.pow(2).sum(-1).sqrt() * target.pow(2).sum(-1).sqrt() + 1e-8
    return (-num / den).mean()


def _temporal_norm(x, color_idx=None):
    """x : (T, R, C) -> (R*len(color_idx), T) normalisé par canal (CHROM-like)."""
    if color_idx is not None:
        x = x[:, :, color_idx]
    T, R, C = x.shape
    flat = x.reshape(T, R * C).astype(np.float32)
    flat = flat / (flat.mean(axis=0, keepdims=True) + 1e-8) - 1.0
    return flat.T   # (R*C, T)


class RegionDataset(Dataset):
    def __init__(self, subject_dirs, clip_len=CLIP_LEN, stride=CLIP_LEN, color_idx=None):
        self.items = []   # (x_window (C,T), y_window (T,), fps)
        for d in subject_dirs:
            for npz in sorted(Path(d).glob('*.npz')):
                data = np.load(str(npz), allow_pickle=True)
                x = _temporal_norm(data['x'], color_idx)   # (C, T)
                # y est DÉJÀ filtré passe-bande à l'extraction (_resample_ppg_to_frames)
                # → ne pas re-filtrer (cohérence avec le pipeline PhysNet, single bandpass)
                y = data['y'].astype(np.float32)        # (T,)
                fps = float(data['fps'])
                T = x.shape[1]
                for s in range(0, T - clip_len + 1, stride):
                    xw = x[:, s:s + clip_len]
                    yw = y[s:s + clip_len]
                    yw = (yw - yw.mean()) / (yw.std() + 1e-8)
                    self.items.append((xw.copy(), yw.copy(), fps))
        print(f"  Dataset : {len(self.items)} fenêtres / {len(subject_dirs)} sujets")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        x, y, fps = self.items[i]
        return torch.from_numpy(x), torch.from_numpy(y), fps


def pick_device(cpu):
    if cpu:
        return torch.device('cpu')
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def main():
    ap = argparse.ArgumentParser(description="Entraînement CNN 1D rPPG")
    ap.add_argument('--data', default=str(ROOT / 'Data' / 'region_signals'))
    ap.add_argument('--output', default=str(ROOT / 'weights' / 'cnn1d_rppg.pth'))
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--val-split', type=float, default=0.1)
    ap.add_argument('--test-split', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--colors', default='all',
                    help="Espaces couleur : 'all', 'rgb', ou liste d'indices 0-8 "
                         "(R G B Y U V L a b) séparés par virgule")
    ap.add_argument('--cpu', action='store_true')
    args = ap.parse_args()

    COLOR_NAMES = ['R', 'G', 'B', 'Y', 'U', 'V', 'L', 'a', 'b']
    if args.colors == 'all':
        color_idx = None
    elif args.colors == 'rgb':
        color_idx = [0, 1, 2]
    else:
        color_idx = [int(c) for c in args.colors.split(',')]
    print(f"Canaux couleur : {args.colors}"
          + (f" -> {[COLOR_NAMES[i] for i in color_idx]}" if color_idx else " (9 canaux)"))

    device = pick_device(args.cpu)
    print(f"Device : {device}")

    data_dir = Path(args.data)
    subjects = sorted(d for d in data_dir.iterdir() if d.is_dir())
    if not subjects:
        raise FileNotFoundError(f"Aucun sujet dans {data_dir}")
    random.seed(args.seed)
    random.shuffle(subjects)
    n_val = max(1, int(len(subjects) * args.val_split))
    n_test = max(1, int(len(subjects) * args.test_split))
    test_d, val_d, train_d = subjects[:n_test], subjects[n_test:n_test+n_val], subjects[n_test+n_val:]
    print(f"{len(subjects)} sujets — split {len(train_d)} train / {len(val_d)} val / {len(test_d)} test")

    ds_tr = RegionDataset(train_d, color_idx=color_idx)
    ds_va = RegionDataset(val_d, color_idx=color_idx)
    ds_te = RegionDataset(test_d, color_idx=color_idx)

    in_ch = ds_tr[0][0].shape[0]
    print(f"Canaux d'entrée : {in_ch}")
    model = CNN1D_rPPG(in_channels=in_ch).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"CNN 1D — {n_params:,} paramètres")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr*0.01)

    ld_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True)
    ld_va = DataLoader(ds_va, batch_size=args.batch, shuffle=False)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best = float('inf')

    for ep in range(1, args.epochs + 1):
        model.train()
        tl = 0.0
        for x, y, _ in ld_tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = pearson_loss(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item()
        tl /= max(len(ld_tr), 1)

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for x, y, _ in ld_va:
                x, y = x.to(device), y.to(device)
                vl += pearson_loss(model(x), y).item()
        vl /= max(len(ld_va), 1)
        sched.step()
        print(f"Epoch {ep:3d}/{args.epochs}  train={tl:.4f}  val={vl:.4f}  lr={opt.param_groups[0]['lr']:.2e}")
        if vl < best:
            best = vl
            torch.save(model.state_dict(), out_path)

    # ── Éval test ──
    print(f"\n{'='*55}\nÉvaluation test ({len(ds_te)} fenêtres)\n{'='*55}")
    model.load_state_dict(torch.load(out_path, map_location=device))
    model.eval()
    errs, snrs, pears = [], [], []
    ld_te = DataLoader(ds_te, batch_size=1, shuffle=False)
    with torch.no_grad():
        for x, y, fps in ld_te:
            x = x.to(device)
            p = model(x).squeeze().cpu().numpy()
            g = y.squeeze().numpy()
            fps = float(fps)
            hg, hp = hr_from_fft(g, fps), hr_from_fft(p, fps)
            errs.append(abs(hp - hg))
            snrs.append(snr(p, hg, fps))
            pears.append(float(np.corrcoef(p, g)[0, 1]))
    print(f"  MAE     : {np.mean(errs):.2f} bpm")
    print(f"  RMSE    : {np.sqrt(np.mean(np.array(errs)**2)):.2f} bpm")
    print(f"  SNR     : {np.nanmean(snrs):.2f} dB")
    print(f"  Pearson : {np.nanmean(pears):.3f}")
    print(f"\nModèle sauvegardé : {out_path}")


if __name__ == '__main__':
    main()
