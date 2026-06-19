#!/usr/bin/env python3
"""
Fine-tuning PhysNet sur VitalVideos-Africa-1 (400 samples).

Stratégie :
  - Poids initiaux : SCAMPS (plus généraux qu'UBFC)
  - Loss : corrélation de Pearson négative (standard rPPG)
  - Mixed precision (fp16) + gradient accumulation → VRAM ~1.5 GB (940MX OK)
  - Augmentations temporelles et photométriques pour la robustesse carnation

Format Africa-1 attendu (configurable via --data-root) :
  DATA_ROOT/
    subject_001/
      video.mp4    (ou vid.avi)
      ppg.csv      (colonnes : timestamp_s, pleth)   ← ou ground_truth.txt (UBFC style)
    subject_002/
      ...

Usage :
  python scripts/finetune_physnet.py --data-root Data/africa1 --epochs 30
  python scripts/finetune_physnet.py --data-root Data/africa1 --epochs 30 --batch 2 --accum 8
"""

import argparse
import os
import sys
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from models.physnet import PhysNet_padding_Encoder_Decoder_MAX

WEIGHTS_SCAMPS = os.path.join(ROOT, 'weights/SCAMPS_PhysNet_DiffNormalized.pth')
CLIP_LEN = 128


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
def pearson_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Corrélation de Pearson négative — loss standard rPPG."""
    pred   = pred   - pred.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    num    = (pred * target).sum(dim=-1)
    denom  = (pred.pow(2).sum(dim=-1).sqrt() * target.pow(2).sum(dim=-1).sqrt() + 1e-8)
    return -num / denom   # shape (B,)


# ---------------------------------------------------------------------------
# Dataset Africa-1
# ---------------------------------------------------------------------------
class Africa1Dataset(Dataset):
    """
    Charge des clips pré-extraits (.npz) issus de preextract_clips.py.
    Format npz : x=(128,72,72,3) float16, y=(128,) float32, fps=float32
    """

    def __init__(self, subject_dirs: list, augment: bool = True):
        self.augment = augment
        self.clips   = []   # list of Path → fichier .npz
        self._build_index(subject_dirs)

    def _build_index(self, dirs):
        for d in dirs:
            d = Path(d)
            npz_files = sorted(d.glob('*.npz'))
            self.clips.extend(npz_files)
        print(f"  Dataset : {len(self.clips)} clips issus de {len(dirs)} sujets")

    def _augment(self, x: np.ndarray) -> np.ndarray:
        # x : (128, 72, 72, 3) DiffNormalized float32
        if random.random() < 0.5:
            x = x[::-1].copy()                          # flip temporel
        if random.random() < 0.5:
            x = x[:, :, ::-1].copy()                    # flip horizontal
        if random.random() < 0.5:
            x = x * random.uniform(0.85, 1.15)          # scaling amplitude
        return x

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        data = np.load(str(self.clips[idx]))
        x = data['x'].astype(np.float32)   # (128, 72, 72, 3)
        y = data['y'].astype(np.float32)   # (128,)

        if self.augment:
            x = self._augment(x)

        # (128, 72, 72, 3) → (3, 128, 72, 72) NCDHW pour PhysNet
        x = torch.from_numpy(x).permute(3, 0, 1, 2)
        y = torch.from_numpy(y)
        return x, y


# ---------------------------------------------------------------------------
# Entraînement
# ---------------------------------------------------------------------------
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"\nDevice : {device}")
    if device.type == 'cuda':
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory // 1024**2} Mo")

    # ── Trouver les sujets ───────────────────────────────────────────────
    data_root = Path(args.data_root)
    all_dirs  = sorted([d for d in data_root.iterdir() if d.is_dir()])
    if not all_dirs:
        raise FileNotFoundError(f"Aucun sous-dossier dans {data_root}")
    print(f"\n{len(all_dirs)} sujets trouvés dans {data_root}")

    random.seed(args.seed)
    random.shuffle(all_dirs)
    n_val   = max(1, int(len(all_dirs) * args.val_split))
    n_test  = max(1, int(len(all_dirs) * args.test_split))
    dirs_test  = all_dirs[:n_test]
    dirs_val   = all_dirs[n_test:n_test + n_val]
    dirs_train = all_dirs[n_test + n_val:]
    print(f"Split  : {len(dirs_train)} train / {len(dirs_val)} val / {len(dirs_test)} test")

    ds_train = Africa1Dataset(dirs_train, augment=True)
    ds_val   = Africa1Dataset(dirs_val,   augment=False)
    ds_test  = Africa1Dataset(dirs_test,  augment=False)

    loader_train = DataLoader(ds_train, batch_size=args.batch, shuffle=True,
                              num_workers=2, pin_memory=(device.type == 'cuda'))
    loader_val   = DataLoader(ds_val,   batch_size=args.batch, shuffle=False,
                              num_workers=2, pin_memory=(device.type == 'cuda'))

    # ── Modèle ──────────────────────────────────────────────────────────
    model = PhysNet_padding_Encoder_Decoder_MAX(frames=CLIP_LEN).to(device)
    ckpt  = torch.load(WEIGHTS_SCAMPS, map_location=device)
    model.load_state_dict(ckpt)
    print(f"\nPoids SCAMPS chargés depuis {WEIGHTS_SCAMPS}")

    # Fine-tuning : dégeler toutes les couches (ou juste les dernières)
    if args.freeze_backbone:
        # Geler tout sauf les 2 derniers blocs + ConvBlock10
        for name, p in model.named_parameters():
            if not any(k in name for k in ('ConvBlock9', 'ConvBlock10', 'upsample2', 'poolspa')):
                p.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Mode backbone gelé — {trainable:,} paramètres entraînables")
    else:
        print(f"Fine-tuning complet — {sum(p.numel() for p in model.parameters()):,} paramètres")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    scaler = GradScaler(enabled=(device.type == 'cuda'))

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float('inf')

    # ── Boucle d'entraînement ────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        t0 = time.time()

        for step, (x, y) in enumerate(loader_train):
            x, y = x.to(device), y.to(device)

            with autocast(enabled=(device.type == 'cuda')):
                pred, _, _, _ = model(x)      # pred : (B, T)
                loss = pearson_loss(pred, y).mean() / args.accum

            scaler.scale(loss).backward()

            if (step + 1) % args.accum == 0 or (step + 1) == len(loader_train):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item() * args.accum

        train_loss /= len(loader_train)

        # ── Validation ──────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in loader_val:
                x, y = x.to(device), y.to(device)
                with autocast(enabled=(device.type == 'cuda')):
                    pred, _, _, _ = model(x)
                    loss = pearson_loss(pred, y).mean()
                val_loss += loss.item()
        val_loss /= max(len(loader_val), 1)

        scheduler.step()
        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}  "
              f"lr={lr_now:.2e}  ({elapsed:.0f}s)")

        # Sauvegarde meilleur modèle
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = out_dir / 'physnet_africa1_best.pth'
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ✓ Meilleur modèle sauvegardé → {ckpt_path.name}  (val={val_loss:.4f})")

        # Checkpoint régulier
        if epoch % 10 == 0:
            torch.save(model.state_dict(), out_dir / f'physnet_africa1_ep{epoch:03d}.pth')

    # ── Évaluation finale sur test set ──────────────────────────────────
    print(f"\n{'='*55}\nÉvaluation sur le test set ({len(ds_test)} clips)\n{'='*55}")
    best_ckpt = out_dir / 'physnet_africa1_best.pth'
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    model.eval()

    from scipy.signal import periodogram
    def hr_from_pred(pred_np, fps=30.0):
        f, p = periodogram(pred_np, fs=fps, nfft=512, detrend=False)
        mask = (f >= 0.7) & (f <= 2.5)
        return float(f[mask][np.argmax(p[mask])] * 60)

    def hr_from_gt(gt_np, fps=30.0):
        return hr_from_pred(gt_np, fps)

    errors = []
    loader_test = DataLoader(ds_test, batch_size=1, shuffle=False)
    with torch.no_grad():
        for x, y in loader_test:
            x = x.to(device)
            with autocast(enabled=(device.type == 'cuda')):
                pred, _, _, _ = model(x)
            p_np = pred.squeeze().cpu().numpy()
            g_np = y.squeeze().numpy()
            fps  = ds_test.clips[0][2]   # fps from first clip (approx)
            hr_p = hr_from_pred(p_np, fps)
            hr_g = hr_from_gt(g_np, fps)
            errors.append(abs(hr_p - hr_g))

    mae = np.mean(errors)
    rmse = np.sqrt(np.mean(np.array(errors) ** 2))
    print(f"  MAE  : {mae:.2f} bpm")
    print(f"  RMSE : {rmse:.2f} bpm")
    print(f"\nPoids fine-tunés → {best_ckpt}")
    print("Ajoute ce chemin dans WEIGHTS de scripts/infer_physnet.py pour l'utiliser.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tuning PhysNet — VitalVideos-Africa-1")
    parser.add_argument('--data-root',       default='Data/africa1',
                        help="Dossier racine des sujets Africa-1")
    parser.add_argument('--output',          default='weights/finetune',
                        help="Dossier de sauvegarde des poids")
    parser.add_argument('--epochs',          type=int,   default=30)
    parser.add_argument('--batch',           type=int,   default=2,
                        help="Batch size (2-4 recommandé pour 940MX 2Go)")
    parser.add_argument('--accum',           type=int,   default=8,
                        help="Gradient accumulation steps (batch effectif = batch × accum)")
    parser.add_argument('--lr',              type=float, default=1e-5,
                        help="Learning rate initial (faible pour fine-tuning)")
    parser.add_argument('--val-split',       type=float, default=0.1,
                        help="Fraction sujets pour validation")
    parser.add_argument('--test-split',      type=float, default=0.1,
                        help="Fraction sujets pour test final")
    parser.add_argument('--seed',            type=int,   default=42)
    parser.add_argument('--freeze-backbone', action='store_true',
                        help="Geler le backbone, entraîner seulement les dernières couches")
    parser.add_argument('--cpu',             action='store_true')
    args = parser.parse_args()

    train(args)
