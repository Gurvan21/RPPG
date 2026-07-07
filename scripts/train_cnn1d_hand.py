#!/usr/bin/env python3
"""
Entraîne le CNN1D-MAIN sur les signaux multi-régions de la paume
(scripts/extract_hand_regions.py). Split au niveau SUJET (pas de fuite), éval
HR stratifiée par Fitzpatrick — pour valider le gain d'équité sur peau foncée.

Usage : python scripts/train_cnn1d_hand.py --data Data/hand_signals --epochs 60
"""
import argparse, os, sys, random
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from models.cnn1d_rppg import CNN1D_rPPG
from mp_rppg.metrics import hr_from_fft, snr
from scripts.train_cnn1d import _temporal_norm, pearson_loss, CLIP_LEN


def subj_fitz(d):
    for npz in Path(d).glob('*.npz'):
        return str(np.load(str(npz), allow_pickle=True)['fitz'])
    return '?'


class DS(Dataset):
    def __init__(self, dirs, color_idx=None):
        self.items = []
        for d in dirs:
            for npz in sorted(Path(d).glob('*.npz')):
                dt = np.load(str(npz), allow_pickle=True)
                x = _temporal_norm(dt['x'], color_idx); y = dt['y'].astype(np.float32)
                fps = float(dt['fps']); T = x.shape[1]
                for s in range(0, T - CLIP_LEN + 1, CLIP_LEN):
                    yw = y[s:s+CLIP_LEN]; yw = (yw - yw.mean()) / (yw.std() + 1e-8)
                    self.items.append((x[:, s:s+CLIP_LEN].copy(), yw.copy(), fps))
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        x, y, f = self.items[i]; return torch.from_numpy(x), torch.from_numpy(y), f


def eval_subjects(model, dirs, dev, color_idx):
    """Éval par SCÉNARIO (concatène fenêtres) → HR vs vérité, groupé par Fitz."""
    model.eval(); rows = []
    with torch.no_grad():
        for d in dirs:
            fz = subj_fitz(d)
            for npz in sorted(Path(d).glob('*.npz')):
                dt = np.load(str(npz), allow_pickle=True)
                x = _temporal_norm(dt['x'], color_idx); y = dt['y'].astype(np.float32)
                fps = float(dt['fps']); T = x.shape[1]; pr = []
                for s in range(0, T - CLIP_LEN + 1, CLIP_LEN):
                    pr.append(model(torch.from_numpy(x[:, s:s+CLIP_LEN]).unsqueeze(0).to(dev)).squeeze().cpu().numpy())
                if not pr: continue
                p = np.concatenate(pr); g = y[:len(p)]
                hp, hg = hr_from_fft(p, fps), hr_from_fft(g, fps)
                rows.append((fz, abs(hp - hg), snr(p, hg, fps), float(np.corrcoef(p, g)[0, 1])))
    return rows


def report(rows, tag):
    print(f"\n── {tag} (n={len(rows)}) ──")
    for fz in ['4', '5', '6', '?']:
        g = [r for r in rows if r[0] == fz]
        if not g: continue
        print(f"  Fitz{fz} (n={len(g)}): MAE={np.mean([r[1] for r in g]):.2f}  "
              f"SNR={np.nanmean([r[2] for r in g]):+.1f}  r={np.nanmean([r[3] for r in g]):+.2f}")
    if rows:
        print(f"  GLOBAL: MAE={np.mean([r[1] for r in rows]):.2f}  "
              f"%<5={100*np.mean([r[1] < 5 for r in rows]):.0f}  "
              f"SNR={np.nanmean([r[2] for r in rows]):+.1f}  r={np.nanmean([r[3] for r in rows]):+.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=str(ROOT / 'Data' / 'hand_signals'))
    ap.add_argument('--output', default=str(ROOT / 'weights' / 'cnn1d_hand.pth'))
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--test-frac', type=float, default=0.2)
    ap.add_argument('--colors', default='all')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    color_idx = None if args.colors == 'all' else ([0,1,2] if args.colors == 'rgb'
                                                   else [int(c) for c in args.colors.split(',')])
    dev = torch.device('mps' if torch.backends.mps.is_available()
                       else 'cuda' if torch.cuda.is_available() else 'cpu')
    subs = sorted(d for d in Path(args.data).iterdir() if d.is_dir())
    # split stratifié par Fitz pour garder du F6 en test
    by = {}
    for d in subs: by.setdefault(subj_fitz(d), []).append(d)
    random.seed(args.seed)
    test, train = [], []
    for fz, ds in by.items():
        random.shuffle(ds); k = max(1, int(len(ds) * args.test_frac))
        test += ds[:k]; train += ds[k:]
    print(f"Device {dev} | {len(subs)} sujets — {len(train)} train / {len(test)} test")
    print("Fitz test:", {fz: sum(subj_fitz(d) == fz for d in test) for fz in ['4','5','6']})
    ds_tr = DS(train, color_idx); print(f"  {len(ds_tr)} fenêtres train")
    in_ch = ds_tr[0][0].shape[0]
    model = CNN1D_rPPG(in_channels=in_ch).to(dev)
    print(f"CNN1D-main: {in_ch} canaux d'entrée, {sum(p.numel() for p in model.parameters()):,} params")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr*0.01)
    ld = DataLoader(ds_tr, batch_size=args.batch, shuffle=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    best = 1e9
    for ep in range(1, args.epochs + 1):
        model.train(); tl = 0.0
        for x, y, _ in ld:
            x, y = x.to(dev), y.to(dev); opt.zero_grad()
            loss = pearson_loss(model(x), y); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); tl += loss.item()
        sched.step(); tl /= max(len(ld), 1)
        if ep % 10 == 0 or ep == 1:
            rows = eval_subjects(model, test, dev, color_idx)
            mae = np.mean([r[1] for r in rows]) if rows else 99
            print(f"ep{ep:3d} train={tl:.4f} | held-out MAE={mae:.2f}")
            if mae < best:
                best = mae; torch.save(model.state_dict(), args.output)
    model.load_state_dict(torch.load(args.output, map_location=dev))
    report(eval_subjects(model, test, dev, color_idx), f"HELD-OUT FINAL (best MAE={best:.2f})")
    print(f"\nPoids: {args.output}")


if __name__ == '__main__':
    main()
