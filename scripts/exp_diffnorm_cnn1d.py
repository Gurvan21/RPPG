#!/usr/bin/env python3
"""EXPÉRIENCE 1 : DiffNormalized (dérivée normalisée trame-à-trame) en entrée du
CNN1D-main, au lieu de x/moyenne-1. Même CV 5-fold groupée par personne, sans
fuite. Compare au baseline (x/mean-1) = MAE 0,69."""
import os, sys
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from models.cnn1d_rppg import CNN1D_rPPG
from models.chrom_adaptive import bandpass_numpy
from mp_rppg.metrics import hr_from_fft, snr
from scripts.train_cnn1d import pearson_loss, CLIP_LEN
from scripts.cv_cnn1d_hand import person_groups
from scripts.train_cnn1d_hand import subj_fitz


def diffnorm(x, color_idx=None):
    """(T,R,C) -> (R*C, T) : DiffNormalized par canal, z-scoré, padé à T."""
    if color_idx is not None:
        x = x[:, :, color_idx]
    T, R, C = x.shape
    flat = x.reshape(T, R * C).astype(np.float32)          # (T, RC)
    num = np.diff(flat, axis=0)                              # (T-1, RC)
    den = flat[1:] + flat[:-1] + 1e-8
    d = num / den                                           # (T-1, RC)
    d = (d - d.mean(0, keepdims=True)) / (d.std(0, keepdims=True) + 1e-8)
    d = np.concatenate([d, d[-1:]], axis=0)                 # pad -> T
    return d.T                                              # (RC, T)


class DS(Dataset):
    def __init__(self, dirs):
        self.items = []
        for d in dirs:
            for npz in sorted(Path(d).glob('*.npz')):
                dt = np.load(str(npz), allow_pickle=True)
                x = diffnorm(dt['x']); y = dt['y'].astype(np.float32); T = x.shape[1]
                for s in range(0, T - CLIP_LEN + 1, CLIP_LEN):
                    yw = y[s:s+CLIP_LEN]; yw = (yw - yw.mean()) / (yw.std() + 1e-8)
                    self.items.append((x[:, s:s+CLIP_LEN].copy(), yw.copy(), float(dt['fps'])))
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        a, b, f = self.items[i]; return torch.from_numpy(a), torch.from_numpy(b), f


def eval_dirs(model, dirs, dev):
    model.eval(); rows = []
    with torch.no_grad():
        for d in dirs:
            fz = subj_fitz(d)
            for npz in sorted(Path(d).glob('*.npz')):
                dt = np.load(str(npz), allow_pickle=True)
                x = diffnorm(dt['x']); y = dt['y'].astype(np.float32)
                fps = float(dt['fps']); T = x.shape[1]; pr = []
                for s in range(0, T-CLIP_LEN+1, CLIP_LEN):
                    pr.append(model(torch.from_numpy(x[:, s:s+CLIP_LEN]).unsqueeze(0).to(dev)).squeeze().cpu().numpy())
                if not pr: continue
                p = np.concatenate(pr); g = y[:len(p)]
                rows.append((fz, abs(hr_from_fft(p, fps)-hr_from_fft(g, fps)),
                             snr(p, hr_from_fft(g, fps), fps), float(np.corrcoef(p, g)[0, 1])))
    return rows


def main():
    dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    groups = person_groups()
    print(f"DiffNormalized→CNN1D | {len(groups)} personnes, device {dev}")
    rng = np.random.default_rng(0); idx = rng.permutation(len(groups)); K = 5
    folds = [idx[i::K] for i in range(K)]; all_rows = []
    for k in range(K):
        te = [d for i in folds[k] for d in groups[i]]
        tr = [d for j in range(K) if j != k for i in folds[j] for d in groups[i]]
        ds = DS(tr); ld = DataLoader(ds, batch_size=16, shuffle=True)
        m = CNN1D_rPPG(in_channels=ds[0][0].shape[0]).to(dev)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60, eta_min=1e-5)
        for ep in range(60):
            m.train()
            for x, y, _ in ld:
                x, y = x.to(dev), y.to(dev); opt.zero_grad()
                loss = pearson_loss(m(x), y); loss.backward()
                nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            sch.step()
        r = eval_dirs(m, te, dev); all_rows += r
        print(f"  fold {k+1}/{K}: MAE={np.mean([z[1] for z in r]):.2f} (n={len(r)})")
    print("\n=== DiffNormalized→CNN1D (CV 5-fold, sans fuite) ===")
    for fz in ['4', '5', '6']:
        g = [z for z in all_rows if z[0] == fz]
        if g: print(f"  Fitz{fz} (n={len(g)}): MAE={np.mean([z[1] for z in g]):.2f}  "
                     f"SNR={np.nanmean([z[2] for z in g]):+.1f}  r={np.nanmean([z[3] for z in g]):+.2f}")
    e = [z[1] for z in all_rows]
    print(f"  GLOBAL: MAE={np.mean(e):.2f}  %<5={100*np.mean(np.array(e)<5):.0f}  "
          f"r={np.nanmean([z[3] for z in all_rows]):+.2f}   [baseline x/mean-1 = 0.69]")


if __name__ == '__main__':
    main()
