#!/usr/bin/env python3
"""CONTRÔLE DE FUITE (permutation test) : réentraîne le CNN1D-main avec les labels
MÉLANGÉS (chaque fenêtre d'entrée reçoit la cible d'une autre au hasard), même CV
5-fold groupée par personne. Si le held-out reste bon → fuite/raccourci. S'il
s'effondre → le modèle a besoin du vrai lien entrée→pouls (pas de fuite)."""
import os, sys
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from models.cnn1d_rppg import CNN1D_rPPG
from models.chrom_adaptive import bandpass_numpy
from mp_rppg.metrics import hr_from_fft, snr
from scripts.train_cnn1d import _temporal_norm, pearson_loss, CLIP_LEN
from scripts.cv_cnn1d_hand import person_groups
from scripts.train_cnn1d_hand import subj_fitz


def build(dirs):
    xs, ys, fp = [], [], []
    for d in dirs:
        for npz in sorted(Path(d).glob('*.npz')):
            dt = np.load(str(npz), allow_pickle=True)
            x = _temporal_norm(dt['x']); y = dt['y'].astype(np.float32); f = float(dt['fps']); T = x.shape[1]
            for s in range(0, T-CLIP_LEN+1, CLIP_LEN):
                yw = y[s:s+CLIP_LEN]; yw = (yw-yw.mean())/(yw.std()+1e-8)
                xs.append(x[:, s:s+CLIP_LEN].copy()); ys.append(yw.copy()); fp.append(f)
    return xs, ys, fp


class DS(Dataset):
    def __init__(self, xs, ys):
        self.xs, self.ys = xs, ys
    def __len__(self): return len(self.xs)
    def __getitem__(self, i): return torch.from_numpy(self.xs[i]), torch.from_numpy(self.ys[i])


def eval_dirs(model, dirs, dev):
    model.eval(); errs, ph, gh = [], [], []
    with torch.no_grad():
        for d in dirs:
            for npz in sorted(Path(d).glob('*.npz')):
                dt = np.load(str(npz), allow_pickle=True); x = _temporal_norm(dt['x']); y = dt['y'].astype(np.float32)
                fps = float(dt['fps']); T = x.shape[1]; pr = []
                for s in range(0, T-CLIP_LEN+1, CLIP_LEN):
                    pr.append(model(torch.from_numpy(x[:, s:s+CLIP_LEN]).unsqueeze(0).to(dev)).squeeze().cpu().numpy())
                if not pr: continue
                p = np.concatenate(pr); g = y[:len(p)]
                errs.append(abs(hr_from_fft(p, fps)-hr_from_fft(g, fps))); ph.append(hr_from_fft(p, fps)); gh.append(hr_from_fft(g, fps))
    return np.mean(errs), np.corrcoef(ph, gh)[0, 1] if np.std(ph) > 0 else float('nan')


def run(shuffle_labels, seed=0):
    dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    groups = person_groups(); rng = np.random.default_rng(seed); idx = rng.permutation(len(groups)); K = 5
    folds = [idx[i::K] for i in range(K)]; maes, corrs = [], []
    for k in range(K):
        tr = [d for j in range(K) if j != k for i in folds[j] for d in groups[i]]
        te = [d for i in folds[k] for d in groups[i]]
        xs, ys, fp = build(tr)
        if shuffle_labels:                       # casse le lien entrée→pouls
            perm = rng.permutation(len(ys)); ys = [ys[p] for p in perm]
        ld = DataLoader(DS(xs, ys), batch_size=16, shuffle=True)
        m = CNN1D_rPPG(in_channels=18*9).to(dev)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60, eta_min=1e-5)
        for ep in range(60):
            m.train()
            for x, y in ld:
                x, y = x.to(dev), y.to(dev); opt.zero_grad(); loss = pearson_loss(m(x), y)
                loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            sch.step()
        mae, corr = eval_dirs(m, te, dev); maes.append(mae); corrs.append(corr)
    return np.mean(maes), np.nanmean(corrs)


print("Contrôle de fuite (CNN1D-main, CV 5-fold groupée par personne)\n")
m0, c0 = run(shuffle_labels=False)
print(f"Labels NORMAUX  : MAE={m0:.2f} bpm  r={c0:+.2f}   (référence : ~0.69)")
m1, c1 = run(shuffle_labels=True)
print(f"Labels MÉLANGÉS : MAE={m1:.2f} bpm  r={c1:+.2f}")
print("\n→ Si MÉLANGÉS s'effondre (MAE haute, r~0) : PAS de fuite (le modèle a besoin du vrai pouls).")
print("→ Si MÉLANGÉS reste bon : FUITE / raccourci (alarme).")
