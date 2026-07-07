#!/usr/bin/env python3
"""EXPÉRIENCE : redonner au CNN1D-main le DC (moyenne par canal) en entrée, à côté
du signal normalisé x/moyenne-1. Entrée = 324 canaux (162 normalisés + 162 moyennes
standardisées, broadcastées dans le temps). CV 5-fold groupée par personne, sans
fuite (stats des moyennes calculées SUR LE TRAIN de chaque pli). Baseline = 0.69."""
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

RC = 18*9


def rec_data(npz):
    """Retourne (normed (RC,T), rec_mean (RC,), y (T,), fps)."""
    dt = np.load(str(npz), allow_pickle=True)
    x = dt['x']; T, R, C = x.shape
    flat = x.reshape(T, R*C).astype(np.float32)
    mean = flat.mean(0)                              # DC par canal
    normed = (flat/(mean+1e-8) - 1.0).T              # (RC, T)
    return normed, mean, dt['y'].astype(np.float32), float(dt['fps'])


def load_group_recs(dirs):
    recs = []
    for d in dirs:
        for npz in sorted(Path(d).glob('*.npz')):
            recs.append((subj_fitz(d),) + rec_data(npz))
    return recs


class DS(Dataset):
    def __init__(self, recs, mu, sd):
        self.items = []
        for fz, normed, mean, y, fps in recs:
            mstd = ((mean - mu)/sd).astype(np.float32)          # DC standardisé
            T = normed.shape[1]
            for s in range(0, T-CLIP_LEN+1, CLIP_LEN):
                nw = normed[:, s:s+CLIP_LEN]
                dc = np.repeat(mstd[:, None], CLIP_LEN, axis=1)  # broadcast temps
                aug = np.concatenate([nw, dc], axis=0).astype(np.float32)  # (2RC, T)
                yw = y[s:s+CLIP_LEN]; yw = (yw-yw.mean())/(yw.std()+1e-8)
                self.items.append((aug, yw.copy(), fps))
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        a, b, f = self.items[i]; return torch.from_numpy(a), torch.from_numpy(b), f


def eval_recs(model, recs, mu, sd, dev):
    model.eval(); rows = []
    with torch.no_grad():
        for fz, normed, mean, y, fps in recs:
            mstd = ((mean-mu)/sd).astype(np.float32); T = normed.shape[1]; pr = []
            for s in range(0, T-CLIP_LEN+1, CLIP_LEN):
                dc = np.repeat(mstd[:, None], CLIP_LEN, axis=1)
                aug = np.concatenate([normed[:, s:s+CLIP_LEN], dc], axis=0).astype(np.float32)
                pr.append(model(torch.from_numpy(aug).unsqueeze(0).to(dev)).squeeze().cpu().numpy())
            if not pr: continue
            p = np.concatenate(pr); g = y[:len(p)]
            rows.append((fz, abs(hr_from_fft(bandpass_numpy(p, fps), fps)-hr_from_fft(bandpass_numpy(g, fps), fps)),
                         float(np.corrcoef(p, g)[0, 1])))
    return rows


def main():
    dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    groups = person_groups(); rng = np.random.default_rng(0); idx = rng.permutation(len(groups)); K = 5
    folds = [idx[i::K] for i in range(K)]; allr = []
    print(f"DC (moyenne/canal) en entrée | {len(groups)} personnes, entrée {2*RC} canaux, {dev}")
    for k in range(K):
        te = load_group_recs([d for i in folds[k] for d in groups[i]])
        tr = load_group_recs([d for j in range(K) if j != k for i in folds[j] for d in groups[i]])
        M = np.stack([r[2] for r in tr])                       # moyennes du TRAIN
        mu = M.mean(0); sd = M.std(0) + 1e-6                   # stats sans fuite
        ld = DataLoader(DS(tr, mu, sd), batch_size=16, shuffle=True)
        m = CNN1D_rPPG(in_channels=2*RC).to(dev)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60, eta_min=1e-5)
        for ep in range(60):
            m.train()
            for x, y, _ in ld:
                x, y = x.to(dev), y.to(dev); opt.zero_grad(); loss = pearson_loss(m(x), y)
                loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            sch.step()
        r = eval_recs(m, te, mu, sd, dev); allr += r
        print(f"  fold {k+1}/{K}: MAE={np.mean([z[1] for z in r]):.2f} (n={len(r)})")
    print(f"\n=== DC en entrée (CV 5-fold, sans fuite) ===")
    for fz in ['4', '5', '6']:
        g = [z for z in allr if z[0] == fz]
        if g: print(f"  Fitz{fz} (n={len(g)}): MAE={np.mean([z[1] for z in g]):.2f}  r={np.nanmean([z[2] for z in g]):+.2f}")
    e = [z[1] for z in allr]
    print(f"  GLOBAL: MAE={np.mean(e):.2f}  %<5={100*np.mean(np.array(e)<5):.0f}  "
          f"r={np.nanmean([z[2] for z in allr]):+.2f}   [baseline sans DC = 0.69]")


if __name__ == '__main__':
    main()
