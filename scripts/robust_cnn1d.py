#!/usr/bin/env python3
"""Robustesse CNN1D par augmentation SIGNAL. Baseline (clean) vs robuste
(clean+augmenté), évalués sur held-out propre et dégradé. Parallèle à robust_physnet."""
import os, sys, random
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from models.cnn1d_rppg import CNN1D_rPPG
from models.chrom_adaptive import bandpass_numpy
from mp_rppg.metrics import hr_from_fft
from scripts.train_cnn1d import _temporal_norm, CLIP_LEN, pearson_loss
from scripts.augment import signal_augment, signal_degrade_fixed
dev = torch.device('cpu')                     # CPU : protège un éventuel run GPU concurrent
RS = ROOT/'Data'/'region_signals'


class DS(Dataset):
    def __init__(self, subs, augment=False, degrade=False):
        self.win = []
        for s in subs:
            for f in sorted((RS/s).glob('*.npz')):
                T = np.load(str(f))['x'].shape[0]
                for st in range(0, T-CLIP_LEN+1, CLIP_LEN): self.win.append((f, st))
        self.augment = augment; self.degrade = degrade; self.rng = np.random.default_rng(0)
    def __len__(self): return len(self.win)
    def __getitem__(self, i):
        f, st = self.win[i]; d = np.load(str(f))
        x = d['x'].astype(np.float32); y = d['y'].astype(np.float32); fps = float(d['fps'])
        x = x[st:st+CLIP_LEN]; y = y[st:st+CLIP_LEN]
        if self.degrade: x = signal_degrade_fixed(x)
        elif self.augment and random.random() < 0.5: x = signal_augment(x, self.rng)
        xn = _temporal_norm(x)                            # (23*9, T)
        yb = bandpass_numpy(y, fps); yb = (yb-yb.mean())/(yb.std()+1e-8)
        return torch.from_numpy(xn), torch.from_numpy(yb.astype(np.float32))


def ev(m, subs, degrade):
    m.eval(); errs = []
    with torch.no_grad():
        for s in subs:
            for f in sorted((RS/s).glob('*.npz')):
                d = np.load(str(f)); x = d['x'].astype(np.float32); y = d['y'].astype(np.float32); fps = float(d['fps'])
                if degrade: x = signal_degrade_fixed(x)
                xn = _temporal_norm(x); pr = []
                for st in range(0, xn.shape[1]-CLIP_LEN+1, CLIP_LEN):
                    pr.append(m(torch.from_numpy(xn[:, st:st+CLIP_LEN]).unsqueeze(0).to(dev)).squeeze().cpu().numpy())
                if not pr: continue
                sig = bandpass_numpy(np.concatenate(pr), fps)
                ref = hr_from_fft(bandpass_numpy(y[:len(np.concatenate(pr))], fps), fps)
                errs.append(abs(hr_from_fft(sig, fps) - ref))
    return float(np.mean(errs))


def train(train_subs, augment, epochs=20):
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    m = CNN1D_rPPG(in_channels=23*9).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    ld = DataLoader(DS(train_subs, augment=augment), batch_size=8, shuffle=True, num_workers=4, persistent_workers=True)
    for ep in range(epochs):
        m.train()
        for xn, yb in ld:
            loss = pearson_loss(m(xn.to(dev)), yb.to(dev))
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if (ep+1) % 5 == 0: print(f"    {'ROBUSTE' if augment else 'baseline'} ep {ep+1}/{epochs}", flush=True)
    return m


def main():
    subs = sorted([d.name for d in RS.iterdir() if d.is_dir() and any(d.glob('*.npz'))])
    random.seed(42); random.shuffle(subs)
    nt = max(1, int(len(subs)*0.2)); te, tr = subs[:nt], subs[nt:]
    print(f"CNN1D robustesse : {len(tr)} train / {len(te)} test  | {dev}", flush=True)
    print("=== baseline (clean) ===", flush=True)
    mb = train(tr, False); b_c, b_d = ev(mb, te, False), ev(mb, te, True)
    torch.save(mb.state_dict(), ROOT/'weights'/'cnn1d_base.pth')
    print("=== robuste (clean+augmenté) ===", flush=True)
    mr = train(tr, True); r_c, r_d = ev(mr, te, False), ev(mr, te, True)
    torch.save(mr.state_dict(), ROOT/'weights'/'cnn1d_robust.pth')
    print("modèles sauvés : weights/cnn1d_base.pth, weights/cnn1d_robust.pth")
    print(f"\n{'='*50}\nCNN1D — MAE (bpm) sur held-out\n{'='*50}")
    print(f"                 propre   dégradé")
    print(f"  baseline       {b_c:5.1f}    {b_d:5.1f}")
    print(f"  robuste (aug)  {r_c:5.1f}    {r_d:5.1f}")
    print(f"\n  gain sur DÉGRADÉ : {b_d-r_d:+.1f} bpm | effet sur PROPRE : {b_c-r_c:+.1f} bpm")


if __name__ == '__main__':
    main()
