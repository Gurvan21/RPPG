#!/usr/bin/env python3
"""Entraîne le CNN1D-main AVEC le DC (moyenne/canal) en entrée sur TOUTES les
données hand_signals, et sauve le modèle + les stats de standardisation (mu, sd)."""
import os, sys
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from models.cnn1d_rppg import CNN1D_rPPG
from scripts.train_cnn1d import pearson_loss, CLIP_LEN
from scripts.exp_meancolor_cnn1d import rec_data, DS, RC

dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
recs = []
for npz in sorted((ROOT/'Data'/'hand_signals').glob('*/*.npz')):
    normed, mean, y, fps = rec_data(npz)
    recs.append(('?', normed, mean, y, fps))
M = np.stack([r[2] for r in recs]); mu = M.mean(0); sd = M.std(0) + 1e-6
print(f"{len(recs)} enregistrements, entrée {2*RC} canaux")
ld = DataLoader(DS(recs, mu, sd), batch_size=16, shuffle=True)
m = CNN1D_rPPG(in_channels=2*RC).to(dev)
opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60, eta_min=1e-5)
for ep in range(60):
    m.train()
    for x, y, _ in ld:
        x, y = x.to(dev), y.to(dev); opt.zero_grad(); loss = pearson_loss(m(x), y)
        loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    sch.step()
torch.save(m.state_dict(), ROOT/'weights'/'cnn1d_hand_dc.pth')
np.savez(ROOT/'weights'/'cnn1d_hand_dc_stats.npz', mu=mu, sd=sd)
print("→ weights/cnn1d_hand_dc.pth + cnn1d_hand_dc_stats.npz")
