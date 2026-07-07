#!/usr/bin/env python3
"""Robustesse PhysNet — entraîne baseline ET robuste JUSQU'À CONVERGENCE
(arrêt anticipé sur val), pour une comparaison non biaisée. Augmentation
d'entraînement RAPIDE (block-compress), évaluation sur le VRAI JPEG."""
import os, sys, json, random
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from models.physnet import PhysNet_padding_Encoder_Decoder_MAX
from mp_rppg.metrics import hr_from_fft
from scripts.augment import frame_augment_fast, frame_degrade_fixed, diff_normalize
dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
SPLIT = json.load(open(ROOT/'Data'/'split_fair.json'))
ROOTD = ROOT/'Data'/'clips_tscan'
MAX_EP, PATIENCE = 18, 5


class DS(Dataset):
    def __init__(self, subs, augment=False, degrade=False):
        self.clips = []
        for s in subs: self.clips += sorted((ROOTD/s).glob('*.npz'))
        self.augment = augment; self.degrade = degrade; self.rng = np.random.default_rng(0)
    def __len__(self): return len(self.clips)
    def __getitem__(self, i):
        d = np.load(str(self.clips[i])); y = d['y'].astype(np.float32)
        if self.degrade:
            x = diff_normalize(frame_degrade_fixed(d['xr'].astype(np.float32)))
        elif self.augment and random.random() < 0.5:
            x = diff_normalize(frame_augment_fast(d['xr'].astype(np.float32), self.rng))
        else:
            x = d['x'].astype(np.float32)
        return torch.from_numpy(x).permute(3, 0, 1, 2), torch.from_numpy(y)


def neg_pearson(p, t):
    p = p - p.mean(1, keepdim=True); t = t - t.mean(1, keepdim=True)
    return (1 - (p*t).sum(1)/torch.sqrt((p**2).sum(1)*(t**2).sum(1)+1e-8)).mean()


def ev(m, ds, fps=30.0):
    m.eval(); e = []
    with torch.no_grad():
        for x, y in DataLoader(ds, batch_size=1, num_workers=2):
            p = m(x.to(dev))[0].squeeze().cpu().numpy()
            e.append(abs(hr_from_fft(p, fps) - hr_from_fft(y.squeeze().numpy(), fps)))
    return float(np.mean(e))


def train(augment, tag):
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    m = PhysNet_padding_Encoder_Decoder_MAX(frames=128).to(dev)
    m.load_state_dict(torch.load(ROOT/'weights'/'SCAMPS_PhysNet_DiffNormalized.pth', map_location=dev))
    opt = torch.optim.Adam(m.parameters(), lr=1e-4, weight_decay=1e-4)
    ld = DataLoader(DS(SPLIT['train'], augment=augment), batch_size=2, shuffle=True, num_workers=4, persistent_workers=True)
    va = DS(SPLIT['val']); va_d = DS(SPLIT['val'], degrade=True)
    ckpt = ROOT/'weights'/f'physnet_rob_{tag}.pth'
    best, since = 1e9, 0
    for ep in range(MAX_EP):
        m.train()
        for x, y in ld:
            loss = neg_pearson(m(x.to(dev))[0], y.to(dev))
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        vc = ev(m, va); vd = ev(m, va_d)
        # sélection du checkpoint : sur la val DÉGRADÉE pour le robuste (max robustesse),
        # sur la val propre pour le baseline.
        crit = vd if augment else vc; flag = ''
        if crit < best - 0.2:
            best = crit; since = 0; torch.save(m.state_dict(), ckpt); flag = ' *'
        else: since += 1
        print(f"    {tag} ep {ep+1:2d}  val_propre {vc:5.1f}  val_dégradé {vd:5.1f}{flag}", flush=True)
        if since >= PATIENCE: print(f"    -> convergé (plateau {PATIENCE} ép.)", flush=True); break
    m.load_state_dict(torch.load(ckpt, map_location=dev)); return m


def main():
    te_c, te_d = DS(SPLIT['test']), DS(SPLIT['test'], degrade=True)
    print("=== BASELINE : réutilise le checkpoint converti (physnet_rob_base.pth) ===", flush=True)
    mb = PhysNet_padding_Encoder_Decoder_MAX(frames=128).to(dev)
    mb.load_state_dict(torch.load(ROOT/'weights'/'physnet_rob_base.pth', map_location=dev))
    b_c, b_d = ev(mb, te_c), ev(mb, te_d)
    print(f"  baseline : propre {b_c:.1f} | dégradé {b_d:.1f}", flush=True)
    print("=== ROBUSTE (lr 1e-4, aug douce) — jusqu'à convergence ===", flush=True)
    mr = train(True, 'robust'); r_c, r_d = ev(mr, te_c), ev(mr, te_d)
    print(f"\n{'='*50}\nPhysNet CONVERGÉ — MAE (bpm) held-out\n{'='*50}")
    print(f"                 propre   dégradé")
    print(f"  baseline       {b_c:5.1f}    {b_d:5.1f}")
    print(f"  robuste (aug)  {r_c:5.1f}    {r_d:5.1f}")
    print(f"\n  gain DÉGRADÉ : {b_d-r_d:+.1f} | effet PROPRE : {b_c-r_c:+.1f}")


if __name__ == '__main__':
    main()
