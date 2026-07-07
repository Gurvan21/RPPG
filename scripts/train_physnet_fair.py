#!/usr/bin/env python3
"""Ré-entraîne PhysNet depuis SCAMPS sur le MÊME split partagé que TS-CAN
(clips_tscan), pour une comparaison SANS FUITE. Sauve weights/physnet_fair.pth."""
import os, sys, argparse, random, json
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from models.physnet import PhysNet_padding_Encoder_Decoder_MAX
from mp_rppg.metrics import hr_from_fft, snr


class DS(Dataset):
    def __init__(self, dirs, augment=False):
        self.clips = []
        for d in dirs: self.clips += sorted(Path(d).glob('*.npz'))
        self.augment = augment
        print(f"  {len(self.clips)} clips / {len(dirs)} sujets")
    def __len__(self): return len(self.clips)
    def __getitem__(self, i):
        d = np.load(str(self.clips[i]))
        x = d['x'].astype(np.float32); y = d['y'].astype(np.float32)
        if self.augment:
            if random.random() < 0.5: x, y = x[::-1].copy(), y[::-1].copy()
            if random.random() < 0.5: x = x[:, :, ::-1].copy()
        x = torch.from_numpy(x).permute(3, 0, 1, 2)   # (3,T,H,W)
        return x, torch.from_numpy(y)


def neg_pearson(pred, target):
    p = pred - pred.mean(1, keepdim=True); t = target - target.mean(1, keepdim=True)
    num = (p*t).sum(1); den = torch.sqrt((p**2).sum(1)*(t**2).sum(1)+1e-8)
    return (1 - num/den).mean()


def evaluate(model, ds, dev, fps):
    model.eval(); errs = []
    with torch.no_grad():
        for x, y in DataLoader(ds, batch_size=1, num_workers=2):
            pred = model(x.to(dev))[0].squeeze().cpu().numpy()
            g = y.squeeze().numpy()
            errs.append(abs(hr_from_fft(pred, fps) - hr_from_fft(g, fps)))
    return np.mean(errs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default='Data/clips_tscan')
    ap.add_argument('--split-file', required=True)
    ap.add_argument('--base', default=str(ROOT/'weights'/'SCAMPS_PhysNet_DiffNormalized.pth'))
    ap.add_argument('--out', default=str(ROOT/'weights'/'physnet_fair.pth'))
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--batch', type=int, default=2)
    ap.add_argument('--lr', type=float, default=1e-4)
    a = ap.parse_args()
    dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    root = Path(a.data_root); sp = json.load(open(a.split_file))
    tr = [root/s for s in sp['train']]; va = [root/s for s in sp['val']]; te = [root/s for s in sp['test']]
    print(f"PhysNet fair — split partagé : {len(tr)} tr / {len(va)} va / {len(te)} te | {dev}")
    ds_tr, ds_va, ds_te = DS(tr, True), DS(va), DS(te)
    fps = float(np.load(str(ds_te.clips[0]))['fps'])

    m = PhysNet_padding_Encoder_Decoder_MAX(frames=128).to(dev)
    m.load_state_dict(torch.load(a.base, map_location=dev)); print(f"Base SCAMPS chargée")
    opt = torch.optim.Adam(m.parameters(), lr=a.lr, weight_decay=1e-4)
    ld = DataLoader(ds_tr, batch_size=a.batch, shuffle=True, num_workers=4, persistent_workers=True)
    best = 1e9
    for ep in range(a.epochs):
        m.train(); tot = 0
        for x, y in ld:
            pred = m(x.to(dev))[0]
            loss = neg_pearson(pred, y.to(dev))
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            tot += loss.item()
        vmae = evaluate(m, ds_va, dev, fps)
        flag = ''
        if vmae < best: best = vmae; torch.save(m.state_dict(), a.out); flag = ' *'
        print(f"ep {ep+1:2d}/{a.epochs}  loss {tot/len(ld):.3f}  val MAE {vmae:.2f}{flag}")
    m.load_state_dict(torch.load(a.out, map_location=dev))
    tmae = evaluate(m, ds_te, dev, fps)
    print(f"\nPhysNet FAIR — held-out MAE {tmae:.2f} bpm  → {a.out}")


if __name__ == '__main__':
    main()
