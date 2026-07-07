#!/usr/bin/env python3
"""Fine-tuning / entraînement de TS-CAN sur les clips propres ré-extraits
(Data/clips_tscan, avec flux apparence xr). Éval MAE held-out + comparaison
directe au PhysNet du pipeline sur LE MÊME test set (apples-to-apples).
Cible = dérivée standardisée du BVP (DiffNormalized label). FC par FFT."""
import os, sys, argparse, random
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from models.tscan import TSCAN
from mp_rppg.metrics import hr_from_fft, snr


def diffnorm_label(y):
    dy = np.zeros_like(y); dy[:-1] = y[1:] - y[:-1]
    return (dy - dy.mean()) / (dy.std() + 1e-8)


class ClipDS(Dataset):
    def __init__(self, dirs, augment=False):
        self.clips = []
        for d in dirs: self.clips += sorted(Path(d).glob('*.npz'))
        self.augment = augment
        print(f"  {len(self.clips)} clips / {len(dirs)} sujets")

    def __len__(self): return len(self.clips)

    def __getitem__(self, i):
        d = np.load(str(self.clips[i]))
        x = d['x'].astype(np.float32); xr = d['xr'].astype(np.float32); y = d['y'].astype(np.float32)
        if self.augment:
            if random.random() < 0.5: x, xr, y = x[::-1].copy(), xr[::-1].copy(), y[::-1].copy()
            if random.random() < 0.5: x, xr = x[:, :, ::-1].copy(), xr[:, :, ::-1].copy()
            if random.random() < 0.5: x = x * random.uniform(0.85, 1.15)
        lab = diffnorm_label(y)
        mot = torch.from_numpy(x).permute(0, 3, 1, 2)     # (T,C,H,W)
        app = torch.from_numpy(xr).permute(0, 3, 1, 2)
        return mot, app, torch.from_numpy(lab), torch.from_numpy(y)


def neg_pearson(pred, target):
    p = pred - pred.mean(1, keepdim=True); t = target - target.mean(1, keepdim=True)
    num = (p * t).sum(1); den = torch.sqrt((p**2).sum(1) * (t**2).sum(1) + 1e-8)
    return (1 - num / den).mean()


def eval_tscan(model, ds, dev, fps):
    model.eval(); errs, snrs, prs = [], [], []
    with torch.no_grad():
        for mot, app, lab, y in DataLoader(ds, batch_size=1, num_workers=2):
            B, T = mot.shape[:2]
            m = mot.view(B*T, *mot.shape[2:]).to(dev); a = app.view(B*T, *app.shape[2:]).to(dev)
            out = model(m, a).squeeze().cpu().numpy()
            g = y.squeeze().numpy()
            hg = hr_from_fft(g, fps); hp = hr_from_fft(out, fps)
            errs.append(abs(hp - hg)); snrs.append(snr(out, hg, fps))
            prs.append(float(np.corrcoef(out, diffnorm_label(g))[0, 1]))
    return np.mean(errs), np.mean(snrs), np.nanmean(prs)


def eval_physnet(ds, dev, fps, weights):
    from models.physnet import PhysNet_padding_Encoder_Decoder_MAX
    net = PhysNet_padding_Encoder_Decoder_MAX(frames=128).to(dev)
    net.load_state_dict(torch.load(weights, map_location=dev)); net.eval()
    errs = []
    with torch.no_grad():
        for mot, app, lab, y in DataLoader(ds, batch_size=1, num_workers=2):
            x = mot.permute(0, 2, 1, 3, 4).to(dev)         # (1,C,T,H,W)
            pred = net(x)[0].squeeze().cpu().numpy()
            g = y.squeeze().numpy()
            errs.append(abs(hr_from_fft(pred, fps) - hr_from_fft(g, fps)))
    return np.mean(errs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default='Data/clips_tscan')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch', type=int, default=2)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--test-split', type=float, default=0.15)
    ap.add_argument('--val-split', type=float, default=0.15)
    ap.add_argument('--physnet-weights', default=str(ROOT/'weights'/'clean_physnet_A_pure'/'physnet_africa1_best.pth'))
    a = ap.parse_args()
    dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)

    dirs = sorted([d for d in Path(a.data_root).iterdir() if d.is_dir()])
    random.shuffle(dirs)
    nt = max(1, int(len(dirs)*a.test_split)); nv = max(1, int(len(dirs)*a.val_split))
    te, va, tr = dirs[:nt], dirs[nt:nt+nv], dirs[nt+nv:]
    print(f"Split : {len(tr)} train / {len(va)} val / {len(te)} test  |  device {dev}")
    ds_tr = ClipDS(tr, augment=True); ds_va = ClipDS(va); ds_te = ClipDS(te)
    fps = float(np.load(str(ds_te.clips[0]))['fps'])

    model = TSCAN(frames=128, img_size=72).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=1e-4)
    ld = DataLoader(ds_tr, batch_size=a.batch, shuffle=True, num_workers=4,
                    persistent_workers=True, prefetch_factor=2)
    out_dir = ROOT/'weights'/'tscan'; out_dir.mkdir(parents=True, exist_ok=True)
    best = 1e9
    for ep in range(a.epochs):
        model.train(); tot = 0
        for mot, app, lab, y in ld:
            B, T = mot.shape[:2]
            m = mot.view(B*T, *mot.shape[2:]).to(dev); ap_ = app.view(B*T, *app.shape[2:]).to(dev)
            out = model(m, ap_)
            loss = neg_pearson(out, lab.to(dev))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.item()
        vmae, vsnr, vpr = eval_tscan(model, ds_va, dev, fps)
        flag = ''
        if vmae < best:
            best = vmae; torch.save(model.state_dict(), out_dir/'tscan_best.pth'); flag = ' *'
        print(f"ep {ep+1:2d}/{a.epochs}  loss {tot/len(ld):.3f}  val MAE {vmae:.2f}  SNR {vsnr:+.2f}  r {vpr:.2f}{flag}")

    print(f"\n{'='*52}\nÉVALUATION HELD-OUT ({len(ds_te)} clips test)\n{'='*52}")
    model.load_state_dict(torch.load(out_dir/'tscan_best.pth', map_location=dev))
    tmae, tsnr, tpr = eval_tscan(model, ds_te, dev, fps)
    print(f"  TS-CAN   : MAE {tmae:.2f} bpm | SNR {tsnr:+.2f} dB | r {tpr:.2f}")
    try:
        pmae = eval_physnet(ds_te, dev, fps, a.physnet_weights)
        print(f"  PhysNet  : MAE {pmae:.2f} bpm  (même held-out, réf pipeline)")
        print(f"\n  → {'TS-CAN gagne' if tmae < pmae else 'PhysNet gagne'} de {abs(tmae-pmae):.2f} bpm")
    except Exception as e:
        print(f"  (comparaison PhysNet impossible : {e})")


if __name__ == '__main__':
    main()
