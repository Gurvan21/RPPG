#!/usr/bin/env python3
"""Fine-tuning ÉQUITABLE de TS-CAN : part des poids pré-entraînés PURE_TSCAN
(rPPG-Toolbox) puis fine-tune sur DataVital propre. Compare à PhysNet sur le
MÊME held-out. Éval zero-shot (avant FT) + après FT."""
import os, sys, argparse, random
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from models.tscan_official import TSCAN_official, load_pretrained
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
        lab = diffnorm_label(y)
        inp = np.concatenate([x, xr], axis=-1)            # (T,H,W,6)
        inp = torch.from_numpy(inp).permute(0, 3, 1, 2)   # (T,6,H,W)
        return inp, torch.from_numpy(lab), torch.from_numpy(y)


def neg_pearson(pred, target):
    p = pred - pred.mean(1, keepdim=True); t = target - target.mean(1, keepdim=True)
    num = (p * t).sum(1); den = torch.sqrt((p**2).sum(1) * (t**2).sum(1) + 1e-8)
    return (1 - num / den).mean()


def eval_model(model, ds, dev, fps):
    model.eval(); errs, snrs, prs = [], [], []
    with torch.no_grad():
        for inp, lab, y in DataLoader(ds, batch_size=1, num_workers=2):
            B, T = inp.shape[:2]
            out = model(inp.view(B*T, *inp.shape[2:]).to(dev), T).squeeze().cpu().numpy()
            g = y.squeeze().numpy()
            errs.append(abs(hr_from_fft(out, fps) - hr_from_fft(g, fps)))
            snrs.append(snr(out, hr_from_fft(g, fps), fps))
            prs.append(float(np.corrcoef(out, diffnorm_label(g))[0, 1]))
    return np.mean(errs), np.mean(snrs), np.nanmean(prs)


def eval_physnet(ds, dev, fps, weights):
    from models.physnet import PhysNet_padding_Encoder_Decoder_MAX
    net = PhysNet_padding_Encoder_Decoder_MAX(frames=128).to(dev)
    net.load_state_dict(torch.load(weights, map_location=dev)); net.eval()
    errs = []
    with torch.no_grad():
        for inp, lab, y in DataLoader(ds, batch_size=1, num_workers=2):
            diff = inp[:, :, :3]                          # (1,T,3,H,W)
            x = diff.permute(0, 2, 1, 3, 4).to(dev)       # (1,3,T,H,W)
            pred = net(x)[0].squeeze().cpu().numpy()
            g = y.squeeze().numpy()
            errs.append(abs(hr_from_fft(pred, fps) - hr_from_fft(g, fps)))
    return np.mean(errs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default='Data/clips_tscan')
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-4)          # bas car pré-entraîné
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--pretrained', default=str(ROOT/'weights'/'tscan_pretrained'/'PURE_TSCAN.pth'))
    ap.add_argument('--freeze', default='early', choices=['none', 'early', 'backbone'],
                    help="early=gèle conv1/conv2 ; backbone=gèle toutes les convs (att+dense libres)")
    ap.add_argument('--tag', default='pt')
    ap.add_argument('--split-file', default='')
    ap.add_argument('--physnet-weights', default=str(ROOT/'weights'/'clean_physnet_A_pure'/'physnet_africa1_best.pth'))
    a = ap.parse_args()
    dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)

    root = Path(a.data_root)
    if a.split_file and Path(a.split_file).exists():
        import json
        sp = json.load(open(a.split_file))
        tr = [root/s for s in sp['train']]; va = [root/s for s in sp['val']]; te = [root/s for s in sp['test']]
        print(f"Split PARTAGÉ ({a.split_file}) : {len(tr)} train / {len(va)} val / {len(te)} test | device {dev}")
    else:
        dirs = sorted([d for d in root.iterdir() if d.is_dir()])
        random.shuffle(dirs)
        nt = max(1, int(len(dirs)*0.15)); nv = max(1, int(len(dirs)*0.15))
        te, va, tr = dirs[:nt], dirs[nt:nt+nv], dirs[nt+nv:]
        print(f"Split : {len(tr)} train / {len(va)} val / {len(te)} test | device {dev}")
    ds_tr = ClipDS(tr, augment=True); ds_va = ClipDS(va); ds_te = ClipDS(te)
    fps = float(np.load(str(ds_te.clips[0]))['fps'])

    model = TSCAN_official(frames=128, img_size=72).to(dev)
    load_pretrained(model, a.pretrained)
    print(f"Poids chargés : {Path(a.pretrained).name}")

    # gel partiel du réseau
    if a.freeze == 'early':
        frozen = ['motion_conv1', 'motion_conv2', 'apperance_conv1', 'apperance_conv2']
    elif a.freeze == 'backbone':
        frozen = ['motion_conv', 'apperance_conv']   # toutes les convs
    else:
        frozen = []
    nfr = 0
    for name, p in model.named_parameters():
        if any(name.startswith(f) for f in frozen):
            p.requires_grad = False; nfr += p.numel()
    tot = sum(p.numel() for p in model.parameters())
    print(f"Gel '{a.freeze}' : {nfr/1e6:.2f}M gelés / {tot/1e6:.2f}M ({100*nfr/tot:.0f}%) — {(tot-nfr)/1e6:.2f}M entraînables")

    z_mae, z_snr, z_pr = eval_model(model, ds_te, dev, fps)
    print(f"\n[ZERO-SHOT PURE, avant fine-tune] held-out MAE {z_mae:.2f} bpm | SNR {z_snr:+.2f} | r {z_pr:.2f}")

    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=1e-4)
    ld = DataLoader(ds_tr, batch_size=a.batch, shuffle=True, num_workers=4,
                    persistent_workers=True, prefetch_factor=2)
    out_dir = ROOT/'weights'/'tscan'; out_dir.mkdir(parents=True, exist_ok=True)
    best = 1e9
    for ep in range(a.epochs):
        model.train(); tot = 0
        for inp, lab, y in ld:
            B, T = inp.shape[:2]
            out = model(inp.view(B*T, *inp.shape[2:]).to(dev), T)
            loss = neg_pearson(out, lab.to(dev))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.item()
        vmae, vsnr, vpr = eval_model(model, ds_va, dev, fps)
        flag = ''
        if vmae < best:
            best = vmae; torch.save(model.state_dict(), out_dir/f'tscan_{a.tag}_best.pth'); flag = ' *'
        print(f"ep {ep+1:2d}/{a.epochs}  loss {tot/len(ld):.3f}  val MAE {vmae:.2f}  SNR {vsnr:+.2f}  r {vpr:.2f}{flag}")

    print(f"\n{'='*52}\nÉVALUATION HELD-OUT ({len(ds_te)} clips)\n{'='*52}")
    model.load_state_dict(torch.load(out_dir/f'tscan_{a.tag}_best.pth', map_location=dev))
    tmae, tsnr, tpr = eval_model(model, ds_te, dev, fps)
    print(f"  TS-CAN zero-shot (PURE)     : MAE {z_mae:.2f} bpm")
    print(f"  TS-CAN fine-tuné (PURE→DV)  : MAE {tmae:.2f} bpm | SNR {tsnr:+.2f} | r {tpr:.2f}")
    try:
        pmae = eval_physnet(ds_te, dev, fps, a.physnet_weights)
        print(f"  PhysNet (réf pipeline)      : MAE {pmae:.2f} bpm")
        print(f"\n  → {'TS-CAN gagne' if tmae < pmae else 'PhysNet gagne'} de {abs(tmae-pmae):.2f} bpm")
    except Exception as e:
        print(f"  (PhysNet compare impossible : {e})")


if __name__ == '__main__':
    main()
