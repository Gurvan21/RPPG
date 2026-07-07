#!/usr/bin/env python3
"""rPPG auto-supervisé CONTRASTIF (Contrast-Phys, Sun & Li ECCV 2022) — sans labels.
Deux extraits du MÊME clip → même spectre (positif, ancre sur le vrai pouls) ;
extraits de clips DIFFÉRENTS → spectres différents (InfoNCE → anti-collapse).
+ prior : puissance concentrée dans la bande FC."""
import os, sys
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from models.cnn1d_rppg import CNN1D_rPPG
from models.chrom_adaptive import bandpass_numpy
from mp_rppg.metrics import hr_from_fft
from scripts.train_cnn1d import _temporal_norm, CLIP_LEN
LOW, HIGH, CROP = 0.7, 3.5, 96


class DS(Dataset):
    def __init__(self, dirs):
        self.items = []
        for d in dirs:
            for npz in sorted(Path(d).glob('*.npz')):
                x = _temporal_norm(np.load(str(npz), allow_pickle=True)['x'], None); T = x.shape[1]
                for s in range(0, T - CLIP_LEN + 1, CLIP_LEN):
                    self.items.append(x[:, s:s+CLIP_LEN].copy())
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return torch.from_numpy(self.items[i])


def npsd(sig, fps):
    sig = sig - sig.mean(-1, keepdim=True); sig = sig / (sig.std(-1, keepdim=True) + 1e-6)
    psd = torch.fft.rfft(sig, dim=-1).abs() ** 2
    fr = torch.fft.rfftfreq(sig.shape[-1], 1/fps).to(sig.device)
    band = (fr >= LOW) & (fr <= HIGH); pb = psd[:, band] + 1e-8
    return pb / pb.sum(-1, keepdim=True)


def loss_fn(model, x, fps=30.0):
    c1 = model(x[:, :, :CROP]); c2 = model(x[:, :, CLIP_LEN-CROP:])   # 2 extraits chevauchants
    p1, p2 = npsd(c1, fps), npsd(c2, fps)
    sim = p1 @ p2.t() / 0.08                                          # similarité PSD (B,B)
    lab = torch.arange(x.shape[0], device=x.device)
    nce = 0.5*(nn.functional.cross_entropy(sim, lab) + nn.functional.cross_entropy(sim.t(), lab))
    # prior : concentration autour du pic (signal périodique)
    fr = torch.fft.rfftfreq(CROP, 1/fps).to(x.device); fb = fr[(fr >= LOW) & (fr <= HIGH)]
    peak = fb[p1.argmax(-1)]; bw = (p1 * (fb[None]-peak[:, None])**2).sum(-1).sqrt().mean()
    return nce + 0.3*bw


def main():
    dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    subs = sorted(d for d in (ROOT/'Data'/'region_signals').iterdir() if d.is_dir())
    held = sorted(d for d in (ROOT/'Data'/'region_new').iterdir() if d.is_dir())
    ds = DS(subs); ld = DataLoader(ds, batch_size=48, shuffle=True, drop_last=True)
    print(f"SSL contrastif (sans labels) : {len(ds)} fenêtres / {len(subs)} sujets")
    model = CNN1D_rPPG(in_channels=23*9).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    def ev():
        model.eval(); errs, ph, gh = [], [], []
        with torch.no_grad():
            for d in held:
                for npz in sorted(Path(d).glob('*.npz')):
                    dat = np.load(str(npz), allow_pickle=True); fps = float(dat['fps'])
                    x = _temporal_norm(dat['x'], None); T = x.shape[1]; pr = []
                    for s in range(0, T-CLIP_LEN+1, CLIP_LEN):
                        pr.append(model(torch.from_numpy(x[:, s:s+CLIP_LEN]).unsqueeze(0).to(dev)).squeeze().cpu().numpy())
                    if not pr: continue
                    p = hr_from_fft(bandpass_numpy(np.concatenate(pr), fps), fps)
                    g = hr_from_fft(bandpass_numpy(dat['y'].astype(np.float32), fps), fps)
                    errs.append(abs(p-g)); ph.append(p); gh.append(g)
        e = np.array(errs); corr = np.corrcoef(ph, gh)[0, 1] if np.std(ph) > 0 else float('nan')
        return e.mean(), 100*(e < 5).mean(), corr

    for ep in range(1, 51):
        model.train(); tl = 0.0
        for x in ld:
            x = x.to(dev); opt.zero_grad(); loss = loss_fn(model, x); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); tl += loss.item()
        if ep % 5 == 0 or ep == 1:
            mae, p5, corr = ev()
            print(f"ep{ep:3d} loss={tl/len(ld):.3f} | held-out MAE={mae:.2f} %<5={p5:.0f}% corr={corr:+.2f}")
    torch.save(model.state_dict(), ROOT/'weights'/'cnn1d_ssl_contrast.pth'); print("→ weights/cnn1d_ssl_contrast.pth")


if __name__ == '__main__':
    main()
