#!/usr/bin/env python3
"""
rPPG AUTO-SUPERVISÉ (style SiNC, Speth et al. CVPR 2023) — entraîne CNN1D SANS
AUCUN label PPG. La seule supervision est SPECTRALE :
  - power loss : la puissance doit être DANS la bande FC (0.7–3.5 Hz)
  - bandwidth loss : concentrée autour d'un pic (signal quasi-périodique = pouls)
  - variance loss : les FC prédites doivent VARIER dans le batch (anti-collapse —
    sinon le modèle sort une sinusoïde constante, comme le vieux CNN1D périmé)

Eval : on compare la FC prédite à la vérité-terrain sur le held-out (region_new),
qui n'a JAMAIS servi à l'entraînement et dont les labels ne sont utilisés QUE
pour mesurer, pas pour entraîner.
"""
import os, sys, random
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from models.cnn1d_rppg import CNN1D_rPPG
from models.chrom_adaptive import bandpass_numpy
from mp_rppg.metrics import hr_from_fft
from scripts.train_cnn1d import _temporal_norm, CLIP_LEN
LOW, HIGH = 0.7, 3.5


class DS(Dataset):
    def __init__(self, dirs):
        self.items = []
        for d in dirs:
            for npz in sorted(Path(d).glob('*.npz')):
                x = _temporal_norm(np.load(str(npz), allow_pickle=True)['x'], None)
                T = x.shape[1]
                for s in range(0, T - CLIP_LEN + 1, CLIP_LEN):
                    self.items.append(x[:, s:s+CLIP_LEN].copy())   # PAS de label
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return torch.from_numpy(self.items[i])


def sinc_loss(pred, fps):
    pred = pred - pred.mean(-1, keepdim=True)
    pred = pred / (pred.std(-1, keepdim=True) + 1e-6)
    n = pred.shape[-1]
    psd = torch.fft.rfft(pred, dim=-1).abs() ** 2                 # (B,F)
    freqs = torch.fft.rfftfreq(n, 1/fps).to(pred.device)
    band = (freqs >= LOW) & (freqs <= HIGH)
    fb = freqs[band]
    psd_b = psd[:, band] + 1e-8
    p = psd_b / psd_b.sum(-1, keepdim=True)                       # distribution normalisée
    # (1) power : maximiser la part de puissance DANS la bande
    power = psd_b.sum(-1) / (psd.sum(-1) + 1e-8)
    l_power = (1 - power).mean()
    # (2) bandwidth : concentrer autour du pic (faible dispersion fréquentielle)
    peak = fb[p.argmax(-1)]
    bw = (p * (fb[None, :] - peak[:, None]) ** 2).sum(-1).sqrt()
    l_bw = bw.mean()
    # (3) variance : diversité des FC dans le batch (anti-collapse)
    l_var = -peak.var()
    return l_power + 0.4 * l_bw + 0.3 * l_var, peak


def main():
    dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    subs = sorted(d for d in (ROOT/'Data'/'region_signals').iterdir() if d.is_dir())
    held = sorted(d for d in (ROOT/'Data'/'region_new').iterdir() if d.is_dir())
    ds = DS(subs); ld = DataLoader(ds, batch_size=32, shuffle=True, drop_last=True)
    print(f"Entraînement SSL (sans labels) : {len(ds)} fenêtres / {len(subs)} sujets")
    model = CNN1D_rPPG(in_channels=23*9).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    def eval_hr():
        model.eval(); errs, ph, gh = [], [], []
        with torch.no_grad():
            for d in held:
                for npz in sorted(Path(d).glob('*.npz')):
                    dat = np.load(str(npz), allow_pickle=True); fps = float(dat['fps'])
                    x = _temporal_norm(dat['x'], None); T = x.shape[1]; preds = []
                    for s in range(0, T-CLIP_LEN+1, CLIP_LEN):
                        preds.append(model(torch.from_numpy(x[:, s:s+CLIP_LEN]).unsqueeze(0).to(dev)).squeeze().cpu().numpy())
                    if not preds: continue
                    sig = bandpass_numpy(np.concatenate(preds), fps)
                    gt = bandpass_numpy(dat['y'].astype(np.float32), fps)
                    p, g = hr_from_fft(sig, fps), hr_from_fft(gt, fps)
                    errs.append(abs(p-g)); ph.append(p); gh.append(g)
        e = np.array(errs)
        corr = np.corrcoef(ph, gh)[0, 1] if len(e) > 2 else float('nan')
        return e.mean(), 100*(e < 5).mean(), corr

    for ep in range(1, 41):
        model.train(); tl = 0.0
        for x in ld:
            x = x.to(dev); opt.zero_grad()
            loss, _ = sinc_loss(model(x), 30.0); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); tl += loss.item()
        if ep % 5 == 0 or ep == 1:
            mae, p5, corr = eval_hr()
            print(f"ep{ep:3d}  loss={tl/len(ld):+.3f}  | held-out MAE={mae:.2f}  %<5={p5:.0f}%  corr(FC,vérité)={corr:+.2f}")
    torch.save(model.state_dict(), ROOT/'weights'/'cnn1d_ssl.pth')
    print("→ weights/cnn1d_ssl.pth")


if __name__ == '__main__':
    main()
