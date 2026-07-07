#!/usr/bin/env python3
"""
Réentraînement CNN1D anti-surapprentissage : teste plusieurs configs
(canaux couleur × régions × capacité × dropout × époques) et évalue CHACUNE
sur le HELD-OUT region_new (22 sujets jamais vus), au niveau scénario.

Diagnostic de COLLAPSE (le défaut du modèle actuel) :
  - corr(FC prédite, FC réelle) sur les scénarios held-out  → doit être > 0
  - écart-type des FC prédites                              → doit suivre celui des GT
Un modèle collapsé prédit ~constant (std faible, corr ~0/négative).

Usage : python scripts/retrain_cnn1d_experiment.py
"""
import os, sys, random, json
from pathlib import Path
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.cnn1d_rppg import CNN1D_rPPG
from mp_rppg.metrics import hr_from_fft, snr
from models.chrom_adaptive import bandpass_numpy

CLIP_LEN = 128
ANAT = list(range(7))   # 7 régions anatomiques (le reste = grille 4x4)


def pearson_loss(pred, target):
    pred = pred - pred.mean(-1, keepdim=True)
    target = target - target.mean(-1, keepdim=True)
    num = (pred * target).sum(-1)
    den = pred.pow(2).sum(-1).sqrt() * target.pow(2).sum(-1).sqrt() + 1e-8
    return (-num / den).mean()


def norm(x, region_idx, color_idx):
    """x:(T,R,C) -> (Rsel*Csel, T) normalisé par canal."""
    if region_idx is not None:
        x = x[:, region_idx, :]
    if color_idx is not None:
        x = x[:, :, color_idx]
    T, R, C = x.shape
    flat = x.reshape(T, R * C).astype(np.float32)
    flat = flat / (flat.mean(0, keepdims=True) + 1e-8) - 1.0
    return flat.T


class DS(Dataset):
    def __init__(self, dirs, region_idx, color_idx):
        self.items = []
        for d in dirs:
            for npz in sorted(Path(d).glob('*.npz')):
                dat = np.load(str(npz), allow_pickle=True)
                x = norm(dat['x'], region_idx, color_idx); y = dat['y'].astype(np.float32)
                T = x.shape[1]
                for s in range(0, T - CLIP_LEN + 1, CLIP_LEN):
                    yw = y[s:s+CLIP_LEN]; yw = (yw - yw.mean())/(yw.std()+1e-8)
                    self.items.append((x[:, s:s+CLIP_LEN].copy(), yw.copy()))

    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        x, y = self.items[i]; return torch.from_numpy(x), torch.from_numpy(y)


def heldout_eval(model, dev, heldout_dirs, region_idx, color_idx):
    """Scénario-level sur les sujets held-out. Retourne dict de métriques."""
    model.eval(); pred_hrs, gt_hrs, errs = [], [], []
    with torch.no_grad():
        for d in heldout_dirs:
            for npz in sorted(Path(d).glob('*.npz')):
                dat = np.load(str(npz), allow_pickle=True); fps = float(dat['fps'])
                x = norm(dat['x'], region_idx, color_idx); T = x.shape[1]
                gt = bandpass_numpy(dat['y'].astype(np.float32), fps)
                preds = []
                for s in range(0, T - CLIP_LEN + 1, CLIP_LEN):
                    xw = torch.from_numpy(x[:, s:s+CLIP_LEN]).unsqueeze(0).to(dev)
                    preds.append(model(xw).squeeze().cpu().numpy())
                if not preds: continue
                sig = bandpass_numpy(np.concatenate(preds), fps)
                hp, hg = hr_from_fft(sig, fps), hr_from_fft(gt, fps)
                pred_hrs.append(hp); gt_hrs.append(hg); errs.append(abs(hp-hg))
    e = np.array(errs); ph = np.array(pred_hrs); gh = np.array(gt_hrs)
    corr = float(np.corrcoef(ph, gh)[0, 1]) if len(e) > 2 else float('nan')
    return {'MAE': float(e.mean()), 'med': float(np.median(e)),
            'p5': float(100*(e < 5).mean()), 'corr': corr,
            'pred_std': float(ph.std()), 'gt_std': float(gh.std()), 'n': len(e)}


def train_eval(cfg, train_d, val_d, heldout_d, dev):
    ri = cfg['regions']; ci = cfg['colors']
    ds_tr, ds_va = DS(train_d, ri, ci), DS(val_d, ri, ci)
    in_ch = ds_tr[0][0].shape[0]
    model = CNN1D_rPPG(in_channels=in_ch, hidden=cfg['hidden'], dropout=cfg['dropout']).to(dev)
    npar = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=cfg['wd'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg['epochs'], eta_min=1e-5)
    ld_tr = DataLoader(ds_tr, batch_size=16, shuffle=True)
    ld_va = DataLoader(ds_va, batch_size=16, shuffle=False)
    best, best_state = float('inf'), None
    for ep in range(cfg['epochs']):
        model.train()
        for x, y in ld_tr:
            x, y = x.to(dev), y.to(dev); opt.zero_grad()
            loss = pearson_loss(model(x), y); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval(); vl = 0.0
        with torch.no_grad():
            for x, y in ld_va:
                vl += pearson_loss(model(x.to(dev)), y.to(dev)).item()
        vl /= max(len(ld_va), 1); sched.step()
        if vl < best:
            best = vl; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    m = heldout_eval(model, dev, heldout_d, ri, ci)
    m['in_ch'] = in_ch; m['params'] = npar; m['val'] = best
    return m, model


def main():
    dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device : {dev}")
    train_root = ROOT / 'Data' / 'region_signals'
    heldout_root = ROOT / 'Data' / 'region_new'
    subs = sorted(d for d in train_root.iterdir() if d.is_dir())
    random.seed(42); random.shuffle(subs)
    n_val = max(1, int(len(subs) * 0.12))
    val_d, train_d = subs[:n_val], subs[n_val:]
    heldout_d = sorted(d for d in heldout_root.iterdir() if d.is_dir())
    print(f"train {len(train_d)} / val {len(val_d)} / held-out {len(heldout_d)}\n")

    RGB = [0, 1, 2]
    configs = {
        'A_baseline_207ch_h64':   dict(regions=None, colors=None, hidden=64, dropout=0.1, wd=1e-4, epochs=40),
        'B_rgb_69ch_h32':         dict(regions=None, colors=RGB, hidden=32, dropout=0.3, wd=1e-3, epochs=40),
        'C_anat_rgb_21ch_h32':    dict(regions=ANAT, colors=RGB, hidden=32, dropout=0.3, wd=1e-3, epochs=40),
        'D_anat_rgb_21ch_h16_ep25': dict(regions=ANAT, colors=RGB, hidden=16, dropout=0.4, wd=3e-3, epochs=25),
    }
    print(f"{'config':<26}{'in':>4}{'par':>7}{'MAE':>7}{'med':>6}{'%<5':>5}"
          f"{'corr':>7}{'pstd':>6}{'gstd':>6}")
    print('-' * 74)
    results = {}
    best_name, best_mae, best_model = None, float('inf'), None
    for name, cfg in configs.items():
        m, model = train_eval(cfg, train_d, val_d, heldout_d, dev)
        results[name] = m
        flag = ' COLLAPSE' if m['corr'] < 0.2 or m['pred_std'] < 0.5 * m['gt_std'] else ''
        print(f"{name:<26}{m['in_ch']:>4}{m['params']/1000:>6.0f}k{m['MAE']:>7.2f}"
              f"{m['med']:>6.1f}{m['p5']:>4.0f}%{m['corr']:>7.2f}{m['pred_std']:>6.1f}{m['gt_std']:>6.1f}{flag}")
        # garde le meilleur NON-collapsé par MAE
        if not flag and m['MAE'] < best_mae:
            best_mae, best_name, best_model = m['MAE'], name, model
    print('-' * 74)
    if best_model is not None:
        out = ROOT / 'weights' / 'cnn1d_rppg_v2.pth'
        torch.save(best_model.state_dict(), out)
        cfg = configs[best_name]
        meta = {'config': best_name, 'regions': cfg['regions'], 'colors': cfg['colors'],
                'hidden': cfg['hidden'], 'in_ch': results[best_name]['in_ch'],
                'metrics': results[best_name]}
        (ROOT / 'weights' / 'cnn1d_rppg_v2.json').write_text(json.dumps(meta, indent=2))
        print(f"MEILLEUR non-collapsé : {best_name} (held-out MAE {best_mae:.2f}) → {out}")
    else:
        print("⚠️ Toutes les configs collapsent encore.")


if __name__ == '__main__':
    main()
