#!/usr/bin/env python3
"""Régression de la RESPIRATION depuis la paume (CNN1D à grand champ réceptif).
Cible = signal rr (ceinture de pression) rééchantillonné à 30 fps + passe-bande
respiratoire (0.1-0.5 Hz). Entrée = signaux paume (hand_signals). Éval : fréquence
respiratoire (pic FFT en bande resp.) vs référence, CV groupée par personne."""
import os, sys, json, glob
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from scipy.signal import butter, filtfilt, periodogram
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from models.cnn1d_rppg import CNN1D_rPPG
from scripts.train_cnn1d import _temporal_norm, pearson_loss
dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
RLO, RHI = 0.1, 0.5        # bande respiratoire (6-30 resp/min)


def resp_bp(sig, fps):
    b, a = butter(2, [RLO/(fps/2), RHI/(fps/2)], btype='band')
    return filtfilt(b, a, sig).astype(np.float32)


def resp_rate(sig, fps):
    n = 1
    while n < len(sig): n *= 2
    f, px = periodogram(sig, fs=fps, nfft=n, detrend=False)
    m = (f >= RLO) & (f <= RHI)
    return float(f[m][np.argmax(px[m])] * 60)


def build():
    """(subject_id, x_norm (C,T), rr_bp (T,), fps, rr_rate_ref) par facepalm avec rr."""
    data = []
    for jf in sorted(glob.glob(str(ROOT/"DataVital"/"Subject*"/"*.json"))):
        try: d = json.load(open(jf))
        except: continue
        subj = jf.split('/')[-2]
        for si, sc in enumerate(d.get("scenarios", [])):
            if sc.get("scenario_data",{}).get("scenario") != "facepalm": continue
            rr = sc.get("recordings",{}).get("rr")
            npz = ROOT/"Data"/"hand_signals"/subj.replace(' ', '_')/f"sc{si}.npz"
            if not (rr and rr.get("timeseries") and npz.exists()): break
            dt = np.load(str(npz), allow_pickle=True); fps = float(dt['fps'])
            xn = _temporal_norm(dt['x']); T = xn.shape[1]
            ts = np.array([p[0] for p in rr["timeseries"]], float)
            vs = np.array([p[1] for p in rr["timeseries"]], float)
            new_t = np.arange(T)*1000.0/fps
            rr_res = np.interp(new_t, ts, vs)
            rr_bp = resp_bp(rr_res, fps)
            if T < 128: break
            data.append((subj, xn.astype(np.float32), rr_bp, fps, resp_rate(rr_bp, fps)))
            break
    return data


def main():
    data = build()
    subs = sorted(set(d[0] for d in data))
    print(f"{len(data)} facepalm avec respiration / {len(subs)} sujets, device {dev}")
    rng = np.random.default_rng(0); order = rng.permutation(len(subs)); K = 5
    folds = [set(np.array(subs)[order[i::K]]) for i in range(K)]
    errs = []
    for k in range(K):
        te = [d for d in data if d[0] in folds[k]]
        tr = [d for d in data if d[0] not in folds[k]]
        m = CNN1D_rPPG(in_channels=18*9, dilations=(1, 2, 4, 8, 16, 32)).to(dev)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
        for ep in range(120):
            m.train()
            for subj, xn, rr, fps, _ in tr:
                x = torch.from_numpy(xn).unsqueeze(0).to(dev)
                y = torch.from_numpy((rr-rr.mean())/(rr.std()+1e-8)).unsqueeze(0).to(dev)
                opt.zero_grad(); loss = pearson_loss(m(x), y); loss.backward()
                nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        m.eval(); fold_err = []
        with torch.no_grad():
            for subj, xn, rr, fps, ref in te:
                p = m(torch.from_numpy(xn).unsqueeze(0).to(dev)).squeeze().cpu().numpy()
                pr = resp_rate(resp_bp(p, fps), fps)
                fold_err.append(abs(pr - ref)); errs.append(abs(pr - ref))
        print(f"  fold {k+1}/{K}: n={len(te)} MAE={np.mean(fold_err):.1f} resp/min")
    e = np.array(errs)
    print(f"\n=== RESPIRATION (CV 5-fold, sans fuite) ===")
    print(f"  MAE fréquence resp. = {e.mean():.1f} resp/min   %<3 = {100*(e<3).mean():.0f}   (réf médiane {np.median([d[4] for d in data]):.0f} resp/min)")


if __name__ == '__main__':
    main()
