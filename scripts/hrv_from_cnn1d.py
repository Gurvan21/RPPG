#!/usr/bin/env python3
"""
HRV depuis l'onde rPPG produite par CNN1D (mode VIDÉO ENTIÈRE), comparée à la
référence (pleth CMS), au niveau scénario, sur le held-out.

Détection de battements robuste appliquée IDENTIQUEMENT aux deux signaux :
  bandpass cardiaque -> pics (find_peaks + prominence) -> raffinement parabolique
  -> IBI (ms) -> filtre physiologique (300-2000) + rejet d'artefacts (>25% écart
  à la médiane = battement raté/double) -> RMSSD, SDNN.
Un scénario est jugé fiable si < 30% des IBI sont rejetés.

NB : la HRV ne nécessite PAS de synchronisation rPPG<->référence : chaque signal
fournit sa propre série d'IBI sur la même période.
"""
import os, sys, json
from pathlib import Path
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import numpy as np
import torch
from scipy.signal import butter, filtfilt, find_peaks

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.cnn1d_rppg import CNN1D_rPPG
from scripts.train_cnn1d import _temporal_norm
from scripts.vitals_reference import process_subject  # pour fitz/format si besoin


def _bp(sig, fs, lo=0.7, hi=3.5, order=3):
    nyq = fs / 2.0
    b, a = butter(order, [max(lo/nyq, 1e-4), min(hi/nyq, 0.999)], btype='band')
    return filtfilt(b, a, sig.astype(float))


def _interp_peaks(sig, idx):
    out = []
    for i in idx:
        if 0 < i < len(sig) - 1:
            y0, y1, y2 = sig[i-1], sig[i], sig[i+1]
            den = y0 - 2*y1 + y2
            out.append(i + 0.5*(y0 - y2)/den if den != 0 else float(i))
        else:
            out.append(float(i))
    return np.array(out)


def robust_ibi(sig, fs):
    """Retourne (ibi_ms_corrigés, frac_rejetés, n_beats)."""
    s = _bp(sig, fs)
    s = (s - s.mean()) / (s.std() + 1e-8)
    peaks, _ = find_peaks(s, distance=int(0.33 * fs), prominence=0.3)
    if len(peaks) < 5:
        return np.array([]), 1.0, len(peaks)
    pk = _interp_peaks(s, peaks)
    pk_t = pk / fs
    ibi = np.diff(pk_t) * 1000.0
    n0 = len(ibi)
    phys = (ibi > 300) & (ibi < 2000)
    ibi = ibi[phys]
    if len(ibi) < 4:
        return np.array([]), 1.0, len(peaks)
    med = np.median(ibi)
    keep = np.abs(ibi - med) <= 0.25 * med        # rejet battements ratés/doubles
    ibi_c = ibi[keep]
    frac_rej = 1.0 - len(ibi_c) / n0
    return ibi_c, frac_rej, len(peaks)


def hrv(ibi):
    if len(ibi) < 4:
        return np.nan, np.nan
    return float(np.sqrt(np.mean(np.diff(ibi)**2))), float(np.std(ibi, ddof=1))


def ref_pleth(subject_dir, sc_idx):
    js = [j for j in subject_dir.glob('*.json') if j.name != 'metadata.json']
    if not js:
        return None
    d = json.load(open(js[0]))
    scs = d.get('scenarios', [])
    if sc_idx >= len(scs):
        return None
    cms = scs[sc_idx].get('recordings', {}).get('CMS')
    if not cms or len(cms) < 5:
        return None
    body = cms[1:] if cms[0][0] == 'time' else cms
    arr = np.array([[float(r[0]), float(r[1])] for r in body])
    t = (arr[:, 0] - arr[0, 0]) / 1000.0
    fs = 1.0 / np.median(np.diff(t))
    return arr[:, 1], fs


def main():
    dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    m = CNN1D_rPPG(in_channels=23*9).to(dev)
    m.load_state_dict(torch.load(ROOT/'weights'/'cnn1d_rppg.pth', map_location=dev)); m.eval()
    rdir = ROOT / 'Data' / 'region_new'

    print(f"{'Sujet':<12}{'sc':>3}{'Fz':>3} | {'RMSSD_ref':>10}{'RMSSD_cnn':>10}{'err':>7}"
          f" | {'rejREF':>7}{'rejCNN':>7}")
    print('-'*68)
    rows = []
    for d in sorted(p for p in rdir.iterdir() if p.is_dir()):
        for npz in sorted(d.glob('*.npz')):
            sc = int(npz.stem.replace('sc', ''))
            dat = np.load(str(npz), allow_pickle=True); fps = float(dat['fps'])
            fz = str(dat['region_names']) and '?'  # placeholder
            # --- rPPG CNN1D vidéo entière ---
            x = _temporal_norm(dat['x'], None)
            with torch.no_grad():
                wav = m(torch.from_numpy(x).unsqueeze(0).to(dev)).squeeze().cpu().numpy()
            ibi_c, rej_c, _ = robust_ibi(wav, fps)
            rmssd_c, sdnn_c = hrv(ibi_c)
            # --- référence pleth ---
            pr = ref_pleth(ROOT/'DataVital'/d.name, sc)
            if pr is None:
                continue
            ibi_r, rej_r, _ = robust_ibi(pr[0], pr[1])
            rmssd_r, sdnn_r = hrv(ibi_r)
            ok = (rej_r < 0.30) and (rej_c < 0.30) and np.isfinite(rmssd_r) and np.isfinite(rmssd_c)
            err = abs(rmssd_c - rmssd_r) if ok else np.nan
            rows.append((rmssd_r, rmssd_c, err, ok, rej_r, rej_c))
            mark = '' if ok else '  (écarté: trop d artefacts)'
            print(f"{d.name:<12}{sc:>3}{'':>3} | {rmssd_r:>10.1f}{rmssd_c:>10.1f}"
                  f"{(err if ok else float('nan')):>7.1f} | {100*rej_r:>6.0f}%{100*rej_c:>6.0f}%{mark}")

    valid = [r for r in rows if r[3]]
    if valid:
        rr = np.array([r[0] for r in valid]); rc = np.array([r[1] for r in valid])
        err = np.abs(rc - rr)
        corr = np.corrcoef(rr, rc)[0, 1]
        print('-'*68)
        print(f"Scénarios fiables : {len(valid)}/{len(rows)}")
        print(f"RMSSD — MAE = {err.mean():.1f} ms | médiane err = {np.median(err):.1f} ms"
              f" | corr(ref,cnn) = {corr:.2f}")
        print(f"  RMSSD ref  : {rr.mean():.0f}±{rr.std():.0f} ms   (plage {rr.min():.0f}-{rr.max():.0f})")
        print(f"  RMSSD cnn  : {rc.mean():.0f}±{rc.std():.0f} ms   (plage {rc.min():.0f}-{rc.max():.0f})")


if __name__ == '__main__':
    main()
