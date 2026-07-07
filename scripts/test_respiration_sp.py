#!/usr/bin/env python3
"""Respiration depuis la paume par TRAITEMENT DE SIGNAL (méthode PPG classique),
pas par deep. Trois dérivations respiratoires du pouls (Charlton et al.) :
  RIIV : variation d'intensité de base (baseline wander, canal vert brut)
  RIAV : variation d'amplitude du pouls (enveloppe de Hilbert)
  RIFV : variation de fréquence (RSA, série des IBI)
Chaque candidat -> pic FFT en bande respiratoire -> comparé à la réf ceinture rr.
Aucun entraînement -> aucune fuite. Sur tous les facepalm avec rr."""
import os, sys, json, glob
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
from pathlib import Path
import numpy as np, torch
from scipy.signal import butter, filtfilt, periodogram, find_peaks, hilbert
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from models.cnn1d_rppg import CNN1D_rPPG
from models.chrom_adaptive import bandpass_numpy
from scripts.train_cnn1d import _temporal_norm, CLIP_LEN
dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
m = CNN1D_rPPG(in_channels=18*9).to(dev)
m.load_state_dict(torch.load(ROOT/'weights'/'cnn1d_hand.pth', map_location=dev)); m.eval()
RLO, RHI = 0.1, 0.5


def resp_bp(s, fps):
    b, a = butter(2, [RLO/(fps/2), RHI/(fps/2)], btype='band')
    return filtfilt(b, a, s - np.mean(s)).astype(np.float64)


def resp_rate(s, fps):
    n = 1
    while n < len(s): n *= 2
    f, px = periodogram(s, fs=fps, nfft=n, detrend='linear')
    mk = (f >= RLO) & (f <= RHI)
    return float(f[mk][np.argmax(px[mk])] * 60)


def pulse_of(xn, fps):
    pr = []
    for s in range(0, xn.shape[1]-CLIP_LEN+1, CLIP_LEN):
        with torch.no_grad():
            pr.append(m(torch.from_numpy(xn[:, s:s+CLIP_LEN]).unsqueeze(0).to(dev)).squeeze().cpu().numpy())
    return bandpass_numpy(np.concatenate(pr), fps) if pr else None


def riav(pulse, fps):        # enveloppe d'amplitude
    return resp_rate(resp_bp(np.abs(hilbert(pulse)), fps), fps)


def rifv(pulse, fps):        # RSA : série des intervalles inter-battements
    pk, _ = find_peaks(pulse, distance=int(0.4*fps))
    if len(pk) < 5: return float('nan')
    ibi = np.diff(pk)/fps; t = pk[1:]
    grid = np.arange(pk[0], pk[-1])
    ser = np.interp(grid, t, ibi)
    return resp_rate(resp_bp(ser, fps), fps)


def main():
    rows = []
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
            x = dt['x']; T = x.shape[0]
            if T < CLIP_LEN: break
            ts = np.array([p[0] for p in rr["timeseries"]], float)
            vs = np.array([p[1] for p in rr["timeseries"]], float)
            ref = resp_rate(resp_bp(np.interp(np.arange(T)*1000/fps, ts, vs), fps), fps)
            green = x[:, :, 1].mean(axis=1).astype(np.float64)      # RIIV
            r_riiv = resp_rate(resp_bp(green, fps), fps)
            pulse = pulse_of(_temporal_norm(x), fps)
            r_riav = riav(pulse, fps) if pulse is not None else float('nan')
            r_rifv = rifv(pulse, fps) if pulse is not None else float('nan')
            rows.append((ref, r_riiv, r_riav, r_rifv))
            break
    a = np.array(rows)
    print(f"n = {len(a)} facepalm avec respiration  (réf médiane {np.median(a[:,0]):.0f} resp/min, plage {a[:,0].min():.0f}-{a[:,0].max():.0f})\n")
    for j, nm in [(1, "RIIV (baseline vert)"), (2, "RIAV (amplitude)"), (3, "RIFV (RSA/IBI)")]:
        e = np.abs(a[:, j] - a[:, 0]); e = e[np.isfinite(e)]
        print(f"  {nm:22s}: MAE = {e.mean():5.1f} resp/min   %<3 = {100*(e<3).mean():3.0f}")
    # fusion : médiane des 3 candidats
    fus = np.nanmedian(a[:, 1:], axis=1); e = np.abs(fus - a[:, 0])
    print(f"  {'FUSION (médiane 3)':22s}: MAE = {e.mean():5.1f} resp/min   %<3 = {100*(e<3).mean():3.0f}")
    # baseline bête : prédire toujours la médiane
    e = np.abs(np.median(a[:,0]) - a[:,0])
    print(f"\n  (baseline 'toujours médiane' : MAE = {e.mean():.1f} resp/min)")


if __name__ == '__main__':
    main()
