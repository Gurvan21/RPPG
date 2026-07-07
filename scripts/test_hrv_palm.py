#!/usr/bin/env python3
"""Test HRV : sur enregistrements paume propres, passe-bande FC±10 / FC±20 (phase
nulle) sur le signal rPPG, détecte les battements → SDNN/RMSSD, compare à la
référence PPG de contact CMS (y). Clips ~20s → indicatif."""
import os, sys
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
from pathlib import Path
import numpy as np, torch
from scipy.signal import butter, filtfilt, find_peaks
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from models.cnn1d_rppg import CNN1D_rPPG
from models.chrom_adaptive import bandpass_numpy
from mp_rppg.metrics import hr_from_fft, snr
from scripts.train_cnn1d import _temporal_norm, CLIP_LEN
dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
m = CNN1D_rPPG(in_channels=18*9).to(dev)
m.load_state_dict(torch.load(ROOT/'weights'/'cnn1d_hand.pth', map_location=dev)); m.eval()


def rppg_signal(x, fps):
    xn = _temporal_norm(x); pr = []
    for s in range(0, xn.shape[1]-CLIP_LEN+1, CLIP_LEN):
        with torch.no_grad():
            pr.append(m(torch.from_numpy(xn[:, s:s+CLIP_LEN]).unsqueeze(0).to(dev)).squeeze().cpu().numpy())
    return bandpass_numpy(np.concatenate(pr), fps) if pr else None


def bp(sig, fps, lo, hi):
    ny = fps/2.0; lo = max(lo, 30)/60.0; hi = min(hi, ny*60-1)/60.0
    b, a = butter(3, [lo/ny, hi/ny], btype='band')
    return filtfilt(b, a, sig)


def ibis(sig, fps):
    pk, _ = find_peaks(sig, distance=int(0.4*fps))
    if len(pk) < 4: return None
    return np.diff(pk) / fps * 1000.0  # ms


def hrv(ibi):
    if ibi is None or len(ibi) < 3: return (float('nan'), float('nan'), len(ibi) if ibi is not None else 0)
    sdnn = float(np.std(ibi)); rmssd = float(np.sqrt(np.mean(np.diff(ibi)**2)))
    return sdnn, rmssd, len(ibi)


# sélectionne les enregistrements paume au meilleur SNR rPPG
npzs = sorted(Path(ROOT/'Data'/'hand_signals').glob('*/*.npz'))
scored = []
for p in npzs:
    d = np.load(str(p), allow_pickle=True); fps = float(d['fps'])
    sig = rppg_signal(d['x'], fps)
    if sig is None or len(sig) < 300: continue          # >=10s
    h = hr_from_fft(sig, fps); s = snr(sig, h, fps)
    scored.append((s, p, fps, sig, d['y'].astype(np.float32), h))
scored.sort(key=lambda z: z[0], reverse=True)
top = scored[:6]

print(f"{'enreg.':<22}{'HR':>5}{'SNR':>6}  |  {'REF CMS':>13} {'rPPG bande':>12} {'±20':>10} {'±10':>10}")
print(f"{'':<22}{'':>5}{'':>6}  |  {'SDNN/RMSSD':>13} {'SDNN/RMSSD':>12} {'SD/RM':>10} {'SD/RM':>10}")
rows = []
for s, p, fps, sig, y, h in top:
    ref = hrv(ibis(y, fps))                              # référence contact
    full = hrv(ibis(bp(sig, fps, 42, 150), fps))         # rPPG bande large
    b20 = hrv(ibis(bp(sig, fps, h-20, h+20), fps))
    b10 = hrv(ibis(bp(sig, fps, h-10, h+10), fps))
    name = f"{p.parent.name}/{p.stem}"[:22]
    def f(z): return f"{z[0]:.0f}/{z[1]:.0f}"
    print(f"{name:<22}{h:>5.0f}{s:>+6.1f}  |  {f(ref):>13} {f(full):>12} {f(b20):>10} {f(b10):>10}")
    rows.append((ref, full, b20, b10))

# écart moyen (SDNN) vs référence
print("\nÉcart moyen |SDNN_rPPG - SDNN_ref| (ms) :")
for lbl, i in [("bande large", 1), ("±20 bpm", 2), ("±10 bpm", 3)]:
    diffs = [abs(r[i][0]-r[0][0]) for r in rows if np.isfinite(r[i][0]) and np.isfinite(r[0][0])]
    if diffs: print(f"  {lbl:<12}: {np.mean(diffs):.1f} ms   (RMSSD: "
                     f"{np.mean([abs(r[i][1]-r[0][1]) for r in rows if np.isfinite(r[i][1])]):.1f} ms)")
