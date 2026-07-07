#!/usr/bin/env python3
"""
Estimation de la fréquence respiratoire (RR) depuis la vidéo, validée contre la
ceinture thoracique de VitalVideos.

Trois modulations respiratoires extraites de l'onde rPPG (POS sur région front) :
  - RSA  : oscillation de la série d'intervalles inter-battements (fréquence)
  - RIAV : oscillation de l'amplitude du pouls
  - RIIV : oscillation de la ligne de base de l'intensité de la peau
Fusion = médiane des 3 si elles s'accordent, sinon abstention.

Référence ceinture vérifiée 2 façons (FFT + comptage de pics) pour détecter une
éventuelle harmonique.
"""
import sys, json
from pathlib import Path
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

sys.path.insert(0, '.')
from mp_rppg.methods import pos

ROOT = Path('.')
RESP_LO, RESP_HI = 0.1, 0.5      # 6–30 br/min


def bp(s, fs, lo, hi, order=2):
    s = np.asarray(s, float)
    b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype='band')
    return filtfilt(b, a, s)


def fft_peak_bpm(sig, fs, lo=RESP_LO, hi=RESP_HI):
    sig = sig - sig.mean()
    n = 1 << int(np.ceil(np.log2(len(sig) * 4)))
    f = np.fft.rfftfreq(n, 1 / fs); P = np.abs(np.fft.rfft(sig, n)) ** 2
    band = (f >= lo) & (f <= hi)
    if not band.any():
        return np.nan
    return f[band][np.argmax(P[band])] * 60


def belt_rr(t_ms, force):
    t = (t_ms - t_ms[0]) / 1000.0
    fs = 1.0 / np.median(np.diff(t))
    s = bp(force, fs, RESP_LO, RESP_HI)
    rr_fft = fft_peak_bpm(s, fs)
    pk, _ = find_peaks(s, distance=int(fs / (RESP_HI)))   # ≥ 1 respiration toutes 2s
    dur_min = (t[-1] - t[0]) / 60.0
    rr_peaks = len(pk) / dur_min if dur_min > 0 else np.nan
    return rr_fft, rr_peaks


def resample_uniform(times, vals, fs=4.0, dur=None):
    if len(times) < 4:
        return None, fs
    dur = dur or (times[-1] - times[0])
    grid = np.arange(times[0], times[0] + dur, 1 / fs)
    return np.interp(grid, times, vals), fs


def video_rr(rgb, fps):
    """RSA, RIAV, RIIV (br/min) depuis la région front."""
    pulse = bp(pos(rgb, fps), fps, 0.7, 3.5)
    peaks, _ = find_peaks(pulse, distance=int(0.4 * fps))
    out = {}
    if len(peaks) >= 6:
        pt = peaks / fps
        ibi = np.diff(pt) * 1000.0                       # ms
        sig, fsr = resample_uniform(pt[1:], ibi, 4.0)    # tachogramme
        if sig is not None:
            out['RSA'] = fft_peak_bpm(bp(sig, fsr, RESP_LO, RESP_HI), fsr)
        amp = pulse[peaks]                                # amplitude par battement
        sig2, fsr2 = resample_uniform(pt, amp, 4.0)
        if sig2 is not None:
            out['RIAV'] = fft_peak_bpm(bp(sig2, fsr2, RESP_LO, RESP_HI), fsr2)
    # RIIV : baseline d'intensité (moyenne RGB front)
    inten = rgb.mean(1)
    out['RIIV'] = fft_peak_bpm(bp(inten, fps, RESP_LO, RESP_HI), fps)
    return out


def main():
    rows = []
    for d in sorted((ROOT / 'Data' / 'region_new').iterdir()):
        if not d.is_dir():
            continue
        js = [j for j in (ROOT / 'DataVital' / d.name).glob('*.json') if j.name != 'metadata.json']
        if not js:
            continue
        J = json.load(open(js[0]))
        for npz in sorted(d.glob('*.npz')):
            sc = int(npz.stem.replace('sc', ''))
            if sc >= len(J.get('scenarios', [])):
                continue
            rr = J['scenarios'][sc].get('recordings', {}).get('rr', {}).get('timeseries')
            if not rr or len(rr) < 10:
                continue
            ra = np.array([[float(x[0]), float(x[1])] for x in rr])
            rr_fft, rr_pk = belt_rr(ra[:, 0], ra[:, 1])
            dat = np.load(str(npz), allow_pickle=True); fps = float(dat['fps'])
            rgb = dat['x'][:, 0, :3].astype(np.float32)
            v = video_rr(rgb, fps)
            rows.append((d.name, sc, rr_fft, rr_pk, v))

    # ── 1) Vérif référence ceinture ──
    print("=== Référence ceinture : FFT vs comptage de pics ===")
    bf = np.array([r[2] for r in rows]); bp_ = np.array([r[3] for r in rows])
    print(f"  belt FFT   : {np.nanmean(bf):.1f} ± {np.nanstd(bf):.1f}  (plage {np.nanmin(bf):.0f}-{np.nanmax(bf):.0f})")
    print(f"  belt pics  : {np.nanmean(bp_):.1f} ± {np.nanstd(bp_):.1f}  (plage {np.nanmin(bp_):.0f}-{np.nanmax(bp_):.0f})")
    print(f"  écart moyen FFT-pics : {np.nanmean(np.abs(bf - bp_)):.1f} br/min")
    ref = bp_   # on prend le comptage de pics comme référence (plus robuste aux harmoniques)

    # ── 2) Modulations vidéo + fusion ──
    def fuse(v):
        vals = [v[k] for k in ('RSA', 'RIAV', 'RIIV') if k in v and np.isfinite(v[k])]
        if not vals:
            return np.nan, 99
        return float(np.median(vals)), float(np.std(vals))

    print(f"\n{'Sujet':<12}{'sc':>3}{'BELT':>6}{'RSA':>6}{'RIAV':>6}{'RIIV':>6}{'FUSION':>8}{'std':>6}{'err':>6}")
    print('-' * 60)
    e = {k: [] for k in ('RSA', 'RIAV', 'RIIV', 'fus', 'fus_ok')}
    for (name, sc, _, _, v), rr in zip(rows, ref):
        fr, std = fuse(v)
        for k in ('RSA', 'RIAV', 'RIIV'):
            if k in v and np.isfinite(v[k]):
                e[k].append(abs(v[k] - rr))
        if np.isfinite(fr):
            e['fus'].append(abs(fr - rr))
            if std <= 4:                     # accord des 3 modulations
                e['fus_ok'].append(abs(fr - rr))
        g = lambda k: v.get(k, np.nan)
        print(f"{name:<12}{sc:>3}{rr:>6.1f}{g('RSA'):>6.1f}{g('RIAV'):>6.1f}{g('RIIV'):>6.1f}"
              f"{fr:>8.1f}{std:>6.1f}{abs(fr-rr):>6.1f}")
    print('-' * 60)
    for k, lab in [('RSA', 'RSA seule'), ('RIAV', 'RIAV seule'), ('RIIV', 'RIIV seule'),
                   ('fus', 'FUSION (tous)'), ('fus_ok', 'FUSION (si accord std≤4)')]:
        a = np.array(e[k])
        if len(a):
            print(f"{lab:<26} N={len(a):>3}  MAE={a.mean():.1f}  méd={np.median(a):.1f}  %<3br/min={100*(a<3).mean():.0f}%")


if __name__ == '__main__':
    main()
