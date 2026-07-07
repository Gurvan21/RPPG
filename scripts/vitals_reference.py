#!/usr/bin/env python3
"""
Extraction de la VÉRITÉ-TERRAIN des 3 constantes vitales défendables depuis un
JSON VitalVideos, par scénario :
  - FC (bpm)        : depuis le pleth CMS (FFT + pics)
  - HRV (ms)        : RMSSD, SDNN depuis les intervalles inter-battements (IBI)
                      du pleth CMS  → en réalité PRV (pulse rate variability)
  - Respiration (br/min) : depuis la CEINTURE thoracique ('rr', pression newtons)

Hypothèses de format (vérifiées sur DataVital) :
  - timestamps en MILLISECONDES
  - CMS  : [['time','ppg','hr','spo2'], [t,ppg,hr,spo2], ...]  ~62.5 Hz
  - rr   : {'timeseries': [[t, force_newton], ...]}            ~20 Hz

Usage : python scripts/vitals_reference.py --subject "Subject 100"
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

ROOT = Path(__file__).resolve().parents[1]


def _bp(sig, fs, lo, hi, order=3):
    nyq = fs / 2.0
    lo_n, hi_n = max(lo / nyq, 1e-4), min(hi / nyq, 0.999)
    b, a = butter(order, [lo_n, hi_n], btype='band')
    return filtfilt(b, a, sig)


def _interp_peak(sig, idx):
    """Raffinement sous-échantillon de la position d'un pic (parabole)."""
    out = []
    for i in idx:
        if 0 < i < len(sig) - 1:
            y0, y1, y2 = sig[i-1], sig[i], sig[i+1]
            denom = (y0 - 2*y1 + y2)
            out.append(i + 0.5*(y0 - y2)/denom if denom != 0 else i)
        else:
            out.append(float(i))
    return np.array(out)


def hr_hrv_from_pleth(t_ms, ppg):
    """Retourne (HR_bpm, RMSSD_ms, SDNN_ms, n_beats, fs)."""
    t = (t_ms - t_ms[0]) / 1000.0
    fs = 1.0 / np.median(np.diff(t))
    sig = _bp(ppg.astype(float), fs, 0.5, 4.0)
    # pics = battements ; distance mini ~0.4 s (FC max ~150)
    peaks, _ = find_peaks(sig, distance=int(0.4 * fs))
    if len(peaks) < 4:
        return np.nan, np.nan, np.nan, len(peaks), fs
    pk = _interp_peak(sig, peaks)
    pk_t = np.interp(pk, np.arange(len(t)), t)        # instants (s)
    ibi = np.diff(pk_t) * 1000.0                       # ms
    ibi = ibi[(ibi > 300) & (ibi < 2000)]              # physiologique
    hr = 60000.0 / np.mean(ibi)
    rmssd = np.sqrt(np.mean(np.diff(ibi) ** 2))
    sdnn = np.std(ibi, ddof=1)
    return hr, rmssd, sdnn, len(ibi) + 1, fs


def rr_from_belt(t_ms, force):
    """Fréquence respiratoire (br/min) depuis la ceinture thoracique."""
    t = (t_ms - t_ms[0]) / 1000.0
    fs = 1.0 / np.median(np.diff(t))
    sig = _bp(force.astype(float), fs, 0.1, 0.6, order=2)
    # FFT zero-paddée → pic dans 0.1-0.6 Hz
    n = 1 << int(np.ceil(np.log2(len(sig) * 4)))
    f = np.fft.rfftfreq(n, 1/fs); P = np.abs(np.fft.rfft(sig, n)) ** 2
    band = (f >= 0.1) & (f <= 0.6)
    rr_hz = f[band][np.argmax(P[band])]
    return rr_hz * 60.0, fs


def process_subject(subject_dir: Path):
    js = [j for j in subject_dir.glob('*.json') if j.name != 'metadata.json']
    if not js:
        print(f"  [pas de JSON] {subject_dir.name}"); return []
    d = json.load(open(js[0]))
    fz = d.get('participant', {}).get('fitzpatrick', '?')
    rows = []
    for si, sc in enumerate(d.get('scenarios', [])):
        rec = sc.get('recordings', {})
        out = {'subject': subject_dir.name, 'scenario': si, 'fitz': fz}
        # FC + HRV depuis CMS
        cms = rec.get('CMS')
        if cms and len(cms) > 5:
            body = cms[1:] if cms[0][0] == 'time' else cms
            arr = np.array([[float(r[0]), float(r[1])] for r in body])
            hr, rmssd, sdnn, nb, fs = hr_hrv_from_pleth(arr[:, 0], arr[:, 1])
            out.update(HR=hr, RMSSD=rmssd, SDNN=sdnn, n_beats=nb, pleth_fs=fs)
        # Respiration depuis ceinture
        rr = rec.get('rr', {}).get('timeseries')
        if rr and len(rr) > 5:
            ra = np.array([[float(x[0]), float(x[1])] for x in rr])
            rrate, rfs = rr_from_belt(ra[:, 0], ra[:, 1])
            out.update(RR=rrate, belt_fs=rfs)
        rows.append(out)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subject', nargs='*', help="ex: 'Subject 100'")
    ap.add_argument('--all-heldout', action='store_true',
                    help="tous les sujets de Data/region_new")
    args = ap.parse_args()

    if args.all_heldout:
        names = sorted(p.name for p in (ROOT / 'Data' / 'region_new').iterdir() if p.is_dir())
    elif args.subject:
        names = args.subject
    else:
        ap.error("--subject ou --all-heldout")

    print(f"{'Sujet':<12}{'sc':>3}{'Fz':>3}{'HR':>7}{'RMSSD':>8}{'SDNN':>7}"
          f"{'RR':>7}{'plFs':>6}{'beats':>6}")
    print('-' * 60)
    for nm in names:
        for r in process_subject(ROOT / 'DataVital' / nm):
            print(f"{r['subject']:<12}{r['scenario']:>3}{str(r.get('fitz','?')):>3}"
                  f"{r.get('HR',float('nan')):>7.1f}{r.get('RMSSD',float('nan')):>8.1f}"
                  f"{r.get('SDNN',float('nan')):>7.1f}{r.get('RR',float('nan')):>7.1f}"
                  f"{r.get('pleth_fs',float('nan')):>6.0f}{r.get('n_beats',0):>6d}")


if __name__ == '__main__':
    main()
