#!/usr/bin/env python3
"""Idée : récupérer le PPG du VISAGE en filtrant CHROM/POS autour de la FC de
RÉFÉRENCE (doigt/CMS). + TEST DE VALIDITÉ pour distinguer un vrai pouls faible
d'une FABRICATION (bruit mis en forme) :
  (1) SNR du visage À la FC de référence (énergie in-band vs bruit),
  (2) COHÉRENCE spectrale visage↔doigt à cette FC (verrouillage d'amplitude+phase).
Un vrai pouls : SNR>seuil ET cohérence haute. Sinon → fabrication, on rejette."""
import os, sys, json, glob
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
from pathlib import Path
import numpy as np
from scipy.signal import coherence
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.preextract_clips import (load_video, resample_to_fps, TARGET_FPS,
                                      _resample_ppg_to_frames)
from scripts.extract_regions_bisenet import load_bisenet, extract_video as ef, pick_device
from scripts.run_on_video import FULLSKIN, RGB_IDX
from mp_rppg.methods import chrom, pos
from mp_rppg.metrics import hr_from_fft, snr
from models.chrom_adaptive import bandpass_numpy

COH_MIN, SNR_MIN = 0.5, -3.0


def facepalm_items():
    out = []
    for jf in sorted(glob.glob(str(ROOT/"DataVital"/"Subject*"/"*.json"))):
        try: d = json.load(open(jf))
        except: continue
        fitz = d.get("participant", {}).get("fitzpatrick")
        for sc in d.get("scenarios", []):
            if sc.get("scenario_data", {}).get("scenario") != "facepalm": continue
            rec = sc.get("recordings", {}); rgb = rec.get("RGB"); cms = rec.get("CMS")
            if rgb and cms and len(cms) > 2:
                out.append((Path(jf), fitz, rgb, cms))
            break
    return out


def coh_at(a, b, fps, f_hz):
    n = min(len(a), len(b)); a, b = a[:n], b[:n]
    fr, C = coherence(a, b, fs=fps, nperseg=min(128, n//2))
    return float(C[np.argmin(np.abs(fr - f_hz))])


def main(limit):
    dev = pick_device(); net = load_bisenet(dev)
    items = facepalm_items()[:limit]
    print(f"{len(items)} facepalm — récupération VISAGE guidée par FC doigt + validité\n")
    rows = []
    for k, (jf, fitz, rgb, cms) in enumerate(items):
        vp = jf.parent/rgb['filename']
        if not vp.exists(): continue
        try:
            frames, _ = load_video(str(vp), max_dim=640)
            ft = np.array([t for t, _ in rgb['timeseries']], np.float64)
            n = min(len(frames), len(ft)); frames, ft = frames[:n], ft[:n]
            frames, ft = resample_to_fps(frames, ft, TARGET_FPS); fps = TARGET_FPS
            y = _resample_ppg_to_frames(cms[1:], ft)          # PPG doigt (réf), aligné
            ref_hr = hr_from_fft(y, fps)
            x, _, det = ef(net, dev, frames, 4)
            if det < 0.5: print(f"[{k+1}] F{fitz}: visage {det*100:.0f}% (skip)"); continue
            skin = x[:, FULLSKIN, :][:, RGB_IDX].astype(np.float32)
            fc = bandpass_numpy(chrom(skin, fps), fps); fp = bandpass_numpy(pos(skin, fps), fps)
            # validité À la FC de référence
            s_c = snr(fc, ref_hr, fps); s_p = snr(fp, ref_hr, fps)
            coh_c = coh_at(fc, y, fps, ref_hr/60.0); coh_p = coh_at(fp, y, fps, ref_hr/60.0)
            best_snr = max(s_c, s_p); best_coh = max(coh_c, coh_p)
            real = best_snr > SNR_MIN and best_coh > COH_MIN
            rows.append((fitz, ref_hr, best_snr, best_coh, real))
            print(f"[{k+1}] F{fitz} FC_réf={ref_hr:.0f}  visage SNR@réf={best_snr:+.1f}  "
                  f"cohérence={best_coh:.2f}  → {'VRAI pouls' if real else 'FABRICATION (rejet)'}", flush=True)
        except Exception as e:
            print(f"[{k+1}] ERREUR {e}", flush=True)

    if rows:
        nreal = sum(r[4] for r in rows)
        print(f"\n=== {nreal}/{len(rows)} visages ont un VRAI pouls récupérable "
              f"(SNR@réf>{SNR_MIN} ET cohérence>{COH_MIN}) ===")
        print(f"cohérence moyenne : VRAIS {np.mean([r[3] for r in rows if r[4]] or [0]):.2f}  "
              f"| REJETÉS {np.mean([r[3] for r in rows if not r[4]] or [0]):.2f}")
        for fz in ['4', '5', '6']:
            g = [r for r in rows if r[0] == fz]
            if g: print(f"  Fitz{fz}: {sum(r[4] for r in g)}/{len(g)} récupérables")


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20)
