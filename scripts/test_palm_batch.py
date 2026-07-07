#!/usr/bin/env python3
"""Batch paume vs visage sur les scénarios facepalm, agrégé par Fitzpatrick.
Pour chaque site on garde la meilleure des 2 méthodes (CHROM/POS) au SNR."""
import os, sys, json, glob, time
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.preextract_clips import load_video, resample_to_fps
from scripts.palm_rppg import extract_palm_rgb, interp_nan
from scripts.test_palm_poc import face_rgb, ref_hr
from mp_rppg.methods import chrom, pos
from mp_rppg.metrics import hr_from_fft, snr
from models.chrom_adaptive import bandpass_numpy


def best(rgb_sig, fps):
    s = interp_nan(rgb_sig)
    if s is None: return None
    cands = []
    for fn in (chrom, pos):
        sig = bandpass_numpy(fn(s, fps), fps)
        h = hr_from_fft(sig, fps); cands.append((h, snr(sig, h, fps)))
    return max(cands, key=lambda c: c[1])   # meilleur SNR


def collect():
    items = []
    for jf in sorted(glob.glob("DataVital/Subject*/*.json")):
        try: d = json.load(open(jf))
        except: continue
        fitz = d.get("participant", {}).get("fitzpatrick")
        for sc in d.get("scenarios", []):
            if sc.get("scenario_data", {}).get("scenario") == "facepalm":
                fn = sc.get("recordings", {}).get("RGB", {}).get("filename")
                if fn: items.append((jf, fn, fitz))
    return items


def main(limit):
    items = collect()
    if limit: items = items[:limit]
    print(f"{len(items)} vidéos facepalm\n")
    rows = []
    for k, (jf, video, fitz) in enumerate(items):
        vp = Path(jf).parent / video
        if not vp.exists(): continue
        t0 = time.time()
        try:
            frames, fps = load_video(str(vp), max_dim=512)
            if fps > 32:
                ft = np.arange(len(frames))*1000.0/fps; frames, _ = resample_to_fps(frames, ft, 30.0); fps = 30.0
            ref, _ = ref_hr(jf, video)
            if not ref: continue
            palm, det, _ = extract_palm_rgb(frames)
            pr = best(palm, fps) if det > 0.3 else None
            fc = best(face_rgb(frames), fps)
            pe = abs(pr[0]-ref) if pr else None
            fe = abs(fc[0]-ref) if fc else None
            rows.append((fitz, ref, pr, fc, pe, fe, det))
            print(f"[{k+1}/{len(items)}] {video[:14]} F{fitz} réf{ref:.0f} "
                  f"| PAUME {pr[0]:.0f}({pr[1]:+.1f}) err{pe:.0f} det{100*det:.0f}% "
                  f"| VISAGE {fc[0]:.0f}({fc[1]:+.1f}) err{fe:.0f}  [{time.time()-t0:.0f}s]"
                  if pr else f"[{k+1}] {video[:14]} paume non détectée")
        except Exception as e:
            print(f"[{k+1}] {video[:14]} ERREUR {e}")

    print("\n" + "="*64 + "\nRÉSUMÉ par Fitzpatrick (paume vs visage)")
    for fz in ['4', '5', '6', None]:
        grp = [r for r in rows if r[0] == fz and r[2] and r[3]]
        if not grp: continue
        pmae = np.mean([r[4] for r in grp]); fmae = np.mean([r[5] for r in grp])
        psnr = np.mean([r[2][1] for r in grp]); fsnr = np.mean([r[3][1] for r in grp])
        pw = np.mean([r[4] < r[5] for r in grp])
        print(f"  Fitz{fz} (n={len(grp)}): PAUME MAE={pmae:.1f} SNR={psnr:+.1f} | "
              f"VISAGE MAE={fmae:.1f} SNR={fsnr:+.1f} | paume gagne {100*pw:.0f}%")
    allg = [r for r in rows if r[2] and r[3]]
    if allg:
        print(f"  GLOBAL (n={len(allg)}): paume MAE={np.mean([r[4] for r in allg]):.1f} "
              f"SNR={np.mean([r[2][1] for r in allg]):+.1f} | "
              f"visage MAE={np.mean([r[5] for r in allg]):.1f} SNR={np.mean([r[3][1] for r in allg]):+.1f}")


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
