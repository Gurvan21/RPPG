#!/usr/bin/env python3
"""PTT paume-visage sur les visages VALIDÉS uniquement (récupération guidée FC +
cohérence). Δt par inter-corrélation sous-échantillon (narrow à FC réf, CHROM sur
les 2 sites), corrélé à la BP brassard. Attente : moins délirant qu'avant mais
probablement toujours trop bruité pour une vraie corrélation BP."""
import os, sys, json, glob
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
from pathlib import Path
import numpy as np
from scipy.signal import coherence
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.preextract_clips import (load_video, resample_to_fps, TARGET_FPS, _resample_ppg_to_frames)
from scripts.extract_regions_bisenet import load_bisenet, extract_video as ef, pick_device
from scripts.extract_hand_regions import extract_video as eh
from scripts.run_on_video import FULLSKIN, RGB_IDX
from scripts.test_ptt_phase import narrow, lag_ms
from mp_rppg.methods import chrom
from mp_rppg.metrics import hr_from_fft, snr
from models.chrom_adaptive import bandpass_numpy
COH_MIN, SNR_MIN = 0.5, -3.0
dev = pick_device()


def items():
    out = []
    for jf in sorted(glob.glob(str(ROOT/"DataVital"/"Subject*"/"*.json"))):
        try: d = json.load(open(jf))
        except: continue
        fitz = d.get("participant", {}).get("fitzpatrick"); fp = None; bp = None
        for sc in d.get("scenarios", []):
            rec = sc.get("recordings", {})
            v = rec.get("BP", {}).get("value") if isinstance(rec.get("BP"), dict) else None
            if v and "/" in str(v) and bp is None: bp = str(v)
            if sc.get("scenario_data", {}).get("scenario") == "facepalm" and fp is None:
                if rec.get("RGB") and rec.get("CMS") and len(rec["CMS"]) > 2:
                    fp = (rec["RGB"], rec["CMS"])
        if fp and bp: out.append((Path(jf), fitz, fp[0], fp[1], bp))
    return out


def coh_at(a, b, fps, f):
    n = min(len(a), len(b)); fr, C = coherence(a[:n], b[:n], fs=fps, nperseg=min(128, n//2))
    return float(C[np.argmin(np.abs(fr-f))])


def main(limit):
    net = load_bisenet(dev); it = items()[:limit]
    print(f"{len(it)} facepalm avec BP — PTT sur visages VALIDÉS\n")
    val = []
    for k, (jf, fitz, rgb, cms, bp) in enumerate(it):
        vp = jf.parent/rgb['filename']
        if not vp.exists(): continue
        try:
            frames, _ = load_video(str(vp), max_dim=640)
            ft = np.array([t for t, _ in rgb['timeseries']], np.float64)
            n = min(len(frames), len(ft)); frames, ft = frames[:n], ft[:n]
            frames, ft = resample_to_fps(frames, ft, TARGET_FPS); fps = TARGET_FPS
            y = _resample_ppg_to_frames(cms[1:], ft); ref = hr_from_fft(y, fps)
            xf, _, df = ef(net, dev, frames, 4); xh, _, dh = eh(frames, 3)
            if df < 0.5 or dh < 0.3: continue
            face = bandpass_numpy(chrom(xf[:, FULLSKIN, :][:, RGB_IDX].astype(np.float32), fps), fps)
            palm = bandpass_numpy(chrom(xh[:, 8, :3].astype(np.float32), fps), fps)
            sf = snr(face, ref, fps); sp = snr(palm, ref, fps)
            cf = coh_at(face, y, fps, ref/60.0)
            face_ok = sf > SNR_MIN and cf > COH_MIN; palm_ok = sp > SNR_MIN
            sysbp = int(bp.split("/")[0]); dia = int(bp.split("/")[1])
            tag = "—"
            if face_ok and palm_ok:
                dt = lag_ms(narrow(palm, fps, ref), narrow(face, fps, ref), fps)
                val.append((fitz, ref, dt, sysbp, dia, cf)); tag = f"Δt={dt:+.1f}ms VALIDÉ"
            print(f"[{k+1}] F{fitz} FC={ref:.0f} SNR(p/v)={sp:+.0f}/{sf:+.0f} coh={cf:.2f} BP={bp}  {tag}", flush=True)
        except Exception as e:
            print(f"[{k+1}] ERREUR {e}", flush=True)

    print(f"\n=== {len(val)} sujets avec paume ET visage validés ===")
    if len(val) >= 4:
        dt = np.array([v[2] for v in val]); sys_ = np.array([v[3] for v in val]); dia = np.array([v[4] for v in val])
        print(f"Δt = {dt.mean():+.1f} ± {dt.std():.1f} ms  (dispersion = bruit résiduel)")
        print(f"corr(Δt, systolique) = {np.corrcoef(dt, sys_)[0,1]:+.2f}")
        print(f"corr(Δt, diastolique) = {np.corrcoef(dt, dia)[0,1]:+.2f}")
        print("(PTT court quand BP haute → corrélation NÉGATIVE espérée)")
    else:
        print("Trop peu de sujets validés pour une corrélation.")


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 40)
