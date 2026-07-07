#!/usr/bin/env python3
"""BP via déphasage paume-visage (PTT deux-sites) sur facepalm.
Même méthode (CHROM) sur les deux sites, passe-bande étroit à la FC (phase nulle),
délai par inter-corrélation sous-échantillon, corrélé à la BP brassard.
Attente honnête : inter-sujets confondu (bras/âge) → probablement faible."""
import os, sys, json, glob
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
from pathlib import Path
import numpy as np
from scipy.signal import butter, filtfilt
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.preextract_clips import load_video, resample_to_fps
from scripts.palm_rppg import extract_palm_rgb, interp_nan
from scripts.test_palm_poc import face_rgb
from mp_rppg.methods import chrom
from mp_rppg.metrics import hr_from_fft, snr
from models.chrom_adaptive import bandpass_numpy


def narrow(sig, fps, hr, half=10.0):
    ny = fps/2.0; lo = max(hr-half, 40)/60.0; hi = min(hr+half, ny*60-1)/60.0
    b, a = butter(3, [lo/ny, hi/ny], btype='band')
    return filtfilt(b, a, sig)


def lag_ms(palm, face, fps, maxlag_s=0.30):
    a = (palm-palm.mean())/(palm.std()+1e-8); b = (face-face.mean())/(face.std()+1e-8)
    n = len(a); xc = np.correlate(a, b, mode='full'); mid = n-1
    ml = int(maxlag_s*fps); seg = xc[mid-ml:mid+ml+1]; k = int(np.argmax(seg))
    delta = 0.0
    if 0 < k < len(seg)-1:
        y0, y1, y2 = seg[k-1], seg[k], seg[k+1]; d = (y0-2*y1+y2)
        if d != 0: delta = 0.5*(y0-y2)/d
    return ((k-ml)+delta)/fps*1000.0     # >0 : paume en retard sur visage (attendu)


def collect():
    out = []
    for jf in sorted(glob.glob(str(ROOT/"DataVital"/"Subject*"/"*.json"))):
        try: d = json.load(open(jf))
        except: continue
        fitz = d.get("participant", {}).get("fitzpatrick")
        for sc in d.get("scenarios", []):
            if sc.get("scenario_data", {}).get("scenario") == "facepalm":
                rec = sc.get("recordings", {}); rgb = rec.get("RGB"); bp = rec.get("BP", {})
                val = bp.get("value") if isinstance(bp, dict) else None
                if rgb and val and "/" in str(val):
                    out.append((Path(jf), rgb["filename"], fitz, str(val)))
                break
    return out


def main(limit):
    items = collect()
    if limit: items = items[:limit]
    print(f"{len(items)} facepalm avec BP\n")
    rows = []
    for k, (jf, video, fitz, bp) in enumerate(items):
        vp = jf.parent/video
        if not vp.exists(): continue
        try:
            frames, fps = load_video(str(vp), max_dim=640)
            if fps > 32:
                ft = np.arange(len(frames))*1000.0/fps; frames, _ = resample_to_fps(frames, ft, 30.0); fps = 30.0
            palm = interp_nan(extract_palm_rgb(frames)[0]); face = interp_nan(face_rgb(frames))
            if palm is None or face is None: continue
            ps = bandpass_numpy(chrom(palm, fps), fps); fs = bandpass_numpy(chrom(face, fps), fps)
            hp = hr_from_fft(ps, fps); hf = hr_from_fft(fs, fps)
            sp = snr(ps, hp, fps); sf = snr(fs, hf, fps)
            sysbp = int(bp.split("/")[0]); dia = int(bp.split("/")[1])
            hr = hp if sp >= sf else hf
            dt = lag_ms(narrow(ps, fps, hr), narrow(fs, fps, hr), fps)
            ok = sp > -2 and sf > -2 and abs(hp-hf) < 6      # les 2 sites fiables + même HR
            rows.append((fitz, hr, sp, sf, dt, sysbp, dia, ok))
            print(f"[{k+1}/{len(items)}] F{fitz} HR~{hr:.0f} SNR(p/v) {sp:+.0f}/{sf:+.0f} "
                  f"Δt={dt:+.1f}ms BP={sysbp}/{dia} {'OK' if ok else '(exclu)'}", flush=True)
        except Exception as e:
            print(f"[{k+1}] {video[:12]} ERREUR {e}", flush=True)

    good = [r for r in rows if r[7]]
    print(f"\n=== {len(good)}/{len(rows)} enregistrements fiables (2 sites SNR>-2, même HR) ===")
    if len(good) >= 4:
        dt = np.array([r[4] for r in good]); sys_ = np.array([r[5] for r in good]); dia = np.array([r[6] for r in good])
        print(f"Δt moyen = {dt.mean():+.1f} ± {dt.std():.1f} ms")
        cs = np.corrcoef(dt, sys_)[0, 1]; cd = np.corrcoef(dt, dia)[0, 1]
        print(f"corr(Δt, systolique) = {cs:+.2f}   corr(Δt, diastolique) = {cd:+.2f}")
        print("(PTT plus court attendu quand BP plus haute → corrélation NÉGATIVE espérée)")


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 30)
