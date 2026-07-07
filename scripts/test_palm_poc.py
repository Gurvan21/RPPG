#!/usr/bin/env python3
"""POC : paume vs visage sur UNE vidéo facepalm, comparé à la référence CMS."""
import os, sys, json
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
from pathlib import Path
import numpy as np, cv2
import mediapipe as mp
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.preextract_clips import load_video, resample_to_fps
from scripts.palm_rppg import extract_palm_rgb, interp_nan
from mp_rppg.methods import chrom, pos
from mp_rppg.metrics import hr_from_fft, snr
from models.chrom_adaptive import bandpass_numpy

# régions front+joues via FaceMesh (peau visage, comparaison équitable)
FORE = [10, 67, 69, 109, 108, 151, 337, 299, 297, 338]
LCHEEK = [50, 101, 118, 117, 123, 147, 187, 205, 36]
RCHEEK = [280, 330, 347, 346, 352, 376, 411, 425, 266]


def face_rgb(frames):
    fm = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                         refine_landmarks=False, min_detection_confidence=0.5)
    H, W = frames.shape[1:3]; out = np.full((len(frames), 3), np.nan, np.float32)
    try:
        for i, f in enumerate(frames):
            r = fm.process(f)
            if not r.multi_face_landmarks: continue
            lm = r.multi_face_landmarks[0].landmark
            mask = np.zeros((H, W), np.uint8)
            for poly in (FORE, LCHEEK, RCHEEK):
                pts = np.array([[lm[k].x*W, lm[k].y*H] for k in poly], np.int32)
                cv2.fillConvexPoly(mask, cv2.convexHull(pts), 1)
            sel = mask.astype(bool)
            if sel.sum() < 100: continue
            out[i] = f[sel].astype(np.float32).mean(0)
    finally:
        fm.close()
    return out


def ref_hr(jf, video_name):
    d = json.load(open(jf))
    for sc in d["scenarios"]:
        if sc["recordings"].get("RGB", {}).get("filename") == video_name:
            cms = sc["recordings"]["CMS"]
            hrs = np.array([row[2] for row in cms[1:] if row[2]], float)
            return float(np.median(hrs)) if len(hrs) else None, d["participant"]["fitzpatrick"]
    return None, None


def hr_snr(rgb_sig, fps):
    s = interp_nan(rgb_sig)
    if s is None: return None
    res = {}
    for name, fn in (('CHROM', chrom), ('POS', pos)):
        sig = bandpass_numpy(fn(s, fps), fps)
        h = hr_from_fft(sig, fps); res[name] = (h, snr(sig, h, fps))
    return res


def main(jf, video):
    vp = str(Path(jf).parent / video)
    frames, fps = load_video(vp, max_dim=640)
    if fps > 32:
        ft = np.arange(len(frames))*1000.0/fps; frames, _ = resample_to_fps(frames, ft, 30.0); fps = 30.0
    ref, fitz = ref_hr(jf, video)
    print(f"{video}  Fitz{fitz}  réf CMS={ref:.0f} bpm  ({len(frames)} frames @{fps:.0f})")
    palm, det, area = extract_palm_rgb(frames)
    print(f"  paume détectée : {100*det:.0f}% des frames, aire {100*area:.1f}%")
    pr = hr_snr(palm, fps) if det > 0.3 else None
    fc = hr_snr(face_rgb(frames), fps)
    for label, r in (("PAUME", pr), ("VISAGE", fc)):
        if r is None: print(f"  {label}: — (détection insuffisante)"); continue
        txt = "  ".join(f"{k} {v[0]:.0f}bpm SNR{v[1]:+.1f}" for k, v in r.items())
        errs = "  ".join(f"err{k}={abs(v[0]-ref):.0f}" for k, v in r.items()) if ref else ""
        print(f"  {label:6s}: {txt}   {errs}")


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
