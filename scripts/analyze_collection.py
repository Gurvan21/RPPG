#!/usr/bin/env python3
"""
Analyse batch de la collecte smartphone → tableau final (le résultat de l'article).
Lit Data/collection_smartphone/subject_XX/{*.mp4, meta.json}. meta.json :
  {"fitzpatrick": 6, "takes": {"palm_1": {"ref_hr": 72}, "face_1": {"ref_hr": 72}}}
Sort MAE/SNR/couverture par Fitzpatrick ET par site (paume vs visage).

Usage : python scripts/analyze_collection.py [--root Data/collection_smartphone]
"""
import os, sys, json, argparse
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.preextract_clips import load_video, resample_to_fps
from scripts.train_cnn1d import _temporal_norm, CLIP_LEN
from models.cnn1d_rppg import CNN1D_rPPG
from models.chrom_adaptive import bandpass_numpy
from mp_rppg.methods import chrom, pos
from mp_rppg.metrics import hr_from_fft, snr, hr_candidates
dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
GO_SNR, AGREE = 1.0, 8.0


def analyze(video, mode):
    frames, fps = load_video(str(video), max_dim=720)
    if fps > 32:
        ft = np.arange(len(frames))*1000.0/fps; frames, _ = resample_to_fps(frames, ft, 30.0); fps = 30.0
    sigs = {}
    if mode == 'paume':
        from scripts.extract_hand_regions import extract_video as eh
        from scripts.palm_rppg import extract_palm_rgb, interp_nan
        x, _, det = eh(frames, 3)
        if det > 0.3:
            m = CNN1D_rPPG(in_channels=18*9).to(dev)
            m.load_state_dict(torch.load(ROOT/'weights'/'cnn1d_hand.pth', map_location=dev)); m.eval()
            xn = _temporal_norm(x); pr = []
            for s in range(0, xn.shape[1]-CLIP_LEN+1, CLIP_LEN):
                with torch.no_grad():
                    pr.append(m(torch.from_numpy(xn[:, s:s+CLIP_LEN]).unsqueeze(0).to(dev)).squeeze().cpu().numpy())
            if pr: sigs['CNN1D-main'] = bandpass_numpy(np.concatenate(pr), fps)
        rgb = interp_nan(extract_palm_rgb(frames)[0])
    else:
        from scripts.test_palm_poc import face_rgb
        from scripts.palm_rppg import interp_nan
        rgb = interp_nan(face_rgb(frames))
    if rgb is not None:
        sigs['CHROM'] = bandpass_numpy(chrom(rgb, fps), fps)
        sigs['POS'] = bandpass_numpy(pos(rgb, fps), fps)
    per = [(n, hr_from_fft(s, fps), snr(s, hr_from_fft(s, fps), fps)) for n, s in sigs.items() if s is not None]
    if not per: return None
    per.sort(key=lambda z: z[2], reverse=True)
    prim, hr, sn = per[0]; std = float(np.std([h for _, h, _ in per]))
    _, ambig = hr_candidates(sigs[prim], fps)
    go = sn >= GO_SNR and std <= AGREE and not ambig
    return hr, sn, go


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=str(ROOT/'Data'/'collection_smartphone'))
    args = ap.parse_args()
    root = Path(args.root)
    if not root.exists():
        print(f"❌ {root} n'existe pas encore. Crée-le avec des subject_XX/ + meta.json."); return
    rows = []   # (fitz, site, err, snr, go)
    for subj in sorted(root.glob('subject_*')):
        meta_p = subj / 'meta.json'
        if not meta_p.exists(): continue
        meta = json.load(open(meta_p)); fitz = str(meta.get('fitzpatrick', '?'))
        for take, info in meta.get('takes', {}).items():
            vid = next((subj/f"{take}{e}" for e in ('.mp4', '.mov', '.MOV') if (subj/f"{take}{e}").exists()), None)
            if not vid: print(f"  [manque] {subj.name}/{take}"); continue
            mode = 'paume' if take.lower().startswith(('palm', 'paume', 'main')) else 'visage'
            ref = info.get('ref_hr')
            r = analyze(vid, mode)
            if r is None: print(f"  [échec] {subj.name}/{take}"); continue
            hr, sn, go = r; err = abs(hr - ref) if ref else None
            rows.append((fitz, mode, err, sn, go))
            print(f"  {subj.name} {take:8s} [{mode}] HR {hr:.0f} SNR {sn:+.1f} {'GO' if go else 'abstain'}"
                  + (f" | réf {ref} err {err:.0f}" if err is not None else ""))

    print("\n" + "="*66 + "\nTABLEAU FINAL — par Fitzpatrick × site")
    for site in ('paume', 'visage'):
        print(f"\n  ── {site.upper()} ──")
        for fz in ['1', '2', '3', '4', '5', '6', '?']:
            g = [r for r in rows if r[0] == fz and r[1] == site and r[2] is not None]
            if not g: continue
            mae = np.mean([r[2] for r in g]); msn = np.mean([r[3] for r in g])
            cov = np.mean([r[4] for r in g])
            acc = [r[2] for r in g if r[4]]
            mae_acc = np.mean(acc) if acc else float('nan')
            print(f"    Fitz{fz} (n={len(g)}): MAE={mae:.1f}  SNR={msn:+.1f}  "
                  f"couverture={100*cov:.0f}%  MAE(acceptées)={mae_acc:.1f}")
    # équité : paume Fitz5-6 vs Fitz3-4
    dark = [r[2] for r in rows if r[1] == 'paume' and r[0] in ('5', '6') and r[2] is not None]
    light = [r[2] for r in rows if r[1] == 'paume' and r[0] in ('3', '4') and r[2] is not None]
    if dark and light:
        print(f"\n  ÉQUITÉ paume : Fitz5-6 MAE={np.mean(dark):.1f} (n={len(dark)}) "
              f"vs Fitz3-4 MAE={np.mean(light):.1f} (n={len(light)})  "
              f"→ écart {abs(np.mean(dark)-np.mean(light)):.1f} bpm")


if __name__ == '__main__':
    main()
