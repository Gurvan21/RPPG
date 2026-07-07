#!/usr/bin/env python3
"""PTT paume-visage avec les CNN1D (main + visage) sur facepalm.
DÉMONSTRATION du piège : les CNN1D sont entraînés vers la même référence (CMS doigt)
→ ils alignent les deux sites sur ce timing → le déphasage (PTT) s'effondre.
Comparé à CHROM (non entraîné, phase-fidèle) sur les mêmes signaux."""
import os, sys, json, glob
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
from pathlib import Path
import numpy as np, torch
from scipy.signal import butter, filtfilt
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.preextract_clips import load_video, resample_to_fps
from scripts.extract_hand_regions import extract_video as eh
from scripts.extract_regions_bisenet import load_bisenet, extract_video as ef, pick_device
from scripts.train_cnn1d import _temporal_norm, CLIP_LEN
from models.cnn1d_rppg import CNN1D_rPPG
from models.chrom_adaptive import bandpass_numpy
from mp_rppg.methods import chrom
from mp_rppg.metrics import hr_from_fft, snr
from scripts.test_ptt_phase import narrow, lag_ms
from scripts.run_on_video import FRONT, FULLSKIN, RGB_IDX

dev = pick_device()
_hand = CNN1D_rPPG(in_channels=18*9).to(dev); _hand.load_state_dict(torch.load(ROOT/'weights'/'cnn1d_hand.pth', map_location=dev)); _hand.eval()
_face = CNN1D_rPPG(in_channels=23*9).to(dev); _face.load_state_dict(torch.load(ROOT/'weights'/'cnn1d_rppg.pth', map_location=dev)); _face.eval()


def cnn(x, model, fps):
    xn = _temporal_norm(x); pr = []
    for s in range(0, xn.shape[1]-CLIP_LEN+1, CLIP_LEN):
        with torch.no_grad():
            pr.append(model(torch.from_numpy(xn[:, s:s+CLIP_LEN]).unsqueeze(0).to(dev)).squeeze().cpu().numpy())
    return bandpass_numpy(np.concatenate(pr), fps) if pr else None


def main(limit):
    from scripts.test_ptt_phase import collect
    net = load_bisenet(dev)
    items = collect()[:limit]
    print(f"{len(items)} facepalm — PTT via CNN1D vs CHROM\n")
    cnn_dt, chrom_dt = [], []
    for k, (jf, video, fitz, bp) in enumerate(items):
        vp = jf.parent/video
        if not vp.exists(): continue
        try:
            frames, fps = load_video(str(vp), max_dim=640)
            if fps > 32:
                ft = np.arange(len(frames))*1000.0/fps; frames, _ = resample_to_fps(frames, ft, 30.0); fps = 30.0
            xh, _, dh = eh(frames, 3)
            xf, _, df = ef(net, dev, frames, 4)
            if dh < 0.3 or df < 0.3: continue
            # CNN1D des deux sites
            ph_c, fh_c = cnn(xh, _hand, fps), cnn(xf, _face, fps)
            # CHROM des deux sites (mêmes ROI : paume complète / peau visage complète)
            palm_rgb = xh[:, 8, :3].astype(np.float32); face_rgb = xf[:, FULLSKIN, :][:, RGB_IDX].astype(np.float32)
            ph_h = bandpass_numpy(chrom(palm_rgb, fps), fps); fh_h = bandpass_numpy(chrom(face_rgb, fps), fps)
            hr = hr_from_fft(ph_c, fps)
            dt_cnn = lag_ms(narrow(ph_c, fps, hr), narrow(fh_c, fps, hr), fps)
            dt_chrom = lag_ms(narrow(ph_h, fps, hr), narrow(fh_h, fps, hr), fps)
            cnn_dt.append(dt_cnn); chrom_dt.append(dt_chrom)
            print(f"[{k+1}] F{fitz} HR~{hr:.0f} BP={bp}  Δt CNN1D={dt_cnn:+.1f}ms   Δt CHROM={dt_chrom:+.1f}ms", flush=True)
        except Exception as e:
            print(f"[{k+1}] {video[:12]} ERREUR {e}", flush=True)
    if cnn_dt:
        print(f"\nΔt CNN1D  : moyenne {np.mean(cnn_dt):+.1f} ± {np.std(cnn_dt):.1f} ms")
        print(f"Δt CHROM  : moyenne {np.mean(chrom_dt):+.1f} ± {np.std(chrom_dt):.1f} ms")
        print("→ Si CNN1D s'effondre/désordonne vs CHROM, c'est le piège annoncé.")


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 9)
