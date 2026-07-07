#!/usr/bin/env python3
"""Compare CNN1D BASELINE vs ROBUSTE sur les vidéos perso. Forcé CPU (protège le run
GPU). Extraction régions une fois/vidéo, puis les deux modèles."""
import os, sys
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from models.cnn1d_rppg import CNN1D_rPPG
from models.chrom_adaptive import bandpass_numpy
from scripts.train_cnn1d import _temporal_norm, CLIP_LEN
from scripts.extract_regions_bisenet import load_bisenet, extract_video
from scripts.preextract_clips import load_video, resample_to_fps
from mp_rppg.metrics import hr_from_fft, snr
dev = torch.device('cpu')
VID = ROOT/'DataVital'/'SubjecTestRonel'/'Visage'

TESTS = [
    ("Video50bpm.mp4",        50,  "réf connue"),
    ("Video50MPS54bpm.mp4",   54,  "réf connue"),
    ("VideoTestBPM56.MOV",    56,  "réf connue"),
    ("videoTestBPM54.mov",    54,  "réf connue"),
    ("VideoIssa63.MOV",       63,  "réf connue"),
    ("VID_20260703_122023.mp4", 68, "échouée"),
    ("VID_20260703_154335.mp4", 61, "échouée"),
    ("VID_20260703_154420.mp4", 74, "échouée"),
]


def load_cnn(path):
    m = CNN1D_rPPG(in_channels=23*9).to(dev)
    m.load_state_dict(torch.load(path, map_location=dev)); m.eval(); return m


def hr_of(model, xn, fps):
    pr = []
    for s in range(0, xn.shape[1]-CLIP_LEN+1, CLIP_LEN):
        with torch.no_grad():
            pr.append(model(torch.from_numpy(xn[:, s:s+CLIP_LEN]).unsqueeze(0).to(dev)).squeeze().numpy())
    if not pr: return float('nan'), float('nan')
    sig = bandpass_numpy(np.concatenate(pr), fps)
    hr = hr_from_fft(sig, fps); return hr, snr(sig, hr, fps)


def main():
    net = load_bisenet(dev)
    base = load_cnn(ROOT/'weights'/'cnn1d_base.pth')
    robust = load_cnn(ROOT/'weights'/'cnn1d_robust.pth')
    print(f"{'vidéo':28s}{'réf':>4}{'BASE':>6}{'errB':>5}{'ROBU':>6}{'errR':>5}  catégorie")
    print("-"*74)
    eb, er = [], []
    for fn, ref, cat in TESTS:
        fp = VID/fn
        if not fp.exists(): print(f"{fn:28s} introuvable"); continue
        try:
            frames, fps = load_video(str(fp), max_dim=720)
            if fps > 32:
                ft = np.arange(len(frames))*1000.0/fps; frames, _ = resample_to_fps(frames, ft, 30.0); fps = 30.0
            xr, _, _ = extract_video(net, dev, frames, 4); xn = _temporal_norm(xr)
            hb, sb = hr_of(base, xn, fps); hr_, sr = hr_of(robust, xn, fps)
            errb = abs(hb-ref); errr = abs(hr_-ref); eb.append(errb); er.append(errr)
            print(f"{fn:28s}{ref:>4}{hb:>6.0f}{errb:>5.0f}{hr_:>6.0f}{errr:>5.0f}  {cat}", flush=True)
        except Exception as e:
            print(f"{fn:28s} ERREUR {str(e)[:50]}", flush=True)
    if eb:
        eb, er = np.array(eb), np.array(er)
        print("-"*74)
        print(f"MAE   BASE {eb.mean():.1f} | ROBUSTE {er.mean():.1f} bpm")
        print(f"%<10  BASE {100*(eb<10).mean():.0f} | ROBUSTE {100*(er<10).mean():.0f}")


if __name__ == '__main__':
    main()
