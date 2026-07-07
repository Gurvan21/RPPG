#!/usr/bin/env python3
"""Compare CNN1D-main AVEC bord vs SANS bord (edge-frac) sur des vidéos smartphone.
Chaque modèle est appliqué sur l'extraction qui lui correspond. Pas de vérité-
terrain → on juge au SNR (surtout sur les prises BOUGÉES)."""
import os, sys
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.preextract_clips import load_video, resample_to_fps
from scripts.extract_hand_regions import extract_video as eh
from scripts.train_cnn1d import _temporal_norm, CLIP_LEN
from models.cnn1d_rppg import CNN1D_rPPG
from models.chrom_adaptive import bandpass_numpy
from mp_rppg.metrics import hr_from_fft, snr, hr_candidates
dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

MODELS = {0.0: ROOT/'weights'/'cnn1d_hand.pth', 0.10: ROOT/'weights'/'cnn1d_hand_noedge.pth'}
_cache = {}


def infer(frames, fps, edge_frac):
    x, _, det = eh(frames, 3, edge_frac=edge_frac)
    if det < 0.3: return None
    key = str(MODELS[edge_frac])
    if key not in _cache:
        m = CNN1D_rPPG(in_channels=18*9).to(dev)
        m.load_state_dict(torch.load(MODELS[edge_frac], map_location=dev)); m.eval()
        _cache[key] = m
    m = _cache[key]; xn = _temporal_norm(x); pr = []
    for s in range(0, xn.shape[1]-CLIP_LEN+1, CLIP_LEN):
        with torch.no_grad():
            pr.append(m(torch.from_numpy(xn[:, s:s+CLIP_LEN]).unsqueeze(0).to(dev)).squeeze().cpu().numpy())
    sig = bandpass_numpy(np.concatenate(pr), fps); h = hr_from_fft(sig, fps)
    _, amb = hr_candidates(sig, fps)
    return h, snr(sig, h, fps), amb, det


def main():
    vids = [
        ("Paume/videoDeMain.mp4", "stable ✓"),
        ("PaumeVisage/VideoMainVisage.mp4", "bougé"),
        ("PaumeVisage/VID_20260630_152633.mp4", "bougé"),
        ("PaumeVisage/VID_20260630_152419.mp4", "bougé 19fps"),
    ]
    D = ROOT/"DataVital"/"SubjecTestRonel"
    print(f"{'vidéo':<34}{'AVEC bord':>18}{'SANS bord':>18}")
    for rel, tag in vids:
        p = D/rel
        if not p.exists(): print(f"{rel} introuvable"); continue
        frames, fps = load_video(str(p), max_dim=720)
        if fps > 32:
            ft = np.arange(len(frames))*1000.0/fps; frames, _ = resample_to_fps(frames, ft, 30.0); fps = 30.0
        r0 = infer(frames, fps, 0.0); r1 = infer(frames, fps, 0.10)
        def fmt(r): return f"{r[0]:.0f}bpm SNR{r[1]:+.1f}{'⚠' if r[2] else ''}" if r else "—"
        name = Path(rel).stem[:24]
        print(f"{name:<24}[{tag:<9}] {fmt(r0):>16}   {fmt(r1):>16}")


if __name__ == '__main__':
    main()
