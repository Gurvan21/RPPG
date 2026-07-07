#!/usr/bin/env python3
"""
Teste si ramener la vidéo au fps d'ENTRAÎNEMENT (30) améliore PhysNet/CNN1D,
vs le fps natif iPhone. Compare HR + SNR des deux régimes.
"""
import os, sys
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.cnn1d_rppg import CNN1D_rPPG
from models.chrom_adaptive import bandpass_numpy
from mp_rppg.metrics import hr_from_fft, snr
from scripts.train_cnn1d import _temporal_norm, CLIP_LEN
from scripts.extract_regions_bisenet import load_bisenet, extract_video, pick_device
from scripts.preextract_clips import load_video, resample_to_fps
from scripts.run_on_video import run_physnet
from models.physnet import PhysNet_padding_Encoder_Decoder_MAX

D = ROOT / "DataVital" / "SubjecTestRonel"
VIDEOS = ["J'enaimarre1.mp4", "Video50MPS54bpm.mp4", "VID_20260626_112827.mp4"]
PHYS_W = str(ROOT / "weights" / "clean_physnet_A_pure" / "physnet_africa1_best.pth")


def run_pipeline(frames, fps, net, cnn, dev):
    x_reg, _, _ = extract_video(net, dev, frames, 4)
    physig = run_physnet(frames, fps, PHYS_W, dev)
    hp = hr_from_fft(physig, fps); sp = snr(physig, hp, fps)
    xn = _temporal_norm(x_reg); T = xn.shape[1]; preds = []
    for s in range(0, T - CLIP_LEN + 1, CLIP_LEN):
        with torch.no_grad():
            preds.append(cnn(torch.from_numpy(xn[:, s:s + CLIP_LEN]).unsqueeze(0).to(dev)).squeeze().cpu().numpy())
    cs = bandpass_numpy(np.concatenate(preds), fps) if preds else np.zeros(10)
    hc = hr_from_fft(cs, fps); sc = snr(cs, hc, fps)
    return hp, sp, hc, sc


def main():
    dev = pick_device(); net = load_bisenet(dev)
    cnn = CNN1D_rPPG(in_channels=23 * 9).to(dev)
    cnn.load_state_dict(torch.load(ROOT / "weights" / "cnn1d_rppg.pth", map_location=dev)); cnn.eval()
    print(f"{'Vidéo':<22}{'fps':>6}{'PhysHR':>8}{'PhysSNR':>9}{'CNNhr':>7}{'CNNsnr':>8}")
    print('-' * 60)
    for fn in VIDEOS:
        vid = D / fn
        if not vid.exists():
            continue
        frames, fps = load_video(vid, max_dim=720)
        # natif
        hp, sp, hc, sc = run_pipeline(frames, fps, net, cnn, dev)
        print(f"{fn[:21]:<22}{fps:>6.1f}{hp:>8.1f}{sp:>9.2f}{hc:>7.1f}{sc:>8.2f}  (natif)")
        # ramené à 30 fps (= condition d'entraînement)
        ft = np.arange(len(frames)) * 1000.0 / fps
        fr30, _ = resample_to_fps(frames, ft, 30.0)
        hp2, sp2, hc2, sc2 = run_pipeline(fr30, 30.0, net, cnn, dev)
        print(f"{'  → resamplé':<22}{30.0:>6.1f}{hp2:>8.1f}{sp2:>9.2f}{hc2:>7.1f}{sc2:>8.2f}  (=train)")
        print()
    print("SNR plus élevé après resample 30fps = match domaine d'entraînement utile.")


if __name__ == '__main__':
    main()
