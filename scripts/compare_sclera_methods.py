#!/usr/bin/env python3
"""
Tableau de comparaison sur les vidéos test à référence connue :
  - CHROM-ITA SANS sclère (ITA brut)  vs  CHROM-ITA AVEC sclère
  - + CHROM, POS, PhysNet, CNN1D, et la fusion adaptative
Erreur calculée vs la FC de référence (appareil externe).

Usage : python scripts/compare_sclera_methods.py
"""
import os, sys
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.cnn1d_rppg import CNN1D_rPPG
from models.chrom_adaptive import CHROMAdaptiveConditioned, compute_ita, bandpass_numpy
from mp_rppg.metrics import hr_from_fft, snr
from mp_rppg.methods import chrom, pos, chrom_adaptive
from mp_rppg.fusion import adaptive_fusion
from mp_rppg.skin_ita import sclera_corrected_ita
from scripts.train_cnn1d import _temporal_norm, CLIP_LEN
from scripts.extract_regions_bisenet import load_bisenet, extract_video, pick_device
from scripts.preextract_clips import load_video
from scripts.run_on_video import run_physnet, FRONT, FULLSKIN, RGB_IDX

D = ROOT / "DataVital" / "SubjecTestRonel"
VIDEOS = [   # (fichier, référence bpm)
    ("videoTestBPM54.mov", 54), ("VideoTestBPM56.MOV", 56),
    ("VideoIssa63.MOV", 67), ("VideoTest4.mov", 58),
    ("Video50MPS54bpm.mp4", 54), ("J'enaimarre1.mp4", 50),
]
PHYS_W = ROOT / "weights" / "clean_physnet_A_pure" / "physnet_africa1_best.pth"


def main():
    dev = pick_device()
    net = load_bisenet(dev)
    cnn = CNN1D_rPPG(in_channels=23 * 9).to(dev)
    cnn.load_state_dict(torch.load(ROOT / "weights" / "cnn1d_rppg.pth", map_location=dev)); cnn.eval()
    cmlp = CHROMAdaptiveConditioned()
    cmlp.load_state_dict(torch.load(ROOT / "weights" / "chrom_conditioned_regions.pth",
                                    map_location="cpu")["model_state_dict"]); cmlp.eval()

    hdr = (f"{'Vidéo':<20}{'réf':>4}{'ITAb':>6}{'ITAs':>6}"
           f"{'CHITA_brut':>11}{'CHITA_scl':>11}{'CHROM':>7}{'POS':>7}"
           f"{'PhysN':>7}{'CNN1D':>7}{'FUSION':>8}")
    print(hdr); print('-' * len(hdr))
    agg = {k: [] for k in ['chita_raw', 'chita_scl', 'chrom', 'pos', 'phys', 'cnn', 'fus']}
    for fname, ref in VIDEOS:
        vid = D / fname
        if not vid.exists():
            print(f"{fname:<20} [absent]"); continue
        frames, fps = load_video(vid, max_dim=720)
        x_reg, _, _ = extract_video(net, dev, frames, 4)
        front = x_reg[:, FRONT, :][:, RGB_IDX].astype(np.float32)
        skin = x_reg[:, FULLSKIN, :][:, RGB_IDX].astype(np.float32)
        ita_raw = compute_ita(front.mean(0))
        ita_scl = sclera_corrected_ita(frames, skin_rgb_fallback=front.mean(0))["ita"]

        # CHROM-ITA dans les deux régimes
        chita_raw = hr_from_fft(bandpass_numpy(chrom_adaptive(front, fps, cmlp.get_coefficients(ita_raw)), fps), fps)
        chita_scl = hr_from_fft(bandpass_numpy(chrom_adaptive(front, fps, cmlp.get_coefficients(ita_scl)), fps), fps)
        # autres méthodes
        hr_chrom = hr_from_fft(bandpass_numpy(chrom(skin, fps), fps), fps)
        hr_pos = hr_from_fft(bandpass_numpy(pos(skin, fps), fps), fps)
        physig = run_physnet(frames, fps, str(PHYS_W), dev)
        hr_phys = hr_from_fft(physig, fps)
        xn = _temporal_norm(x_reg); T = xn.shape[1]; preds = []
        for s in range(0, T - CLIP_LEN + 1, CLIP_LEN):
            with torch.no_grad():
                preds.append(cnn(torch.from_numpy(xn[:, s:s + CLIP_LEN]).unsqueeze(0).to(dev)).squeeze().cpu().numpy())
        hr_cnn = hr_from_fft(bandpass_numpy(np.concatenate(preds), fps), fps) if preds else float('nan')

        # fusion (avec CHROM-ITA sclère)
        pm = [("CNN1D", hr_cnn, snr(bandpass_numpy(np.concatenate(preds), fps), hr_cnn, fps)),
              ("PhysNet", hr_phys, snr(physig, hr_phys, fps)),
              ("CHROM-ITA", chita_scl, snr(bandpass_numpy(chrom_adaptive(front, fps, cmlp.get_coefficients(ita_scl)), fps), chita_scl, fps)),
              ("CHROM", hr_chrom, snr(bandpass_numpy(chrom(skin, fps), fps), hr_chrom, fps)),
              ("POS", hr_pos, snr(bandpass_numpy(pos(skin, fps), fps), hr_pos, fps))]
        fus = adaptive_fusion(pm)["hr"]

        def e(x): return abs(x - ref)
        agg['chita_raw'].append(e(chita_raw)); agg['chita_scl'].append(e(chita_scl))
        agg['chrom'].append(e(hr_chrom)); agg['pos'].append(e(hr_pos))
        agg['phys'].append(e(hr_phys)); agg['cnn'].append(e(hr_cnn)); agg['fus'].append(e(fus))
        print(f"{fname[:19]:<20}{ref:>4}{ita_raw:>6.0f}{ita_scl:>6.0f}"
              f"{chita_raw:>7.0f}({e(chita_raw):>2.0f}){chita_scl:>7.0f}({e(chita_scl):>2.0f})"
              f"{hr_chrom:>7.0f}{hr_pos:>7.0f}{hr_phys:>7.0f}{hr_cnn:>7.0f}{fus:>8.0f}")
    print('-' * len(hdr))
    print("MAE :  CHITA_brut={:.1f}  CHITA_scl={:.1f}  CHROM={:.1f}  POS={:.1f}  "
          "PhysN={:.1f}  CNN1D={:.1f}  FUSION={:.1f}".format(
          *[np.mean(agg[k]) for k in ['chita_raw', 'chita_scl', 'chrom', 'pos', 'phys', 'cnn', 'fus']]))


if __name__ == '__main__':
    main()
