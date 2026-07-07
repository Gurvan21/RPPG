#!/usr/bin/env python3
"""
Sanity-check : CHROM (méthode classique, sans aucun apprentissage) sur les mêmes
vidéos DataVital que PhysNet, pour savoir si l'échec catastrophique de PhysNet
(Pearson ~0) vient d'un problème de pipeline/sync général, ou bien spécifiquement
de la généralisation de PhysNet à cette caméra/carnation.

Si CHROM donne aussi un Pearson ~0 ici -> suspect un bug de pipeline (sync, crop...).
Si CHROM donne un signal raisonnable (Pearson nettement > bruit) -> le problème est
spécifique à PhysNet, et le fine-tuning a un sens (le signal est bien extractible).

Usage :
  python scripts/test_chrom_baseline.py --subject-dir DataVital/Subjet1
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mp_rppg.pipeline import extract_rgb
from mp_rppg.methods import chrom
from mp_rppg.metrics import hr_from_fft, snr
from scripts.preextract_clips import load_video, _find_vitalvideos_json, _resample_ppg_to_frames


def eval_subject(subject_dir: Path):
    meta = _find_vitalvideos_json(subject_dir)
    results = []
    for sc_idx, scenario in enumerate(meta.get('scenarios', [])):
        rec = scenario.get('recordings', {})
        rgb, cms = rec.get('RGB'), rec.get('CMS')
        if not rgb or not cms:
            continue
        video_path = subject_dir / rgb['filename']
        if not video_path.exists():
            continue

        frames, _ = load_video(video_path)
        frame_times_ms = np.array([t for t, _ in rgb['timeseries']], dtype=np.float64)
        n = min(len(frames), len(frame_times_ms))
        frames = frames[:n]
        fps = float(rgb.get('device', {}).get('FrameRate', 30))

        gt = _resample_ppg_to_frames(cms[1:], frame_times_ms[:n])

        rgb_sig = extract_rgb(frames, verbose=False)['mean']   # (n,3)
        bvp = chrom(rgb_sig, fps)
        m = min(len(bvp), len(gt))
        bvp, gt_m = bvp[:m], gt[:m]

        hr_g = hr_from_fft(gt_m, fps)
        hr_p = hr_from_fft(bvp, fps)
        err  = abs(hr_p - hr_g)
        s    = snr(bvp, hr_g, fps)
        pear = float(np.corrcoef(bvp, gt_m)[0, 1])
        results.append((sc_idx, err, s, pear, hr_p, hr_g))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject-dir', required=True, action='append')
    args = parser.parse_args()

    print(f"{'Sujet':<10}{'sc':>3}{'HR_pred':>9}{'HR_gt':>9}{'Err':>8}{'SNR':>8}{'Pearson':>9}")
    print('-' * 56)
    all_err, all_snr, all_pear = [], [], []
    for sd in args.subject_dir:
        sd = Path(sd)
        for sc_idx, err, s, pear, hr_p, hr_g in eval_subject(sd):
            print(f"{sd.name:<10}{sc_idx:>3}{hr_p:>9.1f}{hr_g:>9.1f}{err:>8.1f}{s:>8.2f}{pear:>+9.3f}")
            all_err.append(err); all_snr.append(s); all_pear.append(pear)

    print('-' * 56)
    print(f"{'MOYENNE':<13}{'':>9}{'':>9}{np.mean(all_err):>8.1f}{np.mean(all_snr):>8.2f}{np.mean(all_pear):>+9.3f}")


if __name__ == '__main__':
    main()
