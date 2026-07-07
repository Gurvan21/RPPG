#!/usr/bin/env python3
"""
Test d'hypothèse : PhysNet (SCAMPS/UBFC) est pré-entraîné sur des vidéos ~30fps.
Nos clips DataVital sont en 60fps -> 128 frames couvrent 2.13s au lieu des ~4.27s
attendus. Les noyaux conv temporels de PhysNet n'ont aucune notion de fps absolu :
ils reconnaissent une périodicité en NOMBRE DE FRAMES, pas en secondes. À 60fps,
un cycle cardiaque dure ~2x plus de frames qu'à 30fps -> décalage de périodicité
qui pourrait suffire à expliquer un Pearson ~0, indépendamment de la carnation.

Ce script ré-extrait les mêmes sujets en sous-échantillonnant 60fps -> 30fps
(un frame sur deux) avant le clip de 128 frames, et compare le baseline brut
(sans fine-tuning) à la version 60fps déjà testée.

Usage :
  python scripts/test_fps_hypothesis.py --subject-dir DataVital/Subjet1 --weights UBFC
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from models.physnet import PhysNet_padding_Encoder_Decoder_MAX
from mp_rppg.metrics import hr_from_fft, snr
from scripts.preextract_clips import (
    CLIP_LEN, RESIZE, diff_normalize, find_face_bbox, load_video,
    _find_vitalvideos_json, _resample_ppg_to_frames,
)

import cv2

WEIGHTS = {
    'SCAMPS': os.path.join(ROOT, 'weights/SCAMPS_PhysNet_DiffNormalized.pth'),
    'UBFC':   os.path.join(ROOT, 'weights/UBFC-rPPG_PhysNet_DiffNormalized.pth'),
}


def make_clips_30fps(subject_dir: Path, downsample: int = 2):
    """Comme process_subject_vitalvideos, mais sous-échantillonne les frames
    (1 sur `downsample`) avant le découpage en clips -> fps effectif réduit."""
    meta = _find_vitalvideos_json(subject_dir)
    clips = []
    for scenario in meta.get('scenarios', []):
        rec = scenario.get('recordings', {})
        rgb, cms = rec.get('RGB'), rec.get('CMS')
        if not rgb or not cms or len(cms) < 2:
            continue
        video_path = subject_dir / rgb['filename']
        if not video_path.exists():
            continue

        frames, _ = load_video(video_path)
        frame_times_ms = np.array([t for t, _ in rgb['timeseries']], dtype=np.float64)
        n = min(len(frames), len(frame_times_ms))
        frames = frames[:n:downsample]
        times_ds = frame_times_ms[:n:downsample]
        fps = float(rgb.get('device', {}).get('FrameRate', 30)) / downsample

        if len(frames) < CLIP_LEN:
            continue

        ppg = _resample_ppg_to_frames(cms[1:], times_ds)

        bbox = find_face_bbox(frames)
        x1, y1, x2, y2 = bbox
        cropped = np.stack([
            cv2.resize(f[y1:y2, x1:x2], (RESIZE, RESIZE), interpolation=cv2.INTER_AREA)
            for f in frames
        ]).astype(np.float32)

        for start in range(0, len(frames) - CLIP_LEN + 1, CLIP_LEN):
            clip_dn = diff_normalize(cropped[start:start + CLIP_LEN])
            ppg_seg = ppg[start:start + CLIP_LEN]
            ppg_seg = (ppg_seg - ppg_seg.mean()) / (ppg_seg.std() + 1e-8)
            clips.append((clip_dn.astype(np.float32), ppg_seg.astype(np.float32), fps))
    return clips


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject-dir', required=True, action='append',
                        help="Répétable : un ou plusieurs dossiers sujet VitalVideos")
    parser.add_argument('--weights', default='UBFC', choices=list(WEIGHTS.keys()))
    parser.add_argument('--downsample', type=int, default=2,
                        help="Facteur de sous-échantillonnage temporel (2 = 60fps->30fps)")
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    device = torch.device('cpu') if args.cpu or not (torch.cuda.is_available() or torch.backends.mps.is_available()) \
        else (torch.device('cuda') if torch.cuda.is_available() else torch.device('mps'))
    print(f"Device : {device}  |  Poids : {args.weights}  |  downsample x{args.downsample}\n")

    model = PhysNet_padding_Encoder_Decoder_MAX(frames=CLIP_LEN).to(device)
    model.load_state_dict(torch.load(WEIGHTS[args.weights], map_location=device))
    model.eval()

    all_errors, all_snrs, all_pearsons = [], [], []
    for subject_dir in args.subject_dir:
        subject_dir = Path(subject_dir)
        print(f"  {subject_dir.name} : extraction avec sous-échantillonnage x{args.downsample}…")
        clips = make_clips_30fps(subject_dir, downsample=args.downsample)
        for x_np, y_np, fps in clips:
            x = torch.from_numpy(x_np).permute(3, 0, 1, 2).unsqueeze(0).to(device)
            with torch.no_grad():
                pred, _, _, _ = model(x)
            p_np = pred.squeeze().cpu().numpy()

            hr_g = hr_from_fft(y_np, fps)
            hr_p = hr_from_fft(p_np, fps)
            all_errors.append(abs(hr_p - hr_g))
            all_snrs.append(snr(p_np, hr_g, fps))
            all_pearsons.append(float(np.corrcoef(p_np, y_np)[0, 1]))
        print(f"    {len(clips)} clips ({clips[0][2] if clips else '?'} fps effectif)")

    if not all_errors:
        print("Aucun clip généré.")
        return

    mae  = float(np.mean(all_errors))
    rmse = float(np.sqrt(np.mean(np.array(all_errors) ** 2)))
    print(f"\n{'='*50}")
    print(f"  n        : {len(all_errors)} clips")
    print(f"  MAE      : {mae:.2f} bpm")
    print(f"  RMSE     : {rmse:.2f} bpm")
    print(f"  SNR      : {np.mean(all_snrs):.2f} dB")
    print(f"  Pearson  : {np.mean(all_pearsons):+.3f}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
