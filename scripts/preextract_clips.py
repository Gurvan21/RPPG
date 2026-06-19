#!/usr/bin/env python3
"""
Pré-extraction Africa-1 : vidéo brute → clips DiffNormalized (.npz).

Pour chaque sujet :
  - Détecte le visage (Haar Cascade)
  - Découpe en clips de 128 frames (72×72)
  - Applique DiffNormalized
  - Aligne le signal PPG (avec offset de sync)
  - Sauvegarde en .npz (~5-6 Mo/clip)
  - Supprime la vidéo brute si --delete-raw

Format Africa-1 attendu (à ajuster selon readme.json) :
  subject_dir/
    *.mp4              ← vidéo uncompressed
    *.csv              ← PPG (colonnes: timestamp_s, pleth  OU  time, bvp)
    metadata.json      ← optionnel (fps, sync_offset, etc.)

Usage :
    python scripts/preextract_clips.py --subject-dir Data/africa1_raw/001 --output-dir Data/africa1_clips
    python scripts/preextract_clips.py --subject-dir Data/africa1_raw/001 --output-dir Data/africa1_clips --delete-raw
"""

import argparse
import csv
import json
import os
import sys
import shutil
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import butter, filtfilt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CLIP_LEN  = 128
RESIZE    = 72
BOX_COEF  = 1.4
HAAR_XML  = os.path.join(ROOT, 'assets/haarcascade_frontalface_default.xml')

# ── À AJUSTER selon la "note on sync" du Dropbox ─────────────────────────────
# Offset en secondes entre le début de la vidéo et le début du PPG.
# Exemple : si le PPG démarre 0.5s avant la vidéo → SYNC_OFFSET = +0.5
# Lire attentivement le fichier "note on sync" pour chaque dataset.
DEFAULT_SYNC_OFFSET = 0.0   # secondes (PPG - vidéo)
# ─────────────────────────────────────────────────────────────────────────────


def bandpass(sig, fs, lo=0.7, hi=2.5):
    nyq = fs / 2.0
    b, a = butter(4, [lo / nyq, hi / nyq], btype='band')
    return filtfilt(b, a, sig)


def diff_normalize(clip):
    out = np.zeros_like(clip, dtype=np.float32)
    out[:-1] = (clip[1:] - clip[:-1]) / (clip[1:] + clip[:-1] + 1e-7)
    std = out[:-1].std()
    if std > 0:
        out /= std
    out[np.isnan(out)] = 0
    return out


def find_face_bbox(frames_rgb, step=10):
    detector = cv2.CascadeClassifier(HAAR_XML)
    for i in range(0, len(frames_rgb), step):
        gray  = cv2.cvtColor(frames_rgb[i], cv2.COLOR_RGB2GRAY)
        zones = detector.detectMultiScale(gray, 1.1, 5)
        if len(zones):
            x, y, w, h = zones[np.argmax(zones[:, 2])]
            cx, cy = x + w // 2, y + h // 2
            hw = int(w * BOX_COEF / 2)
            hh = int(h * BOX_COEF / 2)
            H, W = frames_rgb[i].shape[:2]
            return (max(0, cx-hw), max(0, cy-hh), min(W, cx+hw), min(H, cy+hh))
    H, W = frames_rgb[0].shape[:2]
    return (0, 0, W, H)


def load_video(path):
    cap    = cv2.VideoCapture(str(path))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.array(frames, dtype=np.uint8), fps


def load_ppg(subject_dir: Path, fps_vid: float, n_frames: int, sync_offset: float):
    """
    Charge le PPG et le resample sur la grille vidéo.
    sync_offset : décalage PPG-vidéo en secondes (voir "note on sync").
    """
    # Cherche CSV
    csvs = list(subject_dir.glob('*.csv'))
    if not csvs:
        return None

    with open(csvs[0]) as f:
        rows = list(csv.DictReader(f))

    # Détection automatique des colonnes
    keys = list(rows[0].keys())
    # Colonne signal : 'pleth', 'bvp', 'ppg', ou dernière colonne
    sig_key = next((k for k in keys if k.lower() in ('pleth', 'bvp', 'ppg', 'signal')),
                   keys[-1])
    # Colonne temps : 'timestamp_s', 'time', 'timestamp', ou None (équidistant 60Hz)
    ts_key  = next((k for k in keys if 'time' in k.lower() or 'ts' in k.lower()), None)

    signal = np.array([float(r[sig_key]) for r in rows], dtype=np.float32)

    if ts_key:
        ts = np.array([float(r[ts_key]) for r in rows], dtype=np.float64)
        fps_ppg = 1.0 / np.median(np.diff(ts))
    else:
        fps_ppg = 60.0
        ts = np.arange(len(signal)) / fps_ppg

    # Appliquer l'offset de sync : décaler le PPG
    ts = ts - sync_offset

    # Grille temporelle vidéo
    duration    = n_frames / fps_vid
    vid_times   = np.linspace(0, duration, n_frames)
    mask        = (ts >= 0) & (ts <= duration)

    if mask.sum() < 2:
        print(f"    [WARN] PPG hors plage après sync offset={sync_offset}s")
        return None

    signal_bp = bandpass(signal, fps_ppg)
    resampled = np.interp(vid_times, ts[mask], signal_bp[mask]).astype(np.float32)
    return resampled


def process_subject(subject_dir: Path, output_dir: Path, sync_offset: float,
                    delete_raw: bool) -> int:
    subject_id = subject_dir.name

    # Trouver la vidéo
    videos = (list(subject_dir.glob('*.mp4')) + list(subject_dir.glob('*.avi'))
              + list(subject_dir.glob('*.mov')))
    if not videos:
        print(f"  [SKIP] {subject_id} : pas de vidéo")
        return 0

    # Lire métadonnées locales si présentes (sync_offset par sujet)
    meta_path = subject_dir / 'metadata.json'
    local_offset = sync_offset
    if meta_path.exists():
        meta = json.load(open(meta_path))
        local_offset = float(meta.get('sync_offset_s', sync_offset))

    print(f"  {subject_id} : chargement vidéo…")
    frames, fps = load_video(videos[0])
    n = len(frames)
    if n < CLIP_LEN:
        print(f"  [SKIP] {subject_id} : trop court ({n} frames)")
        return 0

    # PPG
    ppg = load_ppg(subject_dir, fps, n, local_offset)
    if ppg is None:
        print(f"  [SKIP] {subject_id} : PPG introuvable ou invalide")
        return 0

    # Détection visage sur quelques frames
    bbox = find_face_bbox(frames)
    x1, y1, x2, y2 = bbox

    # Crop + resize toutes les frames
    cropped = np.stack([
        cv2.resize(f[y1:y2, x1:x2], (RESIZE, RESIZE), interpolation=cv2.INTER_AREA)
        for f in frames
    ]).astype(np.float32)   # (T, 72, 72, 3)

    # Découpage en clips
    n_clips = 0
    out_subj = output_dir / subject_id
    out_subj.mkdir(parents=True, exist_ok=True)

    for start in range(0, n - CLIP_LEN + 1, CLIP_LEN):
        clip_raw = cropped[start:start + CLIP_LEN]
        clip_dn  = diff_normalize(clip_raw)              # (128, 72, 72, 3)
        ppg_seg  = ppg[start:start + CLIP_LEN]

        # z-score GT
        ppg_seg = (ppg_seg - ppg_seg.mean()) / (ppg_seg.std() + 1e-8)

        out_path = out_subj / f"clip_{start:05d}.npz"
        np.savez_compressed(str(out_path),
                            x=clip_dn.astype(np.float16),   # fp16 → moitié de taille
                            y=ppg_seg,
                            fps=np.float32(fps))
        n_clips += 1

    size_mb = sum(f.stat().st_size for f in out_subj.glob('*.npz')) / 1e6
    print(f"  {subject_id} : {n_clips} clips → {size_mb:.1f} Mo")

    if delete_raw:
        shutil.rmtree(subject_dir)
        print(f"  {subject_id} : vidéo brute supprimée")

    return n_clips


def main():
    parser = argparse.ArgumentParser(description="Pré-extraction clips Africa-1")
    parser.add_argument('--subject-dir',  required=True)
    parser.add_argument('--output-dir',   default='Data/africa1_clips')
    parser.add_argument('--sync-offset',  type=float, default=DEFAULT_SYNC_OFFSET,
                        help="Décalage PPG-vidéo en secondes (voir note on sync)")
    parser.add_argument('--delete-raw',   action='store_true',
                        help="Supprimer la vidéo brute après extraction")
    args = parser.parse_args()

    n = process_subject(Path(args.subject_dir), Path(args.output_dir),
                        args.sync_offset, args.delete_raw)
    print(f"Total : {n} clips extraits")


if __name__ == '__main__':
    main()
