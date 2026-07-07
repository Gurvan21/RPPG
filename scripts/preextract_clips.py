#!/usr/bin/env python3
"""
Pré-extraction Africa-1 : vidéo brute → clips DiffNormalized (.npz).

Pour chaque sujet :
  - Détecte le visage (MediaPipe FaceMesh)
  - Découpe en clips de 128 frames (72×72)
  - Applique DiffNormalized
  - Aligne le signal PPG (avec offset de sync)
  - Sauvegarde en .npz (~5-6 Mo/clip)
  - Supprime la vidéo brute si --delete-raw

Deux formats de sujet sont supportés (détection automatique) :

1. Format générique :
  subject_dir/
    *.mp4              ← vidéo uncompressed
    *.csv              ← PPG (colonnes: timestamp_s, pleth  OU  time, bvp)
    metadata.json      ← optionnel (fps, sync_offset, etc.)

2. Format VitalVideos (JSON natif, ex. DataVital/) :
  subject_dir/
    <GUID>.json        ← scenarios[].recordings.{RGB,CMS,rr,BP}
    <GUID>_1.mp4, <GUID>_2.mp4  ← une vidéo par scénario
  Le JSON fournit les timestamps (ms depuis le début du scénario) pour
  chaque frame vidéo ET chaque échantillon PPG (CMS) → la synchronisation
  est exacte via cette origine commune, pas besoin de --sync-offset.

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
BBOX_PAD  = 0.1   # padding relatif autour de l'enveloppe des landmarks FaceMesh

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


def skin_neutralize(frame_rgb, min_skin_px=50):
    """
    Remplace les pixels non-peau (fond, cheveux, vêtements) d'une frame déjà
    recadrée par la couleur moyenne de peau de cette frame (masque YCrCb).
    Ne distingue PAS la peau du visage de la peau d'une main/autre — un masque
    colorimétrique ne peut pas faire cette distinction, les deux passent le
    même seuil. Réduit juste le bruit de fond capté par la marge du crop.
    """
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(ycrcb,
                       np.array([0,   110,  60], np.uint8),
                       np.array([255, 190, 145], np.uint8))
    skin_px = frame_rgb[mask > 0]
    if len(skin_px) < min_skin_px:
        return frame_rgb
    out = frame_rgb.copy()
    out[mask == 0] = skin_px.mean(axis=0)
    return out


SAVE_RAW = False   # si True, sauve aussi le flux apparence brut standardisé (xr) pour TS-CAN


def standardize_clip(clip):
    """Standardisation (x-mean)/std sur tout le clip — flux apparence TS-CAN."""
    m = clip.mean(); s = clip.std() + 1e-8
    return (clip - m) / s


def diff_normalize(clip):
    out = np.zeros_like(clip, dtype=np.float32)
    out[:-1] = (clip[1:] - clip[:-1]) / (clip[1:] + clip[:-1] + 1e-7)
    std = out[:-1].std()
    if std > 0:
        out /= std
    out[np.isnan(out)] = 0
    return out


def find_face_bbox(frames_rgb, step=10, pad=BBOX_PAD):
    """Bbox = enveloppe des landmarks MediaPipe FaceMesh + padding relatif."""
    import mediapipe as mp

    H, W = frames_rgb[0].shape[:2]
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1,
        refine_landmarks=True, min_detection_confidence=0.5,
    )
    try:
        for i in range(0, len(frames_rgb), step):
            res = face_mesh.process(frames_rgb[i])
            if not res.multi_face_landmarks:
                continue
            lm = res.multi_face_landmarks[0].landmark
            xs = [p.x for p in lm]
            ys = [p.y for p in lm]
            x1 = max(0, int((min(xs) - pad) * W))
            y1 = max(0, int((min(ys) - pad) * H))
            x2 = min(W, int((max(xs) + pad) * W))
            y2 = min(H, int((max(ys) + pad) * H))
            return (x1, y1, x2, y2)
    finally:
        face_mesh.close()
    return (0, 0, W, H)


def track_face_bboxes(frames_rgb, pad=BBOX_PAD):
    """
    Bbox par frame via MediaPipe FaceMesh en mode tracking (static_image_mode=
    False) : MediaPipe ne refait une détection complète que quand le suivi est
    perdu, sinon il propage les landmarks d'une frame à l'autre -- bien moins
    cher qu'un second réseau de segmentation (BiSeNet), tout en suivant le
    mouvement de tête contrairement à une boîte statique unique.

    Si aucun visage n'est détecté sur une frame, réutilise la dernière bbox
    connue plutôt que de retomber sur la frame entière (évite un saut brutal).

    Retourne une liste de bbox (x1,y1,x2,y2), une par frame.
    """
    import mediapipe as mp

    H, W = frames_rgb[0].shape[:2]
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1,
        refine_landmarks=True, min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    bboxes = []
    last_bbox = None
    try:
        for frame in frames_rgb:
            res = face_mesh.process(frame)
            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark
                xs = [p.x for p in lm]
                ys = [p.y for p in lm]
                x1 = max(0, int((min(xs) - pad) * W))
                y1 = max(0, int((min(ys) - pad) * H))
                x2 = min(W, int((max(xs) + pad) * W))
                y2 = min(H, int((max(ys) + pad) * H))
                last_bbox = (x1, y1, x2, y2)
            bboxes.append(last_bbox if last_bbox is not None else (0, 0, W, H))
    finally:
        face_mesh.close()
    return bboxes


TARGET_FPS = 30.0   # fps commun pour TOUS les pipelines (cohérence + match webcam/poids préentraînés)


def load_video(path, max_dim=None):
    """Charge toutes les frames RGB. Si max_dim est fourni, chaque frame est
    redimensionnée (ratio conservé) pour que sa plus grande dimension soit
    <= max_dim AVANT stockage — indispensable pour les vidéos 4K (sinon la pile
    de frames pleine résolution sature la RAM). Aucun impact sur le signal rPPG,
    qui est temporel et non lié à la résolution spatiale."""
    cap    = cv2.VideoCapture(str(path))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if max_dim is not None:
            h, w = f.shape[:2]
            if max(h, w) > max_dim:
                s = max_dim / max(h, w)
                f = cv2.resize(f, (int(round(w * s)), int(round(h * s))),
                               interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.array(frames, dtype=np.uint8), fps


def resample_to_fps(frames, frame_times_ms, target_fps=TARGET_FPS):
    """
    Ré-échantillonne une vidéo (et sa grille de timestamps) vers `target_fps` par
    sélection de la frame la plus proche (pas d'interpolation d'image → pas de flou).
    Indispensable pour homogénéiser des sujets filmés à 40/60 fps : les réseaux
    comptent la périodicité EN FRAMES, donc un fps mixte fausse l'apprentissage.

    frames : (T,H,W,3) ; frame_times_ms : (T,) ms depuis le début.
    Retourne (frames_rs, times_rs) sur une grille uniforme à target_fps.
    """
    frame_times_ms = np.asarray(frame_times_ms, dtype=np.float64)
    duration_s = (frame_times_ms[-1] - frame_times_ms[0]) / 1000.0
    n_target = int(round(duration_s * target_fps)) + 1
    grid_ms = frame_times_ms[0] + np.arange(n_target) * (1000.0 / target_fps)
    # frame la plus proche pour chaque point de la grille
    idx = np.abs(frame_times_ms[None, :] - grid_ms[:, None]).argmin(axis=1)
    return frames[idx], grid_ms


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


def _find_vitalvideos_json(subject_dir: Path):
    """Retourne le dict JSON VitalVideos (clé 'scenarios') du sujet, ou None si absent."""
    for jp in subject_dir.glob('*.json'):
        if jp.name == 'metadata.json':
            continue
        try:
            data = json.load(open(jp))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if 'scenarios' in data:
            return data
    return None


def _resample_ppg_to_frames(cms_rows, frame_times_ms, lo=0.7, hi=2.5):
    """
    cms_rows : liste de [time_ms, ppg, hr, spo2] (header déjà retiré).
    frame_times_ms : timestamps vidéo (ms depuis début scénario), même origine que cms_rows.
    Filtre passe-bande à la fréquence native du capteur puis interpole sur la grille vidéo.
    """
    times = np.array([r[0] for r in cms_rows], dtype=np.float64)
    ppg   = np.array([r[1] for r in cms_rows], dtype=np.float64)
    fs    = 1000.0 / np.median(np.diff(times))
    ppg_bp = bandpass(ppg, fs, lo, hi)
    return np.interp(frame_times_ms, times, ppg_bp).astype(np.float32)


def process_subject_vitalvideos(subject_dir: Path, output_dir: Path, meta: dict) -> int:
    """
    Sujet au format VitalVideos natif : un .mp4 + un sous-objet 'recordings' par scénario,
    synchronisation PPG/vidéo via les timestamps ms communs (RGB.timeseries / CMS).
    """
    subject_id = subject_dir.name
    out_subj   = output_dir / subject_id
    out_subj.mkdir(parents=True, exist_ok=True)
    n_clips = 0

    for sc_idx, scenario in enumerate(meta.get('scenarios', [])):
        rec = scenario.get('recordings', {})
        rgb, cms = rec.get('RGB'), rec.get('CMS')
        if not rgb or not cms or len(cms) < 2:
            print(f"  [SKIP] {subject_id} scénario {sc_idx} : RGB ou CMS manquant")
            continue

        video_path = subject_dir / rgb['filename']
        if not video_path.exists():
            print(f"  [SKIP] {subject_id} scénario {sc_idx} : vidéo introuvable ({rgb['filename']})")
            continue

        label = scenario.get('scenario_data', {}).get('scenario', '?')
        print(f"  {subject_id} scénario {sc_idx} ({label}) : chargement {rgb['filename']}…")
        frames, _ = load_video(video_path, max_dim=720)   # downscale -> évite l'OOM sur vidéos >1GB

        frame_times_ms = np.array([t for t, _ in rgb['timeseries']], dtype=np.float64)
        n = min(len(frames), len(frame_times_ms))
        if n < CLIP_LEN:
            print(f"  [SKIP] {subject_id} scénario {sc_idx} : trop court ({n} frames)")
            continue
        frames, frame_times_ms = frames[:n], frame_times_ms[:n]

        # Normalisation fps → 30 (cohérence inter-sujets : périodicité comptée en frames)
        frames, frame_times_ms = resample_to_fps(frames, frame_times_ms, TARGET_FPS)
        n = len(frames)
        fps = TARGET_FPS
        if n < CLIP_LEN:
            print(f"  [SKIP] {subject_id} scénario {sc_idx} : trop court après resample ({n})")
            continue

        ppg = _resample_ppg_to_frames(cms[1:], frame_times_ms)

        bboxes = track_face_bboxes(frames)
        cropped = np.stack([
            cv2.resize(skin_neutralize(f[y1:y2, x1:x2]), (RESIZE, RESIZE), interpolation=cv2.INTER_AREA)
            for f, (x1, y1, x2, y2) in zip(frames, bboxes)
        ]).astype(np.float32)

        for start in range(0, n - CLIP_LEN + 1, CLIP_LEN):
            clip_dn = diff_normalize(cropped[start:start + CLIP_LEN])
            ppg_seg = ppg[start:start + CLIP_LEN]
            ppg_seg = (ppg_seg - ppg_seg.mean()) / (ppg_seg.std() + 1e-8)

            out_path = out_subj / f"sc{sc_idx}_clip_{start:05d}.npz"
            arrs = dict(x=clip_dn.astype(np.float16), y=ppg_seg, fps=np.float32(fps))
            if SAVE_RAW:            # flux apparence brut standardisé pour TS-CAN
                arrs['xr'] = standardize_clip(cropped[start:start + CLIP_LEN]).astype(np.float16)
            np.savez_compressed(str(out_path), **arrs)
            n_clips += 1

    size_mb = sum(f.stat().st_size for f in out_subj.glob('*.npz')) / 1e6
    print(f"  {subject_id} : {n_clips} clips → {size_mb:.1f} Mo")
    return n_clips


def process_subject(subject_dir: Path, output_dir: Path, sync_offset: float,
                    delete_raw: bool) -> int:
    subject_id = subject_dir.name

    # Format VitalVideos natif (JSON avec scenarios/recordings) ?
    vv_meta = _find_vitalvideos_json(subject_dir)
    if vv_meta is not None:
        n_clips = process_subject_vitalvideos(subject_dir, output_dir, vv_meta)
        if delete_raw and n_clips > 0:
            shutil.rmtree(subject_dir)
            print(f"  {subject_id} : vidéo brute supprimée")
        return n_clips

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

    # Détection + suivi du visage frame par frame
    bboxes = track_face_bboxes(frames)

    # Crop + resize toutes les frames
    cropped = np.stack([
        cv2.resize(skin_neutralize(f[y1:y2, x1:x2]), (RESIZE, RESIZE), interpolation=cv2.INTER_AREA)
        for f, (x1, y1, x2, y2) in zip(frames, bboxes)
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
        arrs = dict(x=clip_dn.astype(np.float16), y=ppg_seg, fps=np.float32(fps))
        if SAVE_RAW:
            arrs['xr'] = standardize_clip(clip_raw).astype(np.float16)
        np.savez_compressed(str(out_path), **arrs)
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
    parser.add_argument('--save-raw',     action='store_true',
                        help="Sauver aussi le flux apparence brut standardisé (xr) pour TS-CAN")
    args = parser.parse_args()

    global SAVE_RAW
    SAVE_RAW = args.save_raw
    n = process_subject(Path(args.subject_dir), Path(args.output_dir),
                        args.sync_offset, args.delete_raw)
    print(f"Total : {n} clips extraits")


if __name__ == '__main__':
    main()
