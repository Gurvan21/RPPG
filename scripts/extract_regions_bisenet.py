#!/usr/bin/env python3
"""
Extraction des signaux multi-régions / multi-espaces-couleur (BiSeNet face-
parsing + MediaPipe FaceMesh tracking) pour l'entraînement d'un CNN 1D rPPG.

Régions extraites par frame :
  - 7 régions ANATOMIQUES (front, joues, nez, sous-yeux, peau complète)
  - GRID×GRID blocs réguliers sur la boîte du visage (style MSTmap / RhythmNet),
    par défaut 4×4 = 16 blocs
  → N_REGIONS = 7 + GRID² régions

Par région, moyenne des pixels de peau (BiSeNet) dans 3 espaces couleur :
  RGB + YUV + Lab  → 9 canaux couleur par région.

Sortie .npz : x=(T, N_REGIONS, 9) float32, y=(T,) float32, fps, region_names,
              color_names.  Le script d'entraînement choisit quelles régions /
              quels espaces utiliser (pas besoin de re-extraire).

Coûteux (BiSeNet sur chaque frame) mais mis en cache → une seule fois.

Usage :
    python scripts/extract_regions_bisenet.py --data DataVital --out Data/region_signals --grid 4
"""

import argparse
import os
import sys
from pathlib import Path

# BiSeNet (backbone ResNet18) télécharge des poids ImageNet à l'init — inutile
# (checkpoint complet chargé ensuite) et casse hors-ligne.
import torch.utils.model_zoo as _mz
_mz.load_url = lambda *a, **k: {}

import cv2
import mediapipe as mp
import numpy as np
import torch
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "semaine 1"))

from model import BiSeNet
from scripts.preextract_clips import (
    _find_vitalvideos_json, _resample_ppg_to_frames, load_video,
    resample_to_fps, TARGET_FPS,
)

# ── Régions anatomiques (landmarks FaceMesh, identiques à Semaine 2/server.py) ──
FRONT       = [103, 67, 109, 10, 338, 297, 332, 333, 168, 104]
LEFT_CHEEK  = [187, 214, 211, 57, 216, 203, 101, 118, 117, 123]
RIGHT_CHEEK = [411, 434, 431, 287, 436, 423, 330, 347, 346, 352]
NOSE        = [1, 45, 134, 174, 197, 399, 363, 275]
LEFT_EYE    = [24, 23, 22, 121, 47, 100, 119, 228]
RIGHT_EYE   = [252, 253, 254, 448, 348, 329, 277, 350]
ANATOMICAL = {
    'front': FRONT, 'joue_gauche': LEFT_CHEEK, 'joue_droite': RIGHT_CHEEK,
    'nez': NOSE, 'sous_oeil_gauche': LEFT_EYE, 'sous_oeil_droit': RIGHT_EYE,
}
COLOR_NAMES = ['R', 'G', 'B', 'Y', 'U', 'V', 'L', 'a', 'b']   # 9 canaux
N_COLORS = len(COLOR_NAMES)

_ERODE = np.ones((5, 5), np.uint8)
_BISENET_CKPT = ROOT / "DataVital" / "Session 15:06 Arthur" / "79999_iter.pth"
_BBOX_PAD = 0.1
_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def pick_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_bisenet(device):
    net = BiSeNet(n_classes=19)
    net.load_state_dict(torch.load(str(_BISENET_CKPT), map_location='cpu'))
    return net.to(device).eval()


def skin_mask_bisenet(net, frame_bgr, device):
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    inp = cv2.resize(rgb, (512, 512))
    t = _tf(inp).unsqueeze(0).to(device)
    with torch.no_grad():
        parsing = net(t)[0].squeeze(0).argmax(0).cpu().numpy().astype(np.uint8)
    parsing = cv2.resize(parsing, (w, h), interpolation=cv2.INTER_NEAREST)
    return np.where((parsing == 1) | (parsing == 10), np.uint8(255), np.uint8(0))


def skin_masks_bisenet_batch(net, frames_bgr, device, batch=16):
    """Masques de peau BiSeNet pour une liste de frames, batché sur le GPU
    (bien plus rapide que frame par frame). Résultat identique à la version
    unitaire. Retourne une liste de masques uint8 (H,W)."""
    h, w = frames_bgr[0].shape[:2]
    masks = []
    for i in range(0, len(frames_bgr), batch):
        chunk = frames_bgr[i:i + batch]
        ts = torch.stack([_tf(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (512, 512)))
                          for f in chunk]).to(device)
        with torch.no_grad():
            par = net(ts)[0].argmax(1).cpu().numpy().astype(np.uint8)   # (B,512,512)
        for p in par:
            pr = cv2.resize(p, (w, h), interpolation=cv2.INTER_NEAREST)
            masks.append(np.where((pr == 1) | (pr == 10), np.uint8(255), np.uint8(0)))
    return masks


def polygon_mask(shape, lm, indices):
    h, w = shape[:2]
    pts = np.array([[int(lm[i].x * w), int(lm[i].y * h)] for i in indices], np.int32)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(mask, pts, 255)
    return mask


def face_bbox(lm, h, w, pad=_BBOX_PAD):
    xs = [p.x for p in lm]; ys = [p.y for p in lm]
    x1 = max(0, int((min(xs) - pad) * w)); y1 = max(0, int((min(ys) - pad) * h))
    x2 = min(w, int((max(xs) + pad) * w)); y2 = min(h, int((max(ys) + pad) * h))
    return x1, y1, x2, y2


def grid_masks(shape, bbox, grid):
    """GRID×GRID masques rectangulaires couvrant la boîte du visage."""
    h, w = shape[:2]
    x1, y1, x2, y2 = bbox
    cw = max(1, (x2 - x1) // grid); ch = max(1, (y2 - y1) // grid)
    masks = []
    for i in range(grid):
        for j in range(grid):
            m = np.zeros((h, w), np.uint8)
            cx1, cy1 = x1 + j * cw, y1 + i * ch
            cx2 = x2 if j == grid - 1 else cx1 + cw
            cy2 = y2 if i == grid - 1 else cy1 + ch
            m[cy1:cy2, cx1:cx2] = 255
            masks.append(m)
    return masks


def region_colors(bgr, yuv, lab, mask, skin, sigma=2.0):
    """Moyenne (9 canaux RGB+YUV+Lab) des pixels (region ∩ peau), filtre outliers."""
    final = cv2.bitwise_and(mask, skin)
    sel = final > 0
    px_bgr = bgr[sel].astype(np.float32)
    if len(px_bgr) < 30:
        return np.full(N_COLORS, np.nan, np.float32)
    keep = np.ones(len(px_bgr), bool)
    for c in range(3):
        med, std = np.median(px_bgr[:, c]), np.std(px_bgr[:, c])
        if std > 0:
            keep &= np.abs(px_bgr[:, c] - med) <= sigma * std
    if keep.sum() < 30:
        return np.full(N_COLORS, np.nan, np.float32)
    px_yuv = yuv[sel][keep].astype(np.float32)
    px_lab = lab[sel][keep].astype(np.float32)
    px_bgr = px_bgr[keep]
    b, g, r = px_bgr[:, 0].mean(), px_bgr[:, 1].mean(), px_bgr[:, 2].mean()
    yy, u, v = px_yuv[:, 0].mean(), px_yuv[:, 1].mean(), px_yuv[:, 2].mean()
    ll, aa, bb = px_lab[:, 0].mean(), px_lab[:, 1].mean(), px_lab[:, 2].mean()
    return np.array([r, g, b, yy, u, v, ll, aa, bb], np.float32)


def _poly_pts_bbox(lm, indices, h, w, pad=3):
    """Points entiers du polygone + sa boîte englobante (avec marge pour l'érosion)."""
    pts = np.array([[int(lm[i].x * w), int(lm[i].y * h)] for i in indices], np.int32)
    x0 = max(0, pts[:, 0].min() - pad); x1 = min(w, pts[:, 0].max() + pad + 1)
    y0 = max(0, pts[:, 1].min() - pad); y1 = min(h, pts[:, 1].max() + pad + 1)
    return pts, (x0, y0, x1, y1)


def region_colors_local(bgr, yuv, lab, mask_local, skin_local, sigma=2.0):
    """Comme region_colors mais sur des tableaux DÉJÀ recadrés sur la région
    (boîte englobante). Résultat strictement identique à la version plein cadre,
    car la région ∩ peau est entièrement contenue dans la boîte."""
    return region_colors(bgr, yuv, lab, mask_local, skin_local, sigma)


def build_region_names(grid):
    return list(ANATOMICAL.keys()) + ['peau_complete'] + \
           [f'grid_{i}_{j}' for i in range(grid) for j in range(grid)]


def extract_video(net, device, frames_rgb, grid, bisenet_batch=16):
    region_names = build_region_names(grid)
    n_reg = len(region_names)
    n_anat = len(ANATOMICAL)
    T = len(frames_rgb)
    out = np.full((T, n_reg, N_COLORS), np.nan, np.float32)

    # 1) FaceMesh par frame (CPU) — on stocke les landmarks
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    )
    lms, bgrs = [], []
    try:
        for frame_rgb in frames_rgb:
            res = face_mesh.process(frame_rgb)
            lms.append(res.multi_face_landmarks[0].landmark if res.multi_face_landmarks else None)
            bgrs.append(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    finally:
        face_mesh.close()

    # 2) BiSeNet batché sur le GPU (gros gain de vitesse, résultat identique)
    skins = skin_masks_bisenet_batch(net, bgrs, device, batch=bisenet_batch)

    # 3) Moyennes par région (CPU) — chaque région recadrée sur sa boîte englobante
    #    (résultat strictement identique à la version plein cadre, ~10× plus rapide)
    for t, (lm, bgr, skin) in enumerate(zip(lms, bgrs, skins)):
        if lm is None:
            continue
        yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        h, w = bgr.shape[:2]

        # régions anatomiques : masque construit seulement dans la bbox
        for r, name in enumerate(ANATOMICAL):
            pts, (x0, y0, x1, y1) = _poly_pts_bbox(lm, ANATOMICAL[name], h, w)
            mloc = np.zeros((y1 - y0, x1 - x0), np.uint8)
            cv2.fillConvexPoly(mloc, pts - [x0, y0], 255)
            if name.startswith('joue'):
                mloc = cv2.erode(mloc, _ERODE, iterations=1)
            out[t, r] = region_colors_local(
                bgr[y0:y1, x0:x1], yuv[y0:y1, x0:x1], lab[y0:y1, x0:x1],
                mloc, skin[y0:y1, x0:x1])

        # peau complète : recadrée sur la bbox des pixels de peau
        ys, xs = np.where(skin > 0)
        if len(xs):
            x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
            sc = skin[y0:y1, x0:x1]
            out[t, n_anat] = region_colors_local(
                bgr[y0:y1, x0:x1], yuv[y0:y1, x0:x1], lab[y0:y1, x0:x1], sc, sc)

        # blocs de grille : chaque cellule est déjà un rectangle → slice direct
        gx1, gy1, gx2, gy2 = face_bbox(lm, h, w)
        cw = max(1, (gx2 - gx1) // grid); ch = max(1, (gy2 - gy1) // grid)
        for i in range(grid):
            for j in range(grid):
                cx1, cy1 = gx1 + j * cw, gy1 + i * ch
                cx2 = gx2 if j == grid - 1 else cx1 + cw
                cy2 = gy2 if i == grid - 1 else cy1 + ch
                sc = skin[cy1:cy2, cx1:cx2]
                mloc = np.full((cy2 - cy1, cx2 - cx1), 255, np.uint8)
                out[t, n_anat + 1 + i * grid + j] = region_colors_local(
                    bgr[cy1:cy2, cx1:cx2], yuv[cy1:cy2, cx1:cx2], lab[cy1:cy2, cx1:cx2],
                    mloc, sc)

    det_ratio = sum(lm is not None for lm in lms) / max(T, 1)

    # interpolation linéaire des NaN
    for r in range(n_reg):
        for c in range(N_COLORS):
            col = out[:, r, c]
            nans = np.isnan(col)
            if nans.all():
                col[:] = 0.0
            elif nans.any():
                idx = np.arange(T)
                col[nans] = np.interp(idx[nans], idx[~nans], col[~nans])
    return out, region_names, det_ratio


def main():
    ap = argparse.ArgumentParser(description="Extraction multi-régions/multi-couleur BiSeNet")
    ap.add_argument('--data', default=str(ROOT / 'DataVital'))
    ap.add_argument('--out',  default=str(ROOT / 'Data' / 'region_signals'))
    ap.add_argument('--grid', type=int, default=4, help="Subdivision GRID×GRID de la boîte visage")
    ap.add_argument('--shard', type=int, default=None,
                    help="Index du processus (0..n_shards-1) pour extraction parallèle")
    ap.add_argument('--n-shards', type=int, default=None,
                    help="Nombre total de processus parallèles (chacun traite 1 sujet sur n_shards)")
    args = ap.parse_args()

    device = pick_device()
    print(f"Device : {device}  |  grille {args.grid}×{args.grid}")
    net = load_bisenet(device)
    n_reg = len(ANATOMICAL) + 1 + args.grid ** 2
    print(f"BiSeNet chargé — {n_reg} régions × {N_COLORS} canaux couleur")

    data_dir = Path(args.data); out_dir = Path(args.out)
    subjects = sorted(d for d in data_dir.iterdir()
                      if d.is_dir() and d.name.lower().startswith('subject '))
    # Extraction parallèle : chaque shard traite 1 sujet sur n_shards (sujets indépendants)
    if args.shard is not None and args.n_shards:
        subjects = subjects[args.shard::args.n_shards]
        print(f"[shard {args.shard}/{args.n_shards}] ", end='')
    print(f"{len(subjects)} sujets à traiter\n")

    total = 0
    for subj in subjects:
        meta = _find_vitalvideos_json(subj)
        if meta is None:
            continue
        out_subj = out_dir / subj.name
        out_subj.mkdir(parents=True, exist_ok=True)
        for sc_idx, scenario in enumerate(meta.get('scenarios', [])):
            rec = scenario.get('recordings', {})
            rgb_meta, cms = rec.get('RGB'), rec.get('CMS')
            if not rgb_meta or not cms or len(cms) < 2:
                continue
            out_path = out_subj / f"sc{sc_idx}.npz"
            if out_path.exists():
                total += 1
                continue
            video_path = subj / rgb_meta['filename']
            if not video_path.exists():
                print(f"  [SKIP] {subj.name} sc{sc_idx} : vidéo introuvable")
                continue
            print(f"  {subj.name} sc{sc_idx} ({scenario.get('scenario_data',{}).get('scenario','?')}) ...", flush=True)
            frames, _ = load_video(video_path)
            ft = np.array([t for t, _ in rgb_meta['timeseries']], np.float64)
            n = min(len(frames), len(ft)); frames, ft = frames[:n], ft[:n]
            # Normalisation fps → 30 (identique au pipeline PhysNet)
            frames, ft = resample_to_fps(frames, ft, TARGET_FPS)
            fps = TARGET_FPS
            x, region_names, det = extract_video(net, device, frames, args.grid)
            if det < 0.5:
                print(f"    [SKIP] {subj.name} sc{sc_idx} : visage détecté sur seulement {det*100:.0f}% des frames")
                continue
            y = _resample_ppg_to_frames(cms[1:], ft)
            np.savez_compressed(str(out_path), x=x.astype(np.float32),
                                y=y.astype(np.float32), fps=np.float32(fps),
                                region_names=np.array(region_names),
                                color_names=np.array(COLOR_NAMES))
            total += 1
            print(f"    -> {out_path.name}  x={x.shape}  ({len(frames)} frames, visage {det*100:.0f}%)")

    print(f"\nTotal : {total} scénarios extraits dans {out_dir}")


if __name__ == '__main__':
    main()
