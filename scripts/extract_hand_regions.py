#!/usr/bin/env python3
"""
Extraction des signaux multi-régions / multi-espaces-couleur de la PAUME
(MediaPipe Hands), pour entraîner un CNN1D-main analogue au CNN1D visage.

Motivation équité : la paume est la peau la MOINS pigmentée du corps (même
Fitzpatrick 5-6) → meilleur SNR rPPG que le visage sur peau foncée (validé en
POC : SNR paume +3 à +12 dB vs visage négatif). On réplique donc le pipeline
region_signals visage, mais sur la main.

Régions (depuis les 21 landmarks Hands) :
  - 8 régions anatomiques palmaires (centre, thénar, hypothénar, sous-doigts,
    bas-paume, poignet)
  - GRID×GRID blocs réguliers sur la boîte de la paume (masqués par l'enveloppe)
  - paume_complete (enveloppe convexe entière)
Par région : moyenne RGB + YUV + Lab = 9 canaux (mêmes que le visage).

Masque « peau » = enveloppe convexe des landmarks palmaires, érodée + filtre de
luminance (exclut fond / ombres / spéculaire). Pas de réseau de segmentation.

Sortie .npz (format identique à region_signals) :
  x=(T, N_REGIONS, 9), y=(T,), fps, region_names, color_names, fitz, scenario.

Usage :
    python scripts/extract_hand_regions.py --out Data/hand_signals --grid 3
"""
import argparse, os, sys, json, glob, time
from pathlib import Path
import cv2, numpy as np, mediapipe as mp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.preextract_clips import (load_video, resample_to_fps, TARGET_FPS,
                                      _resample_ppg_to_frames)

COLOR_NAMES = ['R', 'G', 'B', 'Y', 'U', 'V', 'L', 'a', 'b']
N_COLORS = 9
HULL_LM = [0, 1, 2, 5, 9, 13, 17]            # enveloppe palmaire (poignet + bases doigts)
# centres des régions anatomiques (moyennes de landmarks)
ANAT = {
    'centre':      [0, 5, 9, 13, 17],
    'thenar':      [1, 2, 5],
    'hypothenar':  [0, 17],
    'sous_index':  [5, 9],
    'sous_majeur': [9, 13],
    'sous_annul':  [13, 17],
    'bas_paume':   [0, 9],
    'poignet':     [0],
}
HAND_SCENARIOS = {'facepalm', 'handheld'}


def _lm_px(lm, idx, W, H):
    return np.array([[lm[i].x * W, lm[i].y * H] for i in idx], np.float32)


def _avg_colors(bgr, yuv, lab, mask):
    sel = mask > 0
    if sel.sum() < 60:
        return np.full(N_COLORS, np.nan, np.float32)
    pb = bgr[sel].astype(np.float32)
    lum = pb.mean(1)
    keep = (lum > 30) & (lum < 245)
    if keep.sum() < 40:
        return np.full(N_COLORS, np.nan, np.float32)
    # filtre outliers ±2σ par canal BGR
    k2 = np.ones(keep.sum(), bool); pbk = pb[keep]
    for c in range(3):
        med, std = np.median(pbk[:, c]), np.std(pbk[:, c])
        if std > 0:
            k2 &= np.abs(pbk[:, c] - med) <= 2.0 * std
    if k2.sum() < 40:
        return np.full(N_COLORS, np.nan, np.float32)
    py = yuv[sel][keep][k2].astype(np.float32)
    pl = lab[sel][keep][k2].astype(np.float32)
    pbk = pbk[k2]
    b, g, r = pbk[:, 0].mean(), pbk[:, 1].mean(), pbk[:, 2].mean()
    return np.array([r, g, b, py[:, 0].mean(), py[:, 1].mean(), py[:, 2].mean(),
                     pl[:, 0].mean(), pl[:, 1].mean(), pl[:, 2].mean()], np.float32)


def region_names(grid):
    return list(ANAT.keys()) + ['paume_complete'] + \
           [f'grid_{i}_{j}' for i in range(grid) for j in range(grid)]


def extract_video(frames_rgb, grid, edge_frac=0.0):
    names = region_names(grid)
    n_anat = len(ANAT); n_reg = len(names)
    T = len(frames_rgb); H, W = frames_rgb.shape[1:3]
    out = np.full((T, n_reg, N_COLORS), np.nan, np.float32)
    hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1,
                                     min_detection_confidence=0.5, min_tracking_confidence=0.5)
    det = 0
    try:
        for t, frgb in enumerate(frames_rgb):
            res = hands.process(frgb)
            if not res.multi_hand_landmarks:
                continue
            det += 1
            lm = res.multi_hand_landmarks[0].landmark
            bgr = cv2.cvtColor(frgb, cv2.COLOR_RGB2BGR)
            yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)
            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
            # enveloppe palmaire (masque « peau »)
            hull = cv2.convexHull(_lm_px(lm, HULL_LM, W, H).astype(np.int32))
            palm = np.zeros((H, W), np.uint8); cv2.fillConvexPoly(palm, hull, 255)
            scale = np.linalg.norm(_lm_px(lm, [0], W, H)[0] - _lm_px(lm, [9], W, H)[0])
            if edge_frac > 0:
                # filtre SANS pixels de bord : ne garde que l'intérieur profond
                # (distance à la frontière >= edge_frac * taille_paume) → réduit
                # les artefacts de mouvement au bord (fond qui entre/sort du masque)
                dist = cv2.distanceTransform(palm, cv2.DIST_L2, 5)
                palm = np.where(dist >= edge_frac * scale, np.uint8(255), np.uint8(0))
            else:
                er = max(3, int(0.015 * min(H, W)) | 1)
                palm = cv2.erode(palm, np.ones((er, er), np.uint8))
            rad = max(4, int(0.16 * scale))
            # régions anatomiques : disque autour du centre ∩ paume
            for r, (nm, idx) in enumerate(ANAT.items()):
                c = _lm_px(lm, idx, W, H).mean(0).astype(int)
                disk = np.zeros((H, W), np.uint8)
                cv2.circle(disk, tuple(c), rad, 255, -1)
                out[t, r] = _avg_colors(bgr, yuv, lab, cv2.bitwise_and(disk, palm))
            # paume complète
            out[t, n_anat] = _avg_colors(bgr, yuv, lab, palm)
            # grille sur la bbox de la paume, masquée par l'enveloppe
            xs, ys = hull[:, 0, 0], hull[:, 0, 1]
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
            cw = max(1, (x2 - x1) // grid); ch = max(1, (y2 - y1) // grid)
            for i in range(grid):
                for j in range(grid):
                    cx1, cy1 = x1 + j * cw, y1 + i * ch
                    cx2 = x2 if j == grid - 1 else cx1 + cw
                    cy2 = y2 if i == grid - 1 else cy1 + ch
                    cell = np.zeros((H, W), np.uint8); cell[cy1:cy2, cx1:cx2] = 255
                    out[t, n_anat + 1 + i * grid + j] = _avg_colors(
                        bgr, yuv, lab, cv2.bitwise_and(cell, palm))
    finally:
        hands.close()
    # interpolation linéaire des NaN
    for r in range(n_reg):
        for c in range(N_COLORS):
            col = out[:, r, c]; nan = np.isnan(col)
            if nan.all(): col[:] = 0.0
            elif nan.any():
                ix = np.arange(T); col[nan] = np.interp(ix[nan], ix[~nan], col[~nan])
    return out, names, det / max(T, 1)


def collect():
    items = []
    for jf in sorted(glob.glob(str(ROOT / "DataVital" / "Subject*" / "*.json"))):
        try: d = json.load(open(jf))
        except: continue
        fitz = d.get("participant", {}).get("fitzpatrick")
        for si, sc in enumerate(d.get("scenarios", [])):
            nm = sc.get("scenario_data", {}).get("scenario")
            if nm in HAND_SCENARIOS:
                rec = sc.get("recordings", {})
                rgb, cms = rec.get("RGB"), rec.get("CMS")
                if rgb and cms and len(cms) > 2:
                    items.append((Path(jf), si, nm, fitz, rgb, cms))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(ROOT / 'Data' / 'hand_signals'))
    ap.add_argument('--grid', type=int, default=3)
    ap.add_argument('--max-dim', type=int, default=640)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--min-det', type=float, default=0.5)
    ap.add_argument('--shard', type=int, default=None)
    ap.add_argument('--n-shards', type=int, default=None)
    ap.add_argument('--edge-frac', type=float, default=0.0,
                    help="Retire les pixels à moins de edge_frac*taille_paume du bord")
    args = ap.parse_args()
    items = collect()
    if args.limit: items = items[:args.limit]
    if args.shard is not None and args.n_shards:
        items = items[args.shard::args.n_shards]
        print(f"[shard {args.shard}/{args.n_shards}] ", end='')
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(items)} scénarios main (facepalm/handheld), grille {args.grid}×{args.grid} "
          f"→ {len(region_names(args.grid))} régions × 9 canaux\n", flush=True)
    n_ok = 0
    for k, (jf, si, nm, fitz, rgb, cms) in enumerate(items):
        subj = jf.parent.name.replace(' ', '_')
        (out_dir / subj).mkdir(parents=True, exist_ok=True)
        op = out_dir / subj / f"sc{si}.npz"
        if op.exists(): n_ok += 1; continue
        vp = jf.parent / rgb['filename']
        if not vp.exists(): continue
        t0 = time.time()
        try:
            frames, _ = load_video(str(vp), max_dim=args.max_dim)
            ft = np.array([t for t, _ in rgb['timeseries']], np.float64)
            n = min(len(frames), len(ft)); frames, ft = frames[:n], ft[:n]
            frames, ft = resample_to_fps(frames, ft, TARGET_FPS)
            x, names, det = extract_video(frames, args.grid, edge_frac=args.edge_frac)
            if det < args.min_det:
                print(f"[{k+1}/{len(items)}] {subj} sc{si} {nm} F{fitz}: SKIP (main {det*100:.0f}%)", flush=True)
                continue
            y = _resample_ppg_to_frames(cms[1:], ft)
            np.savez_compressed(str(op), x=x.astype(np.float32), y=y.astype(np.float32),
                                fps=np.float32(TARGET_FPS), region_names=np.array(names),
                                color_names=np.array(COLOR_NAMES),
                                fitz=np.array(str(fitz)), scenario=np.array(nm))
            n_ok += 1
            print(f"[{k+1}/{len(items)}] {subj} sc{si} {nm} F{fitz}: x={x.shape} "
                  f"main{det*100:.0f}% [{time.time()-t0:.0f}s]", flush=True)
        except Exception as e:
            print(f"[{k+1}/{len(items)}] {subj} sc{si}: ERREUR {e}", flush=True)
    print(f"\n{n_ok} scénarios extraits dans {out_dir}")


if __name__ == '__main__':
    main()
