#!/usr/bin/env python3
"""
Calibre la constante K de l'ITA normalisé-sclère (peau ÷ sclère × K) pour que
l'échelle absolue corresponde au Fitzpatrick, sur DataVital.

Étape 1 : extrait skin_rgb + sclera_rgb (moyennes sur images échantillonnées) +
          Fitzpatrick pour un échantillon de sujets → cache JSON.
Étape 2 : balaie K, calcule l'ITA corrigé, mesure la corrélation de rang avec
          le Fitzpatrick et la médiane d'ITA du groupe Fitz 6, choisit K.

Usage : python scripts/calibrate_sclera_ita.py --n 25
"""
import argparse, json, sys
from pathlib import Path
import cv2, numpy as np
import mediapipe as mp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.chrom_adaptive import compute_ita
from scripts.test_sclera_ita import sclera_rgb, skin_rgb
from scripts.preextract_clips import _find_vitalvideos_json, load_video

CACHE = ROOT / "Data" / "sclera_calib_cache.json"


def extract(n):
    fm = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
        refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    subs = sorted([d for d in (ROOT / 'DataVital').iterdir() if d.is_dir() and d.name.startswith('Subject')])
    rows = []
    for sd in subs:
        if len(rows) >= n:
            break
        meta = _find_vitalvideos_json(sd)
        if meta is None:
            continue
        try:
            j = json.load(open([p for p in sd.glob('*.json') if p.name != 'metadata.json'][0]))
            fz = int(j['participant']['fitzpatrick'])
        except Exception:
            continue
        rec = meta['scenarios'][0]['recordings'].get('RGB', {})
        vid = sd / rec.get('filename', '')
        if not vid.exists():
            continue
        frames, _ = load_video(vid, max_dim=720)
        sk, sc = [], []
        for f in frames[::20]:
            bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR); res = fm.process(f)
            if not res.multi_face_landmarks:
                continue
            lm = res.multi_face_landmarks[0].landmark
            a = skin_rgb(bgr, lm); b = sclera_rgb(bgr, lm)
            if a is not None: sk.append(a)
            if b is not None: sc.append(b)
        if not sk or not sc:
            continue
        rows.append({"subject": sd.name, "fitz": fz,
                     "skin": np.mean(sk, 0).tolist(), "sclera": np.mean(sc, 0).tolist(),
                     "pct_sclera": round(100 * len(sc) / max(1, len(frames[::20]))),
                     "ita_raw": round(compute_ita(np.mean(sk, 0)))})
        print(f"  {sd.name:<12} Fitz{fz}  ITA_brut={rows[-1]['ita_raw']:>4}  sclère {rows[-1]['pct_sclera']}%")
    fm.close()
    CACHE.write_text(json.dumps(rows, indent=2))
    return rows


def corrected_ita(skin, sclera, K):
    return compute_ita(np.clip(np.array(skin) / (np.array(sclera) + 1e-6) * K, 0, 255))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=25)
    ap.add_argument('--reuse', action='store_true', help="réutiliser le cache")
    args = ap.parse_args()

    if args.reuse and CACHE.exists():
        rows = json.loads(CACHE.read_text())
    else:
        print("Extraction skin+sclère (échantillon DataVital)…")
        rows = extract(args.n)
    if len(rows) < 5:
        print("Pas assez de sujets avec sclère détectée."); return
    fz = np.array([r["fitz"] for r in rows])
    raw = np.array([r["ita_raw"] for r in rows])

    # cibles ITA standard par Fitzpatrick (centre de plage)
    TARGET = {1: 60, 2: 48, 3: 34, 4: 19, 5: -10, 6: -40}
    tgt = np.array([TARGET.get(int(f), 0) for f in rows and fz])

    def spearman(a, b):
        ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
        return np.corrcoef(ra, rb)[0, 1]

    print(f"\nSujets exploitables : {len(rows)}  (Fitz {sorted(set(fz.tolist()))})")
    print(f"ITA brut   : corr_rang(ITA,-Fitz) = {spearman(raw, -fz):+.2f}  "
          f"écart-type = {raw.std():.0f}")
    print(f"\n{'K':>5}{'corr_rang':>11}{'erreur/cible':>14}{'ITA Fitz6 méd':>15}")
    print('-' * 46)
    best = None
    for K in range(120, 261, 10):
        ci = np.array([corrected_ita(r["skin"], r["sclera"], K) for r in rows])
        corr = spearman(ci, -fz)
        err = np.mean(np.abs(ci - tgt))
        f6 = np.median(ci[fz == 6]) if (fz == 6).any() else float('nan')
        print(f"{K:>5}{corr:>11.2f}{err:>14.1f}{f6:>15.0f}")
        score = err - 30 * corr     # minimiser l'erreur, maximiser la corrélation
        if best is None or score < best[0]:
            best = (score, K, corr, err, f6)
    print('-' * 46)
    print(f"\n→ K recommandé = {best[1]}  (corr_rang {best[2]:+.2f}, erreur cible {best[3]:.1f}, "
          f"ITA Fitz6 médian {best[4]:.0f})")


if __name__ == '__main__':
    main()
