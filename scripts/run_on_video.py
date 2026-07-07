#!/usr/bin/env python3
"""
Applique toutes les méthodes rPPG sur une vidéo brute (sans PPG de référence).
Extraction BiSeNet multi-régions + FaceMesh, puis CNN 1D / CHROM-ITA / CHROM /
POS + fusion. Affiche la HR prédite et le SNR aveugle de chaque méthode.

Sans vérité-terrain → pas d'erreur calculable ; l'ACCORD entre méthodes est le
meilleur indice de fiabilité.

Usage :
    python scripts/run_on_video.py --video DataVital/videoTest.mov
"""

import argparse
import os
import sys
from pathlib import Path

# MaxPool3d (PhysNet) pas implémenté sur MPS → fallback CPU, avant import torch
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.cnn1d_rppg import CNN1D_rPPG
from models.chrom_adaptive import CHROMAdaptiveConditioned, compute_ita, bandpass_numpy
from mp_rppg.metrics import hr_from_fft, snr
from mp_rppg.methods import chrom, pos, chrom_adaptive
from mp_rppg.fusion import adaptive_fusion
from scripts.train_cnn1d import _temporal_norm, CLIP_LEN
from scripts.extract_regions_bisenet import (load_bisenet, extract_video, pick_device)
from scripts.preextract_clips import (load_video, track_face_bboxes, skin_neutralize,
                                      diff_normalize, RESIZE)
from models.physnet import PhysNet_padding_Encoder_Decoder_MAX
import cv2

FRONT, FULLSKIN = 0, 6
RGB_IDX = [0, 1, 2]


def znorm(s):
    return (s - s.mean()) / (s.std() + 1e-8)


def run_physnet(frames_rgb, fps, weights, dev, stride=64):
    """Crop visage (tracking) + DiffNormalized + PhysNet avec fenêtres glissantes
    À RECOUVREMENT (overlap-add). Chaque instant est prédit par plusieurs fenêtres
    (stride 64 = 50% de recouvrement sur des fenêtres de 128) ; on moyenne les
    prédictions, pondérées par une fenêtre de Hann, pour LISSER les coutures
    (vs l'ancien collage bout-à-bout). Approche standard du rPPG-Toolbox."""
    model = PhysNet_padding_Encoder_Decoder_MAX(frames=128).to(dev)
    model.load_state_dict(torch.load(weights, map_location=dev)); model.eval()
    bboxes = track_face_bboxes(frames_rgb)
    crop = np.stack([
        cv2.resize(skin_neutralize(f[y1:y2, x1:x2]), (RESIZE, RESIZE), interpolation=cv2.INTER_AREA)
        for f, (x1, y1, x2, y2) in zip(frames_rgb, bboxes)
    ]).astype(np.float32)
    dn = diff_normalize(crop)            # (T,72,72,3)
    T = len(dn)
    if T < 128:                          # vidéo très courte → on complète par bord
        dn = np.concatenate([dn, np.repeat(dn[-1:], 128 - T, 0)]); T2 = 128
    else:
        T2 = T
    # fenêtres : départs espacés de `stride`, + une fenêtre finale calée sur la fin
    starts = list(range(0, T2 - 128 + 1, stride))
    if not starts or starts[-1] != T2 - 128:
        starts.append(T2 - 128)
    win = np.hanning(128) + 1e-3         # pondération douce (centre > bords)
    out = np.zeros(T2); wsum = np.zeros(T2)
    for s in starts:
        x = torch.from_numpy(dn[s:s+128]).permute(3, 0, 1, 2).unsqueeze(0).to(dev)
        with torch.no_grad():
            p = model(x)[0].squeeze().cpu().numpy()
        p = (p - p.mean()) / (p.std() + 1e-8)    # même échelle avant moyennage
        out[s:s+128] += p * win
        wsum[s:s+128] += win
    sig = (out / np.maximum(wsum, 1e-8))[:T]
    return bandpass_numpy(sig, fps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--grid', type=int, default=4)
    ap.add_argument('--cnn', default=str(ROOT / 'weights' / 'cnn1d_rppg.pth'))
    ap.add_argument('--chrom', default=str(ROOT / 'weights' / 'chrom_conditioned_regions.pth'))
    ap.add_argument('--physnet', default=str(ROOT / 'weights' / 'clean_physnet_A_pure' / 'physnet_africa1_best.pth'))
    ap.add_argument('--max-dim', type=int, default=720,
                    help="Redimensionne les frames (plus grande dim <= max_dim) au "
                         "chargement, pour éviter l'OOM sur vidéo 4K. 0 = pleine résolution.")
    args = ap.parse_args()

    dev = pick_device()
    print(f"Device : {dev}")
    frames, fps = load_video(args.video, max_dim=(args.max_dim or None))
    h, w = frames.shape[1:3]
    print(f"Vidéo : {len(frames)} frames @ {fps:.1f} fps ({len(frames)/fps:.1f}s) — {w}x{h}"
          + (f" (réduit à max {args.max_dim}px)" if args.max_dim else ""))
    # Les modèles appris sont calibrés à 30 fps. Si la vidéo est plus rapide
    # (40/50/60 fps), on DÉCIME proprement vers 30 (1 image sur 2 à 60 fps) pour
    # matcher l'entraînement. On ne fait JAMAIS d'upsampling (24→30 dupliquerait
    # des images → casse PhysNet/DiffNormalized).
    if fps > 32.0:
        from scripts.preextract_clips import resample_to_fps
        ft = np.arange(len(frames)) * 1000.0 / fps
        frames, _ = resample_to_fps(frames, ft, 30.0)
        print(f"  → décimé à 30 fps ({len(frames)} frames) pour matcher l'entraînement")
        fps = 30.0

    print("Extraction BiSeNet multi-régions (peut prendre 1-2 min)...")
    net = load_bisenet(dev)
    x_reg, _, _ = extract_video(net, dev, frames, args.grid)   # (T, 23, 9)

    # signaux des 4 méthodes
    cnn = CNN1D_rPPG(in_channels=23 * 9).to(dev)
    cnn.load_state_dict(torch.load(args.cnn, map_location=dev)); cnn.eval()
    cmlp = CHROMAdaptiveConditioned()
    cmlp.load_state_dict(torch.load(args.chrom, map_location='cpu')['model_state_dict']); cmlp.eval()

    front = x_reg[:, FRONT, :][:, RGB_IDX].astype(np.float32)
    skin = x_reg[:, FULLSKIN, :][:, RGB_IDX].astype(np.float32)
    # ITA normalisé par la sclère (robuste à l'éclairage) ; repli sur ITA brut
    # (peau région front) si les yeux/sclère ne sont pas assez visibles.
    from mp_rppg.skin_ita import sclera_corrected_ita
    _ita = sclera_corrected_ita(frames, skin_rgb_fallback=front.mean(0))
    ita = _ita["ita"]

    xn = _temporal_norm(x_reg); T = xn.shape[1]
    preds = []
    for s in range(0, T - CLIP_LEN + 1, CLIP_LEN):
        xw = torch.from_numpy(xn[:, s:s + CLIP_LEN]).unsqueeze(0).to(dev)
        with torch.no_grad():
            preds.append(cnn(xw).squeeze().cpu().numpy())
    print("PhysNet fine-tuné (crop visage + DiffNormalized)...")
    physnet_sig = run_physnet(frames, fps, args.physnet, dev)

    sigs = {
        'CNN 1D':    bandpass_numpy(np.concatenate(preds), fps) if preds else None,
        'PhysNet':   physnet_sig,
        'CHROM-ITA': bandpass_numpy(chrom_adaptive(front, fps, cmlp.get_coefficients(ita)), fps),
        'CHROM':     bandpass_numpy(chrom(skin, fps), fps),
        'POS':       bandpass_numpy(pos(skin, fps), fps),
    }

    # CNN1D réentraîné sur données propres (2026-06-25, held-out MAE 4.69, corr
    # 0.65) → réintégré dans la fusion. L'ancien modèle (stale) collapsait.
    FUSION_EXCLUDE = set()

    print(f"\nITA estimé : {ita:.0f}  (>30 clair, <-30 foncé)  "
          f"[{_ita['used']}, brut {_ita['ita_raw']:.0f}, sclère {100*_ita['pct_sclera']:.0f}%]")
    print(f"\n{'Méthode':<12}{'HR (bpm)':>10}{'SNR aveugle':>13}")
    print('-' * 40)
    per_method = []
    for m, sig in sigs.items():
        if sig is None:
            continue
        hr = hr_from_fft(sig, fps); sn = snr(sig, hr, fps)
        flag = '  [exclu fusion]' if m in FUSION_EXCLUDE else ''
        if m not in FUSION_EXCLUDE:
            per_method.append((m, hr, sn))
        print(f"{m:<12}{hr:>10.1f}{sn:>13.2f}{flag}")

    # rBCG : micro-mouvements de la tête (mécanique) → INDÉPENDANT de la peau,
    # complémentaire de l'optique (utile sur peau foncée). Votant de plus.
    from mp_rppg.bcg import bcg_hr
    _bb = track_face_bboxes(frames)
    _bbox = tuple(np.median(np.array(_bb), axis=0).astype(int)) if len(_bb) else None
    bcg_h, bcg_s, _ = bcg_hr(frames, fps, bbox=_bbox)
    if np.isfinite(bcg_h):
        per_method.append(('rBCG', bcg_h, bcg_s))
        print(f"{'rBCG':<12}{bcg_h:>10.1f}{bcg_s:>13.2f}  [mouvement, indép. peau]")

    # ── Fusion ADAPTATIVE : médiane si accord, sélection-SNR si désaccord ──
    fz = adaptive_fusion(per_method)
    print('-' * 35)
    print(f"{'FUSION':<12}{fz['hr']:>10.1f}   [{fz['mode']} → {fz['chosen']}]")

    # ── SQA PAR FENÊTRES (style Binah) : juge chaque fenêtre, garde les bonnes ──
    from mp_rppg.sqa import windowed_sqa, combined_verdict
    extra = {'rBCG': (bcg_h, bcg_s)} if np.isfinite(bcg_h) else None
    sqa = windowed_sqa(sigs, fps, extra=extra, win_s=10.0, stride_s=2.0)
    seq = ''.join({'FIABLE': '✓', 'DOUTE': '·', 'REJET': '✗'}[w['status']] for w in sqa['windows'])
    cov = 100 * sqa['coverage']
    print(f"\nSQA par fenêtres (10s) : [{seq}]  ({sqa['n_fiable']}/{sqa['n_total']} fiables, "
          f"couverture {cov:.0f}%)")

    # ── AMBIGUÏTÉ : le signal primaire (meilleure méthode optique) a-t-il deux
    #    rythmes candidats quasi égaux ? Si oui → DOUTE (à confirmer). ──
    from mp_rppg.metrics import hr_candidates
    opt = [(m, h, s) for (m, h, s) in per_method if m in sigs and sigs[m] is not None]
    prim = max(opt, key=lambda z: z[2])[0] if opt else None
    cands, ambig = hr_candidates(sigs[prim], fps) if prim else ([], False)
    if ambig:
        print(f"\n⚠️  AMBIGUÏTÉ ({prim}) : {cands[0][0]:.0f} ou {cands[1][0]:.0f} bpm "
              f"(2e pic à {cands[1][1]*100:.0f}% du 1er)")

    # ── VERDICT COMBINÉ : accord + qualité SNR + témoin isolé (rBCG exclu) + SQA + ambiguïté ──
    v = combined_verdict(per_method, sqa['coverage'], ambiguous=ambig, candidates=cands)
    print(f"Accord des méthodes : écart-type = {v['std']:.1f} bpm | SNR médian = {v['med_snr']:+.1f} dB")
    print('=' * 56)
    if v['status'] == 'FIABLE':
        hr_final = sqa['hr'] if (sqa['n_fiable'] >= 2 and np.isfinite(sqa['hr'])) else v['hr']
        print(f"✅  FIABLE  →  {hr_final:.0f} bpm  (accord + signal OK + SQA {cov:.0f}%)")
    elif v['status'] == 'REJET':
        print(f"⚠️  MESURE NON FIABLE — méthodes dispersées ({v['std']:.0f} bpm) / signal faible "
              f"(SNR méd {v['med_snr']:+.1f}) / SQA {cov:.0f}%.")
        print("    NE PAS rapporter de valeur. Refaire (débit ≥40 Mbps, mouvement, éclairage, expo).")
    elif ambig:
        print(f"≈   À CONFIRMER  →  {cands[0][0]:.0f} ou {cands[1][0]:.0f} bpm  "
              f"(deux rythmes candidats se valent — ambiguïté spectrale)")
        print("    Refaire / prolonger la mesure pour lever l'ambiguïté.")
    else:
        print(f"≈   À CONFIRMER  →  {v['hr']:.0f} bpm  (signal limite ou SQA faible {cov:.0f}%)")
        print("    Refaire / prolonger la mesure pour confirmer.")
    print('=' * 56)


if __name__ == '__main__':
    main()
