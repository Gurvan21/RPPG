#!/usr/bin/env python3
"""
Évaluation HELD-OUT (sujets jamais vus à l'entraînement), au niveau SCÉNARIO :
5 méthodes (CNN1D, CHROM, POS, CHROM-ITA, PhysNet) + FUSION ADAPTATIVE
(mp_rppg.fusion), avec politique de confiance SNR :
    SNR > -1 → VALIDÉ   |   -3..-1 → douteux   |   SNR < -3 → REJETÉ

Méthodes régions : Data/region_new   |   PhysNet : Data/clips_new
Les deux sont fusionnés par (sujet, scénario).

Usage :
    python scripts/test_heldout_all.py
"""
import argparse, json, sys, os, re
from pathlib import Path
from collections import defaultdict
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.cnn1d_rppg import CNN1D_rPPG
from models.chrom_adaptive import CHROMAdaptiveConditioned, compute_ita, bandpass_numpy
from models.physnet import PhysNet_padding_Encoder_Decoder_MAX
from mp_rppg.metrics import hr_from_fft, snr
from mp_rppg.methods import chrom, pos, chrom_adaptive
from mp_rppg.fusion import adaptive_fusion
from scripts.train_cnn1d import _temporal_norm, CLIP_LEN
from scripts.preextract_clips import bandpass as bp_clip

REGION_FRONT, RGB_IDX = 0, [0, 1, 2]
METHODS = ['CNN1D', 'CHROM', 'POS', 'CHROM-ITA', 'PhysNet']


def fitz_of(name):
    js = [j for j in (ROOT / 'DataVital' / name).glob('*.json') if j.name != 'metadata.json']
    try:
        return json.load(open(js[0]))['participant']['fitzpatrick'] if js else '?'
    except Exception:
        return '?'


def cat(s):
    return 'REJET' if s < -3 else ('douteux' if s < -1 else 'VALIDÉ')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--region', default=str(ROOT / 'Data' / 'region_new'))
    ap.add_argument('--clips',  default=str(ROOT / 'Data' / 'clips_new'))
    ap.add_argument('--cnn',    default=str(ROOT / 'weights' / 'cnn1d_rppg.pth'))
    ap.add_argument('--chrom-mlp', default=str(ROOT / 'weights' / 'chrom_conditioned_regions.pth'))
    ap.add_argument('--physnet', default=str(ROOT / 'weights' / 'clean_physnet_A_pure' / 'physnet_africa1_best.pth'))
    ap.add_argument('--fuse-methods', default=','.join(METHODS),
                    help="méthodes incluses dans la fusion (séparées par ,)")
    ap.add_argument('--cache', default=str(ROOT / 'Data' / 'heldout_cache.json'),
                    help="dump (sujet,sc,gt,{méthode:(hr,snr)}) pour recalibrer sans relancer")
    args = ap.parse_args()
    fuse_methods = [m.strip() for m in args.fuse_methods.split(',') if m.strip()]

    dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    cnn = CNN1D_rPPG(in_channels=23 * 9).to(dev)
    cnn.load_state_dict(torch.load(args.cnn, map_location=dev)); cnn.eval()
    cmlp = CHROMAdaptiveConditioned()
    cmlp.load_state_dict(torch.load(args.chrom_mlp, map_location='cpu')['model_state_dict']); cmlp.eval()
    pnet = PhysNet_padding_Encoder_Decoder_MAX(frames=128).to(dev)
    pnet.load_state_dict(torch.load(args.physnet, map_location=dev)); pnet.eval()

    # store[(subject, sc)] = {'fz':, 'gt_hr':, 'CNN1D':(hr,snr), ...}
    store = defaultdict(dict)

    # ── 1) Méthodes régions ──
    rdir = Path(args.region)
    for d in sorted([d for d in rdir.iterdir() if d.is_dir()]) if rdir.exists() else []:
        fz = fitz_of(d.name)
        for npz in sorted(d.glob('*.npz')):
            sc = npz.stem  # "sc0"
            dat = np.load(str(npz), allow_pickle=True); fps = float(dat['fps'])
            gt = bandpass_numpy(dat['y'].astype(np.float32), fps); hg = hr_from_fft(gt, fps)
            rgb = dat['x'][:, REGION_FRONT, :][:, RGB_IDX].astype(np.float32)
            key = (d.name, sc); store[key]['fz'] = fz; store[key]['gt_hr'] = hg
            sigs = {}
            x = _temporal_norm(dat['x'], None); T = x.shape[1]; preds = []
            for s in range(0, T - CLIP_LEN + 1, CLIP_LEN):
                xw = torch.from_numpy(x[:, s:s + CLIP_LEN]).unsqueeze(0).to(dev)
                with torch.no_grad():
                    preds.append(cnn(xw).squeeze().cpu().numpy())
            if preds:
                sigs['CNN1D'] = bandpass_numpy(np.concatenate(preds), fps)
            sigs['CHROM'] = bandpass_numpy(chrom(rgb, fps), fps)
            sigs['POS'] = bandpass_numpy(pos(rgb, fps), fps)
            ita = compute_ita(rgb.mean(axis=0)); coe = cmlp.get_coefficients(ita)
            sigs['CHROM-ITA'] = bandpass_numpy(chrom_adaptive(rgb, fps, coe), fps)
            for m, sig in sigs.items():
                h = hr_from_fft(sig, fps); store[key][m] = (h, snr(sig, h, fps))

    # ── 2) PhysNet (clips) ──
    cdir = Path(args.clips)
    cs = lambda p: int(re.search(r'clip_(\d+)', p.name).group(1))
    for d in sorted([d for d in cdir.iterdir() if d.is_dir()]) if cdir.exists() else []:
        fz = fitz_of(d.name)
        by = defaultdict(list)
        for npz in d.glob('*.npz'):
            by[npz.name.split('_clip_')[0]].append(npz)
        for sc, files in sorted(by.items()):
            files = sorted(files, key=cs); preds, gts, fps = [], [], None
            for npz in files:
                dd = np.load(str(npz)); fps = float(dd['fps'])
                x = torch.from_numpy(dd['x'].astype(np.float32)).permute(3, 0, 1, 2).unsqueeze(0).to(dev)
                with torch.no_grad():
                    preds.append(pnet(x)[0].squeeze().cpu().numpy())
                gts.append(dd['y'].astype(np.float32))
            fp = bp_clip(np.concatenate(preds), fps); fg = bp_clip(np.concatenate(gts), fps)
            hg = hr_from_fft(fg, fps); hp = hr_from_fft(fp, fps)
            key = (d.name, sc)
            store[key].setdefault('fz', fz); store[key].setdefault('gt_hr', hg)
            store[key]['PhysNet'] = (hp, snr(fp, hp, fps))

    # cache disque (pour recalibrer τ / sous-ensembles sans relancer l'inférence)
    cache = {f"{s}||{sc}": {'fz': v.get('fz', '?'), 'gt': v.get('gt_hr'),
                            **{m: list(v[m]) for m in METHODS if m in v}}
             for (s, sc), v in store.items()}
    Path(args.cache).write_text(json.dumps(cache, ensure_ascii=False, indent=0))
    print(f"[cache → {args.cache}]  fusion sur : {fuse_methods}\n")

    # ── 3) Fusion adaptative + agrégation ──
    per_err = {m: [] for m in METHODS}          # (err, snr)
    fus_err, fus_modes = [], []
    print(f"{'Sujet':<11}{'Fz':>3}{'sc':>4}{'GT':>6}", end='')
    for m in METHODS:
        print(f"{m[:4]:>7}", end='')
    print(f"{'FUSION':>9}{'err':>6}  mode")
    print('-' * 95)
    for (subj, sc), v in sorted(store.items()):
        if 'gt_hr' not in v:
            continue
        hg = v['gt_hr']
        row = f"{subj:<11}{str(v.get('fz','?')):>3}{sc.replace('sc',''):>4}{hg:>6.1f}"
        per_method = []
        for m in METHODS:
            if m in v:
                h, sn = v[m]
                if m in fuse_methods:
                    per_method.append((m, h, sn))
                per_err[m].append((abs(h - hg), sn)); row += f"{h:>7.0f}"
            else:
                row += f"{'--':>7}"
        fz = adaptive_fusion(per_method)
        fe = abs(fz['hr'] - hg); fus_err.append(fe); fus_modes.append(fz['mode'])
        row += f"{fz['hr']:>9.1f}{fe:>6.1f}  {fz['mode'][:4]}→{fz['chosen']}"
        print(row)

    # ── Synthèse ──
    print("\n" + "=" * 78)
    print(f"{'Méthode':<14}{'N':>4}{'MAE':>8}{'méd':>7}{'%<5':>6}"
          f"{'Validés(SNR>-1)':>18}{'MAEval':>9}")
    print('-' * 78)
    for m in METHODS:
        lst = per_err[m]
        if not lst:
            continue
        a = np.array([e for e, _ in lst]); sn = np.array([s for _, s in lst]); val = sn > -1
        mev = a[val].mean() if val.any() else float('nan')
        print(f"{m:<14}{len(a):>4}{a.mean():>8.2f}{np.median(a):>7.2f}{100*(a<5).mean():>5.0f}%"
              f"{int(val.sum()):>9}/{len(a):<7}{mev:>9.2f}")
    fa = np.array(fus_err)
    nc = sum(1 for x in fus_modes if x == 'consensus'); nsel = sum(1 for x in fus_modes if x == 'selection')
    print('-' * 78)
    print(f"{'FUSION adapt.':<14}{len(fa):>4}{fa.mean():>8.2f}{np.median(fa):>7.2f}{100*(fa<5).mean():>5.0f}%"
          f"   [consensus×{nc}, sélection×{nsel}]")


if __name__ == '__main__':
    main()
