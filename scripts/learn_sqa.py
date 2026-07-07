#!/usr/bin/env python3
"""
Apprend un SQA (Signal Quality Assessment) — façon Binah — qui prédit si une
fenêtre rPPG est FIABLE, à partir de features (accord inter-méthodes + SNR).

Remplace les seuils réglés à la main par une frontière APPRISE. Démontré : sur
le held-out propre (290 fenêtres) il donne plus de couverture à fiabilité égale
que les seuils manuels.

⚠️ FUITE : n'entraîner QUE sur des sujets jamais vus par PhysNet/CNN1D (held-out),
sinon le SNR des modèles profonds est gonflé → SQA mal calibré (démontré 2026-06).
Pour utiliser tous les sujets proprement : features out-of-fold (k-fold) ou
collecte smartphone avec held-out dès le départ.

Features par fenêtre : [std(HR), médiane(SNR), max(SNR), min(SNR), n_méthodes_d'accord].
Label : 1 si |HR_fusion - HR_vérité| < tol_bpm.

Sortie : weights/sqa_model.json (coeffs logistiques + normalisation) + report CV.

Usage : python scripts/learn_sqa.py --region Data/region_new --clips Data/clips_new
"""
import os, sys, re, json, argparse
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
from pathlib import Path
from collections import defaultdict
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
from scripts.preextract_clips import bandpass

FEAT_NAMES = ["std_hr", "med_snr", "max_snr", "min_snr", "n_agree"]


def build_features(region_dir, clips_dir, win_s=10.0, stride_s=2.0, tol=5.0):
    dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    cnn = CNN1D_rPPG(in_channels=23 * 9).to(dev)
    cnn.load_state_dict(torch.load(ROOT/'weights'/'cnn1d_rppg.pth', map_location=dev)); cnn.eval()
    cmlp = CHROMAdaptiveConditioned()
    cmlp.load_state_dict(torch.load(ROOT/'weights'/'chrom_conditioned_regions.pth', map_location='cpu')['model_state_dict']); cmlp.eval()
    pnet = PhysNet_padding_Encoder_Decoder_MAX(frames=128).to(dev)
    pnet.load_state_dict(torch.load(ROOT/'weights'/'clean_physnet_A_pure'/'physnet_africa1_best.pth', map_location=dev)); pnet.eval()
    cs = lambda p: int(re.search(r'clip_(\d+)', p.name).group(1))

    PW = {}
    for d in sorted(p for p in Path(clips_dir).iterdir() if p.is_dir()):
        by = defaultdict(list)
        for npz in d.glob('*.npz'):
            by[npz.name.split('_clip_')[0]].append(npz)
        for sc, files in sorted(by.items()):
            files = sorted(files, key=cs); preds = []
            for npz in files:
                dd = np.load(str(npz))
                x = torch.from_numpy(dd['x'].astype(np.float32)).permute(3, 0, 1, 2).unsqueeze(0).to(dev)
                with torch.no_grad():
                    preds.append(pnet(x)[0].squeeze().cpu().numpy())
            PW[(d.name, sc)] = np.concatenate(preds)

    X, y, grp, errs = [], [], [], []
    WIN, STR = int(win_s * 30), int(stride_s * 30)
    for d in sorted(p for p in Path(region_dir).iterdir() if p.is_dir()):
        for npz in sorted(d.glob('*.npz')):
            sc = npz.stem; dat = np.load(str(npz), allow_pickle=True); fps = float(dat['fps'])
            rgb = dat['x'][:, 0, :][:, :3].astype(np.float32)
            gt = bandpass_numpy(dat['y'].astype(np.float32), fps)
            xw = _temporal_norm(dat['x'], None); T = xw.shape[1]; preds = []
            for s0 in range(0, T - CLIP_LEN + 1, CLIP_LEN):
                with torch.no_grad():
                    preds.append(cnn(torch.from_numpy(xw[:, s0:s0+CLIP_LEN]).unsqueeze(0).to(dev)).squeeze().cpu().numpy())
            sig = {'CNN1D': bandpass_numpy(np.concatenate(preds), fps) if preds else None,
                   'CHROM': bandpass_numpy(chrom(rgb, fps), fps), 'POS': bandpass_numpy(pos(rgb, fps), fps),
                   'CHROM-ITA': bandpass_numpy(chrom_adaptive(rgb, fps, cmlp.get_coefficients(compute_ita(rgb.mean(0)))), fps)}
            pw = PW.get((d.name, sc))
            if pw is not None:
                sig['PhysNet'] = bandpass(pw, fps)
            L = min(len(s) for s in sig.values() if s is not None)
            for s0 in (list(range(0, L - WIN + 1, STR)) or [0]):
                pm = []
                for m, s in sig.items():
                    if s is None: continue
                    w = s[s0:s0+WIN] if L >= WIN else s
                    h = hr_from_fft(w, fps); pm.append((m, h, snr(w, h, fps)))
                hrs = np.array([h for _, h, _ in pm]); snrs = np.array([s for _, _, s in pm])
                fz = adaptive_fusion(pm); med = np.median(hrs)
                X.append([np.std(hrs), np.median(snrs), snrs.max(), snrs.min(), int(np.sum(np.abs(hrs - med) <= 5))])
                hg = hr_from_fft(gt[s0:s0+WIN] if L >= WIN else gt, fps)
                e = abs(fz['hr'] - hg); errs.append(e); y.append(int(e < tol)); grp.append(d.name)
    return np.array(X), np.array(y), np.array(grp), np.array(errs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--region', default=str(ROOT/'Data'/'region_new'))
    ap.add_argument('--clips', default=str(ROOT/'Data'/'clips_new'))
    ap.add_argument('--out', default=str(ROOT/'weights'/'sqa_model.json'))
    args = ap.parse_args()
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GroupKFold

    X, y, grp, errs = build_features(args.region, args.clips)
    print(f"Fenêtres : {len(y)}  (sujets {len(set(grp))})  fiables réels {y.sum()}/{len(y)}")

    pred = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, grp):
        scl = StandardScaler().fit(X[tr])
        clf = LogisticRegression(class_weight='balanced', max_iter=1000).fit(scl.transform(X[tr]), y[tr])
        pred[te] = clf.predict(scl.transform(X[te]))
    sel = pred == 1
    print(f"SQA appris (GroupKFold) : couverture {100*sel.mean():.0f}%  "
          f"MAE acceptés {errs[sel].mean():.2f}  %vrai<5 {100*y[sel].mean():.0f}%")

    # modèle final sur tout + sauvegarde JSON (scaler + coeffs)
    scl = StandardScaler().fit(X); clf = LogisticRegression(class_weight='balanced', max_iter=1000).fit(scl.transform(X), y)
    json.dump({"features": FEAT_NAMES, "mean": scl.mean_.tolist(), "scale": scl.scale_.tolist(),
               "coef": clf.coef_[0].tolist(), "intercept": float(clf.intercept_[0]),
               "note": "predict: sigmoid((x-mean)/scale . coef + intercept) > 0.5 = FIABLE"},
              open(args.out, 'w'), indent=2)
    print(f"Modèle SQA → {args.out}")


if __name__ == '__main__':
    main()
