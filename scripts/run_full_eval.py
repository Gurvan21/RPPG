"""
Évaluation complète sur UBFC-rPPG (42 sujets).
Méthodes comparées :
  CHROM/POS  × backends HC, MP-front, MP-mean
  PhysNet    × poids SCAMPS, UBFC

Usage :
  python scripts/run_full_eval.py --data /path/to/UBFC-rPPG/Data --out results/eval_ubfc
  python scripts/run_full_eval.py --data /path/to/UBFC-rPPG/Data --out results/eval_ubfc --subjects 5
"""

import argparse, glob, os, sys, time
import numpy as np
import cv2
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mp_rppg.pipeline import extract_rgb as mp_extract
from mp_rppg.methods  import chrom, pos
from mp_rppg.metrics  import hr_from_fft, snr, aggregate
from models.physnet   import PhysNet_padding_Encoder_Decoder_MAX

HAAR_XML   = os.path.join(ROOT, "assets/haarcascade_frontalface_default.xml")
RESIZE     = 72
BOX_COEF   = 1.5
CHUNK_LEN  = 128
STRIDE     = 64
DEVICE     = torch.device('cpu')


# ── I/O ────────────────────────────────────────────────────────────────────────
def read_video(path):
    import imageio.v3 as iio
    meta   = iio.immeta(path, plugin='pyav')
    fps    = float(meta.get('fps', 30.0))
    frames = list(iio.imiter(path, plugin='pyav'))
    return np.asarray(frames, dtype=np.uint8), fps


def read_gt(path):
    """UBFC GT : 3 lignes → ligne0=BVP, ligne1=HR/frame, ligne2=timestamps."""
    with open(path) as f:
        lines = f.readlines()
    bvp = np.array([float(x) for x in lines[0].strip().split()])
    hr_vals = np.array([float(x) for x in lines[1].strip().split()])
    return bvp, float(np.mean(hr_vals))


# ── Crop HC ────────────────────────────────────────────────────────────────────
def crop_hc(frames):
    detector = cv2.CascadeClassifier(HAAR_XML)
    bbox = None
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        zones = detector.detectMultiScale(gray, 1.1, 5)
        if len(zones):
            x, y, w, h = zones[np.argmax(zones[:, 2])]
            cx, cy = x+w//2, y+h//2
            hw, hh = int(w*BOX_COEF/2), int(h*BOX_COEF/2)
            H, W = frame.shape[:2]
            bbox = (max(0,cx-hw), max(0,cy-hh), min(W,cx+hw), min(H,cy+hh))
            break
    if bbox is None:
        H, W = frames[0].shape[:2]
        bbox = (0, 0, W, H)
    x1,y1,x2,y2 = bbox
    crops = []
    for f in frames:
        crops.append(cv2.resize(f[y1:y2,x1:x2], (RESIZE,RESIZE),
                                interpolation=cv2.INTER_AREA).astype(np.float32))
    return np.asarray(crops)


# ── DiffNormalized ─────────────────────────────────────────────────────────────
def diff_normalize(data):
    T = len(data)
    out = np.zeros_like(data, dtype=np.float32)
    for t in range(T-1):
        out[t] = (data[t+1]-data[t]) / (data[t+1]+data[t]+1e-7)
    std = np.std(out[:-1])
    if std > 0:
        out /= std
    out[np.isnan(out)] = 0
    return out


# ── Inférence PhysNet ──────────────────────────────────────────────────────────
def run_physnet(frames_norm, model):
    T = len(frames_norm)
    bvp_sum = np.zeros(T)
    weight  = np.zeros(T)
    hann    = np.hanning(CHUNK_LEN)
    starts  = list(range(0, max(1, T-CHUNK_LEN+1), STRIDE))
    model.eval()
    with torch.no_grad():
        for s in starts:
            chunk = frames_norm[s:s+CHUNK_LEN]
            if len(chunk) < CHUNK_LEN:
                pad = np.zeros((CHUNK_LEN-len(chunk),)+chunk.shape[1:], np.float32)
                chunk = np.concatenate([chunk, pad])
            x = torch.from_numpy(chunk).permute(3,0,1,2).unsqueeze(0)
            pred = model(x)[0].squeeze().cpu().numpy()
            end = min(s+CHUNK_LEN, T)
            ln  = end-s
            bvp_sum[s:end] += pred[:ln]*hann[:ln]
            weight[s:end]  += hann[:ln]
    mask = weight > 0
    bvp_sum[mask] /= weight[mask]
    return bvp_sum


# ── Chargement modèles PhysNet ─────────────────────────────────────────────────
def load_physnet(path):
    model = PhysNet_padding_Encoder_Decoder_MAX(frames=CHUNK_LEN).to(DEVICE)
    ckpt  = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt)
    model.eval()
    return model


# ── Évaluation d'un sujet ──────────────────────────────────────────────────────
def eval_subject(vid_path, gt_path, models_physnet, verbose=False):
    frames, fps = read_video(vid_path)
    gt_bvp, hr_gt = read_gt(gt_path)
    # HR GT par FFT sur signal BVP (plus robuste que la moyenne directe)
    hr_gt_fft = hr_from_fft(gt_bvp, fps)

    results = {}

    # ── HC + CHROM/POS ────────────────────────────────────────────────────────
    crops = crop_hc(frames)
    rgb_hc = np.array([c.mean(axis=(0,1)) for c in crops])
    for method_name, method_fn in [('CHROM', chrom), ('POS', pos)]:
        bvp = method_fn(rgb_hc, fps)
        hr  = hr_from_fft(bvp, fps)
        results[f'HC/{method_name}'] = {
            'hr': hr, 'err': abs(hr-hr_gt_fft),
            'snr': snr(bvp, hr_gt_fft, fps)
        }

    # ── MediaPipe + CHROM/POS ─────────────────────────────────────────────────
    mp_rgb = mp_extract(frames, verbose=False)
    for region in ('front', 'mean'):
        for method_name, method_fn in [('CHROM', chrom), ('POS', pos)]:
            bvp = method_fn(mp_rgb[region], fps)
            hr  = hr_from_fft(bvp, fps)
            results[f'MP-{region}/{method_name}'] = {
                'hr': hr, 'err': abs(hr-hr_gt_fft),
                'snr': snr(bvp, hr_gt_fft, fps)
            }

    # ── PhysNet ───────────────────────────────────────────────────────────────
    crops_f32 = crop_hc(frames)
    frames_norm = diff_normalize(crops_f32)
    for w_name, model in models_physnet.items():
        bvp = run_physnet(frames_norm, model)
        hr  = hr_from_fft(bvp, fps)
        results[f'PhysNet-{w_name}'] = {
            'hr': hr, 'err': abs(hr-hr_gt_fft),
            'snr': snr(bvp, hr_gt_fft, fps)
        }

    if verbose:
        for k, v in results.items():
            print(f"    {k:25s}  HR={v['hr']:5.1f}  err={v['err']:5.1f}  SNR={v['snr']:5.1f} dB")

    return results, hr_gt_fft


# ── Résumé agrégé ──────────────────────────────────────────────────────────────
def print_summary(all_results, gt_hrs):
    labels = list(all_results[0].keys())
    print("\n" + "═"*72)
    print("  RÉSULTATS UBFC-rPPG — métriques agrégées")
    print("═"*72)
    print(f"  {'Méthode':<28} {'MAE':>6} {'RMSE':>6} {'SNR':>8}  {'<5bpm%':>7}")
    print("  " + "─"*60)
    rows = []
    for label in labels:
        errs = [r[label]['err'] for r in all_results]
        snrs = [r[label]['snr'] for r in all_results]
        mae  = np.mean(errs)
        rmse = np.sqrt(np.mean(np.array(errs)**2))
        snr_m = np.mean(snrs)
        pct5  = 100*np.mean(np.array(errs) < 5)
        rows.append((mae, label, rmse, snr_m, pct5))
        print(f"  {label:<28} {mae:>6.2f} {rmse:>6.2f} {snr_m:>7.2f}  {pct5:>6.0f}%")
    print("═"*72)
    print(f"  Sujets : {len(all_results)}   |   HR GT moy : {np.mean(gt_hrs):.1f} bpm")
    best = sorted(rows)[0]
    print(f"\n  Meilleure MAE : {best[1]}  →  {best[0]:.2f} bpm")


# ── Graphiques ─────────────────────────────────────────────────────────────────
def save_plots(all_results, gt_hrs, out_dir):
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    labels = list(all_results[0].keys())

    # Palette : rouge=HC, bleu=MP, vert=PhysNet
    def color(l):
        if l.startswith('HC'):       return '#E64B35'
        if l.startswith('MP'):       return '#3182BD'
        return '#2CA02C'

    # ── Figure 1 : MAE bar ────────────────────────────────────────────────────
    maes  = [np.mean([r[l]['err']  for r in all_results]) for l in labels]
    rmses = [np.sqrt(np.mean(np.array([r[l]['err'] for r in all_results])**2)) for l in labels]
    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(max(10, len(labels)*1.1), 5))
    b1 = ax.bar(x-w/2, maes,  w, label='MAE',
                color=[color(l) for l in labels], alpha=0.85)
    b2 = ax.bar(x+w/2, rmses, w, label='RMSE',
                color=[color(l) for l in labels], alpha=0.45, hatch='//')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=8)
    ax.set_ylabel('Erreur HR (bpm)')
    ax.set_title(f'MAE / RMSE sur UBFC-rPPG ({len(all_results)} sujets)\n'
                 f'Rouge=HC  Bleu=MediaPipe  Vert=PhysNet')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    for bar in b1:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'mae_comparison.png'), dpi=150)
    plt.close()

    # ── Figure 2 : scatter HR estimé vs GT ───────────────────────────────────
    n_cols = min(4, len(labels))
    n_rows = (len(labels)+n_cols-1)//n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.5*n_cols, 4*n_rows), squeeze=False)
    for idx, label in enumerate(labels):
        ax = axes[idx//n_cols][idx%n_cols]
        hr_pred = [r[label]['hr'] for r in all_results]
        lo = min(min(hr_pred), min(gt_hrs))-5
        hi = max(max(hr_pred), max(gt_hrs))+5
        ax.scatter(gt_hrs, hr_pred, s=25, alpha=0.7, color=color(label))
        ax.plot([lo,hi],[lo,hi],'r--',lw=1)
        mae = np.mean([r[label]['err'] for r in all_results])
        ax.set_title(f'{label}\nMAE={mae:.1f} bpm', fontsize=8)
        ax.set_xlabel('HR GT (bpm)', fontsize=7)
        ax.set_ylabel('HR estimé (bpm)', fontsize=7)
        ax.grid(alpha=0.25)
    for idx in range(len(labels), n_rows*n_cols):
        axes[idx//n_cols][idx%n_cols].set_visible(False)
    plt.suptitle(f'HR estimé vs GT — UBFC-rPPG ({len(all_results)} sujets)',
                 fontsize=10, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'scatter_hr.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ── Figure 3 : SNR ────────────────────────────────────────────────────────
    snrs = [np.mean([r[l]['snr'] for r in all_results]) for l in labels]
    fig, ax = plt.subplots(figsize=(max(10, len(labels)*1.1), 4))
    cols = ['#2ca02c' if s > 0 else '#d62728' for s in snrs]
    ax.bar(labels, snrs, color=cols, alpha=0.85)
    ax.axhline(0, color='black', lw=0.8, linestyle='--')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=8)
    ax.set_ylabel('SNR moyen (dB)')
    ax.set_title('SNR moyen par méthode  (>0 dB = signal > bruit)')
    ax.grid(axis='y', alpha=0.3)
    for i, (v, bar_x) in enumerate(zip(snrs, range(len(snrs)))):
        ax.text(bar_x, v+(0.1 if v>=0 else -0.5), f'{v:.1f}',
                ha='center', va='bottom', fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'snr_comparison.png'), dpi=150)
    plt.close()

    # ── Figure 4 : box plot erreurs ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(10, len(labels)*1.1), 5))
    data = [[r[l]['err'] for r in all_results] for l in labels]
    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops=dict(color='black', lw=1.5))
    for patch, l in zip(bp['boxes'], labels):
        patch.set_facecolor(color(l))
        patch.set_alpha(0.7)
    ax.set_xticks(range(1, len(labels)+1))
    ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=8)
    ax.set_ylabel('Erreur absolue HR (bpm)')
    ax.set_title('Distribution des erreurs — UBFC-rPPG')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'boxplot_errors.png'), dpi=150)
    plt.close()

    print(f"\n  Graphiques sauvegardés dans {out_dir}/")
    for f in ['mae_comparison.png','scatter_hr.png','snr_comparison.png','boxplot_errors.png']:
        print(f"    {f}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',     required=True,
                        help='Dossier UBFC-rPPG contenant les dossiers subject*/')
    parser.add_argument('--out',      default=os.path.join(ROOT, 'results/eval_ubfc'))
    parser.add_argument('--subjects', type=int, default=None,
                        help='Limiter à N sujets (test rapide)')
    parser.add_argument('--no-mp',    action='store_true',
                        help='Sauter MediaPipe (plus rapide)')
    args = parser.parse_args()

    subject_dirs = sorted(glob.glob(os.path.join(args.data, 'subject*')))
    if not subject_dirs:
        sys.exit(f"Aucun sujet trouvé dans {args.data}")
    if args.subjects:
        subject_dirs = subject_dirs[:args.subjects]

    # Chargement des modèles PhysNet
    print("Chargement des modèles PhysNet...")
    physnet_weights = {
        'SCAMPS': os.path.join(ROOT, 'weights/SCAMPS_PhysNet_DiffNormalized.pth'),
        'UBFC':   os.path.join(ROOT, 'weights/UBFC-rPPG_PhysNet_DiffNormalized.pth'),
    }
    models_physnet = {}
    for name, path in physnet_weights.items():
        models_physnet[name] = load_physnet(path)
        print(f"  PhysNet-{name} chargé")

    print(f"\nDataset : {args.data}  ({len(subject_dirs)} sujets)\n")

    all_results = []
    gt_hrs = []
    t0 = time.time()

    for i, sdir in enumerate(subject_dirs):
        vid_path = os.path.join(sdir, 'vid.avi')
        gt_path  = os.path.join(sdir, 'ground_truth.txt')
        if not (os.path.exists(vid_path) and os.path.exists(gt_path)):
            continue

        elapsed = time.time() - t0
        eta = elapsed/(i+0.01) * (len(subject_dirs)-i) if i > 0 else 0
        print(f"[{i+1:2d}/{len(subject_dirs)}] {os.path.basename(sdir)}  "
              f"(écoulé={elapsed:.0f}s  ETA={eta:.0f}s)")

        try:
            r, hr_gt = eval_subject(vid_path, gt_path, models_physnet, verbose=True)
            all_results.append(r)
            gt_hrs.append(hr_gt)
        except Exception as e:
            print(f"  ERREUR : {e}")

    print_summary(all_results, gt_hrs)
    save_plots(all_results, gt_hrs, args.out)


if __name__ == '__main__':
    main()
