"""
Entraîne et évalue CHROM adaptatif sur les sujets DataVital (format VitalVideos
natif : <GUID>.json + une vidéo par scénario), au lieu d'UBFC.

Mêmes étapes que eval_chrom_adaptive.py :
  1. Extraction RGB (front, MediaPipe) pour chaque scénario, mise en cache (.npz)
  2. Split train/test au niveau SUJET (pas scénario, pour éviter une fuite
     du même sujet entre train et test)
  3. Entraînement de CHROMAdaptatif (Pearson loss vs PPG de référence)
  4. Comparaison CHROM De Haan vs CHROM adaptatif sur le set de test

Usage :
    python scripts/eval_chrom_adaptive_datavital.py --data DataVital --epochs 100
"""

import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, ROOT)

from models.chrom_adaptive import (CHROMAdaptatif, DEHAAN_COEFFICIENTS,
                                    bandpass_numpy, bandpass_straight_through,
                                    loss_pearson)
from mp_rppg.metrics import hr_from_fft, snr
from mp_rppg.methods import chrom, chrom_adaptive
from mp_rppg.pipeline import extract_rgb
from scripts.preextract_clips import (
    _find_vitalvideos_json, _resample_ppg_to_frames, load_video,
)


def extract_features(data_dir, cache_dir, region):
    """Retourne {subject_name: [(sample_id, rgb, fps, bvp), ...]} — un sujet
    peut avoir plusieurs scénarios, regroupés sous la même clé sujet."""
    os.makedirs(cache_dir, exist_ok=True)
    by_subject = {}

    # "subject " (avec espace) uniquement -- exclut les anciens dossiers sans
    # espace (Subject2, Subject3, Subject4) qui sont des doublons des mêmes
    # personnes que Subject 2/6/7 (même GUID), pour éviter une fuite train/test.
    subject_dirs = sorted(d for d in os.listdir(data_dir)
                          if os.path.isdir(os.path.join(data_dir, d))
                          and d.lower().startswith('subject '))

    for subject in subject_dirs:
        subject_path = os.path.join(data_dir, subject)
        meta = _find_vitalvideos_json(__import__('pathlib').Path(subject_path))
        if meta is None:
            continue

        samples = []
        for sc_idx, scenario in enumerate(meta.get('scenarios', [])):
            rec = scenario.get('recordings', {})
            rgb_meta, cms = rec.get('RGB'), rec.get('CMS')
            if not rgb_meta or not cms or len(cms) < 2:
                continue

            sample_id = f"{subject}_sc{sc_idx}"
            cache_path = os.path.join(cache_dir, f"{sample_id}.npz")

            if os.path.exists(cache_path):
                npz = np.load(cache_path)
                samples.append((sample_id, npz['rgb'], float(npz['fps']), npz['bvp']))
                continue

            video_path = os.path.join(subject_path, rgb_meta['filename'])
            if not os.path.exists(video_path):
                print(f"  [SKIP] {sample_id} : vidéo introuvable")
                continue

            print(f"  {sample_id} : extraction...")
            frames, _ = load_video(video_path)
            frame_times_ms = np.array([t for t, _ in rgb_meta['timeseries']], dtype=np.float64)
            n = min(len(frames), len(frame_times_ms))
            frames = frames[:n]
            fps = float(rgb_meta.get('device', {}).get('FrameRate', 30))

            bvp = _resample_ppg_to_frames(cms[1:], frame_times_ms[:n])
            rgb_regions = extract_rgb(frames, verbose=False)
            rgb = rgb_regions[region]

            np.savez(cache_path, rgb=rgb, fps=fps, bvp=bvp)
            samples.append((sample_id, rgb, fps, bvp))

        if samples:
            by_subject[subject] = samples

    n_samples = sum(len(v) for v in by_subject.values())
    print(f"{len(by_subject)} sujets, {n_samples} scénarios extraits")
    return by_subject


def _flatten(by_subject, subject_names):
    flat = {}
    for s in subject_names:
        for sample_id, rgb, fps, bvp in by_subject[s]:
            flat[sample_id] = (rgb, fps, bvp)
    return flat


def train_chrom_adaptive(features, epochs, lr):
    model = CHROMAdaptatif()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    print(f"\nCoefficients initiaux (De Haan) : {model.get_coefficients()}\n")

    for epoch in range(epochs):
        total_loss, n_valid = 0.0, 0
        for name, (rgb, fps, bvp) in features.items():
            R = torch.tensor((rgb[:, 0] / (rgb[:, 0].mean() + 1e-8)).astype(np.float32))
            G = torch.tensor((rgb[:, 1] / (rgb[:, 1].mean() + 1e-8)).astype(np.float32))
            B = torch.tensor((rgb[:, 2] / (rgb[:, 2].mean() + 1e-8)).astype(np.float32))
            label = torch.tensor(bandpass_numpy(bvp, fps).astype(np.float32))

            sig = model(R, G, B)
            sig_t = bandpass_straight_through(sig, fps)

            loss = loss_pearson(sig_t, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_valid += 1

        scheduler.step()
        if epoch % 10 == 0:
            c = model.get_coefficients()
            print(f"Epoch {epoch:3d} | Loss: {total_loss / n_valid:.4f} | "
                  f"[{c['a1']:.3f}, {c['a2']:.3f}, {c['a3']:.3f}, {c['a4']:.3f}, {c['a5']:.3f}]")

    return model


def evaluate(features, names, coeffs, label):
    results = {}
    for name in names:
        rgb, fps, bvp = features[name]
        bvp_pred = chrom(rgb, fps) if label == 'dehaan' else chrom_adaptive(rgb, fps, coeffs)

        hr_gt = hr_from_fft(bvp, fps)
        hr_pred = hr_from_fft(bvp_pred, fps)
        results[name] = {
            'hr_pred': hr_pred,
            'hr_gt': hr_gt,
            'err': abs(hr_pred - hr_gt),
            'snr': snr(bvp_pred, hr_gt, fps),
        }
    return results


def main():
    parser = argparse.ArgumentParser(description="CHROM adaptatif sur DataVital")
    parser.add_argument('--data', default=os.path.join(ROOT, 'DataVital'))
    parser.add_argument('--region', default='front', choices=['front', 'left', 'right', 'mean'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--test-frac', type=float, default=0.2)
    parser.add_argument('--cache', default=os.path.join(ROOT, 'results', 'chrom_adaptive_datavital_cache'))
    parser.add_argument('--out', default=os.path.join(ROOT, 'results', 'chrom_adaptive_datavital'))
    parser.add_argument('--save', default=os.path.join(ROOT, 'weights', 'chrom_adaptive_datavital.pth'))
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    by_subject = extract_features(args.data, args.cache, args.region)
    subjects = sorted(by_subject.keys())

    rng = np.random.RandomState(args.seed)
    rng.shuffle(subjects)
    n_test = max(1, int(len(subjects) * args.test_frac))
    test_subjects = sorted(subjects[:n_test])
    train_subjects = sorted(subjects[n_test:])

    print(f"\nSujets train ({len(train_subjects)}) : {train_subjects}")
    print(f"Sujets test  ({len(test_subjects)}) : {test_subjects}")

    train_features = _flatten(by_subject, train_subjects)
    test_features  = _flatten(by_subject, test_subjects)
    print(f"Scénarios train : {len(train_features)}  |  test : {len(test_features)}")

    model = train_chrom_adaptive(train_features, args.epochs, args.lr)
    coeffs = model.get_coefficients()

    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    torch.save({'model_state_dict': model.state_dict(),
                'coefficients': coeffs, 'fps': None}, args.save)

    print("\n" + "═" * 60)
    print("  Coefficients appris vs De Haan")
    print("═" * 60)
    for k in ('a1', 'a2', 'a3', 'a4', 'a5'):
        print(f"  {k} : {coeffs[k]:7.4f}   (De Haan = {DEHAAN_COEFFICIENTS[k]})")

    test_names = sorted(test_features.keys())
    res_dehaan = evaluate(test_features, test_names, None, 'dehaan')
    res_adapt  = evaluate(test_features, test_names, coeffs, 'adaptive')

    print("\n" + "═" * 60)
    print("  Résultats sur le set de test (par scénario)")
    print("═" * 60)
    print(f"  {'Scénario':<16} {'GT':>6} {'DeHaan':>8} {'err':>6} {'SNR':>7} | "
          f"{'Adapt':>8} {'err':>6} {'SNR':>7}")
    for name in test_names:
        d, a = res_dehaan[name], res_adapt[name]
        print(f"  {name:<16} {d['hr_gt']:6.1f} {d['hr_pred']:8.1f} {d['err']:6.1f} {d['snr']:7.2f} | "
              f"{a['hr_pred']:8.1f} {a['err']:6.1f} {a['snr']:7.2f}")

    mae_dehaan = np.mean([r['err'] for r in res_dehaan.values()])
    mae_adapt  = np.mean([r['err'] for r in res_adapt.values()])
    snr_dehaan = np.mean([r['snr'] for r in res_dehaan.values()])
    snr_adapt  = np.mean([r['snr'] for r in res_adapt.values()])
    print("  " + "─" * 56)
    print(f"  {'MOYENNE':<16} {'':>6} {'':>8} {mae_dehaan:6.1f} {snr_dehaan:7.2f} | "
          f"{'':>8} {mae_adapt:6.1f} {snr_adapt:7.2f}")
    print("═" * 60)

    os.makedirs(args.out, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    x = np.arange(len(test_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(6, len(test_names) * 0.6), 4))
    ax.bar(x - width / 2, [res_dehaan[n]['snr'] for n in test_names], width, label='CHROM De Haan')
    ax.bar(x + width / 2, [res_adapt[n]['snr'] for n in test_names], width, label='CHROM adaptatif')
    ax.set_xticks(x)
    ax.set_xticklabels(test_names, rotation=45, ha='right')
    ax.set_ylabel('SNR (dB)')
    ax.set_title('SNR par scénario (set de test) — CHROM De Haan vs adaptatif (DataVital)')
    ax.legend()
    fig.tight_layout()
    out_path = os.path.join(args.out, 'snr_comparison.png')
    fig.savefig(out_path, dpi=120)
    print(f"\n[Graphique] {out_path}")

    fig2, ax2 = plt.subplots(figsize=(4, 4))
    ax2.bar(['De Haan', 'Adaptatif'], [snr_dehaan, snr_adapt], color=['#888888', '#22aa77'])
    ax2.set_ylabel('SNR moyen (dB)')
    ax2.set_title('SNR moyen — set de test (DataVital)')
    fig2.tight_layout()
    out_path2 = os.path.join(args.out, 'snr_mean_comparison.png')
    fig2.savefig(out_path2, dpi=120)
    print(f"[Graphique] {out_path2}")


if __name__ == '__main__':
    main()
