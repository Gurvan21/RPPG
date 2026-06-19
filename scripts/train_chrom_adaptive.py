"""
Entraîne CHROM adaptatif (models/chrom_adaptive.CHROMAdaptatif) sur un
dossier de sujets, au format rPPG-Toolbox (.npy) ou CSV MediaPipe.

Usage :
    # UBFC préprocessé par rPPG-Toolbox (signal BVP de référence disponible)
    python scripts/train_chrom_adaptive.py \\
        --data /chemin/vers/ubfc_preprocessed \\
        --epochs 100 --lr 0.01 \\
        --save weights/chrom_adaptive_ubfc.pth

    # Données terrain (CSV MediaPipe), HR oxymètre seulement, par phototype
    python scripts/train_chrom_adaptive.py \\
        --data /chemin/vers/donnees_terrain --csv --hr-only \\
        --skin-type 5 --save weights/chrom_adaptive_skin5.pth
"""

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, ROOT)

from models.chrom_adaptive import (CHROMAdaptatif, DEHAAN_COEFFICIENTS,
                                    bandpass_numpy, bandpass_straight_through,
                                    loss_hr_fft, loss_pearson)
from mp_rppg.datasets import RPPGSignalDataset


def train(data_dir, fps=30, epochs=100, lr=0.01, use_csv=False,
          use_label_signal=True, save_path="chrom_adaptive.pth"):
    """
    use_label_signal=True  : nécessite un signal BVP complet (UBFC, PURE)
    use_label_signal=False : nécessite seulement le HR en bpm (données terrain)
    """
    dataset = RPPGSignalDataset(data_dir, fps=fps, use_csv=use_csv)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = CHROMAdaptatif()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    print(f"\nCoefficients initiaux (De Haan) : {model.get_coefficients()}\n")

    for epoch in range(epochs):
        total_loss = 0.0
        n_valid = 0

        for batch in loader:
            Rn = batch['Rn'].squeeze(0)
            Gn = batch['Gn'].squeeze(0)
            Bn = batch['Bn'].squeeze(0)

            sig = model(Rn, Gn, Bn)
            sig_t = bandpass_straight_through(sig, fps)

            if use_label_signal and 'label' in batch:
                label = batch['label'].squeeze(0)
                label_filt = bandpass_numpy(label.numpy(), fps)
                label_t = torch.tensor(label_filt, dtype=torch.float32)
                loss = loss_pearson(sig_t, label_t)

            elif 'hr' in batch:
                loss = loss_hr_fft(sig_t, batch['hr'].squeeze(), fps)

            else:
                continue

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_valid += 1

        scheduler.step()

        if epoch % 10 == 0 and n_valid > 0:
            coeffs = model.get_coefficients()
            print(
                f"Epoch {epoch:3d} | "
                f"Loss: {total_loss / n_valid:.4f} | "
                f"[{coeffs['a1']:.3f}, {coeffs['a2']:.3f}, "
                f"{coeffs['a3']:.3f}, {coeffs['a4']:.3f}, {coeffs['a5']:.3f}]"
            )

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'coefficients': model.get_coefficients(),
        'fps': fps,
    }, save_path)

    print(f"\nModèle sauvegardé : {save_path}")
    print(f"Coefficients finaux  : {model.get_coefficients()}")
    print(f"Coefficients De Haan : {DEHAAN_COEFFICIENTS}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Entraînement de CHROM adaptatif")
    parser.add_argument('--data', default=None,
                        help="Dossier de sujets (préprocessé .npy ou CSV MediaPipe)")
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--csv', action='store_true',
                        help="Format CSV MediaPipe (rppg_rgb.csv + hr_reference.txt)")
    parser.add_argument('--hr-only', action='store_true',
                        help="N'utiliser que le HR de référence (pas de signal BVP)")
    parser.add_argument('--skin-type', type=int, choices=range(1, 7), default=None,
                        help="Phototype (1-6) — utilisé pour nommer le checkpoint "
                             "si --save n'est pas fourni")
    parser.add_argument('--save', default=None,
                        help="Chemin de sauvegarde du modèle (.pth)")
    parser.add_argument('--config', default=None,
                        help="Config YAML (DATA/TRAIN/MODEL) — sert de valeur "
                             "par défaut pour les autres options")
    args = parser.parse_args()

    if args.config:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        data_cfg, train_cfg, model_cfg = cfg.get('DATA', {}), cfg.get('TRAIN', {}), cfg.get('MODEL', {})
        if args.data is None:
            args.data = data_cfg.get('PATH')
        if 'FORMAT' in data_cfg and not args.csv:
            args.csv = data_cfg['FORMAT'] == 'csv'
        args.fps = data_cfg.get('FS', args.fps)
        args.epochs = train_cfg.get('EPOCHS', args.epochs)
        args.lr = train_cfg.get('LR', args.lr)
        if 'USE_LABEL_SIGNAL' in train_cfg:
            args.hr_only = not train_cfg['USE_LABEL_SIGNAL']
        if args.skin_type is None and 'SKIN_TYPE' in model_cfg:
            args.skin_type = model_cfg['SKIN_TYPE']
        if args.save is None and 'SAVE_DIR' in model_cfg:
            name = f"chrom_adaptive_skin{args.skin_type}.pth" if args.skin_type \
                else "chrom_adaptive.pth"
            args.save = os.path.join(ROOT, model_cfg['SAVE_DIR'], name)

    if args.data is None:
        sys.exit("[ERREUR] --data requis (ou DATA.PATH dans --config)")

    save_path = args.save
    if save_path is None:
        name = f"chrom_adaptive_skin{args.skin_type}.pth" if args.skin_type \
            else "chrom_adaptive.pth"
        save_path = os.path.join(ROOT, "weights", name)

    train(
        data_dir=args.data,
        fps=args.fps,
        epochs=args.epochs,
        lr=args.lr,
        use_csv=args.csv,
        use_label_signal=not args.hr_only,
        save_path=save_path,
    )


if __name__ == '__main__':
    main()
