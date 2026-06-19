"""
Dataset pour l'entraînement de CHROM adaptatif (models/chrom_adaptive.py).

Supporte deux formats de dossier sujet :

  - Format rPPG-Toolbox préprocessé :
        subject1/
        ├── input_frames.npy   # [T, H, W, C] frames
        └── label.npy          # [T] signal BVP de référence

  - Format CSV du pipeline MediaPipe (mp_rppg.pipeline) :
        subject1/
        ├── rppg_rgb.csv       # colonnes front_r, front_g, front_b, ...
        └── hr_reference.txt   # HR de référence (bpm), optionnel
"""

import os

import numpy as np
import torch
from torch.utils.data import Dataset


class RPPGSignalDataset(Dataset):
    """
    Charge un dossier de sujets et expose, pour chacun, le signal RGB
    normalisé (Rn, Gn, Bn) ainsi que le label disponible (signal BVP
    complet et/ou HR de référence en bpm).
    """

    def __init__(self, data_dir, fps=30, use_csv=False):
        self.fps = fps
        self.use_csv = use_csv
        self.samples = []
        self._load(data_dir)

    def _load(self, data_dir):
        import pandas as pd

        for subject in sorted(os.listdir(data_dir)):
            subject_path = os.path.join(data_dir, subject)
            if not os.path.isdir(subject_path):
                continue

            if self.use_csv:
                csv_path = os.path.join(subject_path, "rppg_rgb.csv")
                hr_path  = os.path.join(subject_path, "hr_reference.txt")
                if not os.path.exists(csv_path):
                    continue

                df  = pd.read_csv(csv_path).dropna()
                rgb = df[["front_r", "front_g", "front_b"]].values.astype(np.float32)

                hr = None
                if os.path.exists(hr_path):
                    hr = float(open(hr_path).read().strip())

                self.samples.append({'rgb': rgb, 'hr': hr, 'subject': subject})

            else:
                frames_path = os.path.join(subject_path, "input_frames.npy")
                label_path  = os.path.join(subject_path, "label.npy")
                if not os.path.exists(frames_path):
                    continue

                frames = np.load(frames_path)  # [T, H, W, C]
                label  = np.load(label_path) if os.path.exists(label_path) else None

                # Signal RGB moyen sur le tiers supérieur (front)
                h, w = frames.shape[1], frames.shape[2]
                roi = frames[:, :h // 4, w // 4:3 * w // 4, :]
                rgb = roi.mean(axis=(1, 2)).astype(np.float32)  # [T, 3]

                self.samples.append({'rgb': rgb, 'label': label, 'subject': subject})

        print(f"Dataset chargé : {len(self.samples)} sujets")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        rgb = sample['rgb']

        R = (rgb[:, 0] / (rgb[:, 0].mean() + 1e-8)).astype(np.float32)
        G = (rgb[:, 1] / (rgb[:, 1].mean() + 1e-8)).astype(np.float32)
        B = (rgb[:, 2] / (rgb[:, 2].mean() + 1e-8)).astype(np.float32)

        result = {
            'Rn': torch.tensor(R),
            'Gn': torch.tensor(G),
            'Bn': torch.tensor(B),
            'subject': sample['subject'],
        }

        if sample.get('hr') is not None:
            result['hr'] = torch.tensor(sample['hr'], dtype=torch.float32)

        if sample.get('label') is not None:
            result['label'] = torch.tensor(sample['label'].astype(np.float32))

        return result
