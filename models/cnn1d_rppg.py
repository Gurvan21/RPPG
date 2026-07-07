"""
CNN 1D pour rPPG : prend les signaux RGB multi-régions (issus de BiSeNet +
FaceMesh, voir scripts/extract_regions_bisenet.py) et apprend à produire le
signal BVP, end-to-end.

Entrée  : (B, C, T)  avec C = n_regions * 3 canaux (RGB par région)
Sortie  : (B, T)     signal BVP

Architecture : convolutions 1D dilatées (champ réceptif temporel large sans
trop de paramètres) + tête de projection à 1 canal. Bien plus léger qu'un 3D
CNN (PhysNet) → entraînable sur peu de sujets.

Apprend implicitement : pondération des régions + projection chrominance +
filtrage temporel — la version apprenable du pipeline CHROM/POS multi-ROI.
"""

import torch
import torch.nn as nn


class _TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilation, ks=3, p=0.1):
        super().__init__()
        pad = (ks - 1) // 2 * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, ks, padding=pad, dilation=dilation)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, ks, padding=pad, dilation=dilation)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.drop  = nn.Dropout(p)
        self.relu  = nn.ReLU(inplace=True)
        self.down  = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        res = x if self.down is None else self.down(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        return self.relu(out + res)


class CNN1D_rPPG(nn.Module):
    """
    in_channels = n_regions * 3 (par défaut 7 régions → 21).
    Champ réceptif temporel ~ via dilations 1,2,4,8.
    """

    def __init__(self, in_channels=21, hidden=64, dilations=(1, 2, 4, 8), dropout=0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, hidden, 1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[
            _TemporalBlock(hidden, hidden, dilation=d, p=dropout) for d in dilations
        ])
        self.head = nn.Conv1d(hidden, 1, 1)

    def forward(self, x):           # x : (B, C, T)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)            # (B, 1, T)
        return x.squeeze(1)         # (B, T)
