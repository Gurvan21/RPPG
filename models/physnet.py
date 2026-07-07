""" PhysNet
We repulicate the net pipeline of the orginal paper, but set the input as diffnormalized data.
orginal source:
Remote Photoplethysmograph Signal Measurement from Facial Videos Using Spatio-Temporal Networks
British Machine Vision Conference (BMVC)} 2019,
By Zitong Yu, 2019/05/05
Only for research purpose, and commercial use is not allowed.
MIT License
Copyright (c) 2019
"""

import math
import pdb

import torch
import torch.nn as nn
from torch.nn.modules.utils import _triple


class SeparableMaxPool3d(nn.Module):
    """
    Remplacement de nn.MaxPool3d (kernel=stride=2, non chevauchant) par des
    reshape+amax — opérations nativement supportées sur MPS (Apple GPU), alors
    que MaxPool3d ne l'est pas (et tombe sinon en fallback CPU très lent).

    Résultat numériquement IDENTIQUE à nn.MaxPool3d(k, stride=k, ceil_mode=False)
    → les poids entraînés restent valides. Gère les dimensions impaires en
    tronquant le reste (comportement floor de MaxPool3d).
    """

    def __init__(self, temporal=False):
        super().__init__()
        self.temporal = temporal   # True = pool aussi en temps (2,2,2), False = spatial seul (1,2,2)

    def forward(self, x):                       # x : (B, C, T, H, W)
        B, C, T, H, W = x.shape
        if self.temporal:
            T2 = T - (T % 2)
            x = x[:, :, :T2].reshape(B, C, T2 // 2, 2, H, W).amax(dim=3)
            T = T2 // 2
        H2, W2 = H - (H % 2), W - (W % 2)
        x = x[:, :, :, :H2, :W2].reshape(B, C, T, H2 // 2, 2, W2 // 2, 2)
        return x.amax(dim=(4, 6))


class SpatialAvgPool3d(nn.Module):
    """
    Remplace nn.AdaptiveAvgPool3d((frames,1,1)) quand la dim temporelle vaut
    déjà `frames` (toujours le cas dans PhysNet après les upsample) : c'est
    alors une simple moyenne spatiale globale (H,W) → MPS-natif, pas de fallback.
    """

    def __init__(self, frames):
        super().__init__()
        self.frames = frames
        self._fallback = nn.AdaptiveAvgPool3d((frames, 1, 1))

    def forward(self, x):                       # (B, C, T, H, W)
        if x.shape[2] == self.frames:
            return x.mean(dim=(3, 4), keepdim=True)
        return self._fallback(x)                # cas générique (rare)


class PhysNet_padding_Encoder_Decoder_MAX(nn.Module):
    def __init__(self, frames=128):
        super(PhysNet_padding_Encoder_Decoder_MAX, self).__init__()

        self.ConvBlock1 = nn.Sequential(
            nn.Conv3d(3, 16, [1, 5, 5], stride=1, padding=[0, 2, 2]),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock2 = nn.Sequential(
            nn.Conv3d(16, 32, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock3 = nn.Sequential(
            nn.Conv3d(32, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock4 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock5 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock6 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock7 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock8 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock9 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[
                4, 1, 1], stride=[2, 1, 1], padding=[1, 0, 0]),  # [1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[
                4, 1, 1], stride=[2, 1, 1], padding=[1, 0, 0]),  # [1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )

        self.ConvBlock10 = nn.Conv3d(64, 1, [1, 1, 1], stride=1, padding=0)

        self.MaxpoolSpa = SeparableMaxPool3d(temporal=False)     # ≡ MaxPool3d((1,2,2))
        self.MaxpoolSpaTem = SeparableMaxPool3d(temporal=True)   # ≡ MaxPool3d((2,2,2))

        # self.poolspa = nn.AdaptiveMaxPool3d((frames,1,1))    # pool only spatial space
        self.poolspa = SpatialAvgPool3d(frames)   # ≡ AdaptiveAvgPool3d((frames,1,1)), MPS-natif

    def forward(self, x):  # Batch_size*[3, T, 128,128]
        x_visual = x
        [batch, channel, length, width, height] = x.shape

        x = self.ConvBlock1(x)  # x [3, T, 128,128]
        x = self.MaxpoolSpa(x)  # x [16, T, 64,64]

        x = self.ConvBlock2(x)  # x [32, T, 64,64]
        x_visual6464 = self.ConvBlock3(x)  # x [32, T, 64,64]
        # x [32, T/2, 32,32]    Temporal halve
        x = self.MaxpoolSpaTem(x_visual6464)

        x = self.ConvBlock4(x)  # x [64, T/2, 32,32]
        x_visual3232 = self.ConvBlock5(x)  # x [64, T/2, 32,32]
        x = self.MaxpoolSpaTem(x_visual3232)  # x [64, T/4, 16,16]

        x = self.ConvBlock6(x)  # x [64, T/4, 16,16]
        x_visual1616 = self.ConvBlock7(x)  # x [64, T/4, 16,16]
        x = self.MaxpoolSpa(x_visual1616)  # x [64, T/4, 8,8]

        x = self.ConvBlock8(x)  # x [64, T/4, 8, 8]
        x = self.ConvBlock9(x)  # x [64, T/4, 8, 8]
        x = self.upsample(x)  # x [64, T/2, 8, 8]
        x = self.upsample2(x)  # x [64, T, 8, 8]

        # x [64, T, 1,1]    -->  groundtruth left and right - 7
        x = self.poolspa(x)
        x = self.ConvBlock10(x)  # x [1, T, 1,1]

        rPPG = x.view(-1, length)

        return rPPG, x_visual, x_visual3232, x_visual1616
