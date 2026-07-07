#!/usr/bin/env python3
"""TS-CAN (Liu et al. 2020, "Multi-Task Temporal Shift Attention Networks for
On-Device Contactless Vitals Measurement"). Double branche :
  - apparence (frame brute standardisée) -> masques d'attention
  - mouvement (frame DiffNormalized, avec Temporal Shift) -> dérivée du pouls
Sortie : dérivée du BVP par frame (N, T). La FC s'obtient par FFT (la dérivée
conserve la fréquence fondamentale)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def temporal_shift(x, n_frames, fold_div=3):
    """Temporal Shift Module (zero-pad aux bords). x: (N=batch*T, C, H, W).
    Implémenté en slicing + cat (compatible MPS, évite l'affectation indexée
    qui retombe sur CPU)."""
    nt, c, h, w = x.shape
    b = nt // n_frames
    x = x.view(b, n_frames, c, h, w)
    fold = c // fold_div
    x1, x2, x3 = x[:, :, :fold], x[:, :, fold:2*fold], x[:, :, 2*fold:]
    z1 = torch.zeros_like(x1[:, :1])
    z2 = torch.zeros_like(x2[:, :1])
    x1 = torch.cat([x1[:, 1:], z1], dim=1)          # shift vers le futur
    x2 = torch.cat([z2, x2[:, :-1]], dim=1)         # shift vers le passé
    return torch.cat([x1, x2, x3], dim=2).view(nt, c, h, w)


class TSCAN(nn.Module):
    def __init__(self, frames=128, img_size=72, f1=32, f2=64, k=3, drop=0.25, dense=128):
        super().__init__()
        self.frames = frames
        pad = k // 2
        # branche mouvement (Temporal Shift appliqué au forward)
        self.m_conv1 = nn.Conv2d(3, f1, k, padding=pad)
        self.m_conv2 = nn.Conv2d(f1, f1, k, padding=pad)
        self.m_conv3 = nn.Conv2d(f1, f2, k, padding=pad)
        self.m_conv4 = nn.Conv2d(f2, f2, k, padding=pad)
        # branche apparence
        self.a_conv1 = nn.Conv2d(3, f1, k, padding=pad)
        self.a_conv2 = nn.Conv2d(f1, f1, k, padding=pad)
        self.a_conv3 = nn.Conv2d(f1, f2, k, padding=pad)
        self.a_conv4 = nn.Conv2d(f2, f2, k, padding=pad)
        # masques d'attention (1x1 conv -> 1 canal)
        self.att1 = nn.Conv2d(f1, 1, 1)
        self.att2 = nn.Conv2d(f2, 1, 1)
        self.pool = nn.AvgPool2d(2)
        self.drop = nn.Dropout(drop)
        s = img_size // 4                            # 2 poolings
        self.fc1 = nn.Linear(f2 * s * s, dense)
        self.fc2 = nn.Linear(dense, 1)

    def _attn(self, a, att):
        m = torch.sigmoid(att(a))                    # (N,1,H,W)
        # normalisation "softmax spatial" façon TS-CAN
        b, _, h, w = m.shape
        norm = m / (2 * m.sum(dim=(2, 3), keepdim=True) / (h * w) + 1e-7)
        return norm

    def forward(self, motion, appearance):
        # entrées : (N=batch*T, 3, H, W)
        n = self.frames
        d = torch.tanh(self.m_conv1(temporal_shift(motion, n)))
        d = torch.tanh(self.m_conv2(temporal_shift(d, n)))
        a = torch.tanh(self.a_conv1(appearance))
        a = torch.tanh(self.a_conv2(a))
        g = d * self._attn(a, self.att1)             # gating par attention
        g = self.drop(self.pool(g))
        a = self.drop(self.pool(a))

        d = torch.tanh(self.m_conv3(temporal_shift(g, n)))
        d = torch.tanh(self.m_conv4(temporal_shift(d, n)))
        a = torch.tanh(self.a_conv3(a))
        a = torch.tanh(self.a_conv4(a))
        g = d * self._attn(a, self.att2)
        g = self.drop(self.pool(g))

        g = g.flatten(1)
        g = self.drop(torch.tanh(self.fc1(g)))
        out = self.fc2(g)                            # (N,1)
        return out.view(-1, n)                       # (batch, T)
