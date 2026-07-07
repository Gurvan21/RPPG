#!/usr/bin/env python3
"""Port FIDÈLE du TS-CAN de rPPG-Toolbox (ubicomplab), pour charger leurs poids
pré-entraînés PURE_TSCAN.pth (noms de couches + paddings identiques).
Entrée : (N=B*T, 6, H, W) = concat[DiffNormalized(3), Raw standardisé(3)].
Vérifié : charge le state_dict PURE en strict=True (img_size=72 -> flatten 16384)."""
import torch
import torch.nn as nn


def tsm(x, n_frames, fold_div=3):
    """Temporal Shift Module (zero-pad, slicing+cat -> compatible MPS)."""
    nt, c, h, w = x.shape
    b = nt // n_frames
    x = x.view(b, n_frames, c, h, w)
    fold = c // fold_div
    x1, x2, x3 = x[:, :, :fold], x[:, :, fold:2*fold], x[:, :, 2*fold:]
    z1 = torch.zeros_like(x1[:, :1]); z2 = torch.zeros_like(x2[:, :1])
    x1 = torch.cat([x1[:, 1:], z1], dim=1)
    x2 = torch.cat([z2, x2[:, :-1]], dim=1)
    return torch.cat([x1, x2, x3], dim=2).view(nt, c, h, w)


class TSCAN_official(nn.Module):
    def __init__(self, frames=128, in_ch=3, f1=32, f2=64, k=3, d1=0.25, d2=0.5,
                 dense=128, img_size=72):
        super().__init__()
        self.frames = frames
        # branche mouvement (conv1/3 pad=1 "same", conv2/4 pad=0 "valid")
        self.motion_conv1 = nn.Conv2d(in_ch, f1, k, padding=(1, 1))
        self.motion_conv2 = nn.Conv2d(f1, f1, k)
        self.motion_conv3 = nn.Conv2d(f1, f2, k, padding=(1, 1))
        self.motion_conv4 = nn.Conv2d(f2, f2, k)
        # branche apparence
        self.apperance_conv1 = nn.Conv2d(in_ch, f1, k, padding=(1, 1))
        self.apperance_conv2 = nn.Conv2d(f1, f1, k)
        self.apperance_conv3 = nn.Conv2d(f1, f2, k, padding=(1, 1))
        self.apperance_conv4 = nn.Conv2d(f2, f2, k)
        # masques d'attention
        self.apperance_att_conv1 = nn.Conv2d(f1, 1, 1)
        self.apperance_att_conv2 = nn.Conv2d(f2, 1, 1)
        self.avg_pooling_1 = nn.AvgPool2d((2, 2))
        self.avg_pooling_2 = nn.AvgPool2d((2, 2))
        self.avg_pooling_3 = nn.AvgPool2d((2, 2))
        self.dropout_1 = nn.Dropout(d1); self.dropout_2 = nn.Dropout(d1)
        self.dropout_3 = nn.Dropout(d1); self.dropout_4 = nn.Dropout(d2)
        s = img_size
        for _ in range(2): s = (s + 2 - 2*1 - 1 + 1)  # conv "same" garde s
        # calcul réel du flatten via forward-dummy à l'init serait plus sûr, mais
        # 72 -> conv2 valid(70) -> pool(35) -> conv4 valid(33) -> pool(16) -> 64*16*16
        flat = f2 * 16 * 16 if img_size == 72 else f2 * ((img_size)//4) ** 2
        self.final_dense_1 = nn.Linear(flat, dense)
        self.final_dense_2 = nn.Linear(dense, 1)

    def _att(self, mask):
        b, _, h, w = mask.shape
        return mask / (2 * mask.sum(dim=(2, 3), keepdim=True) + 1e-7) * h * w

    def forward(self, inputs, n_frames=None):
        n = n_frames or self.frames
        diff = inputs[:, :3]; raw = inputs[:, 3:]
        d = torch.tanh(self.motion_conv1(tsm(diff, n)))
        d = torch.tanh(self.motion_conv2(tsm(d, n)))
        r = torch.tanh(self.apperance_conv1(raw))
        r = torch.tanh(self.apperance_conv2(r))
        g = torch.sigmoid(self.apperance_att_conv1(r)); g = self._att(g)
        d = self.dropout_1(self.avg_pooling_1(d * g))
        r = self.dropout_2(self.avg_pooling_2(r))

        d = torch.tanh(self.motion_conv3(tsm(d, n)))
        d = torch.tanh(self.motion_conv4(tsm(d, n)))
        r = torch.tanh(self.apperance_conv3(r))
        r = torch.tanh(self.apperance_conv4(r))
        g = torch.sigmoid(self.apperance_att_conv2(r)); g = self._att(g)
        d = self.dropout_3(self.avg_pooling_3(d * g))

        d = d.reshape(d.size(0), -1)
        d = self.dropout_4(torch.tanh(self.final_dense_1(d)))
        out = self.final_dense_2(d)
        return out.view(-1, n)


def load_pretrained(model, path):
    sd = torch.load(path, map_location='cpu')
    sd = sd.get('state_dict', sd) if isinstance(sd, dict) else sd
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    return model
