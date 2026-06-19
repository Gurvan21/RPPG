"""
CHROM adaptatif — variante de De Haan & Jeanne (IEEE TBME 2013) dont les
5 coefficients de combinaison des canaux sont appris par descente de
gradient au lieu d'être fixés à [3, 2, 1.5, 1, 1.5].

Signal = (a1*Rn - a2*Gn) - alpha * (a3*Rn + a4*Gn - a5*Bn)
avec alpha = std(Xs) / std(Ys)

Tourne entièrement sur CPU, compatible Python 3.8 / PyTorch.
"""

import os

import numpy as np
import torch
import torch.nn as nn
from scipy import signal as sp_signal

# Coefficients originaux de De Haan (2013), utilisés comme initialisation
# et comme valeurs par défaut quand aucun modèle entraîné n'est disponible.
DEHAAN_COEFFICIENTS = {'a1': 3.0, 'a2': 2.0, 'a3': 1.5, 'a4': 1.0, 'a5': 1.5}


class CHROMAdaptatif(nn.Module):
    """CHROM avec 5 coefficients apprenables, initialisés aux valeurs De Haan."""

    def __init__(self):
        super().__init__()
        self.a1 = nn.Parameter(torch.tensor(DEHAAN_COEFFICIENTS['a1']))
        self.a2 = nn.Parameter(torch.tensor(DEHAAN_COEFFICIENTS['a2']))
        self.a3 = nn.Parameter(torch.tensor(DEHAAN_COEFFICIENTS['a3']))
        self.a4 = nn.Parameter(torch.tensor(DEHAAN_COEFFICIENTS['a4']))
        self.a5 = nn.Parameter(torch.tensor(DEHAAN_COEFFICIENTS['a5']))

    def forward(self, Rn, Gn, Bn):
        """
        Rn, Gn, Bn : tenseurs [T] normalisés par leur moyenne.
        Retourne le signal rPPG [T].
        """
        Xs = self.a1 * Rn - self.a2 * Gn
        Ys = self.a3 * Rn + self.a4 * Gn - self.a5 * Bn
        alpha = Xs.std() / (Ys.std() + 1e-8)
        return Xs - alpha * Ys

    def get_coefficients(self):
        return {
            'a1': self.a1.item(),
            'a2': self.a2.item(),
            'a3': self.a3.item(),
            'a4': self.a4.item(),
            'a5': self.a5.item(),
        }


def _checkpoint_path_for_skin_type(model_path, skin_type):
    """
    Si model_path est un dossier, cherche un checkpoint spécifique au
    phototype `skin_type` (chrom_adaptive_skin{N}.pth), sinon retombe sur
    chrom_adaptive.pth / chrom_adaptive_default.pth dans ce dossier.
    """
    if os.path.isdir(model_path):
        candidates = []
        if skin_type is not None:
            candidates.append(os.path.join(model_path, f"chrom_adaptive_skin{skin_type}.pth"))
        candidates.append(os.path.join(model_path, "chrom_adaptive.pth"))
        candidates.append(os.path.join(model_path, "chrom_adaptive_default.pth"))
        for c in candidates:
            if os.path.exists(c):
                return c
        raise FileNotFoundError(
            f"Aucun checkpoint CHROM adaptatif trouvé dans {model_path} "
            f"(essayé : {candidates})"
        )
    return model_path


def load_coefficients(model_path=None, skin_type=None):
    """
    Charge les coefficients CHROM adaptatif appris.

    Args:
        model_path : chemin vers un fichier .pth, ou dossier contenant
                      plusieurs checkpoints (un par phototype). Si None,
                      retourne les coefficients De Haan par défaut.
        skin_type  : phototype (1-6), utilisé uniquement si model_path est
                      un dossier contenant un checkpoint par phototype.

    Returns:
        dict {'a1', 'a2', 'a3', 'a4', 'a5'}
    """
    if model_path is None:
        return dict(DEHAAN_COEFFICIENTS)

    path = _checkpoint_path_for_skin_type(model_path, skin_type)
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    return checkpoint['coefficients']


# ══════════════════════════════════════════════════════════════
# LOSS & FILTRAGE — utilisés par scripts/train_chrom_adaptive.py
# ══════════════════════════════════════════════════════════════

def bandpass_numpy(sig, fps, low=0.75, high=3.5, order=4):
    nyq = fps / 2.0
    b, a = sp_signal.butter(order, [low / nyq, high / nyq], btype='band')
    return np.ascontiguousarray(sp_signal.filtfilt(b, a, sig))


def bandpass_straight_through(sig, fps, low=0.75, high=3.5, order=4):
    """
    Filtre passe-bande appliqué à `sig` (tenseur avec grad), via scipy.

    Le filtrage scipy n'est pas différentiable : on utilise un estimateur
    "straight-through" — la valeur retournée est le signal filtré, mais le
    gradient est routé directement sur `sig` (filtre traité comme l'identité
    pour la rétropropagation). Sans cela, la loss ne dépend plus des
    paramètres du modèle et l'entraînement n'a aucun effet.
    """
    sig_filt = bandpass_numpy(sig.detach().numpy(), fps, low, high, order)
    sig_filt_t = torch.tensor(sig_filt, dtype=sig.dtype, device=sig.device)
    return sig + (sig_filt_t - sig.detach())


def loss_pearson(pred, target):
    """Loss = 1 - corrélation de Pearson (même loss que PhysNet)."""
    pred_m = pred - pred.mean()
    target_m = target - target.mean()
    r = (pred_m * target_m).sum() / (pred_m.norm() * target_m.norm() + 1e-8)
    return 1 - r


def loss_hr_fft(signal_pred, hr_ref, fps=30):
    """
    Loss = écart entre le pic FFT du signal prédit et le HR de référence
    (en bpm). Utilisée quand on n'a que le HR oxymètre, pas le BVP complet.
    """
    N = len(signal_pred)
    window = torch.hann_window(N, device=signal_pred.device)
    fft = torch.abs(torch.fft.rfft(signal_pred * window, n=N * 4))
    freqs = torch.fft.rfftfreq(N * 4, d=1.0 / fps).to(signal_pred.device) * 60

    mask = (freqs >= 45) & (freqs <= 180)
    hr_est = freqs[mask][torch.argmax(fft[mask])]
    return torch.abs(hr_est - hr_ref)


# ══════════════════════════════════════════════════════════════
# INFÉRENCE
# ══════════════════════════════════════════════════════════════

def infer_hr(signal_rgb, model_path=None, skin_type=None, fps=30):
    """
    Estime le HR (bpm) depuis un signal RGB avec les coefficients appris.

    Args:
        signal_rgb : array [T, 3]
        model_path : .pth ou dossier de checkpoints (voir load_coefficients)
        skin_type  : phototype 1-6, optionnel
        fps        : fréquence d'échantillonnage

    Returns:
        (hr_bpm, coeffs)
    """
    coeffs = load_coefficients(model_path, skin_type)

    R = signal_rgb[:, 0].astype(np.float32)
    G = signal_rgb[:, 1].astype(np.float32)
    B = signal_rgb[:, 2].astype(np.float32)

    Rn = torch.tensor(R / (R.mean() + 1e-8))
    Gn = torch.tensor(G / (G.mean() + 1e-8))
    Bn = torch.tensor(B / (B.mean() + 1e-8))

    model = CHROMAdaptatif()
    with torch.no_grad():
        model.a1.copy_(torch.tensor(coeffs['a1']))
        model.a2.copy_(torch.tensor(coeffs['a2']))
        model.a3.copy_(torch.tensor(coeffs['a3']))
        model.a4.copy_(torch.tensor(coeffs['a4']))
        model.a5.copy_(torch.tensor(coeffs['a5']))
        sig = model(Rn, Gn, Bn).numpy()

    sig_filt = bandpass_numpy(sig, fps)

    N = len(sig_filt)
    fft = np.abs(np.fft.rfft(sig_filt * np.hanning(N), n=N * 4))
    freqs = np.fft.rfftfreq(N * 4, d=1.0 / fps) * 60
    mask = (freqs >= 45) & (freqs <= 180)
    hr = freqs[mask][np.argmax(fft[mask])]
    return float(hr), coeffs
