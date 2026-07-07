"""
Calcul des métriques d'évaluation rPPG.
Utilise les fonctions du toolbox (evaluation/post_process.py) quand disponibles,
sinon re-implémente les mêmes formules.
"""

import sys
import os
import numpy as np
from scipy.signal import butter, filtfilt, periodogram

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def _next_pow2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def hr_from_fft(bvp, fs, low=0.7, high=2.5):
    """HR en bpm par FFT (pic dominant dans [low, high] Hz)."""
    N = _next_pow2(len(bvp))
    f, pxx = periodogram(bvp, fs=fs, nfft=N, detrend=False)
    mask = (f >= low) & (f <= high)
    return float(f[mask][np.argmax(pxx[mask])] * 60)


def hr_candidates(bvp, fs, low=0.7, high=2.5, ratio=0.6, min_sep_bpm=4.0):
    """Détecte l'AMBIGUÏTÉ : deux rythmes candidats dont les pics spectraux se
    valent presque. Même estimateur que hr_from_fft (periodogram boxcar+zeropad).

    Retourne (candidats, ambigu) où candidats = liste [(hr_bpm, puissance_rel)]
    triée par puissance décroissante, et ambigu=True si le 2e pic (maximum LOCAL,
    séparé de >min_sep_bpm, NON-harmonique) atteint >= `ratio` fois le 1er.
    """
    from scipy.signal import find_peaks
    N = _next_pow2(len(bvp))
    f, pxx = periodogram(bvp, fs=fs, nfft=N, detrend=False)
    mask = (f >= low) & (f <= high)
    fb, pb = f[mask] * 60.0, pxx[mask]
    if pb.max() <= 0:
        return [(float(fb[np.argmax(pb)]), 1.0)], False
    loc, _ = find_peaks(pb)                       # maxima LOCAUX (vrais pics)
    if len(loc) == 0:
        loc = np.array([int(np.argmax(pb))])
    loc = loc[np.argsort(pb[loc])[::-1]]          # triés par hauteur
    top = pb[loc[0]]
    cands = [(float(fb[i]), float(pb[i] / top)) for i in loc[:3]]
    ambiguous = False
    if len(cands) >= 2:
        (h1, _), (h2, r2) = cands[0], cands[1]
        harm = h2 > 0 and (abs(h1 / h2 - 2) < 0.15 or abs(h2 / h1 - 2) < 0.15)
        if r2 >= ratio and abs(h1 - h2) > min_sep_bpm and not harm:
            ambiguous = True
    return cands, ambiguous


def snr(bvp, hr_gt_bpm, fs, low=0.7, high=2.5):
    """
    SNR (dB) : puissance aux harmoniques 1+2 du HR vrai
               vs reste de la bande [low, high] Hz.
    """
    N = _next_pow2(len(bvp))
    f, pxx = periodogram(bvp, fs=fs, nfft=N, detrend=False)
    dev = 6 / 60
    h1, h2 = hr_gt_bpm / 60, 2 * hr_gt_bpm / 60
    sig_mask  = (((f >= h1 - dev) & (f <= h1 + dev)) |
                 ((f >= h2 - dev) & (f <= h2 + dev)))
    noise_mask = (f >= low) & (f <= high) & ~sig_mask
    sp = np.sum(pxx[sig_mask])
    np_ = np.sum(pxx[noise_mask])
    return float(10 * np.log10(sp / np_)) if np_ > 0 else 0.0


def aggregate(errors, snrs):
    """Calcule MAE, RMSE, Pearson à partir des listes d'erreurs absolues et SNR."""
    e = np.array(errors)
    s = np.array(snrs)
    mae  = float(np.mean(np.abs(e)))
    rmse = float(np.sqrt(np.mean(e ** 2)))
    return {'MAE': mae, 'RMSE': rmse, 'SNR_mean': float(np.mean(s))}
