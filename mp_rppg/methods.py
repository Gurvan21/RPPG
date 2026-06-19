"""
Implémentations CHROM et POS adaptées pour recevoir un signal RGB (T,3) pré-extrait.
Réplique fidèle des méthodes du toolbox rPPG-Toolbox.
"""

import math
import numpy as np
from scipy import signal, sparse


def _detrend(sig, lam=100):
    n = len(sig)
    H = np.identity(n)
    ones = np.ones(n)
    D = sparse.spdiags(np.array([ones, -2 * ones, ones]), [0, 1, 2],
                       n - 2, n).toarray()
    return (H - np.linalg.inv(H + lam ** 2 * D.T @ D)) @ sig


def chrom(rgb, fs):
    """
    CHROM — De Haan & Jeanne, IEEE TBME 2013.

    Args:
        rgb : (T, 3) float — signal RGB moyen par frame
        fs  : fréquence d'échantillonnage (FPS)

    Returns:
        bvp : (N,) signal BVP normalisé
    """
    FN = len(rgb)
    Nyq = fs / 2
    B, A = signal.butter(3, [0.7 / Nyq, 2.5 / Nyq], 'bandpass')

    WinL = math.ceil(1.6 * fs)
    if WinL % 2:
        WinL += 1
    NWin = math.floor((FN - WinL // 2) / (WinL // 2))
    total = (WinL // 2) * (NWin + 1)
    S = np.zeros(total)
    WinS, WinM, WinE = 0, WinL // 2, WinL

    for _ in range(NWin):
        base = np.mean(rgb[WinS:WinE], axis=0)
        Rn = rgb[WinS:WinE] / (base + 1e-8)
        Xs = 3 * Rn[:, 0] - 2 * Rn[:, 1]
        Ys = 1.5 * Rn[:, 0] + Rn[:, 1] - 1.5 * Rn[:, 2]
        Xf = signal.filtfilt(B, A, Xs)
        Yf = signal.filtfilt(B, A, Ys)
        alpha = np.std(Xf) / (np.std(Yf) + 1e-8)
        SWin = (Xf - alpha * Yf) * signal.windows.hann(WinL)
        S[WinS:WinM] += SWin[:WinL // 2]
        S[WinM:WinE]  = SWin[WinL // 2:]
        WinS = WinM
        WinM = WinS + WinL // 2
        WinE = WinS + WinL
    return S


def chrom_adaptive(rgb, fs, coeffs=None):
    """
    CHROM avec coefficients de combinaison appris (voir models/chrom_adaptive.py).

    Même fenêtrage glissant que chrom(), mais Xs/Ys sont calculés avec
    a1..a5 au lieu des constantes [3, 2, 1.5, 1, 1.5] de De Haan.

    Args:
        rgb    : (T, 3) float — signal RGB moyen par frame
        fs     : fréquence d'échantillonnage (FPS)
        coeffs : dict {'a1','a2','a3','a4','a5'} ; si None, valeurs De Haan

    Returns:
        bvp : (N,) signal BVP normalisé
    """
    if coeffs is None:
        coeffs = {'a1': 3.0, 'a2': 2.0, 'a3': 1.5, 'a4': 1.0, 'a5': 1.5}
    a1, a2, a3, a4, a5 = (coeffs['a1'], coeffs['a2'], coeffs['a3'],
                          coeffs['a4'], coeffs['a5'])

    FN = len(rgb)
    Nyq = fs / 2
    B, A = signal.butter(3, [0.7 / Nyq, 2.5 / Nyq], 'bandpass')

    WinL = math.ceil(1.6 * fs)
    if WinL % 2:
        WinL += 1
    NWin = math.floor((FN - WinL // 2) / (WinL // 2))
    total = (WinL // 2) * (NWin + 1)
    S = np.zeros(total)
    WinS, WinM, WinE = 0, WinL // 2, WinL

    for _ in range(NWin):
        base = np.mean(rgb[WinS:WinE], axis=0)
        Rn = rgb[WinS:WinE] / (base + 1e-8)
        Xs = a1 * Rn[:, 0] - a2 * Rn[:, 1]
        Ys = a3 * Rn[:, 0] + a4 * Rn[:, 1] - a5 * Rn[:, 2]
        Xf = signal.filtfilt(B, A, Xs)
        Yf = signal.filtfilt(B, A, Ys)
        alpha = np.std(Xf) / (np.std(Yf) + 1e-8)
        SWin = (Xf - alpha * Yf) * signal.windows.hann(WinL)
        S[WinS:WinM] += SWin[:WinL // 2]
        S[WinM:WinE]  = SWin[WinL // 2:]
        WinS = WinM
        WinM = WinS + WinL // 2
        WinE = WinS + WinL
    return S


def pos(rgb, fs):
    """
    POS — Wang et al., IEEE TBME 2017.

    Args:
        rgb : (T, 3) float — signal RGB moyen par frame
        fs  : fréquence d'échantillonnage (FPS)

    Returns:
        bvp : (T,) signal BVP normalisé
    """
    N = len(rgb)
    H = np.zeros(N)
    l = math.ceil(1.6 * fs)
    P = np.array([[0, 1, -1], [-2, 1, 1]])

    for n in range(N):
        m = n - l
        if m < 0:
            continue
        win_mean = np.mean(rgb[m:n], axis=0)
        Cn = rgb[m:n] / (win_mean + 1e-8)
        S = P @ Cn.T
        h = S[0] + (np.std(S[0]) / (np.std(S[1]) + 1e-8)) * S[1]
        H[m:n] += h - np.mean(h)

    H = _detrend(H, 100)
    b, a = signal.butter(1, [0.75 / fs * 2, 3.0 / fs * 2], btype='bandpass')
    return signal.filtfilt(b, a, H.astype(np.float64))
