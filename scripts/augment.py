#!/usr/bin/env python3
"""Augmentations 'capture dure' pour rPPG. Le pouls (label) est INVARIANT à ces
dégradations -> on dégrade l'entrée, on garde le label -> robustesse apprise.
- niveau FRAME (PhysNet) : compression JPEG, mouvement (jitter affine), lumière, bruit
- niveau SIGNAL (CNN1D)  : bruit, amplitude, dérive, quantification temporelle
"""
import numpy as np, cv2


# ---------- niveau FRAME (pour PhysNet : (T,H,W,3)) ----------
def _to_u8(clip):
    lo, hi = clip.min(), clip.max()
    return ((clip - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8), (lo, hi)


def _from_u8(u8, lohi):
    lo, hi = lohi
    return u8.astype(np.float32) / 255 * (hi - lo) + lo


def diff_normalize(clip):
    out = np.zeros_like(clip, dtype=np.float32)
    out[:-1] = (clip[1:] - clip[:-1]) / (clip[1:] + clip[:-1] + 1e-7)
    return out


def standardize(clip):
    return (clip - clip.mean()) / (clip.std() + 1e-8)


def jpeg_compress(u8, q):
    out = np.empty_like(u8)
    for t in range(len(u8)):
        ok, enc = cv2.imencode('.jpg', u8[t], [cv2.IMWRITE_JPEG_QUALITY, int(q)])
        out[t] = cv2.imdecode(enc, cv2.IMREAD_COLOR) if ok else u8[t]
    return out


def motion_jitter(u8, rng, max_shift=4, max_rot=6):
    H, W = u8.shape[1:3]; out = np.empty_like(u8)
    for t in range(len(u8)):
        dx = rng.uniform(-max_shift, max_shift); dy = rng.uniform(-max_shift, max_shift)
        ang = rng.uniform(-max_rot, max_rot)
        M = cv2.getRotationMatrix2D((W/2, H/2), ang, 1.0); M[0, 2] += dx; M[1, 2] += dy
        out[t] = cv2.warpAffine(u8[t], M, (W, H), borderMode=cv2.BORDER_REFLECT)
    return out


def frame_augment(raw, rng, strength=1.0):
    """raw : (T,H,W,3) float (standardisé ou brut). Renvoie raw augmenté."""
    u8, lohi = _to_u8(raw)
    if rng.random() < 0.8*strength:                      # compression (le tueur WhatsApp)
        u8 = jpeg_compress(u8, q=rng.uniform(8, 30))
    if rng.random() < 0.7*strength:                      # mouvement
        u8 = motion_jitter(u8, rng, max_shift=4*strength, max_rot=6*strength)
    aug = _from_u8(u8, lohi)
    if rng.random() < 0.7*strength:                      # lumière (gamma + gain)
        g = rng.uniform(0.6, 1.6); aug = np.sign(aug)*np.abs(aug)**g * rng.uniform(0.7, 1.3)
    if rng.random() < 0.5*strength:                      # bruit capteur
        aug = aug + rng.normal(0, 0.03*np.std(aug), aug.shape)
    return aug.astype(np.float32)


def block_compress(clip, k):
    """Compression rapide (numpy vectorisé) : moyenne par blocs k×k puis ré-agrandit
    -> effet de blocs/perte de résolution, ~instantané. clip: (T,H,W,C)."""
    T, H, W, C = clip.shape
    Hk, Wk = H//k*k, W//k*k
    c = clip[:, :Hk, :Wk]
    small = c.reshape(T, Hk//k, k, Wk//k, k, C).mean(axis=(2, 4))
    big = np.repeat(np.repeat(small, k, axis=1), k, axis=2)
    out = clip.copy(); out[:, :Hk, :Wk] = big
    return out


def frame_augment_fast(raw, rng, strength=1.0):
    """Version RAPIDE (tout numpy) pour l'entraînement : compression par blocs +
    translation (roll) + lumière + bruit. ~100× plus rapide que le vrai JPEG."""
    x = raw.astype(np.float32).copy()
    if rng.random() < 0.8*strength:                      # compression (blocs)
        x = block_compress(x, int(rng.integers(2, 4)))
    if rng.random() < 0.7*strength:                      # mouvement (translation)
        x = np.roll(x, (int(rng.integers(-4, 5)), int(rng.integers(-4, 5))), axis=(1, 2))
    if rng.random() < 0.7*strength:                      # lumière
        x = np.sign(x)*np.abs(x)**rng.uniform(0.6, 1.6) * rng.uniform(0.7, 1.3)
    if rng.random() < 0.5*strength:                      # bruit
        x = x + rng.normal(0, 0.03*np.std(x), x.shape)
    return x.astype(np.float32)


def frame_degrade_fixed(raw):
    """Dégradation LOURDE fixe (pour l'éval 'capture dure')."""
    rng = np.random.default_rng(0)
    u8, lohi = _to_u8(raw)
    u8 = jpeg_compress(u8, q=12)
    u8 = motion_jitter(u8, rng, max_shift=4, max_rot=6)
    aug = _from_u8(u8, lohi) * 0.8
    return aug.astype(np.float32)


# ---------- niveau SIGNAL (pour CNN1D : (T, R, C)) ----------
def signal_augment(x, rng, strength=1.0):
    """x : (T, R, C) signaux de régions. Imite l'effet des captures dures."""
    x = x.astype(np.float32).copy(); T = x.shape[0]
    if rng.random() < 0.8*strength:                      # bruit (compression/capteur)
        x = x + rng.normal(0, 0.05*strength*np.std(x), x.shape)
    if rng.random() < 0.6*strength:                      # amplitude réduite (signal enfoui)
        x = x * rng.uniform(0.5, 1.0)
    if rng.random() < 0.6*strength:                      # dérive de base (lumière qui varie)
        drift = np.cumsum(rng.normal(0, 0.02*strength, T))[:, None, None]
        x = x + drift * np.std(x)
    if rng.random() < 0.5*strength:                      # quantification temporelle (bas débit)
        k = rng.integers(2, 4)
        x = np.repeat(x[::k], k, axis=0)[:T]
    return x


def signal_degrade_fixed(x):
    rng = np.random.default_rng(0); x = x.astype(np.float32).copy(); T = x.shape[0]
    x = x + rng.normal(0, 0.05*np.std(x), x.shape)
    x = x * 0.6
    x = np.repeat(x[::3], 3, axis=0)[:T]
    return x
