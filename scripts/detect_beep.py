#!/usr/bin/env python3
"""Détecte l'instant PRÉCIS d'un bip (ou clap) dans la piste audio d'une vidéo.
But : valider qu'on peut retrouver un événement de synchro à la ms dans la vidéo
du téléphone. Extrait l'audio via ffmpeg, calcule l'enveloppe d'énergie, repère
le/les onset(s), et sauvegarde une figure (onde + enveloppe + spectrogramme).

Usage: python scripts/detect_beep.py chemin/vers/video.mp4 [--fmin 800 --fmax 4000]
"""
import sys, os, subprocess, argparse, tempfile
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, spectrogram
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def extract_audio(video, sr=48000):
    wav = tempfile.mktemp(suffix='.wav')
    subprocess.run(['ffmpeg', '-y', '-i', video, '-vn', '-ac', '1', '-ar', str(sr),
                    '-loglevel', 'error', wav], check=True)
    fs, x = wavfile.read(wav); os.remove(wav)
    x = x.astype(np.float64)
    if x.ndim > 1: x = x.mean(1)
    x /= (np.abs(x).max() + 1e-9)
    return fs, x


def envelope(x, fs, win_ms=5.0):
    w = max(1, int(fs*win_ms/1000))
    return np.sqrt(np.convolve(x**2, np.ones(w)/w, mode='same'))


def find_onsets(env, fs, k=8.0, refractory_s=0.6, min_frac=0.25):
    """Onsets = franchissements montants d'un seuil robuste (médiane + k*MAD).
    Rejette les échos : période réfractaire + énergie >= min_frac du pic le plus fort."""
    med = np.median(env); mad = np.median(np.abs(env-med)) + 1e-9
    thr = med + k*mad
    above = env > thr
    rises = np.where((~above[:-1]) & (above[1:]))[0] + 1
    peak_glob = env.max()
    onsets, last = [], -1e9
    for r in rises:
        t = r/fs
        if t-last < refractory_s: continue
        if env[r:r+int(0.15*fs)].max() < med + min_frac*(peak_glob-med): continue
        # onset = DÉBUT de la montée (recul jusqu'à 10% du pic local -> attaque)
        seg = env[r:r+int(0.15*fs)]
        if len(seg) < 2: continue
        pk = seg.max(); foot = med + 0.1*(pk-med)
        j = r
        while j > 0 and env[j] > foot and r-j < int(0.05*fs): j -= 1
        onsets.append((j/fs, pk, thr)); last = t
    return onsets, thr


def dominant_freq(x, fs, t0, dur=0.08):
    a = x[int(t0*fs):int((t0+dur)*fs)]
    if len(a) < 16: return float('nan')
    f = np.fft.rfftfreq(len(a), 1/fs); P = np.abs(np.fft.rfft(a*np.hanning(len(a))))
    return float(f[np.argmax(P)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('video')
    ap.add_argument('--fmin', type=float, default=0, help='passe-bande min (Hz) si bip tonal')
    ap.add_argument('--fmax', type=float, default=0, help='passe-bande max (Hz)')
    ap.add_argument('--k', type=float, default=8.0, help='sensibilité seuil (MAD)')
    a = ap.parse_args()
    fs, x = extract_audio(a.video)
    dur = len(x)/fs
    print(f"Audio : {dur:.2f} s @ {fs} Hz")
    xf = x
    if a.fmax > a.fmin > 0:
        b, aa = butter(4, [a.fmin/(fs/2), a.fmax/(fs/2)], btype='band')
        xf = filtfilt(b, aa, x)
    env = envelope(xf, fs)
    onsets, thr = find_onsets(env, fs, k=a.k)
    print(f"\n{len(onsets)} événement(s) détecté(s) :")
    for i, (t, pk, _) in enumerate(onsets):
        fr = dominant_freq(x, fs, t)
        print(f"  #{i+1}  t = {t:8.3f} s  ({t*1000:.0f} ms)   énergie={pk:.3f}   f≈{fr:.0f} Hz")
    if len(onsets) == 1:
        print(f"\n  => 1 bip net à {onsets[0][0]*1000:.0f} ms. Précision d'onset ~ 1 frame audio ({1000/fs:.2f} ms).")
    # figure
    fig, ax = plt.subplots(3, 1, figsize=(12, 8))
    tt = np.arange(len(x))/fs
    ax[0].plot(tt, x, lw=0.4); ax[0].set_title('Forme d\'onde'); ax[0].set_ylabel('amp')
    ax[1].plot(tt, env, lw=0.6); ax[1].axhline(thr, color='r', ls='--', lw=0.8, label='seuil')
    for t, _, _ in onsets: ax[1].axvline(t, color='g', lw=1)
    ax[1].set_title('Enveloppe d\'énergie + onsets'); ax[1].legend()
    f, tg, S = spectrogram(x, fs, nperseg=1024, noverlap=768)
    ax[2].pcolormesh(tg, f, 10*np.log10(S+1e-12), shading='gouraud', cmap='magma')
    for t, _, _ in onsets: ax[2].axvline(t, color='c', lw=1)
    ax[2].set_ylim(0, 6000); ax[2].set_title('Spectrogramme'); ax[2].set_xlabel('temps (s)'); ax[2].set_ylabel('Hz')
    plt.tight_layout()
    out = os.path.splitext(a.video)[0] + '_beep.png'
    plt.savefig(out, dpi=110); print(f"\nFigure -> {out}")


if __name__ == '__main__':
    main()
