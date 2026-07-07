#!/usr/bin/env python3
"""
Détecteur de bips dans l'audio d'une vidéo + alignement temporel (côté sync).

1. Extrait la piste audio de la vidéo (ffmpeg).
2. Détecte les bips (sinus 3 kHz) → leurs instants dans l'horloge VIDÉO.
3. Aligne l'horloge vidéo sur l'horloge de référence (beeps.json de l'émetteur) :
     - 2 bips → offset CONSTANT + correction de DÉRIVE (étirement linéaire)
     - 1 bip  → offset constant seul (pas de correction de dérive)
4. Donne un mapping  temps_vidéo → temps_référence  + un score de qualité.

Usage (test) :
    python scripts/beep_sync_detect.py --video ma_video.mov --beeps Data/collection/test/beeps.json

Sortie : affiche l'offset, la dérive, et écrit sync.json à côté de beeps.json.
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, hilbert, correlate

BEEP_FREQ = 3000.0
BEEP_DUR = 0.06


def _beep_template(sr, freq=BEEP_FREQ, dur=BEEP_DUR):
    t = np.arange(int(sr * dur)) / sr
    w = np.sin(2 * np.pi * freq * t)
    nf = max(1, int(sr * 0.003))
    w[:nf] *= np.linspace(0, 1, nf); w[-nf:] *= np.linspace(1, 0, nf)
    return w / np.linalg.norm(w)


def extract_audio(video_path, sr=44100):
    """Extrait l'audio mono via ffmpeg → (signal float32, sr)."""
    tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    tmp.close()
    cmd = ['ffmpeg', '-y', '-i', str(video_path), '-ac', '1', '-ar', str(sr),
           '-vn', tmp.name]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    rate, data = wavfile.read(tmp.name)
    Path(tmp.name).unlink(missing_ok=True)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    if np.abs(data).max() > 0:
        data /= np.abs(data).max()
    return data, rate


def detect_beeps(audio, sr, freq=BEEP_FREQ, dur=BEEP_DUR, min_gap_s=2.0):
    """
    Détection par FILTRE ADAPTÉ : on corrèle l'audio avec la forme EXACTE du bip
    (sinus fenêtré). Le pic de corrélation pointe l'instant du bip au sample
    près, de façon CONSTANTE d'un bip à l'autre (contrairement à un pic d'enveloppe).
    Retourne la liste des instants (s).
    """
    nyq = sr / 2
    b, a = butter(4, [(freq - 200) / nyq, (freq + 200) / nyq], btype='band')
    filt = filtfilt(b, a, audio)
    tmpl = _beep_template(sr, freq, dur)
    corr = np.abs(correlate(filt, tmpl, mode='valid'))   # pic = onset du bip
    corr = corr / (corr.max() + 1e-12)

    thr = max(0.3, corr.mean() + 6 * corr.std())
    min_gap = int(sr * min_gap_s)
    beeps, i, n = [], 0, len(corr)
    while i < n:
        if corr[i] > thr:
            j = min(n, i + min_gap)
            peak = i + int(np.argmax(corr[i:j]))         # onset précis du bip
            beeps.append((peak / sr, float(corr[peak])))
            i = peak + min_gap
        else:
            i += 1
    return beeps, corr, thr


def main():
    ap = argparse.ArgumentParser(description="Détection bips + alignement")
    ap.add_argument('--video', required=True)
    ap.add_argument('--beeps', required=True, help="beeps.json produit par beep_sync_emit.py")
    args = ap.parse_args()

    ref = json.loads(Path(args.beeps).read_text())
    ref_beeps = ref['beeps_ref_s']
    # fréquence/durée du bip lues depuis le json → le détecteur reste cohérent
    # avec l'émetteur même si on change le bip
    freq = float(ref.get('beep_freq_hz', BEEP_FREQ))
    dur = float(ref.get('beep_dur_s', BEEP_DUR))
    print(f"Référence : {len(ref_beeps)} bip(s) à {[round(x,3) for x in ref_beeps]} s "
          f"(bip {freq:.0f}Hz / {dur*1000:.0f}ms)")

    print("Extraction audio…")
    audio, sr = extract_audio(args.video, sr=44100)
    print(f"  {len(audio)/sr:.1f}s d'audio @ {sr} Hz")

    found, env, thr = detect_beeps(audio, sr, freq=freq, dur=dur)
    print(f"\nBips détectés dans la vidéo : {len(found)}")
    for k, (t, amp) in enumerate(found):
        print(f"  bip #{k+1} : t_vidéo = {t:.3f}s  (amplitude {amp:.2f})")

    if len(found) < len(ref_beeps):
        print("\n⚠️  Moins de bips détectés que prévu — vérifie le volume / le bruit ambiant.")
        print("   (sync impossible de façon fiable)")
        return

    vid_t = [t for t, _ in found[:len(ref_beeps)]]

    # ── Alignement ──
    if len(ref_beeps) >= 2:
        # offset + dérive : temps_ref = a * temps_vidéo + b  (régression sur les bips)
        a = (ref_beeps[-1] - ref_beeps[0]) / (vid_t[-1] - vid_t[0])
        b = ref_beeps[0] - a * vid_t[0]
        drift_ppm = (a - 1.0) * 1e6
        print(f"\n=== ALIGNEMENT (2 ancrages) ===")
        print(f"  temps_référence = {a:.6f} × temps_vidéo + {b:+.3f}")
        print(f"  offset au début : {b*1000:+.0f} ms")
        print(f"  dérive d'horloge : {drift_ppm:+.0f} ppm "
              f"({'OK' if abs(drift_ppm) < 5000 else 'ÉLEVÉE — vérifier'})")
        # contrôle : l'intervalle vidéo doit ≈ l'intervalle référence
        di_ref = ref_beeps[-1] - ref_beeps[0]
        di_vid = vid_t[-1] - vid_t[0]
        print(f"  intervalle bips : référence={di_ref:.3f}s  vidéo={di_vid:.3f}s  "
              f"écart={abs(di_ref-di_vid)*1000:.0f} ms")
        quality = "BON ✅" if abs(di_ref - di_vid) < 0.1 else "À VÉRIFIER ⚠️"
    else:
        a, b = 1.0, ref_beeps[0] - vid_t[0]
        print(f"\n=== ALIGNEMENT (1 ancrage) ===")
        print(f"  offset constant : {b*1000:+.0f} ms (pas de correction de dérive)")
        quality = "OFFSET SEUL"

    print(f"\n  Qualité sync : {quality}")
    print(f"  → pour convertir un instant vidéo t en horloge PPG : t_ppg = {a:.6f}*t {b:+.3f}")

    sync = {'video': str(args.video), 'scale_a': a, 'offset_b': b,
            'video_beeps_s': vid_t, 'ref_beeps_s': ref_beeps, 'quality': quality}
    out = Path(args.beeps).with_name('sync.json')
    out.write_text(json.dumps(sync, indent=2))
    print(f"\nMapping sauvegardé → {out}")


if __name__ == '__main__':
    main()
