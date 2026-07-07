#!/usr/bin/env python3
"""
Émetteur de bips de synchronisation (côté laptop) — POUR TESTER LA SYNC.

Joue un bip de DÉBUT puis un bip de FIN (séparés par la durée d'enregistrement),
et sauvegarde l'instant exact de chaque bip dans une horloge de référence
(time.monotonic — la même que celle du logging PPG dans la version finale).

Le téléphone, lui, filme (vidéo + AUDIO) et capte ces bips dans sa piste audio.
Ensuite `beep_sync_detect.py` retrouve les bips dans l'audio et aligne la vidéo
sur cette horloge.

Bip = sinus court à 3 kHz (fréquence peu courante dans l'environnement → détection
robuste), encadré d'un fondu pour un transitoire net.

Usage (test) :
    python scripts/beep_sync_emit.py --duration 20 --out Data/collection/test
      → bip de début, attend 20s (tu filmes), bip de fin, sauvegarde beeps.json

    python scripts/beep_sync_emit.py --manual
      → ESPACE pour déclencher chaque bip manuellement, Q pour finir
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import sounddevice as sd

SR = 44100          # fréquence d'échantillonnage audio
BEEP_FREQ = 3000.0  # Hz
BEEP_DUR = 0.06     # s — bip COURT (transitoire net = meilleure précision de sync)
BEEP_AMP = 0.95     # amplitude FORTE (proche du max sans saturer)


def make_beep():
    t = np.arange(int(SR * BEEP_DUR)) / SR
    wave = BEEP_AMP * np.sin(2 * np.pi * BEEP_FREQ * t)
    # fondu entrée/sortie (3 ms) pour éviter les clics et garder un transitoire propre
    n_fade = int(SR * 0.003)
    env = np.ones_like(wave)
    env[:n_fade] = np.linspace(0, 1, n_fade)
    env[-n_fade:] = np.linspace(1, 0, n_fade)
    return (wave * env).astype(np.float32)


BEEP = make_beep()


def play_beep_and_timestamp():
    """Joue le bip et retourne l'instant de référence (monotonic) du déclenchement."""
    t = time.monotonic()
    sd.play(BEEP, SR)
    sd.wait()
    return t


def main():
    ap = argparse.ArgumentParser(description="Émetteur de bips de sync")
    ap.add_argument('--duration', type=float, default=20.0,
                    help="Durée d'enregistrement entre les 2 bips (mode auto)")
    ap.add_argument('--manual', action='store_true',
                    help="Mode manuel : ESPACE déclenche un bip, Q termine")
    ap.add_argument('--out', default='Data/collection/test',
                    help="Dossier de sortie (beeps.json y est écrit)")
    ap.add_argument('--subject', default='test')
    ap.add_argument('--countdown', type=float, default=None,
                    help="Compte à rebours (s) au lieu d'attendre ENTRÉE — pour lancement non interactif")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    beeps = []

    print("\n" + "=" * 60)
    print("  ÉMETTEUR DE BIPS — assure-toi que le TÉLÉPHONE FILME")
    print("  (vidéo + audio) AVANT de continuer.")
    print("=" * 60)

    if args.manual:
        print("\nESPACE = bip   |   Q = terminer\n")
        import sys, termios, tty
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while True:
                ch = sys.stdin.read(1)
                if ch == ' ':
                    t = play_beep_and_timestamp()
                    beeps.append(t - t0)
                    sys.stdout.write(f"\r  BIP #{len(beeps)} à t={t-t0:.3f}s\n")
                elif ch.lower() == 'q':
                    break
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
    else:
        if args.countdown is not None:
            print(f"\n>>> DÉMARRE LA VIDÉO SUR TON TÉLÉPHONE MAINTENANT (audio activé)")
            for s in range(int(args.countdown), 0, -1):
                print(f"    1er bip dans {s}s…", flush=True)
                time.sleep(1)
        else:
            input("\nAppuie sur ENTRÉE quand le téléphone filme et que le PPG tourne… ")
        print("\n>>> BIP DE DÉBUT")
        beeps.append(play_beep_and_timestamp() - t0)
        print(f"    enregistre pendant {args.duration:.0f}s — reste immobile, silence total…")
        time.sleep(args.duration)
        print(">>> BIP DE FIN")
        beeps.append(play_beep_and_timestamp() - t0)

    meta = {
        'subject': args.subject,
        'beep_freq_hz': BEEP_FREQ,
        'beep_dur_s': BEEP_DUR,
        'beeps_ref_s': beeps,          # instants des bips dans l'horloge de référence
        'clock': 'time.monotonic (origine t0 = début du script)',
    }
    out_path = out_dir / 'beeps.json'
    out_path.write_text(json.dumps(meta, indent=2))
    print(f"\n{len(beeps)} bip(s) sauvegardés → {out_path}")
    print("Arrête maintenant l'enregistrement du téléphone, puis lance beep_sync_detect.py.")


if __name__ == '__main__':
    main()
