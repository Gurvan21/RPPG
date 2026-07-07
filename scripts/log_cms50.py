#!/usr/bin/env python3
"""
Logger CMS50D+ autonome pour macOS (référence FC pendant une prise smartphone).
Auto-détecte le port, affiche la FC en direct, enregistre un CSV, et donne la
FC MÉDIANE à la fin (= la référence de la prise). À lancer PENDANT que tu filmes.

Usage :
    python scripts/log_cms50.py --subject 01 --take palm_1
    (Ctrl-C pour arrêter → FC médiane affichée + CSV sauvegardé)
"""
import argparse, glob, sys, time, csv
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]


def find_port():
    for pat in ("/dev/cu.usbserial*", "/dev/cu.SLAB*", "/dev/cu.wchusbserial*",
                "/dev/cu.usbmodem*", "/dev/tty.usbserial*"):
        m = glob.glob(pat)
        if m:
            return m[0]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subject', required=True)
    ap.add_argument('--take', required=True, help="ex: palm_1, face_2")
    ap.add_argument('--port', default=None)
    ap.add_argument('--out', default=str(ROOT / 'Data' / 'collection_smartphone'))
    args = ap.parse_args()
    import serial
    port = args.port or find_port()
    if not port:
        print("❌ Aucun port série détecté. Branche le CMS50D+ et installe le driver "
              "USB (CP210x/CH340). Ports vus :", glob.glob("/dev/cu.*"))
        sys.exit(1)
    try:
        ser = serial.Serial(port, baudrate=19200, timeout=1)
    except Exception as e:
        print(f"❌ {port} inaccessible ({e})"); sys.exit(1)
    print(f"✅ CMS50D+ sur {port} — enregistrement '{args.take}' (Ctrl-C pour arrêter)\n")

    subdir = Path(args.out) / f"subject_{args.subject}"; subdir.mkdir(parents=True, exist_ok=True)
    rows = []; buf = bytearray(); t0 = time.time(); last = 0
    try:
        while True:
            buf.extend(ser.read(ser.in_waiting or 1))
            while len(buf) >= 5:
                if not (buf[0] & 0x80) or (buf[1] & 0x80):
                    buf.pop(0); continue
                pkt = buf[:5]; buf = buf[5:]
                spo2 = pkt[1] & 0x7F
                bpm = ((pkt[2] & 0x40) << 1) | (pkt[3] & 0x7F)
                pleth = pkt[4] & 0x7F
                ts = time.time() - t0
                rows.append((ts, pleth, spo2, bpm))
                if time.time() - last > 0.5 and bpm > 0:      # affichage 2 Hz
                    print(f"\r  t={ts:5.1f}s   FC={bpm:3d} bpm   SpO2={spo2:3d}%   ", end='', flush=True)
                    last = time.time()
    except KeyboardInterrupt:
        pass
    if not rows:
        print("\n❌ Aucune donnée reçue (doigt bien inséré ?)"); sys.exit(1)
    bpms = np.array([r[3] for r in rows]); valid = bpms[bpms > 0]
    ref = float(np.median(valid)) if len(valid) else float('nan')
    csv_path = subdir / f"{args.take}_ppg.csv"
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['t', 'pleth', 'spo2', 'bpm']); w.writerows(rows)
    print(f"\n\n📊 Prise '{args.take}' : FC RÉFÉRENCE (médiane) = {ref:.0f} bpm  "
          f"[{len(rows)} échantillons, {rows[-1][0]:.0f}s]")
    print(f"   CSV : {csv_path}")
    print(f"   → note {ref:.0f} dans meta.json de subject_{args.subject}")


if __name__ == '__main__':
    main()
