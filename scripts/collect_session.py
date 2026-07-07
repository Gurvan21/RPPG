#!/usr/bin/env python3
"""
Outil de collecte rPPG façon VitalVideos — SANS webcam.

Le TÉLÉPHONE filme en local (vidéo + audio). Ce script, côté laptop :
  - affiche un FORMULAIRE (âge, sexe, Fitzpatrick, position, pression sys/dia…)
  - logge le PPG du CMS50D+ (ou simulation si capteur absent)
  - émet un BIP au DÉBUT et à la FIN de l'enregistrement (horodatés dans la même
    horloge que le PPG) → le téléphone capte les bips dans son audio → sync via
    scripts/beep_sync_detect.py

Sortie par sujet : Data/collection/<id>/{ppg.csv, beeps.json, meta.json}

Pré-requis capteur réel : pip install pyserial   (sinon : PPG simulé)
Lance-le dans TON terminal (fenêtre Tk) :
    python scripts/collect_session.py
"""

import csv
import json
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path

import numpy as np
import sounddevice as sd

ROOT = Path(__file__).resolve().parents[1]
SR = 44100
BEEP_FREQ = 3000.0
BEEP_DUR = 0.06       # bip COURT (transitoire net)
BEEP_AMP = 0.95       # FORT

lock = threading.Lock()
state = {
    "recording": False,
    "t_start": None,                # time.time() au démarrage de l'enregistrement
    "ppg_rows": [],                 # (t_rel, pleth, spo2, bpm)
    "ppg_latest": {"connected": False, "spo2": 0, "bpm": 0, "pleth": 0},
    "beeps_rel": [],                # instants des bips (s depuis t_start)
}


# ── Bip ────────────────────────────────────────────────────────────────────
def _make_beep():
    t = np.arange(int(SR * BEEP_DUR)) / SR
    w = BEEP_AMP * np.sin(2 * np.pi * BEEP_FREQ * t)
    nf = int(SR * 0.003)
    w[:nf] *= np.linspace(0, 1, nf); w[-nf:] *= np.linspace(1, 0, nf)
    return w.astype(np.float32)


BEEP = _make_beep()


def play_beep_logged():
    """Joue le bip et enregistre son instant (relatif à t_start) si on enregistre."""
    sd.play(BEEP, SR)
    with lock:
        if state["recording"] and state["t_start"] is not None:
            state["beeps_rel"].append(time.time() - state["t_start"])
    sd.wait()


# ── Lecteur PPG (CMS50D+ réel, sinon simulation) ────────────────────────────
def cms50_reader(port):
    try:
        import serial
        ser = serial.Serial(port, baudrate=19200, timeout=1)
        print(f"[PPG] CMS50D+ connecté sur {port}")
    except Exception as e:
        print(f"[PPG] {port or 'aucun port'} inaccessible ({e}) → SIMULATION")
        _simulate()
        return
    buf = bytearray()
    while True:
        buf.extend(ser.read(ser.in_waiting or 1))
        while len(buf) >= 5:
            if not (buf[0] & 0x80) or (buf[1] & 0x80):
                buf.pop(0); continue
            pkt = buf[:5]; buf = buf[5:]
            spo2 = pkt[1] & 0x7F
            bpm = ((pkt[2] & 0x40) << 1) | (pkt[3] & 0x7F)
            pleth = pkt[4] & 0x7F
            _push(True, spo2, bpm, pleth)


def _simulate():
    t0 = time.time()
    while True:
        t = time.time() - t0
        pleth = int(64 + 63 * np.sin(2 * np.pi * 1.1 * t))
        _push(False, 0, 0, pleth)
        time.sleep(1 / 60)


def _push(connected, spo2, bpm, pleth):
    with lock:
        state["ppg_latest"] = {"connected": connected, "spo2": spo2, "bpm": bpm, "pleth": pleth}
        if state["recording"] and state["t_start"] is not None:
            state["ppg_rows"].append((time.time() - state["t_start"], pleth, spo2, bpm))


# ── Interface (formulaire + contrôle) ───────────────────────────────────────
FIELDS = [
    ("subject",  "ID sujet",            "001"),
    ("age",      "Âge",                 ""),
    ("gender",   "Sexe (M/F)",          "M"),
    ("fitzpatrick", "Fitzpatrick (1-6)", ""),
    ("scenario", "Scénario",            "faceonly"),
    ("position", "Position",            "Assis"),
    ("location", "Lieu",                ""),
    ("environment", "Environnement",    "intérieur"),
    ("bp_sys",   "Pression systolique", ""),
    ("bp_dia",   "Pression diastolique", ""),
    ("notes",    "Notes",               ""),
]


class CollectApp:
    def __init__(self, root, port=None):
        self.root = root
        root.title("Collecte rPPG — VitalVideos style (téléphone + bips)")
        self.entries = {}

        frm = ttk.Frame(root, padding=14)
        frm.grid()
        ttk.Label(frm, text="Formulaire sujet", font=("", 14, "bold")).grid(
            row=0, column=0, columnspan=2, pady=(0, 10), sticky="w")
        for i, (key, label, default) in enumerate(FIELDS, start=1):
            ttk.Label(frm, text=label).grid(row=i, column=0, sticky="w", pady=2)
            e = ttk.Entry(frm, width=28)
            e.insert(0, default)
            e.grid(row=i, column=1, pady=2)
            self.entries[key] = e

        r = len(FIELDS) + 1
        # état PPG live
        self.ppg_lbl = ttk.Label(frm, text="PPG : —", font=("", 12))
        self.ppg_lbl.grid(row=r, column=0, columnspan=2, pady=(12, 4), sticky="w")
        # statut enregistrement
        self.status = ttk.Label(frm, text="● prêt", foreground="gray", font=("", 12, "bold"))
        self.status.grid(row=r + 1, column=0, columnspan=2, sticky="w")
        # boutons
        self.btn = ttk.Button(frm, text="DÉMARRER (bip)", command=self.toggle)
        self.btn.grid(row=r + 2, column=0, columnspan=2, pady=12, ipadx=10, ipady=6)

        self._tick()

    def _tick(self):
        with lock:
            p = state["ppg_latest"]; rec = state["recording"]
            n = len(state["ppg_rows"])
        conn = "CMS50D+" if p["connected"] else "SIMULÉ"
        self.ppg_lbl.config(text=f"PPG [{conn}] : pleth={p['pleth']:3d}  bpm={p['bpm']}  spo2={p['spo2']}")
        if rec:
            self.status.config(text=f"● ENREGISTRE — {n} échantillons", foreground="red")
        self.root.after(100, self._tick)

    def toggle(self):
        if not state["recording"]:
            self.start()
        else:
            self.stop()

    def start(self):
        if not self.entries["subject"].get().strip():
            messagebox.showerror("Erreur", "Renseigne au moins l'ID sujet."); return
        with lock:
            state["recording"] = True
            state["t_start"] = time.time()
            state["ppg_rows"] = []
            state["beeps_rel"] = []
        self.btn.config(text="ARRÊTER (bip)")
        self.status.config(text="● BIP DE DÉBUT…", foreground="red")
        threading.Thread(target=play_beep_logged, daemon=True).start()

    def stop(self):
        # bip de fin (bloquant court) puis sauvegarde
        threading.Thread(target=self._stop_worker, daemon=True).start()

    def _stop_worker(self):
        play_beep_logged()
        with lock:
            state["recording"] = False
            rows = list(state["ppg_rows"])
            beeps = list(state["beeps_rel"])
            connected = state["ppg_latest"]["connected"]
        meta = {k: e.get().strip() for k, e in self.entries.items()}
        self._save(meta, rows, beeps, connected)
        self.btn.config(text="DÉMARRER (bip)")
        self.status.config(text=f"● sauvegardé ({len(rows)} échantillons, {len(beeps)} bips)",
                           foreground="green")

    def _save(self, meta, rows, beeps, connected):
        sid = meta["subject"]
        out = ROOT / "Data" / "collection" / sid
        out.mkdir(parents=True, exist_ok=True)
        # PPG
        with open(out / "ppg.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["timestamp_s", "pleth", "spo2", "bpm_device"])
            w.writerows(rows)
        # bips (pour sync avec l'audio du téléphone)
        (out / "beeps.json").write_text(json.dumps({
            "subject": sid, "beep_freq_hz": BEEP_FREQ, "beep_dur_s": BEEP_DUR,
            "beeps_ref_s": beeps,
            "clock": "relatif au démarrage (t=0) — même horloge que ppg.csv",
        }, indent=2))
        # métadonnées formulaire
        meta["date"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        meta["ppg_device"] = "CMS50D+" if connected else "simulation"
        meta["ppg_hz_nominal"] = 60
        meta["n_ppg_samples"] = len(rows)
        (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        print(f"[OK] sujet {sid} : {len(rows)} échantillons PPG, {len(beeps)} bips → {out}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', default=None, help="Port série CMS50D+ (ex: /dev/tty.usbserial). Vide = simulation")
    args = ap.parse_args()
    threading.Thread(target=cms50_reader, args=(args.port,), daemon=True).start()
    root = tk.Tk()
    CollectApp(root, args.port)
    root.mainloop()


if __name__ == '__main__':
    main()
