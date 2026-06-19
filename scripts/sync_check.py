#!/usr/bin/env python3
"""
Vérifie et visualise la synchronisation entre vidéo et PPG après collecte.

Usage:
    python sync_check.py Data/collection/subject_001/
"""

import sys
import json
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def load_session(folder: Path):
    """Load most recent video+ppg+meta files from a subject folder."""
    # Find files
    videos = sorted(folder.glob("video_*.avi"))
    ppgs   = sorted(folder.glob("ppg_*.csv"))
    tss    = sorted(folder.glob("video_*.timestamps.npy"))
    metas  = sorted(folder.glob("meta_*.json"))

    if not videos:
        raise FileNotFoundError(f"Aucune vidéo dans {folder}")
    if not ppgs:
        raise FileNotFoundError(f"Aucun CSV PPG dans {folder}")

    vid_path = videos[-1]
    ppg_path = ppgs[-1]
    ts_path  = tss[-1] if tss else None
    meta     = json.load(open(metas[-1])) if metas else {}

    print(f"Vidéo    : {vid_path.name}")
    print(f"PPG      : {ppg_path.name}")
    print(f"Sujet    : {meta.get('subject', '?')}")
    print(f"Date     : {meta.get('date', '?')}")

    # Load PPG
    ppg_ts, pleth = [], []
    with open(ppg_path) as f:
        for row in csv.DictReader(f):
            ppg_ts.append(float(row["timestamp_s"]))
            pleth.append(int(row["pleth"]))
    ppg_ts = np.array(ppg_ts)
    pleth  = np.array(pleth, dtype=np.float32)

    # Load frame timestamps
    if ts_path and ts_path.exists():
        frame_ts = np.load(str(ts_path))
    else:
        import cv2
        cap = cv2.VideoCapture(str(vid_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        # Reconstruct timestamps from PPG start (approximate)
        t0 = ppg_ts[0]
        frame_ts = t0 + np.arange(n) / fps
        print("  [WARN] Pas de timestamps.npy — reconstruction approximative depuis fps")

    return ppg_ts, pleth, frame_ts, meta


def analyze(folder: str):
    folder = Path(folder)
    ppg_ts, pleth, frame_ts, meta = load_session(folder)

    # Align to common t=0
    t0       = min(ppg_ts[0], frame_ts[0])
    ppg_t    = ppg_ts  - t0
    frame_t  = frame_ts - t0
    duration = max(ppg_t[-1], frame_t[-1])

    # PPG actual sample rate
    dt       = np.diff(ppg_t)
    ppg_hz   = 1.0 / np.median(dt)
    jitter   = dt.std() * 1000  # ms

    print(f"\n--- Diagnostic synchronisation ---")
    print(f"Durée vidéo  : {frame_t[-1]:.2f} s  ({len(frame_t)} frames)")
    print(f"Durée PPG    : {ppg_t[-1]:.2f}  s  ({len(pleth)} échantillons)")
    print(f"PPG réel     : {ppg_hz:.1f} Hz  (jitter={jitter:.1f} ms)")
    print(f"Décalage t0  : {(frame_ts[0]-ppg_ts[0])*1000:.1f} ms  (vidéo − PPG)")
    print(f"Décalage fin : {(frame_ts[-1]-ppg_ts[-1])*1000:.1f} ms")

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    fig.suptitle(f"Sujet {meta.get('subject','?')} — Synchronisation collecte", fontsize=12)

    # Timeline vidéo vs PPG
    ax = axes[0]
    ax.eventplot([frame_t[::5]], lineoffsets=1.5, linelengths=0.8,
                 colors='steelblue', alpha=0.4, label='Trames vidéo (×5)')
    ax.eventplot([ppg_t[::10]], lineoffsets=0.5, linelengths=0.8,
                 colors='tomato', alpha=0.4, label='Échantillons PPG (×10)')
    ax.set_xlim(0, min(duration, 10))
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(['PPG', 'Vidéo'])
    ax.set_xlabel("Temps (s)")
    ax.set_title("Timeline — 10 premières secondes")
    ax.legend(loc='upper right', fontsize=8)

    # Signal PPG pleth
    ax = axes[1]
    ax.plot(ppg_t, pleth, color='tomato', lw=0.8)
    ax.set_ylabel("Pleth (0–127)")
    ax.set_xlabel("Temps (s)")
    ax.set_title("Signal PPG pléthysmogramme")

    # Jitter distribution
    ax = axes[2]
    ax.hist(dt * 1000, bins=50, color='steelblue', edgecolor='none')
    ax.axvline(1000/60, color='red', ls='--', label=f'Théorique {1000/60:.1f} ms (60 Hz)')
    ax.set_xlabel("Intervalle entre échantillons PPG (ms)")
    ax.set_ylabel("Nombre")
    ax.set_title("Distribution du jitter PPG")
    ax.legend(fontsize=9)

    plt.tight_layout()
    out = folder / "sync_report.png"
    plt.savefig(str(out), dpi=120)
    print(f"\n[Graphique] {out}")
    plt.show()


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "Data/collection/subject_001"
    analyze(folder)
