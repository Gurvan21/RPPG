#!/usr/bin/env python3
"""
Collecte synchronisée : caméra téléphone (IP Webcam) + CMS50D+ PPG.

Le téléphone filme et stream via WiFi → le laptop capture, horodatage et sauvegarde.
Même horloge pour vidéo et PPG → synchronisation automatique.

Usage:
    python collect_rppg_data.py --subject 001 --camera "http://192.168.1.34:8080/video"
    python collect_rppg_data.py --subject 001 --camera "http://192.168.1.34:8080/video" --port /dev/ttyUSB0

Contrôle :
    ESPACE  → démarrer / arrêter l'enregistrement
    Q       → quitter
"""

import argparse
import csv
import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# CMS50D+ protocol  (19200 baud, 5-byte packets at ~60 Hz)
# Byte 0: 0x80 | flags   Byte 1: SpO2   Byte 2: PR high   Byte 3: PR low
# Byte 4: plethysmogram (0-127)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
state = {
    "recording":   False,
    "start_time":  None,
    "ppg_rows":    [],        # (timestamp, pleth, spo2, bpm) buffered during recording
    "ppg_latest":  {"connected": False, "spo2": 0, "bpm": 0, "pleth": 0},
    "out_dir":     None,
    "session_ts":  None,
}
lock = threading.Lock()


# ---------------------------------------------------------------------------
# CMS50D+ reader thread
# ---------------------------------------------------------------------------
def cms50_reader(port: str):
    try:
        import serial
        ser = serial.Serial(port, baudrate=19200, timeout=1)
        print(f"[PPG] CMS50D+ connecté sur {port}")
    except Exception as e:
        print(f"[PPG] {port} inaccessible ({e}) → simulation")
        _cms50_simulate()
        return

    buf = bytearray()
    while True:
        chunk = ser.read(ser.in_waiting or 1)
        buf.extend(chunk)
        while len(buf) >= 5:
            if not (buf[0] & 0x80) or (buf[1] & 0x80):
                buf.pop(0)
                continue
            pkt   = buf[:5]; buf = buf[5:]
            spo2  = pkt[1] & 0x7F
            bpm   = ((pkt[2] & 0x40) << 1) | (pkt[3] & 0x7F)
            pleth = pkt[4] & 0x7F
            ts    = time.time()
            with lock:
                state["ppg_latest"] = {"connected": True, "spo2": spo2,
                                        "bpm": bpm, "pleth": pleth}
                if state["recording"]:
                    state["ppg_rows"].append((ts, pleth, spo2, bpm))


def _cms50_simulate():
    t0 = time.time()
    while True:
        t = time.time() - t0
        pleth = int(64 + 63 * np.sin(2 * np.pi * 1.1 * t))
        with lock:
            state["ppg_latest"] = {"connected": False, "spo2": 0,
                                    "bpm": 0, "pleth": pleth}
            if state["recording"]:
                state["ppg_rows"].append((time.time(), pleth, 0, 0))
        time.sleep(1 / 60)


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------
def _save_session():
    with lock:
        rows      = state["ppg_rows"].copy()
        out_dir   = state["out_dir"]
        ts_str    = state["session_ts"]
        state["ppg_rows"] = []

    if not out_dir or not ts_str:
        return

    # PPG CSV
    csv_path = out_dir / f"ppg_{ts_str}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp_s", "pleth", "spo2", "bpm_device"])
        w.writerows(rows)
    print(f"[PPG] {len(rows)} échantillons → {csv_path.name}")

    # Meta JSON
    meta = {
        "subject":       args.subject,
        "date":          ts_str,
        "ppg_device":    "CMS50D+" if state["ppg_latest"]["connected"] else "simulation",
        "ppg_hz":        60,
        "camera_source": args.camera,
        "n_ppg_samples": len(rows),
        "notes":         args.notes,
    }
    with open(out_dir / f"meta_{ts_str}.json", "w") as f:
        json.dump(meta, f, indent=2)


# ---------------------------------------------------------------------------
# Camera capture + display + recording thread
# ---------------------------------------------------------------------------
def _open_cap(src):
    is_url = isinstance(src, str)
    while True:
        if is_url:
            cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            cap = cv2.VideoCapture(src)
        if cap.isOpened():
            return cap
        print(f"[CAM] Stream non disponible, nouvel essai dans 2 s…")
        cap.release(); time.sleep(2)


def camera_loop(src, fps: float):
    cap        = _open_cap(src)
    w          = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h          = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    print(f"[CAM] {w}×{h} @ {actual_fps:.1f} fps")

    writer     = None
    frame_ts   = []
    out_path   = None

    cv2.namedWindow("rPPG Collecte  [ESPACE=Rec  Q=Quitter]", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("rPPG Collecte  [ESPACE=Rec  Q=Quitter]", 800, 450)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[CAM] Frame perdue — reconnexion…")
            cap.release(); time.sleep(1); cap = _open_cap(src)
            continue

        ts = time.time()
        with lock:
            recording = state["recording"]

        # ── Recording logic ──────────────────────────────────────────────
        if recording:
            if writer is None:
                out_dir = Path(args.output) / f"subject_{args.subject}"
                out_dir.mkdir(parents=True, exist_ok=True)
                ts_str  = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_path = out_dir / f"video_{ts_str}.avi"
                fourcc  = cv2.VideoWriter_fourcc(*"MJPG")
                writer  = cv2.VideoWriter(str(out_path), fourcc, actual_fps, (w, h))
                frame_ts = []
                with lock:
                    state["out_dir"]    = out_dir
                    state["session_ts"] = ts_str
                print(f"[CAM] Enregistrement → {out_path.name}")
            writer.write(frame)
            frame_ts.append(ts)
        else:
            if writer is not None:
                writer.release()
                np.save(str(out_path.with_suffix(".timestamps.npy")), np.array(frame_ts))
                print(f"[CAM] {len(frame_ts)} frames → {out_path.name}")
                _save_session()
                writer = None; out_path = None; frame_ts = []

        # ── Overlay ──────────────────────────────────────────────────────
        disp = frame.copy()
        with lock:
            ppg = state["ppg_latest"]

        if recording:
            elapsed = time.time() - state["start_time"]
            cv2.circle(disp, (30, 30), 12, (0, 0, 220), -1)
            cv2.putText(disp, f"REC  {elapsed:.1f}s", (50, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 220), 2)
        else:
            cv2.putText(disp, "ESPACE = Démarrer", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 0), 2)

        ppg_txt = (f"PPG: {ppg['pleth']}  SpO2: {ppg['spo2']}%  {ppg['bpm']} bpm"
                   if ppg["connected"] else "PPG: simulation (pas d'oxymètre)")
        cv2.putText(disp, ppg_txt, (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)

        cv2.imshow("rPPG Collecte  [ESPACE=Rec  Q=Quitter]", disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            if writer:
                writer.release()
                np.save(str(out_path.with_suffix(".timestamps.npy")), np.array(frame_ts))
                _save_session()
            break
        elif key == ord(' '):
            with lock:
                state["recording"] = not state["recording"]
                if state["recording"]:
                    state["start_time"] = time.time()
                    print("[CAM] ● Enregistrement démarré")
                else:
                    print("[CAM] ■ Enregistrement arrêté")

    cap.release()
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collecte rPPG : téléphone + CMS50D+")
    parser.add_argument("--subject", default="001")
    parser.add_argument("--port",   default="/dev/ttyUSB0",
                        help="Port série CMS50D+ (ex: /dev/ttyUSB0)")
    parser.add_argument("--camera", default="0",
                        help="URL IP Webcam (ex: http://192.168.1.34:8080/video) "
                             "ou index webcam locale (0)")
    parser.add_argument("--fps",    type=float, default=30.0)
    parser.add_argument("--output", default="Data/collection")
    parser.add_argument("--notes",  default="")
    args = parser.parse_args()

    # Résoudre source caméra
    cam_src = args.camera
    try:
        cam_src = int(args.camera)
    except ValueError:
        pass

    # CMS50D+ thread
    threading.Thread(target=cms50_reader, args=(args.port,), daemon=True).start()

    # Caméra + affichage (main thread pour OpenCV)
    print("\nContrôle :")
    print("  ESPACE → démarrer / arrêter l'enregistrement")
    print("  Q      → quitter\n")
    camera_loop(cam_src, args.fps)
