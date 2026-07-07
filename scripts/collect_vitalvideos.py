#!/usr/bin/env python3
"""
Collecte rPPG façon VitalVideos — interface soignée, sortie JSON format VitalVideos.

Déroulé : lance la vidéo sur le téléphone → DÉMARRER (bip + PPG) → FIN (bip) →
le formulaire des mesures apparaît → ENREGISTRER (JSON VitalVideos + beeps.json).

Lance avec un Python doté de Tk 8.6 (sinon fenêtre vide sur macOS) :
    /opt/homebrew/bin/python3.11 scripts/collect_vitalvideos.py            # PPG simulé
    /opt/homebrew/bin/python3.11 scripts/collect_vitalvideos.py --port /dev/tty.usbserial-XXXX
"""

import os
import sys
import json
import threading
import time
import uuid

# ── Garde-fou Tk : macOS rend des fenêtres VIDES avec Tk < 8.6 (Python système).
#    Si c'est le cas, on se relance automatiquement avec un Python doté de Tk 8.6.
import tkinter as _tkcheck
if _tkcheck.TkVersion < 8.6:
    for _alt in ("/opt/homebrew/bin/python3.11", "/opt/homebrew/bin/python3.12",
                 "/usr/local/bin/python3.11"):
        if os.path.exists(_alt) and os.path.realpath(_alt) != os.path.realpath(sys.executable):
            print(f"[Tk {_tkcheck.TkVersion} trop vieux → relance avec {_alt}]")
            os.execv(_alt, [_alt] + sys.argv)
    print(f"[ATTENTION] Tk {_tkcheck.TkVersion} : risque de fenêtre vide sur macOS. "
          f"Installe Tk 8.6 :  brew install python-tk@3.11")

import tkinter as tk
from tkinter import ttk, messagebox
from collections import deque
from pathlib import Path
from datetime import datetime

import numpy as np
import sounddevice as sd

ROOT = Path(__file__).resolve().parents[1]
SR = 44100
BEEP_FREQ, BEEP_DUR, BEEP_AMP = 3000.0, 0.06, 0.95

# ── Palette ──────────────────────────────────────────────────────────────────
BG      = "#0f1720"   # fond général (anthracite)
CARD    = "#18222e"   # cartes
HEADER  = "#111b25"
ACCENT  = "#2dd4bf"   # turquoise (signal)
GREEN   = "#22c55e"
RED     = "#ef4444"
AMBER   = "#f59e0b"
TXT     = "#e5e7eb"
MUTED   = "#94a3b8"

lock = threading.Lock()
WAVE = deque(maxlen=400)   # buffer pleth pour le tracé live
state = {
    "recording": False, "t_start": None,
    "ppg_rows": [], "ppg_latest": {"connected": False, "spo2": 0, "bpm": 0, "pleth": 0},
    "beeps_rel": [],
}


def _make_beep():
    t = np.arange(int(SR * BEEP_DUR)) / SR
    w = BEEP_AMP * np.sin(2 * np.pi * BEEP_FREQ * t)
    nf = int(SR * 0.003); w[:nf] *= np.linspace(0, 1, nf); w[-nf:] *= np.linspace(1, 0, nf)
    return w.astype(np.float32)


BEEP = _make_beep()


def play_beep_logged():
    sd.play(BEEP, SR)
    with lock:
        if state["recording"] and state["t_start"] is not None:
            state["beeps_rel"].append(time.time() - state["t_start"])
    sd.wait()


def cms50_reader(port):
    try:
        import serial
        ser = serial.Serial(port, baudrate=19200, timeout=1)
        print(f"[PPG] CMS50D+ connecté sur {port}")
    except Exception as e:
        print(f"[PPG] {port or 'aucun port'} inaccessible ({e}) → SIMULATION")
        _simulate(); return
    buf = bytearray()
    while True:
        buf.extend(ser.read(ser.in_waiting or 1))
        while len(buf) >= 5:
            if not (buf[0] & 0x80) or (buf[1] & 0x80):
                buf.pop(0); continue
            pkt = buf[:5]; buf = buf[5:]
            _push(True, pkt[1] & 0x7F, ((pkt[2] & 0x40) << 1) | (pkt[3] & 0x7F), pkt[4] & 0x7F)


def _simulate():
    t0 = time.time()
    while True:
        tt = time.time() - t0
        pleth = int(64 + 55 * np.sin(2 * np.pi * 1.15 * tt) + 8 * np.sin(2 * np.pi * 2.3 * tt))
        _push(False, 97, 72, max(0, min(127, pleth)))
        time.sleep(1 / 60)


def _push(connected, spo2, bpm, pleth):
    WAVE.append(pleth)
    with lock:
        state["ppg_latest"] = {"connected": connected, "spo2": spo2, "bpm": bpm, "pleth": pleth}
        if state["recording"] and state["t_start"] is not None:
            state["ppg_rows"].append((time.time() - state["t_start"], pleth, spo2, bpm))


# Champs PATIENT (constants entre scénarios d'un même patient)
PATIENT_FIELDS = [
    ("gender", "Sexe (M/F)", "M"), ("age", "Âge", ""),
    ("fitzpatrick", "Fitzpatrick (1-6)", ""), ("location", "Lieu", ""),
    ("environment", "Environnement", "intérieur"), ("phone", "Téléphone", ""),
]
# Champs de CONDITION / MESURES (remplis après les deux enregistrements) —
# étiquetage « domaine = déploiement » : visage selfie en lumière ambiante variée.
SCENARIO_FIELDS = [
    ("site", "Site", "visage"), ("camera", "Caméra", "avant"),
    ("lighting", "Éclairage", "ambiante_bonne"), ("screen_fill", "Écran appoint", "Non"),
    ("distance_cm", "Distance (cm)", "40"), ("activity", "Activité", "repos"),
    ("position", "Position", "Assis"),
    ("lux", "Luminosité (lux)", ""),
    ("bp_sys", "Pression systolique", ""), ("bp_dia", "Pression diastolique", ""),
    ("spo2", "SpO2 (%)", ""), ("hemoglobin", "Hémoglobine (g/dL)", ""),
    ("notes", "Notes", ""),
]
FORM = PATIENT_FIELDS + SCENARIO_FIELDS

# Les deux enregistrements du protocole (repos puis pression artérielle)
REC_PLAN = [
    ("repos",    "enregistrement 1 — REPOS (60–90 s, brassard NON gonflé)"),
    ("pression", "enregistrement 2 — PRESSION (gonfler le brassard, 30–60 s)"),
]

# Champs à valeurs prédéfinies → sélecteur (combobox éditable : on peut choisir OU taper)
CHOICES = {
    "gender": ["M", "F"],
    "fitzpatrick": ["1", "2", "3", "4", "5", "6"],
    "environment": ["intérieur", "extérieur", "mixte"],
    "site": ["visage", "paume"],
    "camera": ["avant", "arriere"],
    "lighting": ["ambiante_bonne", "ambiante_moyenne", "ambiante_faible",
                 "naturelle_fenetre", "uniforme"],
    "screen_fill": ["Non", "Oui"],
    "activity": ["repos", "apres_effort"],
    "position": ["Assis", "Debout", "Allongé"],
    "phone": ["Samsung A03", "Samsung A13", "Samsung A14", "Samsung Galaxy S21",
              "Tecno Spark", "Tecno Camon", "Infinix Hot", "Infinix Note",
              "Xiaomi Redmi", "Huawei", "iPhone 11", "iPhone 13", "iPhone SE"],
}


class App:
    def __init__(self, root):
        self.root = root
        root.title("Collecte rPPG — VitalVideos")
        root.configure(bg=BG)
        root.geometry("780x1010")
        root.minsize(720, 720)
        root.resizable(True, True)
        self.entries = {}
        self.scenario_start = None
        self.recordings = []   # accumule les 2 enregistrements (repos, pression)

        st = ttk.Style(); st.theme_use("clam")
        st.configure("TFrame", background=BG)
        st.configure("Card.TFrame", background=CARD)
        st.configure("TLabel", background=CARD, foreground=TXT, font=("Helvetica", 12))
        st.configure("Muted.TLabel", background=CARD, foreground=MUTED, font=("Helvetica", 10))
        st.configure("H.TLabel", background=HEADER, foreground=TXT, font=("Helvetica", 18, "bold"))
        st.configure("Sub.TLabel", background=HEADER, foreground=MUTED, font=("Helvetica", 11))
        st.configure("Sect.TLabel", background=CARD, foreground=ACCENT, font=("Helvetica", 12, "bold"))
        st.configure("TEntry", fieldbackground="#0b1220", foreground=TXT, insertcolor=TXT)
        st.configure("Sub2.TLabel", background=CARD, foreground=MUTED,
                     font=("Helvetica", 11, "bold"))
        # Combobox sombre (sélecteur éditable)
        st.configure("Dark.TCombobox", fieldbackground="#0b1220", background="#0b1220",
                     foreground=TXT, arrowcolor=ACCENT, bordercolor="#243140",
                     lightcolor="#243140", darkcolor="#243140", padding=4)
        st.map("Dark.TCombobox",
               fieldbackground=[("readonly", "#0b1220"), ("focus", "#0b1220")],
               foreground=[("readonly", TXT)], arrowcolor=[("active", GREEN)])
        # couleurs de la liste déroulante
        root.option_add("*TCombobox*Listbox.background", "#0b1220")
        root.option_add("*TCombobox*Listbox.foreground", TXT)
        root.option_add("*TCombobox*Listbox.selectBackground", ACCENT)
        root.option_add("*TCombobox*Listbox.selectForeground", "#04201c")
        root.option_add("*TCombobox*Listbox.font", "Helvetica 11")
        st.configure("Go.TButton", font=("Helvetica", 15, "bold"), padding=12,
                     background=GREEN, foreground="#06210f", borderwidth=0)
        st.map("Go.TButton", background=[("active", "#16a34a")])
        st.configure("Stop.TButton", font=("Helvetica", 15, "bold"), padding=12,
                     background=RED, foreground="#2a0707", borderwidth=0)
        st.map("Stop.TButton", background=[("active", "#dc2626")])
        st.configure("Save.TButton", font=("Helvetica", 13, "bold"), padding=10,
                     background=ACCENT, foreground="#04201c", borderwidth=0)

        # ── Header ──
        head = tk.Frame(root, bg=HEADER); head.pack(fill="x")
        tk.Label(head, text="❤  Collecte rPPG", bg=HEADER, fg=TXT,
                 font=("Helvetica", 20, "bold")).pack(anchor="w", padx=20, pady=(14, 0))
        tk.Label(head, text="Format VitalVideos · bips de synchronisation · CMS50D+",
                 bg=HEADER, fg=MUTED, font=("Helvetica", 11)).pack(anchor="w", padx=20, pady=(0, 14))

        body = tk.Frame(root, bg=BG); body.pack(fill="both", expand=True, padx=20, pady=16)

        # ── Carte moniteur (waveform + chiffres) ──
        mon = tk.Frame(body, bg=CARD, highlightbackground="#243140", highlightthickness=1)
        mon.pack(fill="x")
        self.canvas = tk.Canvas(mon, height=120, bg="#0b1220", highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True, padx=14, pady=14)
        nums = tk.Frame(mon, bg=CARD); nums.pack(side="right", padx=18, pady=10)
        self.hr_val = tk.Label(nums, text="--", bg=CARD, fg=ACCENT, font=("Helvetica", 30, "bold"))
        self.hr_val.pack(anchor="e"); tk.Label(nums, text="HR (bpm)", bg=CARD, fg=MUTED,
                 font=("Helvetica", 10)).pack(anchor="e")
        self.spo2_val = tk.Label(nums, text="--", bg=CARD, fg=TXT, font=("Helvetica", 22, "bold"))
        self.spo2_val.pack(anchor="e", pady=(8, 0)); tk.Label(nums, text="SpO2 (%)", bg=CARD,
                 fg=MUTED, font=("Helvetica", 10)).pack(anchor="e")
        self.src_lbl = tk.Label(nums, text="—", bg=CARD, fg=MUTED, font=("Helvetica", 9))
        self.src_lbl.pack(anchor="e", pady=(8, 0))

        # ── Identifiant patient (toujours visible) ──
        pat = tk.Frame(body, bg=BG); pat.pack(fill="x", pady=(14, 0))
        tk.Label(pat, text="N° patient", bg=BG, fg=ACCENT,
                 font=("Helvetica", 12, "bold")).pack(side="left")
        self.patient_id = tk.Entry(pat, width=14, bg="#0b1220", fg=TXT, insertbackground=TXT,
                                   relief="flat", highlightbackground="#243140", highlightthickness=1,
                                   font=("Helvetica", 13))
        self.patient_id.insert(0, "P001"); self.patient_id.pack(side="left", padx=10, ipady=3)
        self.scn_lbl = tk.Label(pat, text="", bg=BG, fg=MUTED, font=("Helvetica", 11))
        self.scn_lbl.pack(side="left", padx=10)

        # ── Statut + bouton ──
        self.status = tk.Label(body, text="● prêt — lance la vidéo sur le téléphone",
                               bg=BG, fg=MUTED, font=("Helvetica", 12, "bold"))
        self.status.pack(anchor="w", pady=(10, 6))
        self.btn = ttk.Button(body, text=f"▶  DÉMARRER — {REC_PLAN[0][1]}",
                              style="Go.TButton", command=self.toggle)
        self.btn.pack(fill="x")

        # récap des enregistrements avec leur heure de début (pour retrouver la vidéo)
        self._recap_lines = []
        self.recap = tk.Label(body, text="", bg=BG, fg=ACCENT, justify="left",
                              font=("Helvetica", 11), anchor="w")
        self.recap.pack(anchor="w", fill="x", pady=(8, 0))

        # ── Carte formulaire (cachée) ──
        self.card = tk.Frame(body, bg=CARD, highlightbackground="#243140", highlightthickness=1)
        inner = tk.Frame(self.card, bg=CARD); inner.pack(fill="x", padx=18, pady=16)
        tk.Label(inner, text="Mesures de la session", bg=CARD, fg=ACCENT,
                 font=("Helvetica", 14, "bold")).grid(row=0, column=0, columnspan=6,
                                                      sticky="w", pady=(0, 10))

        NCOL = 3   # 3 colonnes de champs (étalées sur la largeur → formulaire plus court)

        def add_group(title, fields, start_row):
            ttk.Label(inner, text=title, style="Sub2.TLabel").grid(
                row=start_row, column=0, columnspan=NCOL * 2, sticky="w", pady=(8, 4))
            for i, (k, label, default) in enumerate(fields):
                r = start_row + 1 + i // NCOL
                c = (i % NCOL) * 2
                tk.Label(inner, text=label, bg=CARD, fg=MUTED, font=("Helvetica", 11)).grid(
                    row=r, column=c, sticky="w", padx=(0, 8), pady=4)
                if k in CHOICES:
                    w = ttk.Combobox(inner, values=CHOICES[k], width=13,
                                     style="Dark.TCombobox", font=("Helvetica", 11))
                    w.set(default)
                else:
                    w = tk.Entry(inner, width=14, bg="#0b1220", fg=TXT, insertbackground=TXT,
                                 relief="flat", highlightbackground="#243140",
                                 highlightthickness=1, font=("Helvetica", 11))
                    w.insert(0, default)
                w.grid(row=r, column=c + 1, pady=4, ipady=2, sticky="w", padx=(0, 12))
                self.entries[k] = w
            return start_row + 1 + (len(fields) + NCOL - 1) // NCOL

        nr = add_group("Participant et appareil", PATIENT_FIELDS, 1)
        nr = add_group("Mesures", SCENARIO_FIELDS, nr + 1)
        self.save_btn = ttk.Button(inner, text="💾  ENREGISTRER  (JSON + fiche bilan)",
                                   style="Save.TButton", command=self.save)
        self.save_btn.grid(row=nr + 1, column=0, columnspan=6, pady=(16, 0), sticky="we")

        self._tick()

    def _draw_wave(self):
        c = self.canvas; c.delete("all")
        w = c.winfo_width() or 480; h = c.winfo_height() or 120
        # grille discrète
        for gy in range(1, 4):
            c.create_line(0, h * gy / 4, w, h * gy / 4, fill="#16202b")
        data = list(WAVE)
        if len(data) < 2:
            return
        n = len(data); pts = []
        for i, v in enumerate(data):
            x = w * i / (n - 1)
            y = h - (v / 127.0) * (h - 10) - 5
            pts += [x, y]
        col = RED if state["recording"] else ACCENT
        c.create_line(*pts, fill=col, width=2, smooth=True)

    def _tick(self):
        with lock:
            p = state["ppg_latest"]; rec = state["recording"]; n = len(state["ppg_rows"])
        self.hr_val.config(text=str(p["bpm"]) if p["bpm"] else "--")
        self.spo2_val.config(text=str(p["spo2"]) if p["spo2"] else "--")
        self.src_lbl.config(text="● CMS50D+" if p["connected"] else "● SIMULÉ",
                            fg=GREEN if p["connected"] else AMBER)
        if rec:
            self.status.config(text=f"● ENREGISTREMENT — {n} échantillons", fg=RED)
        self._draw_wave()
        self.root.after(60, self._tick)

    def toggle(self):
        self.start() if not state["recording"] else self.stop()

    def start(self):
        idx = len(self.recordings)               # 0 = repos, 1 = pression
        now = datetime.now()
        if idx == 0:
            self.scenario_start = now.isoformat()
        with lock:
            state["recording"] = True; state["t_start"] = time.time()
            state["ppg_rows"] = []; state["beeps_rel"] = []
        self._rec_start_iso = now.isoformat()
        self._rec_start_clock = now.strftime("%H:%M:%S")
        # affiche l'heure de début → permet d'identifier la vidéo correspondante
        self._recap_lines.append(f"▶ Enr.{idx+1} ({REC_PLAN[idx][0]}) — début {self._rec_start_clock}")
        self.recap.config(text="\n".join(self._recap_lines))
        self.btn.config(text=f"⏹  FIN — {REC_PLAN[idx][0]} (bip)", style="Stop.TButton")
        threading.Thread(target=play_beep_logged, daemon=True).start()

    def stop(self):
        threading.Thread(target=self._stop_worker, daemon=True).start()

    def _stop_worker(self):
        play_beep_logged()
        with lock:
            state["recording"] = False
            rows = list(state["ppg_rows"]); beeps = list(state["beeps_rel"])
        idx = len(self.recordings)
        end_clock = datetime.now().strftime("%H:%M:%S")
        self.recordings.append({"name": REC_PLAN[idx][0], "rows": rows, "beeps": beeps,
                                "start": self._rec_start_iso, "start_clock": self._rec_start_clock,
                                "end_clock": end_clock})
        self._recap_lines[-1] += f" → fin {end_clock}"
        self.recap.config(text="\n".join(self._recap_lines))
        if len(self.recordings) < len(REC_PLAN):     # il reste l'enregistrement 2
            nxt = REC_PLAN[len(self.recordings)]
            self.btn.config(state="normal", text=f"▶  DÉMARRER — {nxt[1]}", style="Go.TButton")
            self.status.config(
                text=f"● {REC_PLAN[idx][0]} OK ({len(rows)} éch.) — prêt pour l'{nxt[1]}",
                fg=GREEN)
        else:                                         # les deux faits → formulaire
            self.btn.config(state="disabled", text="✓ deux enregistrements terminés")
            self.status.config(text="● remplis les mesures puis ENREGISTRER", fg=ACCENT)
            self.card.pack(fill="x", pady=(16, 0))

    def save(self):
        pid = self.patient_id.get().strip()
        if not pid:
            messagebox.showerror("Erreur", "Renseigne le N° patient."); return
        if len(self.recordings) < len(REC_PLAN):
            messagebox.showerror("Erreur", "Les deux enregistrements ne sont pas terminés."); return
        m = {k: e.get().strip() for k, e in self.entries.items()}

        out = ROOT / "Data" / "collection" / pid; out.mkdir(parents=True, exist_ok=True)
        json_path = out / f"{pid}.json"

        # ── Les DEUX enregistrements (repos + pression) dans le même scénario ──
        recordings = {}
        for i, rec in enumerate(self.recordings):
            cms = [["time", "ppg", "hr", "spo2"]] + \
                  [[int(t * 1000), int(pl), int(bp), int(sp)] for (t, pl, sp, bp) in rec["rows"]]
            block = {
                "CMS": cms, "start_time": rec["start"],
                "start_clock": rec.get("start_clock", ""), "end_clock": rec.get("end_clock", ""),
                "RGB": {"parameter": "RGB",
                        "device": {"FrameRate": None, "model": m.get("phone", ""),
                                   "source": "smartphone (local)"},
                        "filename": "", "timeseries": [], "beeps_ref_s": rec["beeps"],
                        "note": "Vidéo téléphone correspondante : voir start_clock/end_clock. "
                                "Aligner via scripts/beep_sync_detect.py"},
            }
            if rec["name"] == "pression":            # la tension est mesurée pendant l'enr. 2
                block["BP"] = {"bp_sys": m["bp_sys"], "bp_dia": m["bp_dia"]}
            recordings[f"rec{i+1}_{rec['name']}"] = block
        recordings["spo2_manual"] = m["spo2"]
        recordings["hemoglobin_gdl"] = m["hemoglobin"]

        # FC de référence = médiane des BPM valides de l'oxymètre sur l'enr. REPOS
        ref_hr = None
        for rec in self.recordings:
            if rec["name"] == "repos":
                bpms = [r[3] for r in rec["rows"] if len(r) > 3 and r[3]]
                if bpms:
                    ref_hr = float(np.median(bpms))
        scenario = {
            "scenario_data": {"site": m["site"], "camera": m["camera"],
                              "lighting": m["lighting"], "screen_fill": m["screen_fill"],
                              "distance_cm": m["distance_cm"], "activity": m["activity"],
                              "position": m["position"], "lux": m["lux"],
                              "ref_hr": ref_hr, "s_notes": m["notes"]},
            "start_time": self.scenario_start,
            "recordings": recordings,
        }

        if json_path.exists():
            vv = json.loads(json_path.read_text()); vv["scenarios"].append(scenario)
        else:
            vv = {
                "GUID": pid, "consent": "", "time": time.strftime("%H%M"),
                "participant": {"gender": m["gender"], "age": m["age"],
                                "fitzpatrick": m["fitzpatrick"], "p_notes": ""},
                "location": {"location": m["location"], "environment": m["environment"]},
                "scenarios": [scenario],
            }
        json_path.write_text(json.dumps(vv, indent=2, ensure_ascii=False))

        sc_idx = len(vv["scenarios"]) - 1
        for i, rec in enumerate(self.recordings):
            (out / f"beeps_sc{sc_idx}_rec{i+1}.json").write_text(json.dumps(
                {"patient": pid, "scenario": sc_idx, "recording": rec["name"],
                 "beep_freq_hz": BEEP_FREQ, "beep_dur_s": BEEP_DUR, "beeps_ref_s": rec["beeps"],
                 "clock": "relatif au démarrage (= time_ms/1000 du CMS)"}, indent=2))

        # ── Fiche bilan (remise au patient) ──
        fiche_path = out / f"fiche_{pid}_sc{sc_idx}.html"
        fiche_path.write_text(self._build_fiche(pid, m), encoding="utf-8")
        try:
            import webbrowser; webbrowser.open(f"file://{fiche_path}")
        except Exception:
            pass

        n1 = len(self.recordings[0]["rows"]); n2 = len(self.recordings[1]["rows"])
        print(f"[OK] {pid} sc{sc_idx} : repos {n1} éch., pression {n2} éch. → {out}")
        messagebox.showinfo("Enregistré",
                            f"Patient {pid} — scénario #{sc_idx} (2 enregistrements) enregistré.\n"
                            f"Fiche bilan : {fiche_path.name}\n\n"
                            f"Prêt pour un autre scénario, ou change le N° patient.")
        self._reset_for_next(len(vv["scenarios"]))

    def _rest_vitals(self):
        """HR et SpO2 médians de l'enregistrement de REPOS (pour la fiche)."""
        rows = self.recordings[0]["rows"]
        bpms = [bp for (_, _, _, bp) in rows if bp]
        spo2s = [sp for (_, _, sp, _) in rows if sp]
        hr = int(round(float(np.median(bpms)))) if bpms else None
        spo2 = int(round(float(np.median(spo2s)))) if spo2s else None
        return hr, spo2

    def _build_fiche(self, pid, m):
        hr, spo2_auto = self._rest_vitals()
        hr_s = str(hr) if hr else "—"
        spo2_s = m["spo2"] or (str(spo2_auto) if spo2_auto else "—")
        bp_s = f"{m['bp_sys']}/{m['bp_dia']} mmHg" if m["bp_sys"] and m["bp_dia"] else "—"
        hb_s = (m["hemoglobin"] + " g/dL") if m["hemoglobin"] else "—"
        hb_note = ""
        try:
            hbv = float(m["hemoglobin"].replace(",", "."))
            seuil = 13.0 if m["gender"].upper().startswith("M") else 12.0
            hb_note = ("possible anémie — consulter" if hbv < seuil else "dans la norme")
        except Exception:
            pass
        date_s = datetime.now().strftime("%d/%m/%Y %H:%M")
        rows_html = "".join(
            f"<tr><td>{lab}</td><td class='v'>{val}</td><td class='n'>{note}</td></tr>"
            for lab, val, note in [
                ("Fréquence cardiaque", f"{hr_s} bpm", "repos"),
                ("Saturation en oxygène (SpO₂)", f"{spo2_s} %", ""),
                ("Pression artérielle", bp_s, ""),
                ("Hémoglobine", hb_s, hb_note),
            ])
        return f"""<!doctype html><html lang="fr"><head><meta charset="utf-8">
<title>Bilan de santé — {pid}</title>
<style>
 body{{font-family:-apple-system,Helvetica,Arial,sans-serif;color:#111;max-width:620px;
   margin:32px auto;padding:0 20px}}
 h1{{font-size:22px;margin:0 0 4px}} .sub{{color:#666;font-size:13px;margin-bottom:18px}}
 table{{width:100%;border-collapse:collapse;margin-top:8px}}
 td{{padding:10px 8px;border-bottom:1px solid #e5e5e5;font-size:15px}}
 td.v{{font-weight:700;text-align:right;white-space:nowrap}}
 td.n{{color:#a15;font-size:12px;text-align:right;width:140px}}
 .meta{{font-size:13px;color:#444;margin:6px 0 14px}}
 .foot{{margin-top:22px;font-size:11px;color:#888;border-top:1px solid #eee;padding-top:10px}}
 @media print{{button{{display:none}}}}
</style></head><body>
<h1>Bilan de santé</h1>
<div class="sub">Mesures non invasives par smartphone — {date_s}</div>
<div class="meta"><b>Participant :</b> {pid} &nbsp;·&nbsp; Sexe : {m['gender'] or '—'}
 &nbsp;·&nbsp; Âge : {m['age'] or '—'} &nbsp;·&nbsp; Phototype : {m['fitzpatrick'] or '—'}</div>
<table>{rows_html}</table>
<button onclick="window.print()" style="margin-top:18px;padding:8px 16px;font-size:14px">
Imprimer</button>
<div class="foot">Ce bilan est informatif et ne constitue pas un diagnostic médical.
En cas de valeur anormale, consultez un professionnel de santé.</div>
</body></html>"""

    def _reset_for_next(self, n_scenarios):
        """Réinitialise pour enchaîner un nouveau scénario (garde le patient)."""
        self.card.pack_forget()
        self.recordings = []
        self._recap_lines = []; self.recap.config(text="")
        for k, _, default in SCENARIO_FIELDS:        # réinitialise les champs mesures
            w = self.entries[k]
            if k in CHOICES:
                w.set(default)
            else:
                w.delete(0, "end"); w.insert(0, default)
        self.btn.config(state="normal", text=f"▶  DÉMARRER — {REC_PLAN[0][1]}", style="Go.TButton")
        self.status.config(text="● prêt — scénario suivant (enr. 1 = repos)", fg=MUTED)
        self.scn_lbl.config(text=f"({n_scenarios} scénario(s) enregistré(s))")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', default=None, help="Port série CMS50D+ (vide = simulation)")
    args = ap.parse_args()
    print(f"[Démarrage] Python {sys.version.split()[0]} · Tk {tk.TkVersion} · "
          f"{'CMS50D+' if args.port else 'PPG simulé'}")
    threading.Thread(target=cms50_reader, args=(args.port,), daemon=True).start()
    root = tk.Tk()
    App(root)
    # forcer la fenêtre au premier plan (macOS l'ouvre parfois derrière)
    root.update_idletasks()
    root.lift()
    root.attributes('-topmost', True)
    root.after(400, lambda: root.attributes('-topmost', False))
    root.focus_force()
    root.mainloop()


if __name__ == '__main__':
    main()
