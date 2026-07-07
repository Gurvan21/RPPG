#!/usr/bin/env python3
"""
Serveur rPPG — reçoit les frames vidéo en temps réel via WebSocket.
Extraction RGB identique à ExtractionRGB.py (MediaPipe Python, BiSeNet, même algo).
Endpoint /analyse lance AnalyseWebRPPG.py sur le CSV produit.

Dépendances :
    pip install fastapi uvicorn websockets torch torchvision pillow numpy pandas opencv-python-headless scipy mediapipe

Lancement :
    python server.py
    # puis ouvrir http://localhost:8000 dans le navigateur
"""

import asyncio
import json
import queue
import sys
import threading
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
import uvicorn

# ── Device (GPU/MPS/CPU) — identique à ExtractionRGB.py ──────────
_device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else
    "cpu"
)
print(f"[Device] {_device}")

# ── Chemins ───────────────────────────────────────────────────────
HERE        = Path(__file__).parent
MODEL_PATH  = HERE / "79999_iter.pth"
HTML_PATH   = HERE / "rppg_live.html"
OUT_CSV     = HERE / "rppg_rgb.csv"
OUT_META    = HERE / "rppg_rgb_meta.json"
ANALYSE_DIR = HERE.parent / "semaine 1"

# ================================================================
# EXTRACTION — copie conforme d'ExtractionRGB.py
# ================================================================

# ── FaceMesh (identique à ExtractionRGB.py) ──────────────────────
_mp_face_mesh = mp.solutions.face_mesh
_face_mesh = _mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# ── BiSeNet ───────────────────────────────────────────────────────
_bisenet_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_bisenet():
    sys.path.insert(0, str(HERE))
    from model import BiSeNet  # type: ignore
    net = BiSeNet(n_classes=19)
    if MODEL_PATH.exists():
        net.load_state_dict(torch.load(str(MODEL_PATH), map_location="cpu"))
        print(f"[BiSeNet] Poids chargés depuis {MODEL_PATH}")
    else:
        print(f"[BiSeNet] ATTENTION : {MODEL_PATH} introuvable — masque désactivé")
    net.to(_device)
    net.eval()
    return net

def face_parsing_mask(bisenet, image_bgr):
    """Identique à ExtractionRGB.py — classes 1 (skin) et 10 (nose)."""
    h, w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    inp = cv2.resize(rgb, (512, 512))
    tensor = _bisenet_transform(inp).unsqueeze(0).to(_device)
    with torch.no_grad():
        out = bisenet(tensor)[0]
    parsing = out.squeeze(0).argmax(0).cpu().numpy().astype(np.uint8)
    parsing = cv2.resize(parsing, (w, h), interpolation=cv2.INTER_NEAREST)
    return np.where((parsing == 1) | (parsing == 10), np.uint8(255), np.uint8(0))

# ── Masques polygone (identique à ExtractionRGB.py) ──────────────
def polygon_mask(shape, landmarks, indices):
    h, w = shape[:2]
    pts = np.array(
        [[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in indices],
        dtype=np.int32,
    )
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, pts, 255)
    return mask

_ERODE_KERNEL = np.ones((5, 5), np.uint8)

FRONT       = [103, 67, 109, 10, 338, 297, 332, 333, 168, 104]
LEFT_CHEEK  = [187, 214, 211, 57, 216, 203, 101, 118, 117, 123]
RIGHT_CHEEK = [411, 434, 431, 287, 436, 423, 330, 347, 346, 352]
NOSE        = [1, 45, 134, 174, 197, 399, 363, 275]
LEFT_EYE    = [24, 23, 22, 121, 47, 100, 119, 228]
RIGHT_EYE   = [252, 253, 254, 448, 348, 329, 277, 350]

# ── Extraction RGB avec filtre outliers (identique à ExtractionRGB.py) ──
def region_rgb(frame_bgr, mask, parsing_mask, sigma=2.0):
    final_mask = cv2.bitwise_and(mask, parsing_mask)
    pixels = frame_bgr[final_mask > 0].astype(np.float32)

    if len(pixels) < 50:
        return np.nan, np.nan, np.nan

    keep = np.ones(len(pixels), dtype=bool)
    for c in range(3):
        med = np.median(pixels[:, c])
        std = np.std(pixels[:, c])
        if std > 0:
            keep &= np.abs(pixels[:, c] - med) <= sigma * std
    pixels = pixels[keep]

    if len(pixels) < 50:
        return np.nan, np.nan, np.nan

    b = np.mean(pixels[:, 0])
    g = np.mean(pixels[:, 1])
    r = np.mean(pixels[:, 2])
    return r, g, b

def extract_frame(bisenet, frame_bgr):
    """
    Traite une frame : MediaPipe Python + BiSeNet + extraction RGB.
    Retourne un dict {roi: {r,g,b}} ou None si pas de visage détecté.
    Identique au pipeline ExtractionRGB.py.
    """
    rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = _face_mesh.process(rgb_frame)

    if not results.multi_face_landmarks:
        return None

    lm = results.multi_face_landmarks[0].landmark
    parsing = face_parsing_mask(bisenet, frame_bgr)

    mask_front  = polygon_mask(frame_bgr.shape, lm, FRONT)
    mask_left   = cv2.erode(polygon_mask(frame_bgr.shape, lm, LEFT_CHEEK),  _ERODE_KERNEL, iterations=1)
    mask_right  = cv2.erode(polygon_mask(frame_bgr.shape, lm, RIGHT_CHEEK), _ERODE_KERNEL, iterations=1)
    mask_nose   = polygon_mask(frame_bgr.shape, lm, NOSE)
    mask_lEye   = polygon_mask(frame_bgr.shape, lm, LEFT_EYE)
    mask_rEye   = polygon_mask(frame_bgr.shape, lm, RIGHT_EYE)

    rf,  gf,  bf  = region_rgb(frame_bgr, mask_front,  parsing)
    rl,  gl,  bl  = region_rgb(frame_bgr, mask_left,   parsing)
    rr,  gr,  br  = region_rgb(frame_bgr, mask_right,  parsing)
    rn,  gn,  bn  = region_rgb(frame_bgr, mask_nose,   parsing)
    rsg, gsg, bsg = region_rgb(frame_bgr, mask_lEye,   parsing)
    rsd, gsd, bsd = region_rgb(frame_bgr, mask_rEye,   parsing)

    return {
        "front":            {"r": rf,  "g": gf,  "b": bf},
        "joue_gauche":      {"r": rl,  "g": gl,  "b": bl},
        "joue_droite":      {"r": rr,  "g": gr,  "b": br},
        "nez":              {"r": rn,  "g": gn,  "b": bn},
        "sous_oeil_gauche": {"r": rsg, "g": gsg, "b": bsg},
        "sous_oeil_droit":  {"r": rsd, "g": gsd, "b": bsd},
    }

ROI_NAMES = ["front", "joue_gauche", "joue_droite", "nez", "sous_oeil_gauche", "sous_oeil_droit"]

# ── Gestionnaire de session WebSocket ────────────────────────────
class Session:
    def __init__(self):
        self.frame_queue: queue.Queue = queue.Queue()
        self.rows: list = []
        self.done = threading.Event()
        self.fps: float = 30.0
        self.bisenet = None
        self.worker = None
        self.frame_count: int = 0
        self.ws_send_cb = None
        self._t_first_frame = None
        self._t_last_frame  = None

    def start(self, bisenet, fps: float, ws_send_cb):
        self.bisenet = bisenet
        self.fps = fps
        self.ws_send_cb = ws_send_cb
        self.done.clear()
        self.rows = []
        self.frame_count = 0
        self._t_first_frame = None
        self._t_last_frame  = None
        self.worker = threading.Thread(target=self._process_loop, daemon=True)
        self.worker.start()
        print(f"[Session] Démarrée — fps client={fps}")

    def push_frame(self, jpeg_bytes: bytes):
        self.frame_queue.put(jpeg_bytes)

    def end(self):
        self.frame_queue.put(None)
        print("[Session] Signal END reçu — attente fin du worker…")

    def _process_loop(self):
        while True:
            item = self.frame_queue.get()
            if item is None:
                break

            t_now = time.monotonic()
            if self._t_first_frame is None:
                self._t_first_frame = t_now
            self._t_last_frame = t_now

            nparr = np.frombuffer(item, np.uint8)
            frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame_bgr is None:
                continue

            rgb = extract_frame(self.bisenet, frame_bgr)
            if rgb is None:
                # pas de visage — ligne NaN comme ExtractionRGB.py
                row = {"frame": self.frame_count}
                for name in ROI_NAMES:
                    row[f"{name}_r"] = np.nan
                    row[f"{name}_g"] = np.nan
                    row[f"{name}_b"] = np.nan
            else:
                row = {"frame": self.frame_count}
                for name in ROI_NAMES:
                    row[f"{name}_r"] = rgb[name]["r"]
                    row[f"{name}_g"] = rgb[name]["g"]
                    row[f"{name}_b"] = rgb[name]["b"]

            self.rows.append(row)
            self.frame_count += 1

            if self.frame_count % 30 == 0:
                print(f"[Session] {self.frame_count} frames traitées")

        # FPS réel mesuré côté serveur
        if self._t_first_frame is not None and self._t_last_frame is not None and self.frame_count > 1:
            elapsed = self._t_last_frame - self._t_first_frame
            measured = (self.frame_count - 1) / elapsed
            print(f"[Session] FPS client={self.fps:.2f} → FPS mesuré={measured:.2f}")
            self.fps = measured

        self._export()
        self.done.set()
        print("[Session] Terminée.")

    def _export(self):
        df = pd.DataFrame(self.rows)
        cols = ["frame"]
        for name in ROI_NAMES:
            cols += [f"{name}_r", f"{name}_g", f"{name}_b"]
        df = df[cols]
        df.to_csv(OUT_CSV, index=False)
        meta = {
            "fps": self.fps,
            "frames_total": self.frame_count,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        OUT_META.write_text(json.dumps(meta, indent=2))
        print(f"[Session] Exporté : {OUT_CSV} ({self.frame_count} frames)")


# ── App FastAPI ───────────────────────────────────────────────────
from typing import Optional

app = FastAPI()
bisenet_model = None
active_session: Optional[Session] = None


@app.on_event("startup")
def startup():
    global bisenet_model
    bisenet_model = load_bisenet()


@app.get("/")
def index():
    if HTML_PATH.exists():
        return FileResponse(str(HTML_PATH))
    return HTMLResponse("<h1>rppg_live.html introuvable</h1>")


@app.get("/download/{filename}")
def download(filename: str):
    path = HERE / filename
    if path.exists():
        return FileResponse(str(path), filename=filename)
    return HTMLResponse("Fichier non trouvé", status_code=404)


@app.get("/analyse")
def analyse_endpoint():
    if not OUT_CSV.exists() or not OUT_META.exists():
        return JSONResponse({"error": "CSV introuvable — lancez d'abord une capture"}, status_code=404)
    try:
        sys.path.insert(0, str(ANALYSE_DIR))
        from AnalyseWebRPPG import analyse  # type: ignore
        df = pd.read_csv(OUT_CSV)
        meta = json.loads(OUT_META.read_text())
        df_results, bpm_final, votes, snr_moyen, coh_moyen = analyse(df, meta["fps"])
        return JSONResponse({
            "bpm_consensus": bpm_final,
            "bpm_votes": votes,
            "bpm_snr": snr_moyen,
            "bpm_coherence": coh_moyen,
            "roi_results": df_results.to_dict(orient="records"),
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    global active_session
    await ws.accept()
    print("[WS] Client connecté")

    session = Session()
    active_session = session
    loop = asyncio.get_event_loop()

    try:
        while True:
            msg = await ws.receive()

            if "text" in msg:
                data = json.loads(msg["text"])

                if data.get("type") == "START":
                    fps = float(data.get("fps", 30))
                    session.start(bisenet_model, fps, None)
                    await ws.send_text(json.dumps({"status": "started"}))

                elif data.get("type") == "END":
                    session.end()
                    await loop.run_in_executor(None, session.done.wait, 600)
                    await ws.send_text(json.dumps({
                        "status": "done",
                        "frames": session.frame_count,
                        "csv_url": "/download/rppg_rgb.csv",
                        "meta_url": "/download/rppg_rgb_meta.json",
                    }))

            elif "bytes" in msg:
                session.push_frame(msg["bytes"])
                await ws.send_text(json.dumps({
                    "status": "processing",
                    "queued": session.frame_queue.qsize(),
                    "done": session.frame_count,
                }))

    except WebSocketDisconnect:
        print("[WS] Client déconnecté")
    except Exception as e:
        print(f"[WS] Erreur : {e}")


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
