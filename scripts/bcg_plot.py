#!/usr/bin/env python3
"""Teste le rBCG (micro-mouvement de tête) sur une vidéo et trace signal + spectre."""
import os, sys
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
from pathlib import Path
import numpy as np
from scipy.signal import periodogram
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.preextract_clips import load_video, resample_to_fps, track_face_bboxes
from mp_rppg.bcg import bcg_hr
from mp_rppg.metrics import hr_from_fft, snr, hr_candidates

video = sys.argv[1]; stem = Path(video).stem
frames, fps = load_video(video, max_dim=720)
if fps > 32:
    ft = np.arange(len(frames))*1000.0/fps; frames, _ = resample_to_fps(frames, ft, 30.0); fps = 30.0
bb = track_face_bboxes(frames)
bbox = tuple(np.median(np.array(bb), axis=0).astype(int)) if len(bb) else None
hr, sn, sig = bcg_hr(frames, fps, bbox=bbox)
print(f"{Path(video).name}  {len(frames)}f @{fps:.1f}  →  rBCG HR={hr:.1f} bpm  SNR={sn:+.1f}")
if sig is None:
    print("   (pas de signal BCG exploitable)"); sys.exit()
cands, ambig = hr_candidates(sig, fps)
print("   candidats: " + ", ".join(f"{h:.0f}({r*100:.0f}%)" for h, r in cands[:3]) + (f"  ⚠️AMBIGU" if ambig else ""))

t = np.arange(len(sig))/fps
nf = 1
while nf < len(sig): nf *= 2
f, px = periodogram(sig, fs=fps, nfft=nf, detrend=False); fr = f*60; b = (fr >= 40) & (fr <= 180)
fig, ax = plt.subplots(2, 1, figsize=(11, 6))
ax[0].plot(t, sig, color='teal', lw=1.0)
ax[0].set_title(f"rBCG (micro-mouvement tête) — {Path(video).name}  |  HR={hr:.0f} bpm  SNR={sn:+.1f} dB")
ax[0].set_xlabel("temps (s)"); ax[0].set_ylabel("déplacement (rel.)"); ax[0].grid(alpha=.3)
ax[1].plot(fr[b], px[b]/px[b].max(), color='navy'); ax[1].axvline(hr, color='crimson', ls='--', label=f"{hr:.0f} bpm")
if ambig: ax[1].axvline(cands[1][0], color='orange', ls=':', lw=2, label=f"rival {cands[1][0]:.0f}")
ax[1].set_title("Spectre rBCG"); ax[1].set_xlabel("fréquence (bpm)"); ax[1].legend(); ax[1].grid(alpha=.3)
plt.tight_layout(); out = ROOT/f"scratch_bcg_{stem}.png"; plt.savefig(out, dpi=100); plt.close()
print(f"   → {out.name}")
