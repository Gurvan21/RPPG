#!/usr/bin/env python3
"""
Check qualité d'UNE prise juste après l'avoir filmée → GO / REFAIRE en ~30 s.
Trace l'onde + spectre, applique les 4 garde-fous (SNR, accord, ambiguïté).

Usage :
    python scripts/check_capture.py <video> [paume|visage] [ref_hr]
"""
import os, sys
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
from pathlib import Path
import numpy as np, torch
from scipy.signal import periodogram
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.preextract_clips import load_video, resample_to_fps
from scripts.train_cnn1d import _temporal_norm, CLIP_LEN
from models.cnn1d_rppg import CNN1D_rPPG
from models.chrom_adaptive import bandpass_numpy
from mp_rppg.methods import chrom, pos
from mp_rppg.metrics import hr_from_fft, snr, hr_candidates

dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
video = sys.argv[1]
mode = sys.argv[2] if len(sys.argv) > 2 else 'paume'
REF = float(sys.argv[3]) if len(sys.argv) > 3 else None
GO_SNR, AGREE = 1.0, 8.0                       # seuils GO


def palm_sigs(frames, fps):
    from scripts.extract_hand_regions import extract_video as eh
    from scripts.palm_rppg import extract_palm_rgb, interp_nan
    x, _, det = eh(frames, 3)
    out = {}
    if det > 0.3:
        m = CNN1D_rPPG(in_channels=18*9).to(dev)
        m.load_state_dict(torch.load(ROOT/'weights'/'cnn1d_hand.pth', map_location=dev)); m.eval()
        xn = _temporal_norm(x); pr = []
        for s in range(0, xn.shape[1]-CLIP_LEN+1, CLIP_LEN):
            with torch.no_grad():
                pr.append(m(torch.from_numpy(xn[:, s:s+CLIP_LEN]).unsqueeze(0).to(dev)).squeeze().cpu().numpy())
        if pr: out['CNN1D-main'] = bandpass_numpy(np.concatenate(pr), fps)
    rgb = interp_nan(extract_palm_rgb(frames)[0])
    if rgb is not None:
        out['CHROM'] = bandpass_numpy(chrom(rgb, fps), fps)
        out['POS'] = bandpass_numpy(pos(rgb, fps), fps)
    return out, det


def face_sigs(frames, fps):
    from scripts.test_palm_poc import face_rgb
    from scripts.palm_rppg import interp_nan
    rgb = interp_nan(face_rgb(frames)); out = {}
    if rgb is not None:
        out['CHROM'] = bandpass_numpy(chrom(rgb, fps), fps)
        out['POS'] = bandpass_numpy(pos(rgb, fps), fps)
    return out, 1.0


frames, fps = load_video(video, max_dim=720)
if fps > 32:
    ft = np.arange(len(frames))*1000.0/fps; frames, _ = resample_to_fps(frames, ft, 30.0); fps = 30.0
sigs, det = (palm_sigs if mode == 'paume' else face_sigs)(frames, fps)
per = [(n, hr_from_fft(s, fps), snr(s, hr_from_fft(s, fps), fps)) for n, s in sigs.items() if s is not None]
if not per:
    print("❌ REFAIRE — aucun signal (paume/visage non détecté)"); sys.exit()
per.sort(key=lambda z: z[2], reverse=True)
prim, hr, sn = per[0]
hrs = [h for _, h, _ in per]; std = float(np.std(hrs))
cands, ambig = hr_candidates(sigs[prim], fps)

# ── verdict GO / REFAIRE ──
reasons = []
if sn < GO_SNR: reasons.append(f"SNR faible ({sn:+.1f})")
if std > AGREE: reasons.append(f"méthodes dispersées ({std:.0f} bpm)")
if ambig: reasons.append(f"ambigu ({cands[0][0]:.0f}/{cands[1][0]:.0f})")
go = not reasons
print(f"\n{'='*50}")
print(f"{Path(video).name}  [{mode}, {100*det:.0f}% détecté, {len(frames)/fps:.0f}s]")
for n, h, s in per: print(f"   {n:11s} {h:5.0f} bpm  SNR {s:+.1f}")
if go:
    print(f"\n✅  GO  →  {hr:.0f} bpm  (SNR {sn:+.1f}, accord {std:.0f} bpm)"
          + (f"  | réf {REF:.0f}, err {abs(hr-REF):.0f}" if REF else ""))
else:
    print(f"\n🔁  REFAIRE — " + " ; ".join(reasons))
    print("    → plus d'immobilité, plus de lumière, paume qui remplit le cadre.")
print('='*50)

# ── onde + spectre ──
sig = sigs[prim]; t = np.arange(len(sig))/fps
nf = 1
while nf < len(sig): nf *= 2
f, px = periodogram(sig, fs=fps, nfft=nf, detrend=False); fr = f*60; b = (fr >= 40) & (fr <= 180)
fig, ax = plt.subplots(2, 1, figsize=(11, 6))
col = 'green' if go else 'darkorange'
ax[0].plot(t, sig, color=col, lw=1.0)
ax[0].set_title(f"{'✅ GO' if go else '🔁 REFAIRE'} — {Path(video).name} ({mode}) | {prim} {hr:.0f} bpm SNR {sn:+.1f}")
ax[0].set_xlabel("temps (s)"); ax[0].grid(alpha=.3)
ax[1].plot(fr[b], px[b]/px[b].max(), color='navy'); ax[1].axvline(hr, color='crimson', ls='--', label=f"{hr:.0f} bpm")
if ambig: ax[1].axvline(cands[1][0], color='orange', ls=':', lw=2, label=f"rival {cands[1][0]:.0f}")
ax[1].set_xlabel("fréquence (bpm)"); ax[1].legend(); ax[1].grid(alpha=.3)
plt.tight_layout(); out = ROOT/f"scratch_check_{Path(video).stem}.png"; plt.savefig(out, dpi=100); plt.close()
print(f"onde+spectre → {out.name}")
