#!/usr/bin/env python3
"""SNR + AMBIGUÏTÉ par méthode sur une vidéo (mêmes signaux que run_on_video).
Pour chaque méthode : HR, SNR, ambigu(oui/non), pics candidats. + figure spectres.
Usage: python scripts/methods_snr_ambig.py <video>"""
import os, sys
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
from pathlib import Path
import numpy as np, torch
from scipy.signal import periodogram
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from models.cnn1d_rppg import CNN1D_rPPG
from models.chrom_adaptive import CHROMAdaptiveConditioned, bandpass_numpy
from mp_rppg.metrics import hr_from_fft, snr, hr_candidates
from mp_rppg.methods import chrom, pos, chrom_adaptive
from mp_rppg.skin_ita import sclera_corrected_ita
from mp_rppg.bcg import bcg_hr
from scripts.train_cnn1d import _temporal_norm, CLIP_LEN
from scripts.extract_regions_bisenet import load_bisenet, extract_video, pick_device
from scripts.preextract_clips import load_video, resample_to_fps, track_face_bboxes
import run_on_video as R
FRONT, FULLSKIN, RGB = 0, 6, [0, 1, 2]

video = sys.argv[1]; dev = pick_device()
frames, fps = load_video(video, max_dim=720)
if fps > 32.0:
    ft = np.arange(len(frames))*1000.0/fps; frames, _ = resample_to_fps(frames, ft, 30.0); fps = 30.0
print(f"{len(frames)} frames @ {fps:.1f} fps")
net = load_bisenet(dev); x_reg, _, _ = extract_video(net, dev, frames, 4)

cnn = CNN1D_rPPG(in_channels=23*9).to(dev)
cnn.load_state_dict(torch.load(ROOT/'weights'/'cnn1d_rppg.pth', map_location=dev)); cnn.eval()
cmlp = CHROMAdaptiveConditioned()
cmlp.load_state_dict(torch.load(ROOT/'weights'/'chrom_conditioned_regions.pth', map_location='cpu')['model_state_dict']); cmlp.eval()
front = x_reg[:, FRONT, :][:, RGB].astype(np.float32); skin = x_reg[:, FULLSKIN, :][:, RGB].astype(np.float32)
ita = sclera_corrected_ita(frames, skin_rgb_fallback=front.mean(0))["ita"]
xn = _temporal_norm(x_reg); preds = []
for s in range(0, xn.shape[1]-CLIP_LEN+1, CLIP_LEN):
    with torch.no_grad():
        preds.append(cnn(torch.from_numpy(xn[:, s:s+CLIP_LEN]).unsqueeze(0).to(dev)).squeeze().cpu().numpy())
physnet_sig = R.run_physnet(frames, fps, str(ROOT/'weights'/'clean_physnet_A_pure'/'physnet_africa1_best.pth'), dev)

sigs = {
    'CNN 1D':    bandpass_numpy(np.concatenate(preds), fps) if preds else None,
    'PhysNet':   physnet_sig,
    'CHROM-ITA': bandpass_numpy(chrom_adaptive(front, fps, cmlp.get_coefficients(ita)), fps),
    'CHROM':     bandpass_numpy(chrom(skin, fps), fps),
    'POS':       bandpass_numpy(pos(skin, fps), fps),
}
rows = []
for m, sig in sigs.items():
    if sig is None: continue
    hr = hr_from_fft(sig, fps); sn = snr(sig, hr, fps)
    cands, ambig = hr_candidates(sig, fps)
    rows.append((m, hr, sn, ambig, cands, sig))

# rBCG
_bb = track_face_bboxes(frames)
_bbox = tuple(np.median(np.array(_bb), axis=0).astype(int)) if len(_bb) else None
bh, bs, bsig = bcg_hr(frames, fps, bbox=_bbox)
if np.isfinite(bh):
    try: c, a = hr_candidates(bsig, fps)
    except Exception: c, a = [(bh, 1.0)], False
    rows.append(('rBCG', bh, bs, a, c, bsig))

print(f"\n{'Méthode':<11}{'HR':>7}{'SNR':>8}  {'ambigu':<8}{'candidats (bpm@force)'}")
print('-'*72)
for m, hr, sn, ambig, cands, _ in rows:
    cs = ', '.join(f'{c[0]:.0f}@{c[1]:.2f}' for c in cands[:3])
    print(f"{m:<11}{hr:>7.1f}{sn:>8.2f}  {'OUI' if ambig else 'non':<8}{cs}")

# figure : un spectre par méthode
n = len(rows); fig, axs = plt.subplots(n, 1, figsize=(11, 2.1*n), sharex=True)
for ax, (m, hr, sn, ambig, cands, sig) in zip(axs, rows):
    nfft = 1
    while nfft < len(sig): nfft *= 2
    f, px = periodogram(sig, fs=fps, nfft=nfft, detrend='linear'); px /= (px.max()+1e-12)
    mk = (f*60 >= 40) & (f*60 <= 180)
    col = 'crimson' if not ambig else 'darkorange'
    ax.plot(f[mk]*60, px[mk], color=col)
    ax.axvline(hr, color='k', ls='--', lw=1)
    for c in cands[1:3]: ax.axvline(c[0], color='gray', ls=':', lw=0.9)
    ax.set_ylabel(m, fontsize=9)
    ax.set_title(f"{m}: HR {hr:.0f} bpm | SNR {sn:+.2f} dB | {'AMBIGU' if ambig else 'net'}", fontsize=9, loc='left')
axs[-1].set_xlabel('bpm')
plt.tight_layout()
out = os.path.splitext(video)[0] + '_methods_spectra.png'
plt.savefig(out, dpi=110); print(f"\nFigure -> {out}")
