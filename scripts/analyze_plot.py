#!/usr/bin/env python3
"""Analyse (HR/SNR) + onde + spectre d'une vidéo, selon le site (paume/visage/both).
Usage: python scripts/analyze_plot.py <video> <paume|visage|both> [verite]"""
import os, sys
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
from pathlib import Path
import numpy as np, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.preextract_clips import load_video, resample_to_fps
from scripts.train_cnn1d import _temporal_norm, CLIP_LEN
from models.cnn1d_rppg import CNN1D_rPPG
from models.chrom_adaptive import CHROMAdaptiveConditioned, bandpass_numpy
from mp_rppg.methods import chrom, pos, chrom_adaptive
from mp_rppg.metrics import hr_from_fft, snr, hr_candidates

dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
video, mode = sys.argv[1], sys.argv[2]
TRUTH = float(sys.argv[3]) if len(sys.argv) > 3 else None
stem = Path(video).stem


def cnn_infer(xn, weights, in_ch, fps):
    m = CNN1D_rPPG(in_channels=in_ch).to(dev)
    m.load_state_dict(torch.load(weights, map_location=dev)); m.eval()
    T = xn.shape[1]; pr = []
    for s in range(0, T-CLIP_LEN+1, CLIP_LEN):
        with torch.no_grad():
            pr.append(m(torch.from_numpy(xn[:, s:s+CLIP_LEN]).unsqueeze(0).to(dev)).squeeze().cpu().numpy())
    return bandpass_numpy(np.concatenate(pr), fps) if pr else None


def palm_signals(frames, fps):
    from scripts.extract_hand_regions import extract_video as eh
    from scripts.palm_rppg import extract_palm_rgb, interp_nan
    x, _, det = eh(frames, 3)
    sigs = {}
    if det > 0.3:
        sigs['CNN1D-main'] = cnn_infer(_temporal_norm(x), ROOT/'weights'/'cnn1d_hand.pth', 18*9, fps)
    prgb = interp_nan(extract_palm_rgb(frames)[0])
    if prgb is not None:
        sigs['CHROM'] = bandpass_numpy(chrom(prgb, fps), fps)
        sigs['POS'] = bandpass_numpy(pos(prgb, fps), fps)
    return sigs, det


def face_signals(frames, fps):
    from scripts.extract_regions_bisenet import load_bisenet, extract_video as ef, pick_device
    from scripts.run_on_video import run_physnet, FRONT, FULLSKIN, RGB_IDX
    from mp_rppg.skin_ita import sclera_corrected_ita
    net = load_bisenet(dev)
    x_reg, _, _ = ef(net, dev, frames, 4)
    front = x_reg[:, FRONT, :][:, RGB_IDX].astype(np.float32)
    skin = x_reg[:, FULLSKIN, :][:, RGB_IDX].astype(np.float32)
    ita = sclera_corrected_ita(frames, skin_rgb_fallback=front.mean(0))['ita']
    cmlp = CHROMAdaptiveConditioned()
    cmlp.load_state_dict(torch.load(ROOT/'weights'/'chrom_conditioned_regions.pth', map_location='cpu')['model_state_dict']); cmlp.eval()
    sigs = {
        'CNN1D-face': cnn_infer(_temporal_norm(x_reg), ROOT/'weights'/'cnn1d_rppg.pth', 23*9, fps),
        'PhysNet': run_physnet(frames, fps, str(ROOT/'weights'/'clean_physnet_A_pure'/'physnet_africa1_best.pth'), dev),
        'CHROM-ITA': bandpass_numpy(chrom_adaptive(front, fps, cmlp.get_coefficients(ita)), fps),
        'CHROM': bandpass_numpy(chrom(skin, fps), fps),
        'POS': bandpass_numpy(pos(skin, fps), fps),
    }
    return sigs, ita


def plot_site(sigs, fps, title, out):
    # signal primaire = meilleur SNR
    scored = [(n, s, snr(s, hr_from_fft(s, fps), fps)) for n, s in sigs.items() if s is not None]
    scored.sort(key=lambda z: z[2], reverse=True)
    name, sig, sn = scored[0]; hr = hr_from_fft(sig, fps); t = np.arange(len(sig))/fps
    # spectre IDENTIQUE à hr_from_fft (periodogram boxcar + zeropad) → le pic
    # affiché correspond exactement à la HR choisie (cohérence visuelle).
    from scipy.signal import periodogram
    nf = 1
    while nf < len(sig): nf *= 2
    ff, ps = periodogram(sig, fs=fps, nfft=nf, detrend=False); fr = ff*60
    b = (fr >= 40) & (fr <= 180)
    cands, ambig = hr_candidates(sig, fps)
    amb_txt = (f"  ⚠️ AMBIGU : {cands[0][0]:.0f} ou {cands[1][0]:.0f} bpm "
               f"(pics {cands[1][1]*100:.0f}% aussi forts)") if ambig else ""
    fig, ax = plt.subplots(3, 1, figsize=(11, 8))
    ax[0].plot(t, sig, lw=.8, color='crimson')
    ax[0].set_title(f"{title}\nsignal primaire = {name} | HR={hr:.0f} bpm  SNR={sn:+.1f} dB"
                    + (f"  (vérité ~{TRUTH:.0f}, err {abs(hr-TRUTH):.0f})" if TRUTH else "") + amb_txt)
    ax[0].set_xlabel("temps (s)"); ax[0].set_ylabel("amplitude (rel.)"); ax[0].grid(alpha=.3)
    z = (t >= 5) & (t <= 13)
    ax[1].plot(t[z], sig[z], lw=1.3, color='crimson', marker='.', ms=2)
    ax[1].set_title("Zoom 5-13 s"); ax[1].set_xlabel("temps (s)"); ax[1].grid(alpha=.3)
    ax[2].plot(fr[b], ps[b]/ps[b].max(), color='navy'); ax[2].axvline(hr, color='crimson', ls='--', label=f"choisi {hr:.0f} bpm")
    if ambig:                                    # marque le 2e candidat concurrent
        ax[2].axvline(cands[1][0], color='orange', ls=':', lw=2, label=f"rival {cands[1][0]:.0f} bpm ({cands[1][1]*100:.0f}%)")
    ax[2].set_title("Spectre" + ("  — ⚠️ AMBIGU" if ambig else "")); ax[2].set_xlabel("fréquence (bpm)"); ax[2].legend(); ax[2].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(out, dpi=100); plt.close()
    return name, hr, sn


def table(sigs, fps, label):
    print(f"  ── {label} ──")
    for n, s in sigs.items():
        if s is None: continue
        h = hr_from_fft(s, fps); sn = snr(s, h, fps)
        e = f"  err {abs(h-TRUTH):.0f}" if TRUTH else ""
        cands, ambig = hr_candidates(s, fps)
        a = f"  ⚠️AMBIGU({cands[0][0]:.0f}/{cands[1][0]:.0f})" if ambig else ""
        print(f"     {n:12s}: {h:5.1f} bpm  SNR {sn:+.1f}{e}{a}")


frames, fps = load_video(video, max_dim=720)
if fps > 32:
    ft = np.arange(len(frames))*1000.0/fps; frames, _ = resample_to_fps(frames, ft, 30.0); fps = 30.0
print(f"\n===== {Path(video).name} — {len(frames)} frames @ {fps:.1f}fps"
      + (f" (vérité ~{TRUTH:.0f})" if TRUTH else "") + " =====")
if mode in ('paume', 'both'):
    ps_, det = palm_signals(frames, fps)
    table(ps_, fps, f"PAUME (main {100*det:.0f}%)")
    nm, hr, sn = plot_site(ps_, fps, f"PAUME — {Path(video).name}", ROOT/f"scratch_wave_{stem}_paume.png")
    print(f"     → onde+spectre: scratch_wave_{stem}_paume.png (primaire {nm})")
if mode in ('visage', 'both'):
    fs_, ita = face_signals(frames, fps)
    table(fs_, fps, f"VISAGE (ITA {ita:.0f})")
    nm, hr, sn = plot_site(fs_, fps, f"VISAGE — {Path(video).name} (ITA {ita:.0f})", ROOT/f"scratch_wave_{stem}_visage.png")
    print(f"     → onde+spectre: scratch_wave_{stem}_visage.png (primaire {nm})")
