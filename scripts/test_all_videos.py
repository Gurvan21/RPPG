#!/usr/bin/env python3
"""Pipeline COMPLET (toutes méthodes + BCG + fusion + verdict + SQA) sur toutes
les vidéos test, avec erreur si référence connue et SNR de chaque méthode."""
import os, sys, numpy as np, torch
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from models.cnn1d_rppg import CNN1D_rPPG
from models.chrom_adaptive import CHROMAdaptiveConditioned, bandpass_numpy
from mp_rppg.metrics import hr_from_fft, snr
from mp_rppg.methods import chrom, pos, chrom_adaptive
from mp_rppg.fusion import adaptive_fusion
from mp_rppg.sqa import verdict, combined_verdict, windowed_sqa
from mp_rppg.skin_ita import sclera_corrected_ita
from mp_rppg.bcg import bcg_hr
from scripts.train_cnn1d import _temporal_norm, CLIP_LEN
from scripts.extract_regions_bisenet import load_bisenet, extract_video, pick_device
from scripts.preextract_clips import load_video, track_face_bboxes, resample_to_fps
from scripts.run_on_video import run_physnet, FRONT, FULLSKIN, RGB_IDX

D = ROOT / "DataVital" / "SubjecTestRonel"
PHYS_W = str(ROOT / "weights" / "clean_physnet_A_pure" / "physnet_africa1_best.pth")
REFS = {  # références connues (bpm) ; les autres = inconnues
    "videoTestBPM54.mov": 54, "VideoTestBPM56.MOV": 56, "Video50MPS54bpm.mp4": 54,
    "VideoIssa63.MOV": 63, "VideoTest4.mov": 58, "IMG_9008.MOV": 67, "J'enaimarre1.mp4": 50,
}
SKIP = {"Video50bpm.mp4",   # doublon de Video50MPS54bpm
        "IMG_9008.MOV"}     # 4K 1808 frames → trop lent en CPU


def main():
    # GPU (MPS) : BiSeNet y est bien plus rapide qu'en CPU, même partagé avec
    # les entraînements. La contention ralentit un peu, mais ça avance.
    dev = pick_device(); net = load_bisenet(dev)
    cnn = CNN1D_rPPG(in_channels=23*9).to(dev)
    cnn.load_state_dict(torch.load(ROOT/'weights'/'cnn1d_rppg.pth', map_location=dev)); cnn.eval()
    cmlp = CHROMAdaptiveConditioned()
    cmlp.load_state_dict(torch.load(ROOT/'weights'/'chrom_conditioned_regions.pth', map_location='cpu')['model_state_dict']); cmlp.eval()

    vids = sorted([p for p in D.iterdir() if p.suffix.lower() in ('.mp4', '.mov', '.mkv') and p.name not in SKIP])
    summary = []
    for vid in vids:
        try:
            frames, fps = load_video(vid, max_dim=720)
            if fps > 32:
                ft = np.arange(len(frames))*1000.0/fps; frames, _ = resample_to_fps(frames, ft, 30.0); fps = 30.0
            x_reg, _, _ = extract_video(net, dev, frames, 4)
            front = x_reg[:, FRONT, :][:, RGB_IDX].astype(np.float32)
            skin = x_reg[:, FULLSKIN, :][:, RGB_IDX].astype(np.float32)
            ita = sclera_corrected_ita(frames, skin_rgb_fallback=front.mean(0))["ita"]
            xn = _temporal_norm(x_reg); T = xn.shape[1]; preds = []
            for s in range(0, T-CLIP_LEN+1, CLIP_LEN):
                with torch.no_grad():
                    preds.append(cnn(torch.from_numpy(xn[:, s:s+CLIP_LEN]).unsqueeze(0).to(dev)).squeeze().cpu().numpy())
            sigs = {'CNN1D': bandpass_numpy(np.concatenate(preds), fps) if preds else None,
                    'PhysNet': run_physnet(frames, fps, PHYS_W, dev),
                    'CHROM-ITA': bandpass_numpy(chrom_adaptive(front, fps, cmlp.get_coefficients(ita)), fps),
                    'CHROM': bandpass_numpy(chrom(skin, fps), fps), 'POS': bandpass_numpy(pos(skin, fps), fps)}
            pm = []
            cells = []
            for m, s in sigs.items():
                if s is None: continue
                h = hr_from_fft(s, fps); sn = snr(s, h, fps); pm.append((m, h, sn))
                cells.append(f"{m} {h:.0f}({sn:+.1f})")
            bb = track_face_bboxes(frames)
            bbox = tuple(np.median(np.array(bb), axis=0).astype(int)) if len(bb) else None
            bh, bs, _ = bcg_hr(frames, fps, bbox=bbox)
            if np.isfinite(bh):
                pm.append(('rBCG', bh, bs)); cells.append(f"rBCG {bh:.0f}({bs:+.1f})")
            sqa = windowed_sqa(sigs, fps, extra={'rBCG': (bh, bs)} if np.isfinite(bh) else None)
            v = combined_verdict(pm, sqa['coverage'])
            ref = REFS.get(vid.name)
            err = abs(v['hr'] - ref) if ref is not None else None
            print(f"\n{vid.name}  (réf {ref if ref else '—'}, ITA {ita:.0f})")
            print("   " + "  ".join(cells))
            print(f"   → FUSION {v['hr']:.0f}  [{v['status']}]  SQA {100*sqa['coverage']:.0f}%"
                  + (f"  | ERREUR {err:.1f} bpm" if err is not None else ""))
            if err is not None and v['status'] != 'REJET':
                summary.append((vid.name, ref, v['hr'], err, v['status']))
        except Exception as e:
            print(f"\n{vid.name} : ERREUR ({e})")

    print("\n" + "="*60 + "\nRÉSUMÉ (vidéos à référence, non rejetées) :")
    if summary:
        for n, r, h, e, st in summary:
            print(f"  {n[:28]:<28} réf {r}  prédit {h:.0f}  err {e:.1f}  [{st}]")
        errs = np.array([s[3] for s in summary])
        print(f"\n  MAE = {errs.mean():.2f} bpm  sur {len(errs)} vidéos acceptées (non rejetées)")


if __name__ == '__main__':
    main()
