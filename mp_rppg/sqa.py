"""
SQA (Signal Quality Assessment) par fenêtres glissantes + agrégation — l'astuce
signature de Binah.AI.

Au lieu d'estimer UNE FC sur tout l'enregistrement, on glisse une fenêtre
(ex. 10 s, pas 2 s), on évalue la qualité de CHAQUE fenêtre (fusion + verdict de
confiance), on ne garde que les fenêtres FIABLES, et on agrège (médiane). Ça
permet de RÉCUPÉRER un enregistrement partiellement propre (ex. 14 s bonnes sur
20 s, le reste bougé) au lieu de tout jeter — et de sortir une COUVERTURE
(fraction de fenêtres fiables) comme indice de confiance.

verdict() : règle de confiance par fenêtre (accord + qualité SNR + témoin isolé),
identique à celle de run_on_video.
"""
import numpy as np

from mp_rppg.fusion import adaptive_fusion
from mp_rppg.metrics import hr_from_fft, snr

# seuils (mêmes que run_on_video)
AGREE_OK, AGREE_DOUBT = 6.0, 12.0
SNR_OK, SNR_REJECT = -3.5, -4.5
SNR_WITNESS, WITNESS_MARGIN = 1.0, 2.0
WITNESS_EXCLUDE = {'rBCG'}        # SNR non calibré comme l'optique → pas témoin isolé
SQA_FIABLE_MIN, SQA_REJECT_MAX = 0.20, 0.05   # couverture SQA dans le verdict


def verdict(per_method):
    """per_method : [(nom, hr, snr)]. Retourne dict {status, hr, std, med_snr}.
    status ∈ {'FIABLE','DOUTE','REJET'}."""
    if not per_method:
        return {'status': 'REJET', 'hr': float('nan'), 'std': 99.0, 'med_snr': -99.0}
    fz = adaptive_fusion(per_method)
    std = fz['std']
    snrs = [s for _, _, s in per_method]
    med_snr = float(np.median(snrs))
    # témoin isolé : 1 méthode au SNR vraiment haut, détachée, non-harmonique.
    # rBCG exclu (son SNR de mouvement n'est pas à la même échelle que l'optique).
    cand = sorted([x for x in per_method if x[0] not in WITNESS_EXCLUDE],
                  key=lambda x: x[2], reverse=True)
    lone = False
    if cand:
        w_hr, w_snr = cand[0][1], cand[0][2]
        s2 = cand[1][2] if len(cand) > 1 else -99.0
        others = [h for _, h, _ in per_method if h != w_hr]
        wm = float(np.median(others)) if others else w_hr
        w_harm = wm > 0 and (abs(w_hr / 2 - wm) <= 0.15 * wm or abs(w_hr * 2 - wm) <= 0.15 * wm)
        lone = w_snr > SNR_WITNESS and (w_snr - s2) >= WITNESS_MARGIN and not w_harm

    if std <= AGREE_OK and med_snr > SNR_OK:
        return {'status': 'FIABLE', 'hr': fz['hr'], 'std': std, 'med_snr': med_snr}
    if lone:
        return {'status': 'FIABLE', 'hr': cand[0][1], 'std': std, 'med_snr': med_snr}
    if std > AGREE_DOUBT or med_snr <= SNR_REJECT:
        return {'status': 'REJET', 'hr': fz['hr'], 'std': std, 'med_snr': med_snr}
    return {'status': 'DOUTE', 'hr': fz['hr'], 'std': std, 'med_snr': med_snr}


def combined_verdict(per_method, sqa_cov, ambiguous=False, candidates=None):
    """Verdict final = verdict scénario MODULÉ par la couverture SQA + ambiguïté.
    FIABLE exige assez de fenêtres fiables ; SQA quasi-nulle → REJET ; et si le
    signal primaire est AMBIGU (deux rythmes candidats quasi égaux) → DOUTE."""
    v = verdict(per_method); st = v['status']
    if st == 'REJET' or sqa_cov < SQA_REJECT_MAX:
        st = 'REJET'
    elif st == 'FIABLE' and sqa_cov < SQA_FIABLE_MIN:
        st = 'DOUTE'                       # FIABLE mais peu de fenêtres fiables → doute
    if ambiguous and st == 'FIABLE':
        st = 'DOUTE'                       # deux BPM candidats se valent → à confirmer
    v['status'] = st; v['sqa_cov'] = sqa_cov
    v['ambiguous'] = ambiguous; v['candidates'] = candidates
    return v


def windowed_sqa(sigs, fps, extra=None, win_s=10.0, stride_s=2.0, lo=0.7, hi=2.5):
    """
    sigs  : dict {nom_méthode: signal_bandpassé (1D)} (CNN1D, PhysNet, CHROM, …).
    extra : votants par enregistrement non fenêtrables, ex. {'rBCG': (hr, snr)}.
    Glisse des fenêtres, juge chacune (verdict), agrège les FIABLES.
    Retourne dict {hr, coverage, n_fiable, n_total, windows}.
    """
    lengths = [len(s) for s in sigs.values() if s is not None]
    if not lengths:
        return {'hr': float('nan'), 'coverage': 0.0, 'n_fiable': 0, 'n_total': 0, 'windows': []}
    T = min(lengths)
    win = int(win_s * fps); stride = max(1, int(stride_s * fps))
    if T < win:                                   # trop court → 1 seule fenêtre
        starts = [0]; win = T
    else:
        starts = list(range(0, T - win + 1, stride))
        if starts[-1] != T - win:
            starts.append(T - win)

    windows, good = [], []
    for s0 in starts:
        pm = []
        for m, sig in sigs.items():
            if sig is None:
                continue
            w = sig[s0:s0 + win]
            h = hr_from_fft(w, fps, low=lo, high=hi)
            pm.append((m, h, snr(w, h, fps, low=lo, high=hi)))
        if extra:
            for m, (h, s) in extra.items():
                pm.append((m, h, s))
        v = verdict(pm)
        windows.append({'t0': s0 / fps, 'status': v['status'], 'hr': v['hr'],
                        'std': v['std'], 'med_snr': v['med_snr']})
        if v['status'] == 'FIABLE':
            good.append(v['hr'])

    n_tot = len(windows); n_ok = len(good)
    final_hr = float(np.median(good)) if good else float('nan')
    return {'hr': final_hr, 'coverage': n_ok / max(1, n_tot),
            'n_fiable': n_ok, 'n_total': n_tot, 'windows': windows}
