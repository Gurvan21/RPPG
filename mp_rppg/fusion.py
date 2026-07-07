"""
Fusion ADAPTATIVE des estimations de FC multi-méthodes.

Motivation expérimentale (2 vidéos peau très foncée) :
  - BPM54 (ITA -56) : les méthodes DIVERGENT (écart-type 15.9 bpm) ; la moyenne est
    polluée par les mauvaises → la SÉLECTION par SNR (CNN1D, err 3) gagne.
  - BPM56 (ITA -65) : les méthodes S'ACCORDENT (écart-type 10.7 bpm) ; le SNR
    sous-note les bonnes (CHROM/POS exacts mais faible signal) → la MÉDIANE
    (err ~0) gagne, la sélection-SNR seule se trompe.

→ Aucune règle unique ne marche sur les deux. La fusion adaptative bascule :
    accord (faible écart-type)   → CONSENSUS : médiane des FC
    désaccord (fort écart-type)  → SÉLECTION : méthode au meilleur SNR (validé),
                                    avec garde anti-harmonique.
"""

import numpy as np

AGREE_THRESH = 12.0     # bpm : seuil d'écart-type séparant accord / désaccord
SNR_VALIDATE = -1.0     # SNR au-dessus duquel une méthode est jugée fiable
HARMONIC_TOL = 0.15     # tolérance relative pour détecter un doublage d'harmonique


def adaptive_fusion(per_method, agree_thresh=AGREE_THRESH,
                    snr_validate=SNR_VALIDATE, harmonic_tol=HARMONIC_TOL):
    """
    per_method : liste de (nom, hr_bpm, snr_db).
    Retourne un dict : {hr, mode, chosen, std, n}.
      mode   = 'consensus' | 'selection'
      chosen = 'médiane'   | nom de la méthode retenue
    """
    items = [(n, float(h), float(s)) for (n, h, s) in per_method
             if h is not None and np.isfinite(h)]
    if not items:
        return {'hr': float('nan'), 'mode': 'vide', 'chosen': None, 'std': 0.0, 'n': 0}

    hrs = np.array([h for _, h, _ in items])
    std = float(np.std(hrs))
    med = float(np.median(hrs))

    # ── Accord : consensus par médiane (robuste aux outliers/harmoniques) ──
    if std <= agree_thresh:
        return {'hr': med, 'mode': 'consensus', 'chosen': 'médiane',
                'std': std, 'n': len(items)}

    # ── Désaccord : sélection par SNR (validés d'abord), garde anti-harmonique ──
    cands = [it for it in items if it[2] > snr_validate] or items
    cands = sorted(cands, key=lambda it: it[2], reverse=True)  # meilleur SNR d'abord

    for name, hr, sn in cands:
        # doublage d'harmonique : hr ≈ 2 × (médiane des autres) ?
        others = [h for _, h, _ in items if h != hr]
        m_others = np.median(others) if others else hr
        if m_others > 0 and abs(hr / 2 - m_others) <= harmonic_tol * m_others and hr > 1.5 * m_others:
            continue  # suspect → on passe au candidat suivant
        return {'hr': hr, 'mode': 'selection', 'chosen': name, 'std': std, 'n': len(items)}

    # tous suspects → repli médiane
    return {'hr': med, 'mode': 'selection', 'chosen': 'médiane (repli)',
            'std': std, 'n': len(items)}
