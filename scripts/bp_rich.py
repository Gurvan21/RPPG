#!/usr/bin/env python3
"""BP depuis PPG doigt — jeu de features ENRICHI (~28 vs 12) : points APG a-b-c-d-e
+ tous ratios, aires syst/diast, largeurs multiples, VPG, rigidité, moments.
Compare 12 vs riche, + courbe d'apprentissage sur le riche, pour répondre à
'plus de features aide-t-il ?'."""
import sys, json, glob, re, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from scipy.stats import skew, kurtosis
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.bp_from_ppg import clean, template, load as load12
FS = 60.0


def rich_features(ppg):
    tmpl, ibi = template(ppg)
    if tmpl is None: return None
    t = (tmpl - tmpl.min()) / (np.ptp(tmpl) + 1e-8)
    L = len(t); sp = int(np.argmax(t))
    vpg = np.gradient(t); apg = np.gradient(vpg)
    hr = 60.0/np.median(ibi) if len(ibi) else 0.0
    hrv = float(np.std(ibi)) if len(ibi) else 0.0

    # --- points APG a,b,c,d,e (extrema alternés du 2e dérivé) ---
    mx, _ = find_peaks(apg); mn, _ = find_peaks(-apg)
    ext = sorted([(i, apg[i]) for i in list(mx)+list(mn)])
    a = b = c = d = e = 0.0
    if ext:
        a = ext[0][1] + 1e-8
        vals = [v for _, v in ext[:5]] + [0]*5
        a, b, c, d, e = vals[0]+1e-8, vals[1], vals[2], vals[3], vals[4]
    ba, ca, da, ea = b/a, c/a, d/a, e/a
    aging = (b - c - d - e)/a

    # --- largeurs à multiples hauteurs ---
    def width(h):
        w = np.where(t >= h)[0]; return (w[-1]-w[0])/L if len(w) else 0.0
    W = [width(h) for h in (0.1, 0.25, 0.5, 0.66, 0.75, 0.9)]

    # --- aires systolique / diastolique ---
    a_sys = float(np.trapz(t[:sp+1]))/L
    a_dia = float(np.trapz(t[sp:]))/L
    a_tot = a_sys + a_dia
    ipa = a_dia/(a_sys+1e-8)                         # inflection point area ratio

    # --- VPG ---
    vmax = float(np.max(vpg)); vmin = float(np.min(vpg))
    t_vmax = float(np.argmax(vpg))/L

    # --- onde réfléchie / notch dicrote ---
    tail = t[sp+3:]; ai = 0.0; notch_pos = 0.0; notch_h = 0.0
    if len(tail) > 4:
        sub, _ = find_peaks(tail)
        dips, _ = find_peaks(-tail)
        if len(sub): ai = float(tail[sub[0]])
        if len(dips): notch_pos = float(sp+3+dips[0])/L; notch_h = float(tail[dips[0]])
    ri = ai                                          # reflection index ~ hauteur 2e pic
    si = 1.0/(width(0.1)+1e-6)                        # stiffness index ~ 1/largeur base

    # --- moments ---
    sk = float(skew(t)); ku = float(kurtosis(t))

    f = np.array([hr, hrv, sp/L, t_vmax,
                  *W, a_sys, a_dia, a_tot, ipa,
                  vmax, vmin, ba, ca, da, ea, aging,
                  ai, ri, si, notch_pos, notch_h, sk, ku])
    return f if np.all(np.isfinite(f)) else None


def load_rich():
    X, ys, yd = [], [], []
    for jf in sorted(glob.glob(str(ROOT/'DataVital'/'Subject*'/'*.json'))):
        try: d = json.load(open(jf))
        except: continue
        bp = None; feat = None
        for sc in d.get('scenarios', []):
            rec = sc.get('recordings', {})
            b = rec.get('BP')
            if isinstance(b, dict) and re.match(r'\d+/\d+', str(b.get('value', ''))): bp = b['value']
            cms = rec.get('CMS')
            if cms and len(cms) > 50 and feat is None:
                rows = cms[1:] if cms[0][0] == 'time' else cms
                try:
                    f = rich_features(clean([r[0] for r in rows], [r[1] for r in rows]))
                    if f is not None: feat = f
                except Exception: pass
        if bp and feat is not None:
            s, dd = map(int, bp.split('/')); X.append(feat); ys.append(s); yd.append(dd)
    return np.array(X), np.array(ys), np.array(yd)


def cv(X, y, name):
    kf = KFold(5, shuffle=True, random_state=0); pred = np.zeros_like(y, float)
    for tr, te in kf.split(X):
        m = GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.03)
        m.fit(X[tr], y[tr]); pred[te] = m.predict(X[te])
    e = pred - y
    print(f"  {name:5s}: MAE {np.abs(e).mean():5.1f} | ME {e.mean():+5.1f} ± {e.std():4.1f} | r {np.corrcoef(pred,y)[0,1]:+.2f}")


def main():
    X12, ys, yd, _ = load12()
    Xr, ys2, yd2 = load_rich()
    print(f"12 features : n={len(ys)} | riches : n={len(ys2)} ({Xr.shape[1]} features)\n")
    print("SBP :"); cv(X12, ys, "12f"); cv(Xr, ys2, "riche")
    print("DBP :"); cv(X12, yd, "12f"); cv(Xr, yd2, "riche")


if __name__ == '__main__':
    main()
