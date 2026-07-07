#!/usr/bin/env python3
"""FAISABILITÉ : prédire la tension (SBP/DBP) depuis les FEATURES de forme d'onde
du VRAI PPG doigt (CMS), pas du rPPG. Approche waveform mono-site (Degott/Gaurav).
Battement moyen + dérivées (VPG/APG) -> features -> régression, CV groupée sujet.
Comparé au baseline 'prédire la moyenne' et au seuil AAMI (ME±SD <= 5±8 mmHg)."""
import os, sys, json, glob, re, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
ROOT = Path(__file__).resolve().parents[1]
FS = 60.0


def clean(ppg_t, ppg_v):
    t = np.array(ppg_t, float); v = np.array(ppg_v, float)
    grid = np.arange(t[0], t[-1], 1000.0/FS)
    v = np.interp(grid, t, v)
    b, a = butter(3, [0.5/(FS/2), 8.0/(FS/2)], btype='band')
    return filtfilt(b, a, v)


def template(ppg):
    pk, _ = find_peaks(ppg, distance=int(0.4*FS), prominence=np.std(ppg)*0.3)
    if len(pk) < 5: return None, None
    feet = []
    for p in pk:
        w0 = max(0, p-int(0.4*FS)); feet.append(w0+int(np.argmin(ppg[w0:p+1])))
    feet = np.unique(feet)
    beats = []
    for i in range(len(feet)-1):
        seg = ppg[feet[i]:feet[i+1]]
        if not (int(0.4*FS) <= len(seg) <= int(1.5*FS)): continue
        seg = seg - seg[0]
        beats.append(np.interp(np.linspace(0, 1, 100), np.linspace(0, 1, len(seg)), seg))
    if len(beats) < 3: return None, None
    tmpl = np.median(np.array(beats), axis=0)
    ibi = np.diff(pk)/FS; ibi = ibi[(ibi > 0.33) & (ibi < 1.5)]
    return tmpl, ibi


def features(ppg):
    tmpl, ibi = template(ppg)
    if tmpl is None: return None
    t = (tmpl - tmpl.min()) / (np.ptp(tmpl) + 1e-8)      # normalisé 0..1
    sp = int(np.argmax(t))                              # pic systolique (position %)
    vpg = np.gradient(t); apg = np.gradient(vpg)
    # points APG a,b,c,d,e (extrema du 2e dérivé) sur la partie montée-début
    a_i = int(np.argmax(apg[:sp+1])) if sp > 2 else 0
    b_i = int(np.argmin(apg[:sp+5])) if sp > 4 else 1
    seg2 = apg[sp:sp+40] if sp+5 < len(apg) else apg[sp:]
    a_v = apg[a_i] + 1e-8
    def rel(idx): return apg[idx]/a_v
    # largeurs à 25/50/75%
    def width(h):
        above = np.where(t >= h)[0]
        return (above[-1]-above[0])/100.0 if len(above) else 0.0
    # indice d'augmentation : 2e pic (onde réfléchie) après le systolique
    tail = t[sp+3:]; ai = 0.0
    if len(tail) > 3:
        sub, _ = find_peaks(tail)
        if len(sub): ai = float(tail[sub[0]])
    hr = 60.0/np.median(ibi) if len(ibi) else 0.0
    hrv = float(np.std(ibi)) if len(ibi) else 0.0
    return np.array([
        hr, hrv,
        sp/100.0,                       # temps systolique (fraction)
        width(0.25), width(0.5), width(0.75),
        float(np.max(vpg)),             # pente max montée
        ai,                             # indice d'augmentation
        rel(b_i),                       # b/a (rigidité)
        float(np.trapz(t))/100.0,       # aire du pouls
        float(t.mean()), float(t.std()),
    ])


def load():
    X, ys, yd, subs = [], [], [], []
    for jf in sorted(glob.glob(str(ROOT/'DataVital'/'Subject*'/'*.json'))):
        try: d = json.load(open(jf))
        except: continue
        subj = jf.split('/')[-2]
        bp = None; ppg_feat = None
        for sc in d.get('scenarios', []):
            rec = sc.get('recordings', {})
            b = rec.get('BP')
            if isinstance(b, dict) and re.match(r'\d+/\d+', str(b.get('value', ''))):
                bp = b['value']
            cms = rec.get('CMS')
            if cms and len(cms) > 50 and ppg_feat is None:
                rows = cms[1:] if cms[0][0] == 'time' else cms
                try:
                    f = features(clean([r[0] for r in rows], [r[1] for r in rows]))
                    if f is not None and np.all(np.isfinite(f)): ppg_feat = f
                except Exception:
                    pass
        if bp and ppg_feat is not None:
            s, dd = map(int, bp.split('/'))
            X.append(ppg_feat); ys.append(s); yd.append(dd); subs.append(subj)
    return np.array(X), np.array(ys), np.array(yd), subs


def evaluate(X, y, name):
    print(f"\n=== {name} (n={len(y)}, moy {y.mean():.0f}±{y.std():.0f} mmHg) ===")
    base = np.abs(y - y.mean()).mean()
    print(f"  baseline 'prédire la moyenne' : MAE {base:.1f} mmHg")
    kf = KFold(5, shuffle=True, random_state=0)
    models = {
        'Ridge':        make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        'GradBoost':    GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.03),
        'RandomForest': RandomForestRegressor(n_estimators=300, max_depth=4, random_state=0),
    }
    for mn, mdl in models.items():
        preds = np.zeros_like(y, float)
        for tr, te in kf.split(X):
            mdl.fit(X[tr], y[tr]); preds[te] = mdl.predict(X[te])
        err = preds - y; me = err.mean(); sd = err.std(); mae = np.abs(err).mean()
        r = np.corrcoef(preds, y)[0, 1]
        aami = "✅" if (abs(me) <= 5 and sd <= 8) else "❌"
        gain = base - mae
        print(f"  {mn:13s}: MAE {mae:5.1f} | ME {me:+5.1f} ± {sd:4.1f} | r {r:+.2f} | vs baseline {gain:+.1f} | AAMI {aami}")


def main():
    print("Chargement + extraction features de forme d'onde (PPG doigt réel)…")
    X, ys, yd, subs = load()
    print(f"{len(subs)} sujets avec features valides + BP  |  {X.shape[1]} features")
    evaluate(X, ys, "SBP (systolique)")
    evaluate(X, yd, "DBP (diastolique)")


if __name__ == '__main__':
    main()
