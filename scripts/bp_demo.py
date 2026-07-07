#!/usr/bin/env python3
"""BP : est-ce que l'ÂGE + GENRE (démographie) améliorent la prédiction ?
Compare waveform seule / âge seul / âge+genre / waveform+âge+genre."""
import sys, json, glob, re, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.bp_from_ppg import clean, features


def load():
    rows = []
    for jf in sorted(glob.glob(str(ROOT/'DataVital'/'Subject*'/'*.json'))):
        try: d = json.load(open(jf))
        except: continue
        p = d.get('participant', {})
        try: age = float(p.get('age', ''));
        except: age = None
        gender = 1.0 if p.get('gender') == 'M' else (0.0 if p.get('gender') == 'F' else None)
        bp = None; wf = None
        for sc in d.get('scenarios', []):
            rec = sc.get('recordings', {})
            b = rec.get('BP')
            if isinstance(b, dict) and re.match(r'\d+/\d+', str(b.get('value', ''))): bp = b['value']
            cms = rec.get('CMS')
            if cms and len(cms) > 50 and wf is None:
                r = cms[1:] if cms[0][0] == 'time' else cms
                try:
                    f = features(clean([x[0] for x in r], [x[1] for x in r]))
                    if f is not None and np.all(np.isfinite(f)): wf = f
                except Exception: pass
        if bp and wf is not None and age is not None and gender is not None:
            s, dd = map(int, bp.split('/'))
            rows.append((wf, age, gender, s, dd))
    wf = np.array([r[0] for r in rows]); age = np.array([r[1] for r in rows])
    gen = np.array([r[2] for r in rows]); sbp = np.array([r[3] for r in rows]); dbp = np.array([r[4] for r in rows])
    return wf, age, gen, sbp, dbp


def cv(X, y):
    kf = KFold(5, shuffle=True, random_state=0); pred = np.zeros_like(y, float)
    for tr, te in kf.split(X):
        m = GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.03)
        m.fit(X[tr], y[tr]); pred[te] = m.predict(X[te])
    e = pred - y
    return np.abs(e).mean(), e.std(), np.corrcoef(pred, y)[0, 1]


def bench(y, wf, age, gen, name):
    print(f"\n=== {name} (n={len(y)}, moy {y.mean():.0f}±{y.std():.0f}) ===")
    sets = {
        'waveform (12)':        wf,
        'âge seul':             age[:, None],
        'âge + genre':          np.column_stack([age, gen]),
        'waveform + âge':       np.column_stack([wf, age]),
        'waveform + âge+genre': np.column_stack([wf, age, gen]),
    }
    print(f"  correlation âge↔BP brute : r = {np.corrcoef(age, y)[0,1]:+.2f}")
    for nm, X in sets.items():
        mae, sd, r = cv(np.asarray(X, float), y)
        print(f"  {nm:22s}: MAE {mae:5.1f} | SD {sd:4.1f} | r {r:+.2f}")


def main():
    wf, age, gen, sbp, dbp = load()
    print(f"{len(sbp)} sujets avec waveform + âge + genre + BP")
    print(f"âge : {age.min():.0f}-{age.max():.0f} (moy {age.mean():.0f}) | {int(gen.sum())} H / {int((1-gen).sum())} F")
    bench(sbp, wf, age, gen, "SBP")
    bench(dbp, wf, age, gen, "DBP")


if __name__ == '__main__':
    main()
