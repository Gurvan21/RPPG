#!/usr/bin/env python3
"""Courbe d'apprentissage BP avec waveform + âge + genre : le plancher est-il
assez bas pour que 1000 sujets donnent du clinique ?"""
import sys, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.bp_demo import load
from sklearn.ensemble import GradientBoostingRegressor
from scipy.optimize import curve_fit


def curve(X, y, name):
    print(f"\n=== {name} (n={len(y)}, moy {y.mean():.0f}±{y.std():.0f}) ===")
    print(f"  {'Ntrain':>7} {'MAE':>6} {'SD':>6} {'r':>6}")
    rng = np.random.default_rng(0); n = len(y)
    sizes = [20, 40, 60, 80, min(100, n-15)]; Ns, MAEs, SDs = [], [], []
    for N in sizes:
        maes, sds, rs = [], [], []
        for _ in range(40):
            idx = rng.permutation(n); tr, te = idx[:N], idx[N:]
            if len(te) < 10: continue
            m = GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.03)
            m.fit(X[tr], y[tr]); p = m.predict(X[te]); e = p - y[te]
            maes.append(np.abs(e).mean()); sds.append(e.std())
            rs.append(np.corrcoef(p, y[te])[0, 1] if np.std(p) > 0 else 0)
        Ns.append(N); MAEs.append(np.mean(maes)); SDs.append(np.mean(sds))
        print(f"  {N:>7} {np.mean(maes):>6.1f} {np.mean(sds):>6.1f} {np.mean(rs):>6.2f}")
    Ns, MAEs, SDs = np.array(Ns, float), np.array(MAEs), np.array(SDs)
    law = lambda N, f, a: f + a/np.sqrt(N)
    try:
        (fm, am), _ = curve_fit(law, Ns, MAEs, p0=[10, 50], maxfev=5000)
        (fs, as_), _ = curve_fit(law, Ns, SDs, p0=[8, 60], maxfev=5000)
        for Nx in [1000, 3000]:
            print(f"    N={Nx}: MAE≈{law(Nx,fm,am):.1f} SD≈{law(Nx,fs,as_):.1f} (AAMI SD≤8: {'✅' if law(Nx,fs,as_)<=8 else '❌'})")
        print(f"    plancher N→∞ : MAE≈{fm:.1f} SD≈{fs:.1f}")
    except Exception as e:
        print(f"    (extrapolation KO: {e})")


def main():
    wf, age, gen, sbp, dbp = load()
    X = np.column_stack([wf, age, gen])
    print(f"{len(sbp)} sujets | waveform+âge+genre ({X.shape[1]} features)")
    curve(X, sbp, "SBP")
    curve(X, dbp, "DBP")


if __name__ == '__main__':
    main()
