#!/usr/bin/env python3
"""Courbe d'apprentissage BP : l'erreur descend-elle encore avec plus de sujets
(-> 1000 aiderait) ou plafonne-t-elle (-> ceiling atteint) ? Extrapolation à 1000."""
import sys, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.bp_from_ppg import load
from sklearn.ensemble import GradientBoostingRegressor
from scipy.optimize import curve_fit


def model():
    return GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.03)


def curve(X, y, name):
    print(f"\n=== {name} (n={len(y)}, moy {y.mean():.0f}±{y.std():.0f}) ===")
    print(f"  {'Ntrain':>7} {'MAE':>6} {'SD':>6} {'r':>6}")
    rng = np.random.default_rng(0); n = len(y)
    sizes = [20, 40, 60, 80, min(100, n-15)]
    Ns, MAEs, SDs = [], [], []
    for N in sizes:
        maes, sds, rs = [], [], []
        for _ in range(40):
            idx = rng.permutation(n); tr, te = idx[:N], idx[N:]
            if len(te) < 10: continue
            m = model(); m.fit(X[tr], y[tr]); p = m.predict(X[te])
            e = p - y[te]; maes.append(np.abs(e).mean()); sds.append(e.std())
            rs.append(np.corrcoef(p, y[te])[0, 1] if np.std(p) > 0 else 0)
        Ns.append(N); MAEs.append(np.mean(maes)); SDs.append(np.mean(sds))
        print(f"  {N:>7} {np.mean(maes):>6.1f} {np.mean(sds):>6.1f} {np.mean(rs):>6.2f}")
    # extrapolation : MAE(N) = plancher + a / sqrt(N)
    Ns, MAEs, SDs = np.array(Ns, float), np.array(MAEs), np.array(SDs)
    def law(N, floor, a): return floor + a / np.sqrt(N)
    try:
        (f_mae, a_mae), _ = curve_fit(law, Ns, MAEs, p0=[10, 50], maxfev=5000)
        (f_sd, a_sd), _ = curve_fit(law, Ns, SDs, p0=[8, 60], maxfev=5000)
        print(f"  → extrapolation loi plancher+a/√N :")
        for Nx in [500, 1000, 3000]:
            print(f"      N={Nx:>4} : MAE≈{law(Nx,f_mae,a_mae):.1f}  SD≈{law(Nx,f_sd,a_sd):.1f}  (AAMI SD≤8 : {'✅' if law(Nx,f_sd,a_sd)<=8 else '❌'})")
        print(f"  plancher estimé (N→∞) : MAE≈{f_mae:.1f}  SD≈{f_sd:.1f}")
    except Exception as e:
        print(f"  (extrapolation impossible : {e})")


def main():
    X, ys, yd, subs = load()
    print(f"{len(subs)} sujets, {X.shape[1]} features")
    curve(X, ys, "SBP")
    curve(X, yd, "DBP")


if __name__ == '__main__':
    main()
