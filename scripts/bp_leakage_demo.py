"""DÉMONSTRATION de la fuite qui gonfle les résultats PPG-BP de la littérature.
Features PAR BATTEMENT (chaque battement = 1 échantillon, label = BP du sujet).
- Split ALÉATOIRE par battement (fuite : même sujet dans train ET test) -> erreur basse (bidon)
- Split PAR SUJET (GroupKFold, honnête) -> erreur réelle
L'écart entre les deux = l'ampleur de la fuite."""
import sys, json, glob, re, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import skew
from sklearn.model_selection import KFold, GroupKFold
from sklearn.ensemble import GradientBoostingRegressor
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.bp_from_ppg import clean
FS = 60.0


def beat_feats(ppg):
    """Un vecteur de features PAR BATTEMENT."""
    pk, _ = find_peaks(ppg, distance=int(0.4*FS), prominence=np.std(ppg)*0.3)
    if len(pk) < 5: return []
    feet = []
    for p in pk:
        w0 = max(0, p-int(0.4*FS)); feet.append(w0+int(np.argmin(ppg[w0:p+1])))
    feet = np.unique(feet)
    out = []
    for i in range(len(feet)-1):
        seg = ppg[feet[i]:feet[i+1]]
        if not (int(0.4*FS) <= len(seg) <= int(1.5*FS)): continue
        b = np.interp(np.linspace(0, 1, 100), np.linspace(0, 1, len(seg)), seg)
        amp = np.ptp(b); b = (b-b.min())/(amp+1e-8); sp = int(np.argmax(b))
        vpg = np.gradient(b)
        def w(h):
            a = np.where(b >= h)[0]; return (a[-1]-a[0])/100 if len(a) else 0.0
        ibi = len(seg)/FS
        out.append([60/ibi if ibi > 0 else 0, amp, sp/100, w(0.25), w(0.5), w(0.75),
                    float(np.max(vpg)), float(np.trapz(b))/100, float(skew(b))])
    return out


def load_beats():
    X, y, g = [], [], []
    for jf in sorted(glob.glob(str(ROOT/'DataVital'/'Subject*'/'*.json'))):
        try: d = json.load(open(jf))
        except: continue
        bp = None; feats = None
        for sc in d.get('scenarios', []):
            rec = sc.get('recordings', {})
            b = rec.get('BP')
            if isinstance(b, dict) and re.match(r'\d+/\d+', str(b.get('value', ''))): bp = b['value']
            cms = rec.get('CMS')
            if cms and len(cms) > 50 and feats is None:
                r = cms[1:] if cms[0][0] == 'time' else cms
                try: feats = beat_feats(clean([x[0] for x in r], [x[1] for x in r]))
                except Exception: feats = []
        if bp and feats:
            s = int(bp.split('/')[0])
            for f in feats:
                if np.all(np.isfinite(f)): X.append(f); y.append(s); g.append(jf)
    return np.array(X), np.array(y), np.array(g)


def run(X, y, splitter, groups=None):
    pred = np.zeros_like(y, float)
    it = splitter.split(X, y, groups) if groups is not None else splitter.split(X)
    for tr, te in it:
        m = GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=0.05)
        m.fit(X[tr], y[tr]); pred[te] = m.predict(X[te])
    e = pred - y
    return np.abs(e).mean(), e.std(), np.corrcoef(pred, y)[0, 1]


def main():
    X, y, g = load_beats()
    print(f"{len(X)} battements / {len(set(g))} sujets  (label = SBP du sujet)\n")
    m1, s1, r1 = run(X, y, KFold(5, shuffle=True, random_state=0))
    print(f"  Split ALÉATOIRE par battement (FUITE)  : MAE {m1:5.1f} | SD {s1:4.1f} | r {r1:+.2f}   <- résultat 'littérature' gonflé")
    m2, s2, r2 = run(X, y, GroupKFold(5), groups=g)
    print(f"  Split PAR SUJET (honnête, GroupKFold)   : MAE {m2:5.1f} | SD {s2:4.1f} | r {r2:+.2f}   <- réalité")
    print(f"\n  => La fuite fait passer le MAE de {m2:.1f} à {m1:.1f} mmHg (×{m2/m1:.1f} plus 'beau') sans aucune vraie amélioration.")


if __name__ == '__main__':
    main()
