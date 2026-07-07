#!/usr/bin/env python3
"""
Validation croisée 5-fold du CNN1D-main, GROUPÉE PAR PERSONNE (pas par dossier) :
les dossiers Subject_X qui partagent un GUID = même humain → même fold (zéro fuite).
Entraîne un modèle frais par fold, agrège les prédictions held-out, rapporte par
Fitzpatrick. C'est le chiffre honnête et stable (vs un seul split chanceux).
"""
import os, sys, glob, json
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from models.cnn1d_rppg import CNN1D_rPPG
from mp_rppg.metrics import hr_from_fft, snr
from scripts.train_cnn1d import _temporal_norm, pearson_loss, CLIP_LEN
from scripts.train_cnn1d_hand import DS, subj_fitz, eval_subjects, report

HAND = ROOT / "Data" / os.environ.get("HAND_DIR", "hand_signals")


def person_groups():
    """Union-find des dossiers Subject_* partageant un GUID → groupes-personnes."""
    folder_guids = {}
    for jf in glob.glob(str(ROOT / "DataVital" / "Subject*" / "*.json")):
        try: g = json.load(open(jf)).get("GUID")
        except: continue
        folder_guids.setdefault(jf.split('/')[-2].replace(' ', '_'), set()).add(g)
    dirs = [d.name for d in HAND.iterdir() if d.is_dir()]
    parent = {d: d for d in dirs}
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b): parent[find(a)] = find(b)
    # relie les dossiers partageant un GUID
    guid_dirs = {}
    for d in dirs:
        for g in folder_guids.get(d, []):
            guid_dirs.setdefault(g, []).append(d)
    for g, ds in guid_dirs.items():
        for d in ds[1:]: union(ds[0], d)
    groups = {}
    for d in dirs: groups.setdefault(find(d), []).append(HAND / d)
    return list(groups.values())


def main():
    dev = torch.device('mps' if torch.backends.mps.is_available()
                       else 'cuda' if torch.cuda.is_available() else 'cpu')
    groups = person_groups()
    nd = sum(len(g) for g in groups)
    print(f"{nd} dossiers → {len(groups)} personnes (après fusion GUID), device {dev}")
    rng = np.random.default_rng(0); idx = rng.permutation(len(groups))
    K = 5; folds = [idx[i::K] for i in range(K)]
    all_rows = []
    for k in range(K):
        te_g = [groups[i] for i in folds[k]]
        tr_g = [groups[i] for i in np.concatenate([folds[j] for j in range(K) if j != k])]
        test_dirs = [d for g in te_g for d in g]
        train_dirs = [d for g in tr_g for d in g]
        ds_tr = DS(train_dirs)
        ld = DataLoader(ds_tr, batch_size=16, shuffle=True)
        model = CNN1D_rPPG(in_channels=ds_tr[0][0].shape[0]).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60, eta_min=1e-5)
        for ep in range(60):
            model.train()
            for x, y, _ in ld:
                x, y = x.to(dev), y.to(dev); opt.zero_grad()
                loss = pearson_loss(model(x), y); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            sched.step()
        rows = eval_subjects(model, test_dirs, dev, None)
        all_rows += rows
        mae = np.mean([r[1] for r in rows]) if rows else float('nan')
        print(f"  fold {k+1}/{K}: {len(test_dirs)} dossiers test, MAE={mae:.2f} (n={len(rows)})")
    report(all_rows, "CV 5-FOLD AGRÉGÉE (groupée par personne, sans fuite)")


if __name__ == '__main__':
    main()
