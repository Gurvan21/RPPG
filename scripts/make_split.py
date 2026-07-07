#!/usr/bin/env python3
"""Génère un split déterministe (seed 42, 70/15/15) des sujets d'un dossier de
clips, écrit dans un JSON partagé par les deux entraînements."""
import sys, json, random
from pathlib import Path
root = Path(sys.argv[1] if len(sys.argv) > 1 else 'Data/clips_tscan')
out = sys.argv[2] if len(sys.argv) > 2 else 'Data/split_fair.json'
dirs = sorted([d.name for d in root.iterdir() if d.is_dir()])
random.seed(42); random.shuffle(dirs)
n = len(dirs); nt = max(1, int(n*0.15)); nv = max(1, int(n*0.15))
sp = {'test': dirs[:nt], 'val': dirs[nt:nt+nv], 'train': dirs[nt+nv:]}
json.dump(sp, open(out, 'w'), indent=1)
print(f"{n} sujets -> {len(sp['train'])} train / {len(sp['val'])} val / {len(sp['test'])} test -> {out}")
