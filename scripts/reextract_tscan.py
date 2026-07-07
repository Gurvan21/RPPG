#!/usr/bin/env python3
"""Ré-extrait les clips des sujets 'propres' (ceux de Data/clips_clean) en gardant
le flux apparence brut (xr) requis par TS-CAN. Sortie : Data/clips_tscan/<Subject>/."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import scripts.preextract_clips as P

P.SAVE_RAW = True
CLEAN = ROOT / 'Data' / 'clips_clean'
SRC   = ROOT / 'DataVital'
OUT   = ROOT / 'Data' / 'clips_tscan'

subjects = sorted([d.name for d in CLEAN.iterdir() if d.is_dir()])
print(f"{len(subjects)} sujets à ré-extraire → {OUT}")
total = 0
for i, s in enumerate(subjects):
    src = SRC / s
    if not src.is_dir():
        print(f"[{i+1}/{len(subjects)}] {s} : source absente, skip"); continue
    try:
        n = P.process_subject(src, OUT, P.DEFAULT_SYNC_OFFSET, delete_raw=False)
        total += n
    except Exception as e:
        print(f"[{i+1}/{len(subjects)}] {s} : ERREUR {e}")
print(f"\nTOTAL : {total} clips ré-extraits (avec xr) → {OUT}")
