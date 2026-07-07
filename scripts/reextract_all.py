#!/usr/bin/env python3
"""Ré-extrait avec xr TOUS les sujets DataVital utilisables non encore présents
dans Data/clips_tscan. Chaque sujet est extrait dans un SOUS-PROCESSUS isolé :
l'OS libère la mémoire entre sujets, et un OOM sur un sujet ne tue pas le batch."""
import sys, subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'DataVital'
OUT = ROOT / 'Data' / 'clips_tscan'
OUT.mkdir(parents=True, exist_ok=True)

subs = sorted([d.name for d in SRC.iterdir() if d.is_dir() and d.name.startswith('Subject')])
done = {d.name for d in OUT.iterdir() if d.is_dir() and any(d.glob('*.npz'))}
todo = []
for s in subs:
    if s in done: continue
    src = SRC / s
    has_vid = any(src.glob('*.mp4')) or any(src.glob('*.mov')) or any(src.glob('*.MOV'))
    if has_vid and any(src.glob('*.json')):
        todo.append(s)

print(f"Déjà extraits : {len(done)} | à extraire : {len(todo)}", flush=True)
ok = 0
for i, s in enumerate(todo):
    print(f"[{i+1}/{len(todo)}] {s} …", flush=True)
    r = subprocess.run(
        [sys.executable, str(ROOT/'scripts'/'preextract_clips.py'),
         '--subject-dir', str(SRC/s), '--output-dir', str(OUT), '--save-raw'],
        capture_output=True, text=True)
    nc = sum(1 for _ in (OUT/s).glob('*.npz')) if (OUT/s).is_dir() else 0
    if r.returncode == 0 and nc > 0:
        ok += 1; print(f"    OK : {nc} clips", flush=True)
    else:
        print(f"    ÉCHEC (rc={r.returncode}, clips={nc}) : {r.stderr.strip()[-150:]}", flush=True)
print(f"\nTerminé : {ok}/{len(todo)} nouveaux OK | total clips_tscan : "
      f"{sum(1 for d in OUT.iterdir() if d.is_dir() and any(d.glob('*.npz')))} sujets", flush=True)
