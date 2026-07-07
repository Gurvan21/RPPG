#!/bin/bash
# Extraction parallèle des signaux multi-régions (BiSeNet) sur N processus.
# Chaque processus traite 1 sujet sur N (sujets indépendants → parallélisable).
# Profite des multiples cœurs CPU (le goulot est le calcul des régions, pas le GPU).
#
# Usage : bash scripts/extract_parallel.sh [N_SHARDS] [DATA_DIR] [OUT_DIR]
#   N_SHARDS : nb de processus parallèles (défaut 6 ; ~6-8 conseillé sur Mac 10 cœurs)

set -e
N=${1:-6}
DATA=${2:-DataVital}
OUT=${3:-Data/region_signals}
cd "$(dirname "$0")/.."

echo "Extraction parallèle : $N processus, data=$DATA, out=$OUT"
pids=()
for i in $(seq 0 $((N-1))); do
  caffeinate -i python3 scripts/extract_regions_bisenet.py \
    --data "$DATA" --out "$OUT" --shard "$i" --n-shards "$N" \
    > "/tmp/extract_shard_${i}.log" 2>&1 &
  pids+=($!)
  echo "  shard $i lancé (PID $!) → /tmp/extract_shard_${i}.log"
done

echo "Attente des $N processus..."
for pid in "${pids[@]}"; do wait "$pid"; done
echo "Extraction parallèle terminée."
