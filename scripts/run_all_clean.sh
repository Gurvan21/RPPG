#!/bin/bash
# Orchestrateur "base propre" v2 (réordonné) :
#   Phase 1 — extraction fraîche PARALLÈLE (region_signals + clips PhysNet), fps=30
#             REPREND là où l'extraction précédente s'est arrêtée (skip des déjà-faits)
#   Phase 2 — entraînement : CNN 1D + CHROM-ITA D'ABORD (rapides), PhysNet EN DERNIER (long)
#
# Logs : /tmp/run_all_clean.log + /tmp/{region_shard_*,physnet_shard_*,train_*}.log

cd "$(dirname "$0")/.."
LOG=/tmp/run_all_clean.log
echo "=== START v2 (réordonné) $(date) ===" >> $LOG

REGION_SHARDS=4
PHYSNET_SHARDS=3
mkdir -p Data/region_signals Data/clips_clean

# ── Phase 1a : region_signals — reprend (extract_regions skippe les sc déjà faits) ──
echo "[1a] region_signals : $REGION_SHARDS shards (reprise) $(date)" >> $LOG
region_pids=()
for i in $(seq 0 $((REGION_SHARDS-1))); do
  caffeinate -i python3 scripts/extract_regions_bisenet.py \
    --data DataVital --out Data/region_signals --shard $i --n-shards $REGION_SHARDS \
    > /tmp/region_shard_$i.log 2>&1 &
  region_pids+=($!)
done

# ── Phase 1b : clips PhysNet — reprend (skip des sujets déjà extraits) ──
echo "[1b] clips PhysNet : $PHYSNET_SHARDS shards (reprise) $(date)" >> $LOG
rm -f /tmp/physnet_shard_*.txt
idx=0
for d in DataVital/Subject\ *; do
  [ -d "$d" ] || continue
  echo "$d" >> "/tmp/physnet_shard_$((idx % PHYSNET_SHARDS)).txt"
  idx=$((idx+1))
done
physnet_pids=()
for i in $(seq 0 $((PHYSNET_SHARDS-1))); do
  ( while IFS= read -r d; do
      name="$(basename "$d")"
      ls "Data/clips_clean/$name"/*.npz >/dev/null 2>&1 && continue   # déjà extrait → skip
      caffeinate -i python3 scripts/preextract_clips.py --subject-dir "$d" --output-dir Data/clips_clean
    done < "/tmp/physnet_shard_$i.txt" ) > /tmp/physnet_shard_$i.log 2>&1 &
  physnet_pids+=($!)
done

for p in "${region_pids[@]}"; do wait "$p"; done
echo "[1a] region_signals DONE $(date) — $(find Data/region_signals -name '*.npz' | wc -l) scénarios" >> $LOG
for p in "${physnet_pids[@]}"; do wait "$p"; done
echo "[1b] clips PhysNet DONE $(date) — $(find Data/clips_clean -name '*.npz' | wc -l) clips" >> $LOG

# ── Phase 2a : CNN 1D (rapide, meilleur global) ──
echo "[2a] CNN 1D $(date)" >> $LOG
caffeinate -i python3 scripts/train_cnn1d.py \
  --data Data/region_signals --output weights/clean_cnn1d.pth --epochs 60 \
  > /tmp/train_cnn1d.log 2>&1 && echo "[2a] OK" >> $LOG || echo "[2a] ÉCHEC (voir /tmp/train_cnn1d.log)" >> $LOG

# ── Phase 2b : CHROM-ITA (rapide, meilleur sur peaux foncées) ──
echo "[2b] CHROM-ITA $(date)" >> $LOG
caffeinate -i python3 scripts/eval_chrom_conditioned_regions.py \
  --epochs 150 --save weights/clean_chrom_conditioned.pth \
  > /tmp/train_chrom.log 2>&1 && echo "[2b] OK" >> $LOG || echo "[2b] ÉCHEC (voir /tmp/train_chrom.log)" >> $LOG

# ── Phase 2c : PhysNet fine-tune (long, en dernier) ──
echo "[2c] PhysNet fine-tune $(date)" >> $LOG
caffeinate -i python3 scripts/finetune_physnet.py \
  --data-root Data/clips_clean --output weights/clean_physnet_A_pure \
  --base-weights PURE --no-augment --lr 1e-4 --epochs 30 --batch 2 --accum 8 \
  > /tmp/train_physnet.log 2>&1 && echo "[2c] OK" >> $LOG || echo "[2c] ÉCHEC (voir /tmp/train_physnet.log)" >> $LOG

echo "=== ALL DONE $(date) ===" >> $LOG
