#!/usr/bin/env bash
# ============================================================
# Téléchargement Africa-1 par batch + pré-extraction immédiate
# Nécessite : lftp  (sudo apt install lftp)
#
# Usage :
#   chmod +x scripts/download_africa1.sh
#   FTP_USER=xxx FTP_PASS=xxx bash scripts/download_africa1.sh
#
# Le script :
#   1. Liste les sujets disponibles sur le serveur
#   2. Télécharge BATCH_SIZE sujets à la fois
#   3. Lance la pré-extraction (vidéo → clips .npz, ~30 Mo/sujet)
#   4. Supprime la vidéo brute (~4 Go/sujet)
#   5. Recommence jusqu'à tout traiter
# ============================================================

set -euo pipefail

FTP_HOST="${FTP_HOST:-u378752-sub18.your-storagebox.de}"
FTP_USER="${FTP_USER:-}"
FTP_PASS="${FTP_PASS:-}"
BATCH_SIZE="${BATCH_SIZE:-20}"     # sujets par batch (~88 Go)
LOCAL_RAW="Data/africa1_raw"       # vidéos brutes temporaires
LOCAL_CLIPS="Data/africa1_clips"   # clips pré-extraits (gardés)
PYTHON="/home/kemnhou/miniconda3/envs/rppg-toolbox/bin/python"

if [[ -z "$FTP_USER" || -z "$FTP_PASS" ]]; then
  echo "Usage : FTP_USER=xxx FTP_PASS=xxx bash $0"
  exit 1
fi

mkdir -p "$LOCAL_RAW" "$LOCAL_CLIPS"

# ── 1. Lister les dossiers sujets sur le serveur ─────────────
echo "[1] Listage des sujets sur le serveur..."
SUBJECT_LIST=$(lftp -u "$FTP_USER","$FTP_PASS" "ftp://$FTP_HOST" -e "
  set ssl:verify-certificate no
  ls
  quit
" 2>/dev/null | awk '{print $NF}' | grep -E '^[0-9]+$|^subject|^S' | sort)

if [[ -z "$SUBJECT_LIST" ]]; then
  echo "  Impossible de lister. Vérifier les credentials ou la structure du serveur."
  echo "  Lance FileZilla sur $FTP_HOST pour voir l'arborescence."
  exit 1
fi

TOTAL=$(echo "$SUBJECT_LIST" | wc -l)
echo "  $TOTAL sujets trouvés."

# ── 2. Traitement par batch ───────────────────────────────────
DONE=0
while IFS= read -r batch; do
  SUBJECTS=($batch)
  echo ""
  echo "═══ Batch $((DONE/BATCH_SIZE + 1)) : ${#SUBJECTS[@]} sujets ═══"

  # Télécharger le batch
  for subj in "${SUBJECTS[@]}"; do
    echo "  ↓ Téléchargement $subj..."
    lftp -u "$FTP_USER","$FTP_PASS" "ftp://$FTP_HOST" -e "
      set ssl:verify-certificate no
      mirror --parallel=4 $subj $LOCAL_RAW/$subj
      quit
    " 2>/dev/null
  done

  # Pré-extraire les clips (vidéo → .npz)
  echo "  ⚙ Pré-extraction des clips..."
  for subj in "${SUBJECTS[@]}"; do
    $PYTHON scripts/preextract_clips.py \
      --subject-dir "$LOCAL_RAW/$subj" \
      --output-dir  "$LOCAL_CLIPS" \
      --delete-raw
  done

  DONE=$((DONE + ${#SUBJECTS[@]}))
  echo "  ✓ $DONE/$TOTAL sujets traités"

done < <(echo "$SUBJECT_LIST" | xargs -n "$BATCH_SIZE")

echo ""
echo "═══ Terminé : clips dans $LOCAL_CLIPS ═══"
du -sh "$LOCAL_CLIPS"
