# Contexte projet — mp-rppg-eval

## Objectif
Pipeline rPPG (Remote PhotoPlethysmoGraphy) pour estimer la fréquence cardiaque depuis une vidéo de visage. Évaluation sur UBFC-rPPG et vidéos personnelles. Fine-tuning sur VitalVideos-Africa-1 pour améliorer les performances sur peaux foncées (Fitzpatrick 4-6).

## Méthodes implémentées
- **CHROM** et **POS** : méthodes classiques (De Haan / Wang), via MediaPipe FaceMesh (front + joues)
- **BKF+CHROM / BKF+POS** : Bounded Kalman Filter pour le tracking ROI + signal CHROM/POS (H channel abandonné — artefact ~48 bpm sur peaux foncées)
- **PhysNet** : réseau 3D CNN, poids SCAMPS et UBFC pré-entraînés, inférence par fenêtre glissante 128 frames

## Résultats clés (UBFC-rPPG, 5 sujets)
- BKF+CHROM : 5/5 corrects (MAE ≤ 4.4 bpm), meilleure méthode classique
- PhysNet SCAMPS : fonctionne bien sur UBFC (sujet1 = 0 bpm d'erreur), mauvais sur vidéo smartphone H.264
- PhysNet UBFC : encore moins généralisable
- Carnations foncées (subject13, 32) : SNR plus bas sur toutes méthodes → motivation Africa-1

## Environnement conda
```bash
conda activate rppg-toolbox
# Python : /home/kemnhou/miniconda3/envs/rppg-toolbox/bin/python
```
GPU : NVIDIA GeForce 940MX (2 GB VRAM)

## Données
- UBFC-rPPG : `Data/subject{N}/` (vid.avi + ground_truth.txt, single-row BVP)
- Vidéos personnelles : `results/personal_video/*.mp4` (ROnel, Arthur36, Arthur10, Nouvelle, VID_20260617_*)
- Africa-1 (accès obtenu) : ~114 sujets, en cours de téléchargement depuis FTP `u378752-sub18.your-storagebox.de`

## Scripts principaux
| Script | Rôle |
|--------|------|
| `scripts/analyze_personal_videos.py` | CHROM/POS sur vidéos perso |
| `scripts/eval_bkf_ubfc.py` | BKF (H/CHROM/POS) sur UBFC + vidéos perso |
| `scripts/infer_physnet.py` | Inférence PhysNet (SCAMPS/UBFC/AFRICA1) |
| `scripts/run_physnet_uncompressed.py` | PhysNet streaming sur vidéo 1080p |
| `scripts/collect_rppg_data.py` | Collecte synchronisée webcam téléphone + CMS50D+ |
| `scripts/sync_check.py` | Vérification sync vidéo/PPG après collecte |
| `scripts/preextract_clips.py` | Vidéo brute → clips .npz (pré-traitement Africa-1) |
| `scripts/finetune_physnet.py` | Fine-tuning PhysNet sur clips Africa-1 |
| `scripts/download_africa1.sh` | Téléchargement FTP Africa-1 par batch |

## Collecte de données (pipeline smartphone)
- Téléphone filme via **IP Webcam** app (Android) → stream `http://192.168.1.34:8080/video`
- Laptop capture via OpenCV + enregistre PPG via CMS50D+ (USB, `/dev/ttyUSB0`)
- Contrôle : fenêtre OpenCV sur écran laptop, ESPACE = start/stop, Q = quitter
- Sortie : `Data/collection/subject_XXX/{video.avi, timestamps.npy, ppg.csv, meta.json}`

## Fine-tuning Africa-1 — plan
1. Lire `readme.json` + `note on sync` (Dropbox) pour connaître format exact + offset sync
2. Installer lftp (`sudo apt install lftp`)
3. Télécharger par batch 20 sujets : `FTP_USER=xxx FTP_PASS=xxx bash scripts/download_africa1.sh`
4. Pré-extraire clips : `python scripts/preextract_clips.py --subject-dir ... --delete-raw`
5. Fine-tuner : `python scripts/finetune_physnet.py --data-root Data/africa1_clips --epochs 30 --batch 2 --accum 8`
- **IMPORTANT** : ajuster `DEFAULT_SYNC_OFFSET` dans `preextract_clips.py` après lecture de "note on sync"
- Poids sortie : `weights/finetune/physnet_africa1_best.pth`
- Utiliser ensuite avec `--weights AFRICA1` dans `infer_physnet.py`

## Bugs connus / décisions
- Canal H de HSV non fiable pour BKF : artefact ~47-53 bpm (3e harmonique respiration) sur peaux foncées → utiliser BKF+CHROM à la place
- PhysNet échoue sur vidéo smartphone H.264 : DiffNormalized amplifie artefacts de compression temporelle
- Affichage OpenCV : utiliser `DISPLAY=:1` (pas `:0`) sur cette machine
- UBFC data path : `Data/subject{N}/` (pas de sous-dossier DATASET_2)
