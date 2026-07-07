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

## Environnement
- **Machine de dev actuelle : MacBook Pro M5** (GPU **MPS**). L'ancienne config Linux/940MX ci-dessous est historique.
- Lancer les entraînements longs en `nohup python3 -u ... &` + `disown` (les wrappers `bash -c` meurent, le python simple survit).
- Vidéos DataVital lourdes (1-2 GB) → `load_video(..., max_dim=720)` sinon **OOM kill** (jetsam macOS tue silencieusement).
- Ancien (Linux) : `conda activate rppg-toolbox` — GPU NVIDIA 940MX (2 GB).

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

---

# TRAVAUX RÉCENTS (juillet 2026) — résultats & leçons

## Dataset DataVital (116 sujets, format VitalVideos)
- `DataVital/Subject N/*.json` + `*.mp4`. JSON par scénario : `recordings.{CMS, rr, RGB, BP}` + `participant.{gender, age, fitzpatrick}`.
- **CMS** = PPG doigt contact (~60 Hz, header `['time','ppg','hr','spo2']`). **BP** = brassard Omron M7, `value:"151/97"` (1 mesure/sujet). **rr** = ceinture respiration (~20 Hz). Cohorte : 88 F / 27 H, surtout **Fitzpatrick 5-6** (cible équité).
- `clips_clean/` = 89 sujets curés (clips DiffNormalized `x=(128,72,72,3)`, `y`, `fps`). `clips_tscan/` = clips ré-extraits **avec `xr`** (flux apparence brut standardisé, requis par TS-CAN) — ~102 sujets (`preextract_clips.py --save-raw`).

## Fine-tuning SOTA (TS-CAN) — NÉGATIF
- Port fidèle rPPG-Toolbox (`models/tscan_official.py`), poids pré-entraînés téléchargés (`weights/tscan_pretrained/{PURE,SCAMPS}_TSCAN.pth`, img_size 72).
- **3 runs concordants** : from-scratch **18,9** | PURE-ft **19,6** | SCAMPS — tous ~19-20 bpm MAE vs **PhysNet ~11**. **L'architecture n'est pas le levier** ; le pré-entraînement massif (SCAMPS) + qualité capture le sont. Ne pas courir après PhysFormer/RhythmFormer.
- Split partagé zéro-fuite : `scripts/make_split.py` → `Data/split_fair.json` ; `train_physnet_fair.py`, `finetune_tscan_pt.py`.

## Faisabilité TENSION (BP) — NÉGATIF quantifié (`scripts/bp_*.py`)
- Waveform features du PPG doigt → SBP **r≈0,45** (MAE 16, échoue AAMI). + **âge+genre** → **r≈0,58** (features démographiques aident, features waveform en plus = surapprentissage → PIRE).
- **Courbe d'apprentissage plafonne** : même extrapolé à 3000 sujets, SD~19 mmHg → **AAMI ❌** (le volume ne suffit pas).
- **Démo de fuite** (`bp_leakage_demo.py`) : split par battement (fuite) r 0,63 vs par sujet (honnête) r 0,33 → explique les « bons » chiffres littérature (MIMIC).
- **Seul levier = calibration par sujet** (1 brassard de réf → offset perso). Casse le cas d'usage dépistage (patient nouveau) → BP = outil de **suivi**, pas de dépistage. PTT (option SCG+doigt) : même plafond calibration, déconseillé.
- Binah & co : FC réelle (signal processing, marche), mais BP/hémoglobine/glucose = **claims wellness non homologués**. Leur moat = robustesse + SQA sur données diverses, pas un algo secret.

## Robustesse par augmentation — NÉGATIF (avec leçon méthodo)
- `scripts/augment.py` : dégradations frame (JPEG/mouvement/lumière/bruit ; `frame_augment_fast` block-compress rapide) + signal (CNN1D). Entraînement = clean+augmenté **mélangés** (50/50).
- **CNN1D** (`robust_cnn1d.py`) : aug signal **n'aide pas** (la normalisation temporelle la neutralise + proxy trop grossier).
- **PhysNet** (`robust_physnet.py`) : aug frame **déstabilise l'entraînement** (val oscille 9,8↔63,4, effondrement). À convergence : robuste **≈ ou pire** que baseline (propre 8,9/dégradé 19,5 vs robuste ~11/~20). Le « +11 bpm de gain » initial était un **ARTEFACT DE SOUS-ENTRAÎNEMENT**.
- **LEÇON MÉTHODO CLÉ : toujours comparer à CONVERGENCE.** Baseline 8 époques = 26 MAE (faux) ; 17 époques lr 3e-4 = ~6 val. On a failli conclure faux.

## Synchronisation collecte (`scripts/detect_beep.py`)
- Vidéo↔PPG : corrélation croisée du pouls (auto-sync) MAIS risque de saut de cycle (±1 battement). **Vraie synchro = horloge unique** (timestamp à l'arrivée) + **fenêtre bornée ±0,3 s** + **bip commun au départ** (repère non-périodique). NTP (ClockSync Android + `sntp` Mac) cale à ~100 ms — assez pour la FC, pas pour HRV/PTT.
- `detect_beep.py` **validé sur vraie vidéo** : retrouve 4 bips à la ms (3 kHz), rejette les échos.
- Respiration : rr trop courte (20 s) → régression échoue (MAE 9 ≈ constante). Besoin d'enregistrements longs (60-90 s de la collecte à venir).

## Nouveaux scripts clés
| Script | Rôle |
|--------|------|
| `models/tscan_official.py` | TS-CAN port rPPG-Toolbox (charge poids PURE/SCAMPS) |
| `scripts/finetune_tscan_pt.py`, `train_physnet_fair.py`, `make_split.py` | comparaison fair TS-CAN vs PhysNet, split partagé |
| `scripts/bp_from_ppg.py`, `bp_demo.py`, `bp_learning_curve.py`, `bp_leakage_demo.py` | faisabilité BP (waveform + démographie + fuite) |
| `scripts/augment.py`, `robust_physnet.py`, `robust_cnn1d.py`, `viz_degrade.py` | robustesse par augmentation |
| `scripts/detect_beep.py` | détection bip synchro dans l'audio vidéo |
| `scripts/reextract_all.py` | ré-extraction clips_tscan (subprocess isolé, évite OOM) |
