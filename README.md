# mp-rppg-eval

Pipeline rPPG (CHROM / POS / PhysNet) avec extraction de signal par **MediaPipe FaceMesh**
(régions front / joues / moyenne) et **Haar Cascade**, évalué sur le dataset
[UBFC-rPPG](https://sites.google.com/view/ybenezeth/ubfcrppg).

Projet d'apprentissage / validation extrait du
[rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox) (ubicomplab).

## Structure

```
mp_rppg/        package principal
  pipeline.py     extraction RGB par régions MediaPipe FaceMesh (front/joues/mean)
  backends.py     extraction RGB via Haar Cascade (et Y5F, non fourni ici)
  methods.py      implémentations CHROM (De Haan 2013) et POS (Wang 2017)
  metrics.py      HR par FFT, SNR, agrégation MAE/RMSE
  plots.py        graphiques de comparaison

models/
  physnet.py      architecture PhysNet (Yu et al., BMVC 2019)

weights/
  SCAMPS_PhysNet_DiffNormalized.pth   poids pré-entraînés (SCAMPS, MIT license)
  UBFC-rPPG_PhysNet_DiffNormalized.pth  poids pré-entraînés (UBFC, recherche uniquement)

assets/
  haarcascade_frontalface_default.xml

scripts/
  run_full_eval.py     évaluation complète UBFC : CHROM/POS (HC, MP-front, MP-mean) + PhysNet (SCAMPS, UBFC)
  evaluate_ubfc.py      comparaison HC vs MediaPipe (front/left/right/mean) avec CHROM/POS
  infer_physnet.py      inférence PhysNet sur une vidéo personnelle
  analyze_video.py       analyse CHROM/POS sur une vidéo personnelle (HC/MP)
  visualize_rois.py       visualisation des ROI MediaPipe sur des frames

results/
  eval_ubfc/      graphiques de l'évaluation sur UBFC-rPPG (3 sujets)
  mp_eval/        comparaison SNR backends sur vidéo personnelle
  personal_video/ visualisations ROI + spectres PhysNet sur vidéo personnelle
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Obtenir le dataset UBFC-rPPG

Le dataset n'est **pas inclus** dans ce repo (trop volumineux, licence académique).
Demander l'accès via le formulaire officiel :
https://sites.google.com/view/ybenezeth/ubfcrppg

Structure attendue :
```
UBFC-rPPG/
  subject1/
    vid.avi
    ground_truth.txt
  subject3/
    ...
```

## Usage

```bash
# Évaluation complète sur UBFC-rPPG (CHROM/POS + PhysNet SCAMPS/UBFC)
python scripts/run_full_eval.py --data /path/to/UBFC-rPPG --out results/eval_ubfc

# Test rapide sur 3 sujets
python scripts/run_full_eval.py --data /path/to/UBFC-rPPG --subjects 3

# Comparaison HC vs MediaPipe (front/left/right/mean) avec CHROM/POS
python scripts/evaluate_ubfc.py --data /path/to/UBFC-rPPG

# Sur une vidéo personnelle (sans ground truth)
python scripts/analyze_video.py --video ma_video.avi --backend MP --debug --ref-hr 70
python scripts/infer_physnet.py --video ma_video.avi --weights UBFC --ref-hr 70
python scripts/visualize_rois.py --video ma_video.avi
```

## Résultats (3 sujets UBFC : subject1, subject10, subject11)

| Méthode          | MAE (bpm) | RMSE (bpm) | SNR moy (dB) | <5bpm |
|------------------|-----------|------------|--------------|-------|
| HC/CHROM         | 3.48      | 5.55       | -4.22        | 67%   |
| HC/POS           | 0.29      | 0.51       | -2.28        | 100%  |
| MP-front/CHROM   | 9.04      | 15.67      | -3.34        | 67%   |
| MP-front/POS     | 0.29      | 0.51       | -1.67        | 100%  |
| MP-mean/CHROM    | 0.29      | 0.51       | -2.14        | 100%  |
| MP-mean/POS      | 0.29      | 0.51       | -1.05        | 100%  |
| PhysNet-SCAMPS   | 0.29      | 0.51       | 0.44         | 100%  |
| PhysNet-UBFC     | 0.29      | 0.51       | 2.35         | 100%  |

Voir `results/eval_ubfc/` pour les graphiques (MAE/RMSE, scatter HR vs GT, SNR, boxplot).

### Interprétation

- **POS et PhysNet sont robustes** sur les 3 sujets ; **CHROM** (surtout sur ROI étroite type
  HC ou MP-front) décroche occasionnellement vers une fréquence erronée (erreurs de 9 à 27 bpm).
- Le **SNR** mesure la concentration du spectre du signal BVP autour de la fréquence
  cardiaque (fondamentale + harmonique 2) vs le reste de la bande [42,150] bpm — PhysNet
  produit un signal "propre" car le réseau apprend à filtrer le bruit non corrélé au pouls,
  contrairement à CHROM/POS qui restent du traitement de signal sur le RGB brut.
- Les valeurs de HR identiques au bpm près entre méthodes très différentes (CHROM/POS/PhysNet)
  s'expliquent par la résolution du périodogramme (~0.9 bpm/bin pour ces longueurs de signal),
  pas par une coïncidence : tous les algos retombent sur le même bin de fréquence quand le
  signal cardiaque domine clairement le spectre.
- Échantillon de **3 sujets seulement** : le classement ci-dessus est indicatif, pas une
  validation statistique.

## Licences

- Code de ce repo : voir `LICENSE` du rPPG-Toolbox original (Responsible AI License).
- `weights/UBFC-rPPG_PhysNet_DiffNormalized.pth` et `weights/SCAMPS_PhysNet_DiffNormalized.pth` :
  poids PhysNet (Zitong Yu, BMVC 2019) — **usage recherche uniquement, usage commercial interdit**.
- `assets/haarcascade_frontalface_default.xml` : OpenCV (BSD).
