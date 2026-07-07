# Comment entraîner les réseaux (mp-rppg-eval)

> Tous les entraînements tournent sur MacBook M5 (MPS). Lancer en `nohup python3 -u ... &` + `disown`
> pour les runs longs. **Toujours comparer à convergence** (un modèle sous-entraîné donne des chiffres faux).

## 0. Préparer les données (clips)

Les réseaux mangent des **clips pré-extraits** `.npz`. Deux formats selon le modèle :

- **PhysNet / TS-CAN** = frames `x=(128,72,72,3)` (DiffNormalized) + `y=(128,)` (+ `xr` brut pour TS-CAN).
  ```bash
  # 1 sujet DataVital (VitalVideos) → clips 72×72 avec flux apparence xr
  python scripts/preextract_clips.py --subject-dir "DataVital/Subject 1" \
      --output-dir Data/clips_tscan --save-raw
  # tous les sujets (sous-processus isolé, évite l'OOM sur vidéos >1GB)
  python scripts/reextract_all.py
  ```
- **CNN1D** = signaux de régions `x=(T,23,9)` (BiSeNet) + `y=(T,)`.
  ```bash
  python scripts/extract_regions_bisenet.py --data DataVital --out Data/region_signals --grid 4
  ```

Split partagé **sans fuite** (groupé par sujet, seed 42, 70/15/15) :
```bash
python scripts/make_split.py Data/clips_tscan Data/split_fair.json
```

## 1. PhysNet (3D-CNN, le meilleur ici — MAE ~9-11)

**Fine-tuning depuis une base pré-entraînée** (SCAMPS recommandé) :
```bash
python scripts/finetune_physnet.py --data-root Data/clips_tscan \
    --base-weights SCAMPS --epochs 20 --batch 2
# → weights/finetune/physnet_africa1_best.pth
```
**Sur split partagé (comparaison fair, zéro fuite)** :
```bash
python scripts/train_physnet_fair.py --split-file Data/split_fair.json \
    --base weights/SCAMPS_PhysNet_DiffNormalized.pth --out weights/physnet_fair.pth \
    --epochs 25 --batch 2
```
- Loss = **negative Pearson** ; lr 1e-4 (trop bas = convergence lente ; 3e-4 = plus rapide mais peut osciller avec augmentation).
- Éval : `hr_from_fft(pred)` vs `hr_from_fft(y)`. Prévoir ~18-25 époques (converge à val ~6).

## 2. TS-CAN (2 branches + Temporal Shift — MAE ~19, PERD contre PhysNet)

Nécessite le flux `xr` (clips avec `--save-raw`). **Fine-tune depuis poids officiels rPPG-Toolbox** :
```bash
python scripts/finetune_tscan_pt.py \
    --pretrained weights/tscan_pretrained/SCAMPS_TSCAN.pth \
    --split-file Data/split_fair.json --freeze none --tag fair \
    --physnet-weights weights/physnet_fair.pth --epochs 25 --batch 4
```
- Entrée = concat `[DiffNormalized(3), Raw standardisé(3)]` = 6 canaux ; `models/tscan_official.py` charge les poids PURE/SCAMPS (img_size 72).
- **Conclusion établie** : ne bat pas PhysNet, quelle que soit l'init → l'archi n'est pas le levier.

## 3. CNN1D (conv 1D sur signaux de régions — léger, ~113K params)

```bash
python scripts/train_cnn1d.py            # entraîne sur Data/region_signals
python scripts/cv_cnn1d_hand.py          # CV 5-fold groupée par personne (paume)
```
- Entrée `_temporal_norm(x)` → `(23*9, T)`, fenêtres `CLIP_LEN=128`. Loss = Pearson. Très rapide (~5 min).

## 4. Robustesse par augmentation (NÉGATIF — voir CLAUDE.md)

```bash
python scripts/robust_physnet.py   # baseline vs robuste (clean+augmenté), éval propre/dégradé
python scripts/robust_cnn1d.py     # idem niveau signal
```
- `scripts/augment.py` : `frame_augment_fast` (block-compress rapide, entraînement) vs `frame_degrade_fixed` (vrai JPEG, éval).
- **Résultat** : l'augmentation synthétique n'a pas débloqué la robustesse (déstabilise PhysNet, neutralisée sur CNN1D). Sélectionner le checkpoint sur la **val dégradée** pour lui donner sa meilleure chance.

## Pièges appris
- **Comparer à convergence** : baseline 8 ép. = 26 MAE (faux) ; 18 ép. = ~6. On a failli conclure faux.
- **OOM** : `load_video(..., max_dim=720)` sur vidéos DataVital 1-2 GB.
- **nohup `bash -c` meurt** ; `nohup python3` simple survit.
- Augmentation par blocs trop agressive → frames plates → DiffNormalized = bruit → entraînement s'effondre (val saute à 63).
