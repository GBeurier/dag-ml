# DAG-ML Use Cases v1

Statut: design v1. Compagnon des specs `dag_ml_externalization_from_code.md`
et `ml_data_specification_v1.md`. Chaque use case (UC) materialise le DSL,
le DAG compile, les invariants, l'ordre des phases et les artifacts.

Notation employee dans tout le document:

- Phases canoniques: `COMPILE -> PLAN -> FIT_CV -> SELECT -> REFIT -> PREDICT [-> EXPLAIN]`.
- Types references: voir `ml_data/contract.py` (`SourceDescriptor`, `DataBlock`,
  `DataView`, `FeatureTable`, `ModelInputSpec`, `FusionPolicy`, `DataPlan`,
  `SampleRelation`, `AggregationPolicy`, `SplitPolicy`, `AugmentationPolicy`,
  `SeedContext`, `SearchSpace`, `SubgraphNodeSpec`).
- Identifiants des noeuds: `<kind>:<short>`. Edges: `A.out -> B.in`.
- Conventions: `OOF` = out-of-fold predictions, `MS` = multi-source.

Table des matieres:

| UC  | Theme principal                                       | Axes couverts                          |
|-----|-------------------------------------------------------|----------------------------------------|
| UC1 | Multi-source heterogene -> RandomForest               | DataPlan, early fusion, missing sources|
| UC2 | NIRS multi-instrument (3 spectrometres)               | MS homogene, by_source, concat/stack   |
| UC3 | Repetitions: plusieurs X pour un Y                    | SampleRelation, group-aware split, agg |
| UC4 | Entites (patients, parcelles) split-unit              | SplitPolicy(split_unit="group")        |
| UC5 | Augmentation train-only + OOF correct                 | AugmentationPolicy, origin tracking    |
| UC6 | Stacking multi-niveau                                 | Branch+merge predictions, meta-modele  |
| UC7 | Generateurs + tuning bayesien                         | _cartesian_, _grid_, TunerAdapter      |
| UC8 | Branches par metadata/tag, merge concat               | Separation branches, by_metadata       |
| UC9 | Refit complet + bundle predict new heterogene         | schema_fingerprint, replay             |
| UC10| Sous-DAG reifie comme noeud                           | SubgraphNodeSpec, inline vs opaque     |
| UC11| OOF train preds refusees par defaut                   | Erreurs invariants, leakage opt-in     |
| UC12| Mixed merge: features + predictions OOF               | Cross-source validation                |

---

## UC1 - Multi-source heterogene vers RandomForest

### 1.1 Contexte metier

Prediction de la teneur en proteines d'une variete de ble a partir de cinq
modalites par echantillon: 1 spectre NIRS, 2 photos RGB (vue dessus, vue cote),
1 patrimoine genotypique (SNP dosage), 1 serie meteo journaliere sur la saison
de culture, plus quelques metadata categorielles (variete, parcelle, annee).

### 1.2 Donnees impliquees

| Source       | Type ML_DATA       | Modality      | Granularity         | Native rep                  | Sample key  | N samples | Notes |
|--------------|---------------------|---------------|---------------------|-----------------------------|-------------|-----------|-------|
| `nir`        | `dense_signal`      | spectroscopy  | `per_sample`        | `signal_with_processings`   | `sample_id` | 800       | 512 wl 950-2500 nm |
| `photo_top`  | `image_rgb`         | image         | `per_sample`        | `rgb_image`                 | `sample_id` | 800       | 224x224x3 |
| `photo_side` | `image_rgb`         | image         | `per_sample`        | `rgb_image`                 | `sample_id` | 760       | 40 missing |
| `geno`       | `genotype_matrix`   | genotype      | `per_sample`        | `variant_matrix`            | `sample_id` | 800       | 12000 SNPs, int8 |
| `weather`    | `time_series`       | meteo         | `per_sample_sequence`| `series_mv`                | `sample_id` | 800       | 180 days x 6 vars |
| `meta`       | `table`             | metadata      | `per_sample`        | `tabular_mixed`             | `sample_id` | 800       | variety/plot/year |
| target `y`   | `table`             | -             | `per_sample`        | `tabular_numeric`           | `sample_id` | 800       | proteine (%) |

Repetitions: aucune. `group_ids = plot_id` (40 parcelles, ~20 samples/parcelle).

### 1.3 Pipeline DSL souhaite

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from nirs4all.operators.transforms import SNV, SavitzkyGolay
from ml_data.contract import FusionPolicy, AlignmentPolicy

pipeline = [
    {"sources": ["nir", "photo_top", "photo_side", "geno", "weather", "meta"]},
    {"y_processing": "standardize"},

    # Per-source preprocessing (ML_DATA adapters declared here)
    {"by_source": {
        "nir":        [SNV(), SavitzkyGolay(window=11, deriv=1)],
        "photo_top":  [{"adapter": "image.embedding", "params": {"backbone": "resnet18", "out_dim": 256}}],
        "photo_side": [{"adapter": "image.embedding", "params": {"backbone": "resnet18", "out_dim": 256}}],
        "geno":       [{"adapter": "genotype.pca", "params": {"k": 32}}],
        "weather":    [{"adapter": "weather.aggregate", "params": {"stats": ["mean","std","min","max","p25","p75"]}}],
        "meta":       [{"adapter": "tabular.encoder", "params": {"categorical": "one_hot"}}],
    }},

    GroupKFold(n_splits=5),

    {"fusion": FusionPolicy(
        mode="concat_features",
        target_representation="tabular_numeric",
        alignment=AlignmentPolicy(join="left", reference_source="nir",
                                   on_missing_sample="mask"),
        missing_source="indicator",
        namespace_columns=True,
        allow_lossy_adapters=True,            # image emb + pca explicit
        max_output_features=2048,
    )},

    {"split_policy": {"split_unit": "group", "group_key": "plot_id"}},

    {"model": RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=0)},
]
```

### 1.4 DAG resultant

```text
                       +----------------------+
                       | SourcesNode          |
                       | id=src:all           |
                       +----------+-----------+
                                  |
            +--------+--------+--------+--------+--------+--------+
            |        |        |        |        |        |        |
            v        v        v        v        v        v        |
        materialize materialize materialize materialize materialize materialize
          src:nir   src:photo_top src:photo_side src:geno src:weather src:meta
            |         |         |         |         |         |
   [SNV+SG] |  [image.emb]      |   [geno.pca k=32] |  [weather.agg]   [tab.encoder]
            v         v         v         v         v         v
        adapt:nir  adapt:imgT  adapt:imgS adapt:geno adapt:wx adapt:meta
            \________\_______ \_______ /_______/_______/
                              align:by_sample_id (left,nir,mask)
                                       |
                                       v
                              fusion.feature_joiner (concat, indicator)
                                       |
                                       v
                              split:GroupKFold(5, plot_id)
                                       |
                              5 folds, OOF predictions
                                       v
                              model:RF(500) -- fit per fold
                                       |
                                       v
                              predictions OOF (per sample, fold_id, partition=val)
                                       |
                              SELECT (rmsecv)
                                       v
                              REFIT (single fold "final")
                                       v
                              ExecutionBundle (artifacts + DataPlan + RF model)
```

Liste des noeuds (NodeSpec):

| Node id              | kind        | Operator / adapter                    | Phase scope |
|----------------------|-------------|----------------------------------------|-------------|
| `src:nir`            | materialize | `MLDataset.materialize("nir")`         | all         |
| `src:photo_top`      | materialize | idem                                   | all         |
| `src:photo_side`     | materialize | idem (presence mask)                   | all         |
| `src:geno`           | materialize | idem                                   | all         |
| `src:weather`        | materialize | idem                                   | all         |
| `src:meta`           | materialize | idem                                   | all         |
| `adapt:nir`          | adapt       | `SNV` -> `SavitzkyGolay` -> `spectra.flatten` | all  |
| `adapt:imgT`         | adapt       | `image.embedding`                      | stateful    |
| `adapt:imgS`         | adapt       | `image.embedding`                      | stateful    |
| `adapt:geno`         | adapt       | `genotype.dosage` -> `genotype.pca`    | stateful    |
| `adapt:wx`           | adapt       | `weather.aggregate`                    | stateless   |
| `adapt:meta`         | adapt       | `tabular.encoder`                      | stateful    |
| `align:fuse`         | align       | `AlignmentPolicy(left, mask)`          | all         |
| `join:fuse`          | join        | `fusion.feature_joiner`                | stateful    |
| `split:cv`           | split       | `GroupKFold(5, plot_id)`               | FIT_CV      |
| `model:rf`           | model       | `RandomForestRegressor`                | FIT_CV+REFIT|
| `pred:rf`            | prediction  | per-fold OOF + agg                     | FIT_CV      |

Edges (selection):

```text
src:nir.out         -> adapt:nir.in
adapt:nir.out       -> align:fuse.in[nir]
adapt:imgT.out      -> align:fuse.in[photo_top]
adapt:imgS.out      -> align:fuse.in[photo_side]   (presence mask -> indicator column)
adapt:geno.out      -> align:fuse.in[geno]
adapt:wx.out        -> align:fuse.in[weather]
adapt:meta.out      -> align:fuse.in[meta]
align:fuse.out      -> join:fuse.in
join:fuse.out       -> split:cv.X
target("y")         -> split:cv.y
split:cv.folds      -> model:rf.folds
model:rf.pred_val   -> pred:rf.oof
```

### 1.5 Invariants ML cles

- OOF: `model:rf` emet predictions `partition="val"`, alignees par `sample_id`.
- No leakage: tous les `adapt:*` stateful (`image.embedding`, `genotype.pca`,
  `tabular.encoder`) ont `fit_scope="fold_train"`. `DataPlanner.execute_fit`
  les fit sur les samples du fold train uniquement.
- Split unit: `group` sur `plot_id` (GroupKFold) -> aucun plot n'apparait dans
  train et val du meme fold.
- Reproductibilite: `SeedContext.child(node_id="adapt:imgT", fold_id=k)` derive
  le seed du backbone (dropout, init head). Persiste dans `LineageRecord`.
- Refit: meme `DataPlan` reapplique sur train complet, `fold_id="final"`,
  emission de nouveaux `FittedAdapter` pour `image.embedding`, `genotype.pca`,
  `tabular.encoder`, `fusion.feature_joiner`, `RandomForestRegressor`.
- Missing sources: `photo_side` manque pour 40 samples -> presence mask
  propagee -> 1 colonne `indicator__photo_side_present` ajoutee par le joiner.

### 1.6 Points de friction

1. `image.embedding` est lossy + stateful. Faut-il forcer
   `policy.allow_lossy_adapters=True` explicitement ou exiger un choix utilisateur
   (escalation `requires_user_choice`) ?
2. La fusion `concat_features` peut exploser la dimension (256+256+32+36+~50+512
   ~= 1142). Faut-il imposer `max_output_features` ou warning seulement ?
3. Pour `photo_side` manquant, quelle politique par defaut: `indicator` (RF aime),
   `impute` (image -> embedding moyen train), ou `drop` ? Choix est ML, pas data.
4. `image.embedding` doit-il etre refit par fold (fit_scope="fold_train") ou
   shared `train_only` (un seul fit sur train complet, fige pour toute la CV) ?
   Premiere option = correct OOF mais cher; seconde option = leakage acceptable
   si backbone non finetuned.

### 1.7 ML_DATA vs DAG-ML

| ML_DATA                                                  | DAG-ML                                                   |
|----------------------------------------------------------|----------------------------------------------------------|
| Declare `SourceDescriptor` pour les 6 sources            | Compile le DSL en `GraphSpec`                            |
| `find_path(src, target="tabular_numeric", policy)`       | Decide `allow_lossy_adapters`, declenche refusal si flou |
| Execute `materialize`, `adapt`, `align`, `join`          | Choisit la phase (`fit_cv` -> `fold_train` scope)        |
| Fournit `PresenceMask` pour `photo_side`                 | Decide `missing_source="indicator"`                      |
| Calcule `schema_fingerprint`                             | Verifie le fingerprint au predict                        |
| Fournit `SampleRelation` avec `group_ids=plot_id`        | Passe `groups` au splitter `GroupKFold`                  |

### 1.8 Resultat attendu

- Artifacts: 5 `FittedAdapter` (image_top, image_side, geno_pca, tab_enc, joiner)
  + 1 `RandomForestRegressor` final + per-fold artifacts.
- Predictions: `PredictionBlock` OOF par fold + `fold_id="final"` apres refit.
- Exports: `ExecutionBundle` (graph + plan + adapters + RF) avec
  `schema_fingerprint`. Format `.bundle` (joblib + JSON manifest).
- Lineage: 800 samples x 5 folds = 4000 enregistrements de predictions OOF.
- Metric: `rmsecv` + `r2_cv` par fold + macro mean.

---

## UC2 - NIRS multi-instrument (3 spectrometres)

### 2.1 Contexte metier

Validation cross-instrument: mesurer le meme echantillon physique sur 3
spectrometres (FOSS NIRS-DS2500, Bruker MPA, ASD FieldSpec). Predire la
matiere seche (MS, regression). Comparer concat early-fusion vs branch
`by_source` (un modele specialise par instrument puis ensemble).

### 2.2 Donnees impliquees

| Source        | Type      | Modality      | Sample key  | Native rep              | N    | Range          |
|---------------|-----------|---------------|-------------|-------------------------|------|----------------|
| `nir_foss`    | `dense_signal` | spectroscopy | `sample_id` | `signal_1d` 1050 wl  | 600  | 400-2500 nm    |
| `nir_bruker`  | `dense_signal` | spectroscopy | `sample_id` | `signal_1d` 2074 wl  | 600  | 800-2500 nm    |
| `nir_asd`     | `dense_signal` | spectroscopy | `sample_id` | `signal_1d` 2151 wl  | 580  | 350-2500 nm    |
| target `y`    | `table`        | -            | `sample_id` | `tabular_numeric`    | 600  | MS (%)         |

Repetitions: 1 spectre / instrument / sample. Pas de groupes.

### 2.3 Pipeline DSL souhaite

```python
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold

# Variante A: early fusion par resampling sur un axe commun
pipeline_concat = [
    {"sources": ["nir_foss", "nir_bruker", "nir_asd"]},
    {"by_source": {
        "nir_foss":   [{"adapter": "spectra.resample", "params": {"target_axis": "wl_800_2500_512"}}, SNV()],
        "nir_bruker": [{"adapter": "spectra.resample", "params": {"target_axis": "wl_800_2500_512"}}, SNV()],
        "nir_asd":    [{"adapter": "spectra.resample", "params": {"target_axis": "wl_800_2500_512"}}, SNV()],
    }},
    {"fusion": FusionPolicy(mode="stack_channels",
                            target_representation="signal_with_processings",
                            alignment=AlignmentPolicy(join="inner"))},
    KFold(n_splits=5),
    {"model": PLSRegression(n_components=12)},
]

# Variante B: branche par source + ensemble OOF
pipeline_branch = [
    {"branch": {"by_source": True, "steps": {
        "nir_foss":   [SNV(), {"_range_": [5, 25, 5]}, {"model": PLSRegression()}],
        "nir_bruker": [SNV(), {"_range_": [5, 25, 5]}, {"model": PLSRegression()}],
        "nir_asd":    [SNV(), {"_range_": [5, 25, 5]}, {"model": PLSRegression()}],
    }}},
    {"merge": "predictions"},                  # OOF predictions as features
    KFold(n_splits=5),
    {"model": Ridge(alpha=1.0)},               # meta-modele
]
```

### 2.4 DAG resultant (variante A: stack_channels)

```text
src:nir_foss --resample(wl_512)--> snv --+
                                         |
src:nir_bruker --resample--------> snv --+--> align(inner) --> stack_channels --> PLS(12)
                                         |
src:nir_asd  --resample-----------> snv -+
                                                         |
                                                         v
                                              (samples, 3, 512)   <-- preserved 3D
                                                         |
                                                         v
                                              PLS-3D adapter: flatten last 2 dims
                                                or 3D PLS variant (POP-PLS)
```

DAG resultant (variante B: by_source branches):

```text
                       split:KFold(5) (outer)
                              |
        +---------------------+---------------------+
        |                     |                     |
        v                     v                     v
ForkNode by_source -> {foss, bruker, asd}
        |                     |                     |
   SNV+PLS(_range_)      SNV+PLS(_range_)      SNV+PLS(_range_)
   5 variants            5 variants            5 variants
        |                     |                     |
   OOF preds_foss        OOF preds_bruker     OOF preds_asd
        |                     |                     |
        +--------+ PredictionJoin (by sample_id, validate OOF) +-+
                              |
                              v
                  meta features: 3 columns (1 per instrument best model)
                              |
                              v
                       Ridge(alpha=1.0) on outer fold
                              |
                              v
                       OOF meta predictions
```

### 2.5 Invariants ML cles

- OOF (variante B): chaque PLS produit predictions `partition="val"` au niveau
  outer fold; le meta-modele ne consomme que des predictions OOF (refus par
  defaut si train preds, voir UC11).
- No leakage: les 3 instruments partagent les memes `sample_id` -> meme fold
  outer pour les 3 branches.
- Split unit: `sample`. Pas de groupes.
- Reproductibilite: meme `KFold(random_state=42)` pour les 3 branches dans
  variante B (sinon decalage OOF).
- `nir_asd` n'a que 580/600 samples: `AlignmentPolicy(join="inner")` -> N=580
  effectifs. Decision a tracer dans le manifest.

### 2.6 Points de friction

1. Variante A: que faire si les 3 instruments couvrent des plages differentes
   (400-2500 vs 800-2500 vs 350-2500) ? Resample sur intersection ou union avec
   masque ? Intersection 800-2500 perd l'info VIS pour FOSS et ASD.
2. Variante B: les `_range_` n_components sont differents par branche. Comment
   selectionner ? Best per branch puis stacker ? Best global apres meta ?
3. Stack_channels: PLS classique attend 2D, donc faut-il un PLS-3D specialise
   (POP-PLS) ou flatten ? Le contrat doit etre explicite cote model adapter.

### 2.7 ML_DATA vs DAG-ML

| ML_DATA                                                | DAG-ML                                              |
|--------------------------------------------------------|-----------------------------------------------------|
| Adapter `spectra.resample(target_axis)`                | Branche `by_source` (ForkNode + MapNode)            |
| Stack/concat sur axe `processing` ou `channel`         | Generateur `_range_` enumere les variants n_comp    |
| AlignmentPolicy(inner) selectionne les 580            | PredictionJoin valide OOF avant meta-modele         |
| Fournit `AxisSpec.coordinates` (wavelengths)           | Choisit ranking metric et top_k                     |

### 2.8 Resultat attendu

- Variante A: 1 modele PLS sur tenseur stacke (3 instruments x 512 wl). Bundle = 1.
- Variante B: 15 PLS variants (3 inst x 5 n_comp) + 1 Ridge meta. Bundle = 16.
- Predictions OOF par variant + meta-OOF + final refit.
- Comparison reporting: RMSECV variante A vs variante B.

---

## UC3 - Repetitions: plusieurs X pour un Y

### 3.1 Contexte metier

Mesure NIRS de poudre vegetale: pour chaque echantillon physique (`sample_id`),
le technicien acquiert 3 spectres (positions A/B/C dans la coupelle) et 1
mesure chimique de reference (proteines). Le Y existe au niveau du sample, pas
de l'observation. Le modele NIRS doit predire par observation, mais l'agregation
finale se fait au niveau sample (mean ou median), avec exclusion d'outliers.

### 3.2 Donnees impliquees

| Source       | Type            | Modality      | Granularity              | observation_id    | sample_id   | target_id | group_id  |
|--------------|-----------------|---------------|--------------------------|-------------------|-------------|-----------|-----------|
| `nir`        | `dense_signal`  | spectroscopy  | `per_sample_repeated`    | `nir_S001_A`...   | `S001`...   | `y_S001`  | `lot_42`  |
| `chem`       | `table`         | reference     | `per_sample`             | `chem_S001`       | `S001`      | `y_S001`  | `lot_42`  |

400 samples logiques, 1200 observations NIRS. 25 lots de production (15-20
samples/lot). `target_id == "y_" + sample_id`.

### 3.3 Pipeline DSL souhaite

```python
pipeline = [
    {"sources": ["nir"]},
    {"y_processing": "standardize"},
    [SNV(), SavitzkyGolay(window=11, deriv=2)],

    {"split_policy": SplitPolicy(
        split_unit="target",           # toutes les obs partageant target_id ensemble
        forbid_origin_cross_fold=True,
    )},
    GroupKFold(n_splits=5),            # groups = target_id derives de SampleRelation

    {"model": PLSRegression(n_components=15)},

    {"aggregate": AggregationPolicy(
        level="sample",
        method="robust_mean",
        keep_observation_predictions=True,  # garde aussi obs-level preds
        exclude_outliers={"enabled": True, "threshold": 0.95},
    )},
]
```

### 3.4 DAG resultant

```text
materialize("nir", view=v)
   -> DataBlock(sample_ids=("S001","S001","S001","S002",...))    [1200 obs]
   -> SampleRelation(observation_ids, sample_ids, target_ids="y_<sample>", group_ids="lot_<n>")

sample_relation.target_ids -> SplitPolicy(split_unit="target")
                            -> GroupKFold(groups=target_ids)

adapt:snv -> adapt:sg2 -> shape (1200, n_wl)
                              |
                              v
                       split:GroupKFold(5)         [folds carry obs indices]
                              |
                              v
                       model:PLS(15)
                              |
                              v
                       PredictionBlock(observation_level)
                              |
                              v
                       AggregatorNode(level="sample", method="robust_mean")
                              |
                              v
                       PredictionBlock(sample_level)
                              |
                              +-- side output: observation_predictions (kept)
```

Nodes:

| Node id           | kind         | Notes                                            |
|-------------------|--------------|--------------------------------------------------|
| `src:nir`         | materialize  | granularity per_sample_repeated, returns 1200 obs|
| `rel:nir`         | sample_rel   | exposes target_id / group_id                     |
| `adapt:snv`       | adapt        | applied per-observation                          |
| `adapt:sg2`       | adapt        | 2nd derivative                                   |
| `split:cv`        | split        | GroupKFold on target_id, 5 folds                 |
| `model:pls`       | model        | PLS(15), trained per fold                        |
| `pred:obs`        | prediction   | observation-level                                |
| `pred:agg`        | aggregator   | sample-level, robust_mean, excl outliers         |

### 3.5 Invariants ML cles

- Les 3 obs d'un sample S001 vont **toutes** dans le meme fold (train ou val),
  jamais splittees. `split_unit="target"` garantit cela.
- OOF: les predictions `partition="val"` sont produites au niveau observation,
  l'agregation sample est faite apres et stockee separement avec
  `aggregation_level="sample"`.
- `keep_observation_predictions=True` => DAG-ML stocke deux PredictionBlocks
  (un obs-level, un sample-level), partageant le meme `fold_id`.
- Reproductibilite: meme `SampleRelation` au predict (verif via `schema_fingerprint`).
- Outlier exclusion (0.95): n'agit que sur l'agregation, jamais sur l'OOF brut.

### 3.6 Points de friction

1. Quel niveau pour ranker/selectionner: RMSE obs-level (1200 preds) ou
   sample-level (400 preds) ? Le second est ce qui compte metier, mais le
   premier reduit la variance d'estimation.
2. Si une observation NIRS d'un sample est tres bruitee, son inclusion biaise
   le sample-mean. Faut-il un `SpectralQualityFilter` upstream comme
   `{"exclude": SpectralQualityFilter()}` avant aggregation, ou un filtre sur
   les predictions ?
3. Agregation `vote` pour classification multi-classes: comment gerer les
   egalites (3 votes pour 3 classes differentes) ?
4. `target_id` peut-il etre derive automatiquement de `sample_id` quand 1:1 ?
   Ou doit-on toujours l'expliciter dans `SampleRelation` ?

### 3.7 ML_DATA vs DAG-ML

| ML_DATA                                                        | DAG-ML                                          |
|----------------------------------------------------------------|-------------------------------------------------|
| Fournit `SampleRelation` complet (target_id, group_id)         | Choisit `split_unit="target"`                   |
| Materialize retourne 1200 obs, pas 400 samples                 | Aggregator node consomme PredictionBlock obs    |
| Aucune agregation: garde les obs telles quelles                | Implemente `robust_mean` + outlier exclusion    |
| Stocke `observation_id` dans chaque DataBlock                  | Persiste les 2 niveaux de PredictionBlock       |

### 3.8 Resultat attendu

- 2 PredictionBlocks par fold: obs-level (1200 OOF) et sample-level (400 OOF).
- Bundle: 1 PLS (refit final) + adapters + `SampleRelation` rejoue au predict.
- Metric: `rmsecv_sample` (primary), `rmsecv_obs` (secondary).

---

## UC4 - Split-unit par entite (patients / parcelles / lots)

### 4.1 Contexte metier

Etude clinique multi-patients: prediction d'un score biochimique a partir de
spectres Raman + parametres cliniques. Chaque patient genere 5 a 15 mesures
sur 2 ans. Le risque de leakage temporel et inter-patient impose un split au
niveau `patient_id`.

### 4.2 Donnees impliquees

| Source     | Type            | Granularity              | sample_id        | group_id      | N samples |
|------------|-----------------|--------------------------|------------------|---------------|-----------|
| `raman`    | `dense_signal`  | `per_sample_repeated`    | `meas_<n>`       | `patient_<p>` | 2400      |
| `clin`     | `table`         | `per_sample`             | `meas_<n>`       | `patient_<p>` | 2400      |
| target `y` | `table`         | `per_sample`             | `meas_<n>`       | -             | 2400      |

300 patients, 8 mesures moyennes/patient.

### 4.3 Pipeline DSL souhaite

```python
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

pipeline = [
    {"sources": ["raman", "clin"]},
    {"y_processing": "standardize"},

    {"by_source": {
        "raman": [SNV(), SavitzkyGolay(window=15, deriv=1)],
        "clin":  [{"adapter": "tabular.encoder"}],
    }},

    {"fusion": FusionPolicy(
        mode="concat_features",
        target_representation="tabular_numeric",
        alignment=AlignmentPolicy(join="exact"),
        namespace_columns=True,
    )},

    {"split_policy": SplitPolicy(
        split_unit="group",
        group_key="patient_id",
        forbid_origin_cross_fold=True,
    )},
    StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0),

    {"model": Ridge(alpha=1.0)},
]
```

### 4.4 DAG resultant

```text
materialize(raman)    +-> SNV+SG1 -+
                                   |
materialize(clin)     +-> tab.enc -+--> align(exact) --> fusion.join
                                                                |
                                                                v
                                                       StratifiedGroupKFold(5)
                                                       groups = patient_id (from SampleRelation)
                                                                |
                                                       5 folds:
                                                       fold k: train = patients not in val_k
                                                                |
                                                                v
                                                          Ridge fit per fold
                                                                |
                                                                v
                                                       OOF preds (per meas), grouped by patient
```

Nodes-cles:

| Node id      | kind       | Notes                                            |
|--------------|------------|--------------------------------------------------|
| `rel:raman`  | sample_rel | retourne `group_ids = patient_id`                |
| `split:sgkf` | split      | StratifiedGroupKFold, garantit no-patient-cross  |

### 4.5 Invariants ML cles

- Aucun patient n'apparait simultanement dans train et val du meme fold.
- Stratification sur quantiles de y (regression) ou classes (classif).
- OOF: predictions sont OOF au niveau `meas_<n>`, on peut aussi agreger
  patient-level via un `Aggregator` optionnel.
- Reproductibilite: `random_state=0` + `SeedContext` hierarchique.
- `join="exact"` impose que raman et clin aient exactement les memes
  `sample_id` (mesures jumelees, pas de manquant).

### 4.6 Points de friction

1. Si un patient n'a que 2 mesures, le fold val pourrait avoir trop peu de
   donnees. Quelle taille minimum par patient ? Imposer ou warning ?
2. Stratification + groupes est non-trivial (`StratifiedGroupKFold` est greedy).
   Faut-il accepter un ecart de distribution de y entre folds ?
3. Predict-time: si un nouveau patient arrive avec 1 seule mesure, le bundle
   doit-il accepter ? Oui, `group_id` n'est utilise qu'au split.
4. Comment exposer le mapping `patient_id -> [sample_ids]` dans le manifest
   sans gonfler la taille du bundle ?

### 4.7 ML_DATA vs DAG-ML

| ML_DATA                                              | DAG-ML                                              |
|------------------------------------------------------|-----------------------------------------------------|
| Expose `SampleRelation.group_ids = patient_id`       | Selectionne `split_unit="group"`                    |
| AlignmentPolicy(exact) enforce sample-id parity      | Splitter `StratifiedGroupKFold` consomme groups     |
| Aucun split, aucune connaissance de fold             | Persiste mapping patient -> folds dans manifest     |

### 4.8 Resultat attendu

- PredictionBlocks par fold avec `sample_ids=meas_*` et `group_ids=patient_*`.
- Bundle: Ridge + adapters + `SampleRelation` partiel (les patient_ids du train).
- Metric: rmsecv_meas + rmsecv_patient_mean.

---

## UC5 - Augmentation train-only avec OOF correct

### 5.1 Contexte metier

Dataset NIRS reduit (180 samples). Application d'augmentations (bruit gaussien,
shift wavelength, mixup local) pour ameliorer la generalisation, sans contamination
inter-fold. La validation et le test doivent rester non-augmentes; les OOF
predictions doivent etre faites pour les samples originaux du fold val, pas
pour leurs copies generees au train.

### 5.2 Donnees impliquees

| Source      | Type            | Granularity            | N originaux | apres aug train | sample_id  | origin_id   |
|-------------|-----------------|------------------------|-------------|------------------|------------|-------------|
| `nir`       | `dense_signal`  | `per_sample`           | 180         | +540 (3x)        | `aug_*`    | `S001`...   |
| target `y`  | `table`         | `per_sample`           | 180         | (heritage)       | `aug_*`    | -           |

`inherit_target=True`, `inherit_group=True`.

### 5.3 Pipeline DSL souhaite

```python
from nirs4all.operators.augmentation import (
    GaussianAdditiveNoise, WavelengthShift, LocalMixupAugmenter
)

pipeline = [
    {"sources": ["nir"]},
    {"y_processing": "standardize"},

    [SNV(), SavitzkyGolay()],

    KFold(n_splits=5, shuffle=True, random_state=42),

    # Augmentation node inside the fold scope:
    {"sample_augmentation": [
        GaussianAdditiveNoise(sigma=0.005, multiplier=2),
        WavelengthShift(max_shift=3, multiplier=1),
    ],
     "policy": AugmentationPolicy(
         apply_to="train_only",
         inherit_target=True,
         inherit_group=True,
         forbid_validation_augmentation=True,
         store_origin_mapping=True,
         seed_scope="fold",
     )},

    {"model": PLSRegression(n_components=12)},
]
```

### 5.4 DAG resultant

```text
                       KFold(5, shuffle, rs=42)
                              |
       +----------- For each fold k ------------+
       |                                        |
   train view (144 obs)                   val view (36 obs)
       |                                        |
       v                                        v
[SampleAug(noise + shift)]                (no augmentation)
       |
       v
SampleRelation extended:
  - origin_id = original sample_id for new rows
  - group_id, target_id inherited
       |
       v
DataBlock train: 144 + 144*2 + 144*1 = 576 rows
       |
       v                                        |
PLS(12) fit on 576 augmented train       PLS.predict(val 36 originals)
       \________________________________________/
                              |
                              v
                  PredictionBlock partition="val", sample_ids=originals
                  (no prediction for augmented copies; aug rows have no Y to validate)
                              |
                              v
                  OOF aggregated: 180 originals x 5 folds = 180 OOF preds
                              |
                              v
                  REFIT: aug applied on full train (180 + multipliers), no held-out
                              |
                              v
                  Final PLS + AugmentationAdapter artifact (params, seed) persisted
```

### 5.5 Invariants ML cles

- `forbid_validation_augmentation=True`: l'augmentation tourne sur la view
  `partition="train"` du fold, jamais `val`.
- `origin_ids` est rempli par l'`AugmentationAdapter` -> DAG-ML verifie que
  pour tout sample S de val_k, aucune ligne train_k n'a `origin_id=S`.
  Violation -> `LeakageError`.
- OOF: 180 predictions (1 par original), pas 720. Les copies augmentees ne
  recoivent jamais de prediction OOF (elles n'ont pas de Y de validation).
- Reproductibilite: `SeedContext.child(node_id="aug:noise", fold_id=k)` derive
  un seed deterministe; le bundle stocke la sequence `(seed, multiplier)`.
- Refit: `apply_to="train_only"` (default) signifie "augmente toutes les
  partitions train", ce qui inclut les CV train folds ET le refit final
  (180 -> 720). Le bundle persiste `AugmentationAdapter` mais le predict ne
  le rejoue pas (les nouveaux samples ne sont pas augmentes). Pour un refit
  SANS reaugmentation, utiliser `apply_to="cv_only"` explicitement.

### 5.6 Points de friction

1. Refit final reapplique-t-il l'augmentation ? Resolu (Q18): default
   `apply_to="train_only"` = augmenter en CV ET au refit (le modele shippe
   match la distribution OOF). Opt-out explicite via `apply_to="cv_only"`.
2. `Mixup` croise deux samples train pour creer un nouveau. Quel `origin_id` ?
   Conserver les deux (`origin_ids = (id_a, id_b)`) demande un schema list
   pour `origin_id`. Spec dit single -> faut-il etendre ?
3. Quand un meta-modele consomme des predictions OOF (UC6), faut-il les
   produire au niveau origin ou au niveau augmented ? Reponse: origin only.
4. L'augmentation est lossy par definition. Comment marquer le lineage pour que
   le predict ne l'applique pas par accident ?

### 5.7 ML_DATA vs DAG-ML

| ML_DATA                                                | DAG-ML                                              |
|--------------------------------------------------------|-----------------------------------------------------|
| `AugmentationAdapter.transform()` produit DataBlock + SampleRelation | Decide quand l'appeler (uniquement view train d'un fold) |
| Stocke `origin_id` non-None dans SampleRelation         | Verifie `forbid_validation_augmentation` invariant  |
| Persiste `random_state` dans FittedAdapter             | Refuse de faire OOF sur lignes augmentees           |

### 5.8 Resultat attendu

- 180 OOF predictions (originaux).
- Bundle: PLS + adapters + AugmentationAdapter (avec seed) marque `phase=REFIT`.
- Lineage: chaque AugmentedRow porte `lineage=("aug.noise", "aug.shift")`.

---

## UC6 - Stacking multi-niveau (3 preprocs/models + meta Ridge)

### 6.1 Contexte metier

Estimer une concentration chimique a partir d'un spectre NIRS, en combinant
3 voies preprocessing/modele complementaires (SNV+PLS, MSC+RF, Detrend+SVR)
via un meta-modele Ridge entraine sur les predictions OOF.

### 6.2 Donnees impliquees

| Source     | Type             | Granularity   | N samples |
|------------|------------------|---------------|-----------|
| `nir`      | `dense_signal`   | `per_sample`  | 500       |
| target `y` | `table` (numeric)| `per_sample`  | 500       |

### 6.3 Pipeline DSL souhaite

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge

pipeline = [
    {"sources": ["nir"]},
    {"y_processing": "standardize"},

    KFold(n_splits=5, shuffle=True, random_state=42),

    {"branch": [
        [SNV(),     {"model": PLSRegression(n_components=12)}],
        [MSC(),     {"model": RandomForestRegressor(n_estimators=300, random_state=0)}],
        [Detrend(), {"model": SVR(kernel="rbf", C=10.0)}],
    ]},

    {"merge": "predictions",
     "policy": {"allow_train_predictions_as_features": False,
                "validate_oof": True, "join_on": "sample_id"}},

    KFold(n_splits=5, shuffle=True, random_state=42),
    {"model": Ridge(alpha=1.0)},
]
```

### 6.4 DAG resultant

```text
                       KFold(5, rs=42)  (level-0 outer CV)
                              |
                +-------------+-------------+
                |             |             |
                v             v             v
            Branch 0       Branch 1     Branch 2
            SNV+PLS        MSC+RF       Detrend+SVR
                |             |             |
                v             v             v
       fit per fold     fit per fold   fit per fold
                |             |             |
                v             v             v
       OOF preds b0   OOF preds b1   OOF preds b2
       (partition=val) (partition=val) (partition=val)
                |             |             |
                +------+------+-------------+
                       |
                       v
              PredictionJoinNode (validate OOF, error if any partition=train)
                       |
                       v
              FeatureTable: columns=("b0_pls","b1_rf","b2_svr"), 500 rows OOF
                       |
                       v
              KFold(5, rs=42)  (level-1 inner CV for meta)
                       |
                       v
              Ridge(alpha=1.0) fit per fold
                       |
                       v
              meta-OOF predictions (final)
                       |
              SELECT (rmsecv_meta)
                       v
              REFIT:
                base learners refit on FULL train (180 -> 500)
                meta refit on full OOF predictions
                       v
              Bundle = 3 base models + meta Ridge + 3 OOF predictions cache
```

Nodes:

| Node id              | kind         | Notes                                           |
|----------------------|--------------|-------------------------------------------------|
| `split:outer`        | split        | KFold(5) shared by 3 branches                   |
| `fork:branches`      | fork         | duplication, broadcast same data                |
| `branch:b0..b2`      | subgraph     | preproc + model per branch                      |
| `pred:b0..b2`        | prediction   | OOF per branch                                  |
| `join:pred`          | prediction_join | validates OOF, builds meta FeatureTable      |
| `split:inner`        | split        | inner CV for meta                               |
| `model:ridge`        | model        | meta-modele                                     |
| `refit:final`        | refit        | refit base + meta on full data                  |

### 6.5 Invariants ML cles

- Memes folds outer pour les 3 branches (meme `random_state` + meme `split:outer`).
- `PredictionJoinNode` refuse par defaut tout `PredictionBlock` non-OOF
  (`partition != "val"`). Si une branche a leak son train, raise.
- Le meta-modele Ridge consomme uniquement les OOF -> taille (500, 3).
- Inner CV pour Ridge peut differer; ne change pas la base.
- Refit: base learners refit sur full train. Meta Ridge refit sur les OOF
  produites par l'inner CV pre-refit (pas les preds des base learners refit,
  sinon train leak).
- `allow_train_predictions_as_features=False` est explicite (default).
  Voir UC11 pour l'opt-in.

### 6.6 Points de friction

1. Doit-on stacker les predictions OOF brutes ou des probabilites/quantiles ?
   Pour regression: y_pred direct. Pour classification: y_proba.
2. Inner CV du meta peut-il etre `LeaveOneOut` quand N=500 ? Couteux, mais
   ne change pas la validite du stacking.
3. Si 2 branches sont quasi-identiques (correlation 0.99 entre preds), le meta
   Ridge va surfit ces 2. Faut-il un decorrelator amont (drop, PCA) ?
4. Pour le refit, faut-il refit les base learners avec exactement les memes
   hyperparams que le meilleur fold ou avec l'agg de tous les folds ?

### 6.7 ML_DATA vs DAG-ML

| ML_DATA                                                       | DAG-ML                                              |
|---------------------------------------------------------------|-----------------------------------------------------|
| Materialize nir + applique adapters (SNV, MSC, Detrend)       | Fork/join, validation OOF, refus si non-OOF         |
| Fournit FeatureTable des preds OOF avec source_ids par col    | Refuse train preds par defaut (opt-in flag explicite) |
| Aucune connaissance du meta-modele                            | Inner CV du meta, refit infrastructure              |

### 6.8 Resultat attendu

- 1500 OOF preds base (500 x 3) + 500 OOF preds meta.
- Bundle: 3 base learners refit + Ridge meta refit + 3 caches OOF.
- Metrics: rmsecv par branche + rmsecv_meta + delta vs best single.

---

## UC7 - Generateurs + tuning bayesien

### 7.1 Contexte metier

Recherche d'hyperparametres a grande echelle: croiser plusieurs preprocessings
candidats, plusieurs familles de modeles, plusieurs grid combinations, plus
un tuning bayesien sur les params continus (alpha, gamma). Enumeration lazy.

### 7.2 Donnees impliquees

| Source     | Type             | Granularity   | N samples |
|------------|------------------|---------------|-----------|
| `nir`      | `dense_signal`   | `per_sample`  | 800       |
| target `y` | `table` (numeric)| `per_sample`  | 800       |

### 7.3 Pipeline DSL souhaite

```python
pipeline = [
    {"sources": ["nir"]},
    {"y_processing": "standardize"},

    {"_cartesian_": [
        {"_or_": [SNV(), MSC(), Detrend()]},
        {"_or_": [None, SavitzkyGolay(window=11, deriv=1),
                          SavitzkyGolay(window=21, deriv=2)]},
    ]},

    KFold(n_splits=5),

    {"_chain_": [
        {"_grid_": {"model": [PLSRegression()], "n_components": [5, 10, 15, 20, 25]}},
        {"_grid_": {"model": [Ridge()], "alpha": [0.01, 0.1, 1.0, 10.0]}},
        {"_sample_": {"distribution": "log_uniform", "from": 1e-3, "to": 1e2, "num": 20,
                       "model": SVR(kernel="rbf"), "tune": ["C", "gamma"]}},
    ]},

    {"tuner": {
        "kind": "bayesian",                # TunerAdapter
        "max_trials": 40,
        "objective": "rmsecv",
        "tunes": ["adapt:sg.window", "model.C", "model.gamma"],
    }},
]
```

### 7.4 DAG resultant

```text
SearchSpace (lazy):
  cartesian = (3 preprocs) x (3 sg variants) = 9 preproc branches
  chain     = (5 PLS comp) + (4 Ridge alpha) + (20 SVR sample) = 29 model variants
  total enumerable variants = 9 * 29 = 261 (lazy, not materialized)

PLAN:
  TunerAdapter.suggest(trial_id=k) -> chooses 1 variant + continuous params
  PLAN materialize:
    src:nir -> adapt:<chosen_preproc> -> adapt:<chosen_sg|None>
            -> split:KFold(5) -> model:<chosen_family with chosen params>

FIT_CV (per trial):
  TunerAdapter -> {params: {preproc_id, sg_id, model_kind, n_comp|alpha|C|gamma}}
  Run 5-fold CV, score = mean(rmsecv)
  Record TrialResult{trial_id, params, score}

After 40 trials:
  best = TunerAdapter.best()  -> SelectedGraph
  REFIT: full train run with best variant
```

Nodes:

| Node id           | kind        | Notes                                                |
|-------------------|-------------|------------------------------------------------------|
| `search:cartesian`| search_space| enumere `(preproc, sg)`                              |
| `search:chain`    | search_space| enumere model families with their grid              |
| `tuner:bayes`     | tuner       | suggests / records, prunes early                    |
| `variant:N`       | virtual     | resolves one concrete combination                    |
| `split:cv`        | split       | KFold(5) shared across trials                        |
| `model:<kind>`    | model       | depends on trial                                     |

### 7.5 Invariants ML cles

- Chaque trial respecte l'OOF: 5-fold CV stricte.
- Stateful adapters (SG) sont fit fold-train uniquement.
- Reproductibilite: `SeedContext.child(trial_id=k)` derive `random_state`.
- Tuner separation: `TunerAdapter` ne voit pas les val data, seulement la
  score `rmsecv`.
- Pruning bayesien autorise mais doit logger `trial.state="pruned"`.
- Refit: meilleur variant uniquement, pas tous les trials.

### 7.6 Points de friction

1. Comment definir le `SearchSpace` pour mix discret (`_cartesian_`, `_grid_`)
   et continu (`_sample_`) ? Tuner bayesien attend des bornes continues.
2. Doit-on faire du `early_stopping` (Hyperband, BOHB) pour eviter de finir
   des trials clairement perdants apres 1 fold ?
3. Si un trial fail (operator incompatible), comment marquer sans corrompre
   l'historique du tuner ? `TrialResult.state="error"` + ne pas record score.
4. La transmission des `params` du tuner au noeud `model` et au noeud `adapt`
   simultanement: format unifie ou per-node ?

### 7.7 ML_DATA vs DAG-ML

| ML_DATA                                                       | DAG-ML                                              |
|---------------------------------------------------------------|-----------------------------------------------------|
| Materialize nir + applique l'adapter (SNV/MSC/...) du trial   | Implemente `SearchSpace`, `TunerAdapter`            |
| Aucune connaissance des trials                                | Enumeration lazy, scheduling, pruning              |
| Persiste `random_state` derive                                | Persiste tous les `TrialResult` dans le manifest    |

### 7.8 Resultat attendu

- 40 `TrialResult` (params, score, time, pruned flag).
- 1 `SelectedGraph` correspondant au meilleur trial.
- Bundle refit + journal complet des trials (utile pour reporting Bayes).

---

## UC8 - Branches par metadata, merge concat

### 8.1 Contexte metier

Dataset NIRS multi-site (3 sites de production: A/B/C). Hypothese: chaque site
a des biais d'instrument differents et un modele specifique par site est plus
performant qu'un modele global. Apres entrainement par site, concatenation des
predictions pour produire une PredictionBlock unifiee (alignee `sample_id`).

### 8.2 Donnees impliquees

| Source     | Type             | Granularity   | Metadata `site` | N par site | Total |
|------------|------------------|---------------|-----------------|------------|-------|
| `nir`      | `dense_signal`   | `per_sample`  | "A","B","C"     | 300/250/200| 750   |
| `meta`     | `table`          | `per_sample`  | site, year, op  | 750        | 750   |
| target `y` | `table` (num)    | `per_sample`  | -               | -          | 750   |

### 8.3 Pipeline DSL souhaite

```python
pipeline = [
    {"sources": ["nir", "meta"]},
    {"y_processing": "standardize"},

    [SNV(), SavitzkyGolay()],

    {"branch": {"by_metadata": "site"}},     # separation branches: 3 disjoint groups

    {"by_branch": {
        "A": [{"model": PLSRegression(n_components=12)}],
        "B": [{"model": PLSRegression(n_components=15)}],
        "C": [{"model": Ridge(alpha=1.0)}],
    }},

    {"merge": "concat",                       # reassemble in canonical sample_id order
     "policy": {"on_missing_branch": "error"}},

    KFold(n_splits=5),
]
```

### 8.4 DAG resultant

```text
                       src:nir + src:meta
                              |
                       adapt:snv+sg
                              |
                              v
                       ForkNode(by_metadata="site")
                              |
        +---------------------+---------------------+
        v                     v                     v
   site=A (300 samples)   site=B (250)         site=C (200)
        |                     |                     |
   split:KFold(5)         split:KFold(5)        split:KFold(5)
        |                     |                     |
   PLS(12)               PLS(15)                Ridge(1.0)
        |                     |                     |
   OOF preds A           OOF preds B           OOF preds C
        |                     |                     |
        +---------+ MergeConcatNode (by sample_id, no overlap) +-+
                              |
                              v
                  Unified PredictionBlock (750, ordered)
```

Nodes-cles:

| Node id           | kind        | Notes                                            |
|-------------------|-------------|--------------------------------------------------|
| `fork:by_meta`    | fork        | separation, disjoint sample groups                |
| `branch:A/B/C`    | subgraph    | independent CV + model per branch                |
| `merge:concat`    | prediction_join | reassemble sample order, no overlap            |

### 8.5 Invariants ML cles

- Separation: les 3 branches sont disjointes (intersection vide sur sample_id).
- Chaque branche fait sa propre CV (folds disjoints intra-branche).
- Merge `concat` exige union exacte = ensemble des sample_ids -> erreur si
  un sample n'est dans aucune branche (incoherence metadata).
- OOF: chaque sample a 1 prediction OOF (sa branche), pas 3.
- Refit: 3 modeles refit independamment, bundle contient les 3.
- Predict-time: nouveau sample classe par `site` (lookup metadata), dispatch
  vers le bon modele du bundle.

### 8.6 Points de friction

1. Un site avec trop peu de samples (ex: 50) ne supporte pas 5 folds. Fallback
   automatique vers 3 folds ou erreur ?
2. Comment gerer un nouveau site `D` au predict ? Strategies: error, fallback
   au modele global (s'il existe), nearest neighbour site.
3. Faut-il toujours produire un modele "global" comme reference ? Option du DSL.
4. Si on veut comparer 3 modeles separes vs 1 modele global, on cree deux
   variants. Comment les coexister dans le meme run ?

### 8.7 ML_DATA vs DAG-ML

| ML_DATA                                                | DAG-ML                                              |
|--------------------------------------------------------|-----------------------------------------------------|
| Materialize `meta` + expose colonne `site`             | ForkNode interprete `by_metadata="site"`            |
| Fournit DataView avec sous-ensemble de sample_ids      | Gere CV par branche (folds independants)            |
| Aucune separation par metadata cote data               | Merge concat valide non-overlap                     |

### 8.8 Resultat attendu

- 3 modeles refit + 1 PredictionBlock concatene (750 OOF).
- Bundle: dict des modeles par site + `site -> sample_ids` mapping.
- Metric: rmsecv global + par site.

---

## UC9 - Refit + bundle + predict new heterogene

### 9.1 Contexte metier

Apres entrainement UC1, deployer le bundle sur 50 nouveaux echantillons recus
6 mois plus tard. Verifier la compatibilite du schema (fingerprint), gerer
le cas ou `photo_side` manque pour 5 echantillons, refuser si le `weather`
arrive avec un schema different (changement de capteur).

### 9.2 Donnees impliquees

| Source       | Train (UC1) | Predict (new)   | Schema diff |
|--------------|-------------|------------------|-------------|
| `nir`        | 512 wl      | 512 wl identique | ok          |
| `photo_top`  | 224x224x3   | 224x224x3        | ok          |
| `photo_side` | 224x224x3, 95% presence | 224x224x3, 90% presence | ok (presence mask differe) |
| `geno`       | 12000 SNPs  | 12000 SNPs same panel | ok      |
| `weather`    | 180 days x 6 vars | 180 days x 7 vars (NEW) | DIVERGE -> reject |
| `meta`       | one-hot variety/plot | one-hot identical | ok    |

### 9.3 Pipeline DSL au predict

```python
import dagml

# Load bundle from UC1 training
bundle = dagml.load_bundle("uc1_protein.bundle")

# Apply on new data
result = dagml.predict(
    bundle=bundle,
    new_dataset=new_data,             # MLDataset over 50 samples
    schema_check="strict",            # "strict" | "compatible" | "loose"
    missing_source_policy="inherit",  # use the policy from training
)
```

### 9.4 DAG resultant (predict)

```text
load_bundle("uc1_protein.bundle")
   -> ExecutionBundle{ graph_spec, data_plan, fitted_adapters, schema_fingerprint }

verify(new_dataset):
   schema_fingerprint(new_dataset, fusion=bundle.fusion,
                      adapter_specs=bundle.adapters)
   if fp != bundle.schema_fingerprint:
       if schema_check == "strict":
           raise SchemaFingerprintMismatch(payload={...})
       elif schema_check == "compatible":
           detect deviations:
             - "weather" axes differ: rank 6 -> 7 -> REJECT
             - "photo_side" presence rate differs -> WARNING ok
       elif schema_check == "loose":
           allow if all sources present, raise on missing source

execute_transform(data_plan, new_dataset, view=full, fitted=adapters):
   - materialize each source
   - apply fitted_adapters in order
   - join via fitted FeatureJoiner (re-uses train column order)
   - feed RF.predict()

returns PredictionBlock(partition="predict", sample_ids=new, y_pred=...)
```

### 9.5 Invariants ML cles

- `schema_fingerprint` recalcule avec les memes sources + fusion + adapter
  specs. Mismatch -> default refuse.
- Aucun fit a predict-time. Tous les adapters stateful viennent du bundle.
- `photo_side` manquant: la presence mask de la nouvelle data alimente la
  colonne `indicator__photo_side_present` (logique du joiner train).
- `weather` schema diverge -> erreur claire avec payload structure {expected,
  got, diff}.
- Reproductibilite: meme ordre de colonnes que train (lock par le FeatureJoiner).
- Lineage: PredictionBlock porte `bundle_id`, `bundle_version`.

### 9.6 Points de friction

1. Quels champs comparer dans le fingerprint ? Sources (id+axes), fusion policy,
   adapter ids + versions. Pas les sample_ids (qui changent toujours).
2. Que faire si une source train est `optional=True` dans `ModelInputSpec`
   et absente au predict ? La spec dit OK; mais le RF a appris avec, donc
   degradation possible -> warning.
3. Comment migrer entre majeures (image embedding v1 -> v2) ? Re-fit ou
   refus ? `PluginVersionError` + path de migration explicite.
4. Le DataPlan stocke `adapter_id` + `params`. Si l'utilisateur a redefini
   un adapter avec le meme id mais semantique differente, le bundle est
   silencieusement compromis. Hash de l'implementation requis ?

### 9.7 ML_DATA vs DAG-ML

| ML_DATA                                                       | DAG-ML                                              |
|---------------------------------------------------------------|-----------------------------------------------------|
| Calcule `schema_fingerprint(schema, fusion, adapters)`        | Compare fingerprint et decide accept/refuse         |
| `execute_transform(plan, fitted, view)` rejoue le DataPlan    | Charge le bundle, dispatch predict                  |
| Refuse si plugin manquant ou version out-of-range            | Refuse si schema_check echoue                       |

### 9.8 Resultat attendu

- `PredictionBlock(partition="predict", sample_ids=new50, y_pred=...)`.
- Aucun nouvel artifact (predict est pur read-only sur le bundle).
- Si fingerprint mismatch: `SchemaFingerprintMismatch` avec payload JSON
  detaillant la diff. Pas de prediction silencieuse.

---

## UC10 - Sous-DAG reifie comme noeud (SubgraphNodeSpec)

### 10.1 Contexte metier

Une equipe NIRS a developpe un sous-pipeline "NIR canonical preproc + PLS"
generique reutilise dans 5 projets. Le packager comme `SubgraphNodeSpec`
versionne, puis l'inserer comme noeud dans un nouveau DAG plus large (par
exemple, alimenter un meta-stacking avec ce sous-DAG comme une "base learner"
parmi d'autres). Choisir `inline` vs `opaque` selon le besoin de cache fin.

### 10.2 Donnees impliquees

| Source     | Type             | Granularity   | N samples |
|------------|------------------|---------------|-----------|
| `nir`      | `dense_signal`   | `per_sample`  | 1000      |
| `tab`      | `table`          | `per_sample`  | 1000      |
| target `y` | `table` (num)    | `per_sample`  | 1000      |

### 10.3 Pipeline DSL souhaite

```python
# Sub-DAG defined once and registered:
NIR_CANONICAL_PLS = GraphSpec(
    id="nirs4all.recipes.canonical_pls",
    version="1.2.0",
    interface=GraphInterface(
        inputs=(
            PortSpec(name="X", kind="data", representation="signal_with_processings"),
            PortSpec(name="y", kind="target", representation="tabular_numeric"),
        ),
        outputs=(
            PortSpec(name="pred", kind="prediction", representation="tabular_numeric"),
        ),
    ),
    nodes={...},     # SNV -> SG2 -> PLS(15)
    edges=(...),
)

# Parent pipeline reuses the sub-DAG as a node:
pipeline = [
    {"sources": ["nir", "tab"]},
    {"y_processing": "standardize"},

    KFold(n_splits=5),

    {"branch": [
        # Sub-DAG used as a node (opaque: black-box, its own artifacts)
        SubgraphNodeSpec(
            id="branch:nir_canon",
            graph=SerializableRef(registry="recipes",
                                   type_id="canonical_pls",
                                   version="1.2.0",
                                   object_id="..."),
            input_mapping={"X": "nir", "y": "y"},
            output_mapping={"pred": "pred_canon"},
            inline_policy="opaque",
        ),
        # Sub-DAG inlined (its nodes are merged into the parent for cache reuse)
        SubgraphNodeSpec(
            id="branch:nir_canon_inlined",
            graph=NIR_CANONICAL_PLS,
            input_mapping={"X": "nir", "y": "y"},
            output_mapping={"pred": "pred_canon_inl"},
            inline_policy="inline",
        ),
        # A regular branch:
        [{"adapter": "tabular.encoder"}, {"model": RandomForestRegressor()}],
    ]},

    {"merge": "predictions"},
    {"model": Ridge()},
]
```

### 10.4 DAG resultant

```text
Parent DAG (compiled):

     +----- src:nir, src:tab, y ------+
     |                                 |
     v                                 v
   ForkNode (3 branches)
     |              |              |
     v              v              v
 SubgraphNode    Subgraph     normal subgraph
 opaque          inlined      (tab.encoder + RF)
 [SNV->SG2->PLS] [snv,sg2,pls become parent nodes]
     |              |              |
     v              v              v
 pred_canon    pred_canon_inl   pred_rf
     |              |              |
     +-----+ PredictionJoin (3 cols, validate OOF) +-+
                       |
                       v
                  Ridge meta
```

Schemas:

```text
SubgraphNodeSpec(opaque):
  - executed as a black box
  - artifacts persisted under its own subgraph_id namespace
  - cache key = hash(graph.id, version, params, input lineage)
  - parent scheduler treats it as a single node

SubgraphNodeSpec(inline):
  - planner expands its nodes into the parent graph
  - parent nodes can reuse cache from inner nodes (e.g. SNV result reused
    by another inlined branch)
  - artifacts persisted under parent run, with subgraph_id prefix in node_id

inline_policy="auto":
  - inline if cache reuse potential is detected (shared preproc upstream)
  - opaque otherwise
```

### 10.5 Invariants ML cles

- `GraphInterface` doit etre declaree pour qu'un sous-DAG soit composable.
- Compatibility check: `input_mapping["X"].representation` doit etre compatible
  avec `PortSpec.representation` du sous-DAG.
- OOF: que ce soit inline ou opaque, le sous-DAG respecte son propre invariant
  OOF (fold-train pour les adapters stateful).
- Reproductibilite: `SeedContext.child(node_id="branch:nir_canon", ...)`
  derive le seed du sous-DAG (root_seed + subgraph_id + fold_id).
- Refit: le sous-DAG est refit comme une unite.
- Versioning: `SerializableRef(version="1.2.0")` -> upgrade `1.3.0` doit etre
  explicit (le bundle stocke la version exacte).

### 10.6 Points de friction

1. Si un sous-DAG opaque a son propre SearchSpace, doit-il apparaitre dans le
   tuner du parent ou non ? Reponse v1: non, opacite = scope ferme.
2. Pour `inline`, le planner doit-il refuser si deux sous-DAGs ont des noeuds
   de meme id ? Solution: prefixer par `subgraph_id`.
3. Pour `inline_policy="auto"`, quel critere precis declenche `inline` ? Reuse
   detecte par hash de l'IR amont ? Score d'utilite ?
4. Un sous-DAG peut-il etre un DAG-ML d'une version differente du parent ? Si
   oui, contrats `ml_data.contract` version-stables (v1 -> v1 ok), majeure
   different -> refus.

### 10.7 ML_DATA vs DAG-ML

| ML_DATA                                            | DAG-ML                                              |
|----------------------------------------------------|-----------------------------------------------------|
| Aucune notion de sous-DAG                          | Definit `SubgraphNodeSpec`, `GraphInterface`        |
| Fournit les meme contrats data au sous-DAG         | Planner decide inline vs opaque                     |
| -                                                  | Cache key inclut `subgraph_id + version`            |

### 10.8 Resultat attendu

- Bundle parent contient: 1 entree pour le sous-DAG opaque (refit-as-unit),
  les noeuds individuels pour la version inlined, et le Ridge meta.
- Lineage: chaque PredictionBlock porte `producer_node_id` complet incluant
  le path subgraph (ex: `branch:nir_canon/model:pls`).

---

## UC11 - Piege OOF: train predictions refusees par defaut

### 11.1 Contexte metier

Un utilisateur tente de stacker en utilisant les predictions de train (non-OOF)
des base learners comme features pour le meta-modele. C'est une fuite de
donnees classique: le meta-modele apprend a corriger les surapprentissages
des base learners.

### 11.2 Donnees impliquees

| Source     | Type             | Granularity   | N samples |
|------------|------------------|---------------|-----------|
| `nir`      | `dense_signal`   | `per_sample`  | 500       |

### 11.3 Pipeline DSL souhaite (PIEGE)

```python
# User attempts:
pipeline = [
    {"sources": ["nir"]},
    {"branch": [
        [SNV(), {"model": PLSRegression(n_components=12)}],
        [MSC(), {"model": RandomForestRegressor()}],
    ]},
    {"merge": "predictions",
     "policy": {
         "allow_train_predictions_as_features": True,  # <- DANGEROUS: include train preds
         "join_on": "sample_id",
         "include_partitions": ["train", "val"],
     }},
    {"model": Ridge()},
]
```

### 11.4 DAG resultant + comportement attendu

```text
COMPILE phase:
  parse pipeline -> GraphSpec
  detect PredictionJoinNode with policy.allow_train_predictions_as_features=True
  emit WARNING during compile (lineage records leakage_acknowledged=True)

PLAN phase:
  PredictionJoinNode.declare_required_partitions() -> ("train", "val")

FIT_CV phase, at the moment the PredictionJoinNode executes:

  Default (flag NOT set):
      if any block.partition != "val":
          raise OOFLeakageError(payload={
              "node_id": "merge:pred",
              "violator_block": {"producer": "branch:0/model:pls", "partition": "train"},
              "policy": {"allow_train_predictions_as_features": False},
              "remediation": "Use only OOF predictions (partition='val'), OR set policy.allow_train_predictions_as_features=True explicitly (NOT recommended).",
          })

  Opt-in (flag set to True):
      emit WARNING in MetricsLogger
      record leakage_acknowledged=True in LineageRecord for the join
      append "train_predictions_used" flag to every downstream PredictionBlock.flags
      proceed
```

The flag name is intentionally verbose (`allow_train_predictions_as_features`)
so a `grep` of the codebase finds every leakage opt-in instantly, and code
review surfaces the intent without requiring a second confirmation field.

Diagramme:

```text
                          base learner b0 (PLS)
                                |
              +-----------------+-----------------+
              | partition=train (fit on train)    | partition=val (OOF)
              v                                   v
   PredictionBlock(train_preds)         PredictionBlock(val_preds)
              |                                   |
              +-----+ PredictionJoinNode +--------+
                              |
            +-----------------+-----------------+
            | flag NOT set (default)            | allow_train_predictions_as_features=True
            v                                   v
       OOFLeakageError                  FeatureTable joined (with leakage flag)
       (refuse train block)                     |
                                                v
                                       Ridge meta-model fit
                                       (lineage marked leakage_acknowledged=True)
```

### 11.5 Invariants ML cles

- DEFAULT: refuse train predictions; raise `OOFLeakageError`.
- ESCAPE: single boolean `allow_train_predictions_as_features=True` on the
  `PredictionJoinNode`. The verbose name does the double-confirmation work.
- PredictionBlock porte un champ `oof_safe: bool` derive de partition + fold_id.
- `PredictionJoinNode` validate: `all(block.oof_safe for block in inputs)` sauf
  si le flag d'opt-in est set.
- Lineage: la PredictionBlock du Ridge meta porte
  `flags=["train_predictions_used"]`, et le join porte
  `leakage_acknowledged=True`. Apparait dans le manifest, dans les
  metriques, et dans tout report.
- Selection: `RankingPolicy.exclude_leaky_variants=True` (default) ecarte ces
  variants de la selection top-k automatique.

### 11.6 Points de friction (3-4 questions)

1. Un single flag avec nom verbose suffit-il ou faut-il une seconde confirmation
   pour le webapp UI ? Decision v1: single flag. UI peut imposer une second
   confirmation par-dessus (modal), mais l'API Python reste mono-flag.
2. Faut-il aussi rejeter les predictions test (partition="test") comme features ?
   Oui, par defaut. Sauf cas transfer learning (a discuter).
3. Une PredictionBlock OOF mais avec un nombre de samples != train_full doit-elle
   etre acceptee (cas augmentation: origins only)? Oui, c'est correct.
4. Comment exposer ce message d'erreur de facon comprehensible ? Payload JSON
   structure + traduction i18n via webapp.

### 11.7 ML_DATA vs DAG-ML

| ML_DATA                                            | DAG-ML                                              |
|----------------------------------------------------|-----------------------------------------------------|
| Aucune notion de OOF / partition                   | Toute la logique OOF, OOFLeakageError              |
| Materialize les donnees sans connaitre l'usage     | PredictionJoinNode valide partitions                |
| -                                                  | Definit champs `oof_safe`, `flags`, `leakage_acknowledged` |

### 11.8 Resultat attendu

- En mode safe (default): refus avec `OOFLeakageError` + remediation message.
- En mode opt-in: pipeline tourne mais predictions et bundle portent
  un drapeau permanent `train_predictions_used` + `leakage_acknowledged=True`
  sur le join.

Message d'erreur attendu (extrait JSON):

```json
{
  "error": "OOFLeakageError",
  "node_id": "merge:pred",
  "violator_blocks": [
    {"producer": "branch:0/model:pls", "partition": "train", "fold_id": 0},
    {"producer": "branch:1/model:rf",  "partition": "train", "fold_id": 0}
  ],
  "policy": {"allow_train_predictions_as_features": false},
  "remediation": "Use only OOF predictions (partition='val'). To override (NOT recommended), set policy.allow_train_predictions_as_features=True.",
  "docs_url": "dagml.readthedocs.io/invariants/oof.html"
}
```

---

## UC12 - Mixed merge: features + predictions OOF

### 12.1 Contexte metier

Un stacking ou certaines branches contribuent leurs features transformees
directement (preprocessings utiles tels quels, comme PCA(20)) et d'autres
branches contribuent leurs predictions OOF (modeles non-lineaires comme RF
ou SVR). Le meta-modele consomme l'union des deux.

### 12.2 Donnees impliquees

| Source     | Type             | Granularity   | N samples |
|------------|------------------|---------------|-----------|
| `nir`      | `dense_signal`   | `per_sample`  | 600       |
| target `y` | `table` (num)    | `per_sample`  | 600       |

### 12.3 Pipeline DSL souhaite

```python
pipeline = [
    {"sources": ["nir"]},
    {"y_processing": "standardize"},

    KFold(n_splits=5, shuffle=True, random_state=42),

    {"branch": [
        # Branch 0: features (PCA-20 directly, no model)
        [SNV(), {"adapter": "spectra.flatten"},
                 {"adapter": "tabular.pca", "params": {"k": 20}}],

        # Branch 1: prediction (PLS as base learner -> OOF)
        [MSC(), {"model": PLSRegression(n_components=12)}],

        # Branch 2: prediction (RF)
        [SavitzkyGolay(deriv=1), {"model": RandomForestRegressor()}],
    ]},

    {"merge": "all",
     "policy": {
         "branches_as_features": [0],         # branch 0 contributes its features
         "branches_as_predictions": [1, 2],   # branches 1, 2 contribute OOF preds
         "join_on": "sample_id",
         "namespace_columns": True,
         "validate_oof": True,
     }},

    {"model": Ridge(alpha=0.5)},
]
```

### 12.4 DAG resultant

```text
                       KFold(5, rs=42)
                              |
       +----------------------+----------------------+
       v                      v                      v
   Branch 0              Branch 1               Branch 2
   SNV+flatten+PCA(20)   MSC+PLS(12)            SG1+RF
       |                      |                      |
       v                      v                      v
   FeatureBlock           PredictionBlock        PredictionBlock
   (per fold, train+val   (val/OOF only)         (val/OOF only)
    if stateful PCA fit
    on train -> transform
    both train and val,
    BUT only val rows
    enter the join)
       |                      |                      |
       +-----+ MixedJoinNode (1 feat block + 2 pred blocks) +-+
                              |
                              v
              FeatureTable: columns = ("b0.pc1..pc20", "b1_pls_pred", "b2_rf_pred")
              rows = OOF samples (partition=val of fold)
                              |
                              v
                       Ridge(0.5)
```

Nodes:

| Node id           | kind              | Notes                                     |
|-------------------|-------------------|-------------------------------------------|
| `mixed:join`      | mixed_join        | accepts FeatureBlock + PredictionBlock    |
| `validate:oof`    | invariant_check   | validates partition=val for pred branches |

### 12.5 Invariants ML cles

- Features de branche 0: `tabular.pca` fit sur fold train -> transform train et
  val. Pour le join, seules les lignes val sont utilisees (sinon mismatch avec
  preds OOF de b1 et b2 qui ne contiennent que val).
- Predictions de branches 1, 2: OOF strict, refus si train.
- Alignment: les 3 blocs partagent `sample_ids` du fold val.
- Reproductibilite: meme fold (meme seed) pour les 3 branches.
- Refit: PCA b0 refit sur full train, base learners b1/b2 refit sur full train,
  meta Ridge refit sur OOF concat (cache des OOF utilises pour le fit meta).

### 12.6 Points de friction

1. Comment unifier le contrat ? Branche 0 produit `FeatureBlock`, branche 1/2
   produit `PredictionBlock`. Le `MixedJoinNode` accepte une union avec
   discriminator (`block_kind: "feature" | "prediction"`).
2. Le PCA b0 doit etre fit fold-train et appliquer aux samples val. Mais en
   refit, on fit sur full train -> au predict, pas de fold. Donc deux artifacts
   PCA (CV vs refit). Cache de l'OOF preserve seulement pour le CV.
3. Si la branche 0 a une dimension de sortie tres grande (PCA k=200), les
   colonnes features dominent les 2 colonnes predictions. Faut-il scaler ?
4. Validation cross-source: une feature de b0 pourrait fuiter (PCA non stateful?
   on a dit stateful). S'assurer que `fit_scope=fold_train`.

### 12.7 ML_DATA vs DAG-ML

| ML_DATA                                            | DAG-ML                                              |
|----------------------------------------------------|-----------------------------------------------------|
| Fournit `FeatureTable` (b0) avec stateful PCA      | MixedJoinNode unifie feat + pred                    |
| Fit `tabular.pca` avec fold_train scope            | Valide OOF pour pred blocks                         |
| Aucune notion de partition                         | Selectionne uniquement val rows pour le join        |

### 12.8 Resultat attendu

- 600 OOF preds meta + bundle (b0 PCA refit + b1 PLS refit + b2 RF refit + Ridge).
- 22 colonnes meta features (20 PCA + 2 preds).
- Lineage: chaque colonne porte son `source_id` = `branch:0/...` ou `branch:1/model:pls`.

---

## Decisions design transverses

Synthese des questions de friction recurrentes dans les UC. Ces questions
restent ouvertes et doivent etre tranchees avant implementation v1.

### D1 - Auto-resolution lossy vs explicit user choice

UC1 (image embedding), UC2 (resample), UC5 (augmentation), UC10 (sub-DAG):
- Quand `find_path` trouve une chaine d'adapters lossy (image embedding,
  PCA genotype, resampling avec interpolation), DAG-ML doit-il:
  (a) auto-resolve si `policy.allow_lossy_adapters=True` (defaut current)?
  (b) toujours escalader via `requires_user_choice`?
  (c) escalader uniquement si plusieurs chaines lossy concurrent?

Proposition: (c) par defaut. Plus un mode strict (a) opt-in via DSL.

### D2 - Niveau du ranking et selection

UC3 (obs vs sample), UC4 (meas vs patient), UC6 (per-branch vs meta), UC8
(global vs per-site):
- Quand un pipeline produit des predictions a plusieurs niveaux (observation,
  sample, group, branche, meta), quel niveau utilise-t-on pour ranker?
  Reponse v1: niveau le plus aggrege (sample > group > observation), sauf
  override via `ranking_level` dans `SelectPolicy`.

### D3 - Refit semantics: same hyperparams or aggregate?

UC6 (stacking refit), UC7 (tuner best refit), UC8 (per-branch refit):
- Au refit, doit-on:
  (a) reutiliser exactement les hyperparams du meilleur fold?
  (b) refit avec un agregat (par exemple median des n_components des 5 folds)?
  (c) re-tuner sur full train?

Proposition: (a) par defaut, (c) opt-in.

### D4 - Schema fingerprint scope

UC9 (predict + diverge weather), UC10 (subgraph versioning):
- Le fingerprint doit-il inclure:
  - axe coordinates (wavelengths exactes)?
  - feature names exacts post-encoder?
  - les seeds utilises?
  - les versions des plugins?

Proposition: oui pour axes, feature names, plugin versions. Non pour seeds
(qui changent par run, c'est `LineageRecord` qui les porte).

### D5 - Parallelism granularity et seeding

UC1 (5 folds + 6 sources), UC7 (40 trials), UC10 (inline subgraphs):
- A quel niveau paralleliser?
  - variant (joblib loky, default)
  - fold (utile pour gros models)
  - branche (pour separation)
  - sub-DAG inline (rare)
- Et comment garantir que la parallelisation imbrique n'explose pas les
  threads (BLAS x joblib x torch)?

Proposition: parallelism budget unique au niveau `RunContext`, distribue
top-down par le scheduler. Un seul niveau parallele actif par defaut.

---

## Annexe A - Mapping DSL nirs4all -> DAG-ML IR

| DSL keyword                | NodeKind                         | Notes                             |
|----------------------------|----------------------------------|-----------------------------------|
| `{"sources": [...]}`       | (implicit) source declarations   | declared in `GraphSpec.inputs`    |
| transformer instance        | `adapt`                          | uses ML_DATA adapter              |
| `{"y_processing": ...}`    | `adapt` (target)                 | TargetBlock transform             |
| `{"model": ...}`           | `model`                          | with `ModelInputSpec`             |
| splitter instance           | `split`                          | folds emitted, no DataPlan change |
| `{"branch": [...]}`        | `fork` + `subgraph`              | duplication or separation         |
| `{"merge": "predictions"}` | `prediction_join` (OOF validate) | refuses non-OOF                   |
| `{"merge": "features"}`    | `feature_join`                   | uses `FeatureJoiner`              |
| `{"merge": "concat"}`      | `prediction_join` (reassemble)   | for separation branches           |
| `{"merge": "all"}`         | `mixed_join`                     | features + predictions            |
| `{"tag": ...}`             | `tag`                            | non-destructive label             |
| `{"exclude": ...}`         | `exclude`                        | removes from training only        |
| `{"sample_augmentation"}`  | `adapt` (augmentation)           | + SampleRelation extension        |
| `{"concat_transform"}`     | `adapt` (multi-transform)        | concat outputs as new features    |
| `{"rep_to_sources"}`       | `restructure`                    | rep groups -> multi-source        |
| `{"rep_to_pp"}`            | `restructure`                    | rep groups -> processings axis    |
| `_or_`                      | `search_space` (categorical)    | enumere alternatives              |
| `_grid_`                    | `search_space` (cartesian)      | params grid                       |
| `_range_`                   | `search_space` (linspace)       | linear sweep                      |
| `_log_range_`               | `search_space` (logspace)       | logarithmic sweep                 |
| `_cartesian_`               | `search_space` (stages)         | pipeline stages cross-product     |
| `_zip_`                     | `search_space` (paired)         | zip params                        |
| `_chain_`                   | `search_space` (sequence)       | ordered choices                   |
| `_sample_`                  | `search_space` (random/bayes)   | random sampling                   |
| `SubgraphNodeSpec`          | `subgraph`                      | reified DAG                       |

---

## Annexe B - Invariants summary

| Invariant                        | Owner          | Enforced by                                  |
|----------------------------------|----------------|----------------------------------------------|
| OOF: meta features = val preds   | DAG-ML         | `PredictionJoinNode.validate`                |
| No-leakage: aug origins          | DAG-ML         | check `origin_id` against val fold           |
| Split unit: group/target         | DAG-ML         | `SplitPolicy` + `SampleRelation.group_ids`   |
| Stateful fit on fold train       | DAG-ML         | `AdapterContext.phase + fit_scope`           |
| Schema fingerprint at predict    | DAG-ML+ML_DATA | `schema_fingerprint` recompute + compare     |
| Determinism: seed propagation    | DAG-ML         | `SeedContext.child(node, fold, trial)`       |
| Refit single fold                | DAG-ML         | refit phase replaces splitter                |
| Plugin version compatibility     | ML_DATA        | `requires_plugin_versions` semver check      |
| Block immutability               | ML_DATA        | frozen dataclasses, `arr.flags.writeable`    |
| Alignment determinism            | ML_DATA        | sorted canonical sample_ids                  |

---

## Annexe C - Phases x UC matrix

| Phase     | UC1 | UC2 | UC3 | UC4 | UC5 | UC6 | UC7 | UC8 | UC9 | UC10| UC11| UC12|
|-----------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| COMPILE   | X   | X   | X   | X   | X   | X   | X   | X   | X   | X   | X   | X   |
| PLAN      | X   | X   | X   | X   | X   | X   | X   | X   | X   | X   | X   | X   |
| FIT_CV    | X   | X   | X   | X   | X   | X   | X   | X   | -   | X   | X   | X   |
| SELECT    | X   | X   | X   | X   | X   | X   | X   | X   | -   | X   | -   | X   |
| REFIT     | X   | X   | X   | X   | X   | X   | X   | X   | -   | X   | -   | X   |
| PREDICT   | X   | X   | X   | X   | X   | X   | X   | X   | X   | X   | -   | X   |
| EXPLAIN   | -   | -   | -   | -   | -   | -   | -   | -   | -   | -   | -   | -   |

UC9 ne traverse que `COMPILE -> PLAN -> PREDICT` car il consomme un bundle existant.
UC11 s'arrete a `FIT_CV` quand l'invariant OOF est viole (refus du run).

---

## Annexe D - Artifacts produits par UC

| UC  | FittedAdapter(s)                              | Model(s)                       | Sub-bundles |
|-----|-----------------------------------------------|--------------------------------|-------------|
| UC1 | img_emb_top, img_emb_side, geno_pca, tab_enc, fusion | RandomForestRegressor   | 0           |
| UC2A| spectra_resample x3, snv                      | PLS(12)                        | 0           |
| UC2B| -                                             | 15 PLS variants + Ridge meta   | 0           |
| UC3 | snv, sg2                                      | PLS(15) + Aggregator(robust)   | 0           |
| UC4 | tab_enc, snv, sg1                             | Ridge                          | 0           |
| UC5 | snv, sg, aug_noise, aug_shift                 | PLS(12)                        | 0           |
| UC6 | snv, msc, detrend                             | PLS + RF + SVR + Ridge meta    | 0           |
| UC7 | per trial: snv|msc|detrend, sg variant        | best model from best trial     | 0           |
| UC8 | snv, sg                                       | 3 models per site              | 0           |
| UC9 | reused from UC1 bundle                        | reused                         | 0           |
| UC10| canonical sub-DAG artifacts + parent's        | nested model + Ridge meta      | 1 (opaque)  |
| UC11| (n/a - refused)                               | (n/a)                          | 0           |
| UC12| snv, msc, sg1, pca(20)                        | PLS + RF + Ridge meta          | 0           |

---

## Annexe E - Erreurs typees et messages

| Erreur                          | Levee dans                       | Cause                                    |
|---------------------------------|----------------------------------|------------------------------------------|
| `OOFLeakageError`               | PredictionJoinNode               | partition=train without opt-in flag      |
| `LeakageError` (augmentation)   | AugmentationAdapter consumer     | origin_id of val sample in train         |
| `SchemaFingerprintMismatch`     | bundle predict                   | new dataset diverges from train schema   |
| `NoPlanFoundError`              | DataPlanner.resolve              | no adapter path source -> port repr      |
| `PortArityError`                | DataPlanner.resolve              | multi sources to non-multi port          |
| `FusionError`                   | FeatureJoiner.fit                | max_output_features exceeded             |
| `AlignmentError`                | aligner                          | exact join with mismatched sample_ids    |
| `PluginVersionError`            | ML_DATA bundle load              | plugin missing or major out of range     |
| `StatefulAdapterMisuse`         | RepresentationAdapter.transform  | stateful adapter without fitted          |
| `RepresentationError`           | MLDataset.materialize            | unknown representation id                |

Format payload commun:

```json
{
  "error": "<error_class>",
  "node_id": "...",
  "phase": "FIT_CV",
  "fold_id": 2,
  "details": { ... },
  "remediation": "...",
  "docs_url": "..."
}
```

---

## Annexe F - Ordre des phases par UC (resumee)

| UC  | Phases executees                                                |
|-----|-----------------------------------------------------------------|
| UC1 | COMPILE -> PLAN -> FIT_CV(5) -> SELECT -> REFIT -> bundle       |
| UC2A| COMPILE -> PLAN -> FIT_CV(5) -> SELECT -> REFIT -> bundle       |
| UC2B| COMPILE -> PLAN(15 variants) -> FIT_CV -> SELECT(top per branch) -> meta-FIT_CV -> REFIT -> bundle |
| UC3 | COMPILE -> PLAN -> FIT_CV(5) -> AGGREGATE -> SELECT -> REFIT -> bundle |
| UC4 | COMPILE -> PLAN -> FIT_CV(5,group) -> SELECT -> REFIT -> bundle |
| UC5 | COMPILE -> PLAN -> FIT_CV(5+aug) -> SELECT -> REFIT(+aug) -> bundle |
| UC6 | COMPILE -> PLAN -> FIT_CV(branches) -> PredJoin(OOF) -> meta FIT_CV -> SELECT -> REFIT -> bundle |
| UC7 | COMPILE -> PLAN(lazy) -> FOR_EACH_TRIAL{FIT_CV} -> SELECT(best) -> REFIT -> bundle |
| UC8 | COMPILE -> PLAN -> FORK(by_meta) -> FIT_CV per branch -> MERGE(concat) -> SELECT -> REFIT (3 models) -> bundle |
| UC9 | LOAD_BUNDLE -> SCHEMA_CHECK -> PREDICT                          |
| UC10| COMPILE(with subgraph) -> PLAN -> FIT_CV -> SELECT -> REFIT -> bundle (nested) |
| UC11| COMPILE -> PLAN -> FIT_CV -> PredJoin REFUSE -> raise OOFLeakageError |
| UC12| COMPILE -> PLAN -> FIT_CV(branches mixed) -> MixedJoin(validate OOF) -> meta FIT_CV -> SELECT -> REFIT -> bundle |

---

Fin du document.
