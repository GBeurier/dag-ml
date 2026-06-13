# DAG-ML Use Cases v1

Status: design v1. Companion to `dag_ml_externalization_from_code.md`
and`ml_data_specification_v1.md`. Each use case (UC) materializes the DSL,
the DAG compiles the invariants, the phase order, and the artifacts.

Notation used throughout the document:

- Canonical phases:`COMPILE -> PLAN -> FIT_CV -> SELECT -> REFIT -> PREDICT [-> EXPLAIN]`. - Reference types: see`ml_data/contract.py`(`SourceDescriptor`,`DataBlock`,`DataView`,`FeatureTable`,`ModelInputSpec`,`FusionPolicy`,`DataPlan`,`SampleRelation`,`AggregationPolicy`,`SplitPolicy`,`AugmentationPolicy`,`SeedContext`,`SearchSpace`,`SubgraphNodeSpec`). - Node identifiers:`<kind>:<short>`. Edges:`A.out -> B.in`. - Conventions:`OOF`= out-of-fold predictions,`MS`= multi-source.

Table of contents:

| UC  | Main theme                                            | Covered axes                           |
|-----|-------------------------------------------------------|----------------------------------------|
| UC1 | Heterogeneous multi-source -> RandomForest            | DataPlan, early fusion, missing sources |
| UC2 | NIRS multi-instrument (3 spectrometers)               | Homogeneous MS, by_source, concat/stack |
| UC3 | Repetitions: multiple X values for one Y              | SampleRelation, group-aware split, aggregation |
| UC4 | Entities (patients, plots) split-unit                 | SplitPolicy(split_unit="group")        |
| UC5 | Train-only augmentation + correct OOF                 | AugmentationPolicy, origin tracking     |
| UC6 | Multi-level stacking                                  | Branch+merge predictions, meta-model   |
| UC7 | Generators + Bayesian tuning                          | _cartesian_, _grid_, TunerAdapter      |
| UC8 | Branches by metadata/tag, merge concat                | Branch separation, by_metadata         |
| UC9 | Full refit + bundle predict new heterogeneity         | schema_fingerprint, replay             |
| UC10| SubDAG reifies as node                                | SubgraphNodeSpec, inline vs opaque     |
| UC11| OOF train predictions rejected by default             | Invariant errors, leakage opt-in       |
| UC12| Mixed merge: features + OOF predictions               | Cross-source validation                |

---

## UC1 - Heterogeneous multi-source to RandomForest

### 1.1 Business context

Predicting the protein content of a wheat variety from five
methods per sample: 1 NIRS spectrum, 2 RGB photos (top view, side view),
1 genotypic heritage (SNP dosage), 1 daily weather series over the season
culture, plus some categorical metadata (variety, plot, year).

### 1.2 Involved data

| Source       | Type ML_DATA       | Modality      | Granularity         | Native rep                  | Sample key  | N samples | Notes |
|--------------|---------------------|---------------|---------------------|-----------------------------|-------------|-----------|-------|
| `nir`        | `dense_signal`      | spectroscopy  | `per_sample`        | `signal_with_processings`   | `sample_id` | 800       | 512 wl 950-2500 nm |
| `photo_top`  | `image_rgb`         | image         | `per_sample`        | `rgb_image`                 | `sample_id` | 800       | 224x224x3 |
| `photo_side` | `image_rgb`         | image         | `per_sample`        | `rgb_image`                 | `sample_id` | 760       | 40 missing |
| `geno`       | `genotype_matrix`   | genotype      | `per_sample`        | `variant_matrix`            | `sample_id` | 800       | 12000 SNPs, int8 |
| `weather`    | `time_series`       | meteo         | `per_sample_sequence`| `series_mv`                | `sample_id` | 800       | 180 days x 6 vars |
| `meta`       | `table`             | metadata      | `per_sample`        | `tabular_mixed`             | `sample_id` | 800       | variety/plot/year |
| target `y`   | `table`             | -             | `per_sample`        | `tabular_numeric`           | `sample_id` | 800       | proteine (%) |

Repetitions: none. `group_ids = plot_id` (40 plots, ~20 samples/plot).

### 1.3 Desired DSL pipeline

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

### 1.4 Resulting DAG

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

List of nodes (NodeSpec):

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

### 1.5 Key ML invariants

- OOF:`model:rf`emits`partition="val"`predictions, aligned by`sample_id`. - No leakage: all stateful`adapt:*`(`image.embedding`,`genotype.pca`,`tabular.encoder`) have`fit_scope="fold_train"`.`DataPlanner.execute_fit`did them on the fold train samples only. - Split unit:`group`on`plot_id`(GroupKFold) -> no plot appears in
  train and val of the same fold. - Reproducibility:`SeedContext.child(node_id="adapt:imgT", fold_id=k)`derivative
  the seed of the backbone (dropout, init head). Persists in`LineageRecord`. - Refit: same`DataPlan`reapplied on complete train,`fold_id="final"`,
  emission of new`FittedAdapter`for`image.embedding`,`genotype.pca`,`tabular.encoder`,`fusion.feature_joiner`,`RandomForestRegressor`. - Missing sources:`photo_side`missing for 40 samples -> presence mask
  propagated -> 1`indicator__photo_side_present`column added by the joiner.

### 1.6 Friction points

1.`image.embedding`is lossy + stateful. Should we force`policy.allow_lossy_adapters=True`explicitly or require user choice
   (`requires_user_choice`escalation)? 2.`concat_features`fusion can explode dimension (256+256+32+36+~50+512
   ~= 1142). Should we impose`max_output_features`or warning only? 3. For missing`photo_side`, which default policy:`indicator`(RF likes),`impute`(image -> medium train embedding), or`drop`? Choice is ML, not data. 4. Does`image.embedding`need to be refit by fold (fit_scope="fold_train") or
   shared`train_only`(only one fit on full train, freezes for the entire CV)? First option = correct OOF but expensive; second option = acceptable leakage
   if backbone not finetuned.

### 1.7 ML_DATA vs DAG-ML

| ML_DATA                                                  | DAG-ML                                                   |
|----------------------------------------------------------|----------------------------------------------------------|
| Declare`SourceDescriptor`for the 6 sources            | Compile the DSL into `GraphSpec`                         |
| `find_path(src, target="tabular_numeric", policy)`       | Decide `allow_lossy_adapters`, reject if ambiguous       |
| Execute `materialize`, `adapt`, `align`, `join`          | Choose the phase (`fit_cv` -> `fold_train` scope)        |
| Provides`PresenceMask`for`photo_side`                 | Decide `missing_source="indicator"`                      |
| Computes `schema_fingerprint`                           | Check the fingerprint when predicting                    |
| Provides`SampleRelation`with`group_ids=plot_id`        | Pass `groups` to the `GroupKFold` splitter               |

### 1.8 Expected result

- Artifacts: 5`FittedAdapter`(image_top, image_side, geno_pca, tab_enc, joiner)
  + 1 final`RandomForestRegressor`+ per-fold artifacts. - Predictions:`PredictionBlock`OOF by fold +`fold_id="final"`after refit. - Exports:`ExecutionBundle`(graph + plan + adapters + RF) with`schema_fingerprint`.`.bundle`format (joblib + JSON manifest). - Lineage: 800 samples x 5 folds = 4000 OOF prediction records. - Metric:`rmsecv`+`r2_cv`by fold + macro mean.

---

## UC2 - NIRS multi-instrument (3 spectrometres)

### 2.1 Business context

Cross-instrument validation: measure the same physical sample out of 3
spectrometers (FOSS NIRS-DS2500, Bruker MPA, ASD FieldSpec). Predict the
dry matter (MS, regression). Compare concat early-fusion vs branch`by_source`(a model specialized by instrument then together).

### 2.2 Involved data

| Source        | Type      | Modality      | Sample key  | Native rep              | N    | Range          |
|---------------|-----------|---------------|-------------|-------------------------|------|----------------|
| `nir_foss`    | `dense_signal` | spectroscopy | `sample_id` | `signal_1d` 1050 wl  | 600  | 400-2500 nm    |
| `nir_bruker`  | `dense_signal` | spectroscopy | `sample_id` | `signal_1d` 2074 wl  | 600  | 800-2500 nm    |
| `nir_asd`     | `dense_signal` | spectroscopy | `sample_id` | `signal_1d` 2151 wl  | 580  | 350-2500 nm    |
| target `y`    | `table`        | -            | `sample_id` | `tabular_numeric`    | 600  | MS (%)         |

Repetitions: 1 spectrum / instrument / sample. No groups.

### 2.3 Desired DSL pipeline

```python
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold

# Variant A: early fusion by resampling onto a common axis
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

# Variant B: branch per source + OOF ensemble
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

### 2.4 Resulting DAG (variant A: stack_channels)

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

Resulting DAG (variant B: by_source branches):

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

### 2.5 Key ML invariants

- OOF (variant B): each PLS produces`partition="val"`predictions at the level
  outer fold; the meta-model only consumes OOF predictions (refusal by
  default if train preds, see UC11). - No leakage: the 3 instruments share the same`sample_id`-> same fold
  outer for the 3 branches. - Split unit:`sample`. No groups. - Reproducibility: same`KFold(random_state=42)`for the 3 branches in
  variant B (otherwise OOF shift). -`nir_asd`only has 580/600 samples:`AlignmentPolicy(join="inner")`-> N=580
  workforce. Decision to be traced in the manifest.

### 2.6 Friction points

1. Variant A: what to do if the 3 instruments cover different ranges
   (400-2500 vs 800-2500 vs 350-2500)? Resample on intersection or union with
   mask? Intersection 800-2500 loses VIS info for FOSS and ASD. 2. Variant B: the`_range_`n_components are different per branch. How
   select? Best per branch then stacker? Best overall after meta? 3. Stack_channels: Classic PLS expects 2D, so a specialized PLS-3D is needed
   (POP-PLS) or flatten? The contract must be explicit about the adapted model.

### 2.7 ML_DATA vs DAG-ML

| ML_DATA                                                | DAG-ML                                              |
|--------------------------------------------------------|-----------------------------------------------------|
| Adapter `spectra.resample(target_axis)`                | Branche `by_source` (ForkNode + MapNode)            |
| Stack/concat on`processing`or`channel`axis         | Generateur `_range_` enumere les variants n_comp    |
| AlignmentPolicy(inner) selectionne les 580            | Valid OOF PredictionJoin before meta-model         |
| Fournit `AxisSpec.coordinates` (wavelengths)           | Choisit ranking metric et top_k                     |

### 2.8 Expected result

- Variant A: 1 PLS model on the stacked tensor (3 instruments x 512 wl). Bundle = 1.
- Variant B: 15 PLS variants (3 inst x 5 n_comp) + 1 Ridge meta. Bundle = 16.
- Predictions OOF par variant + meta-OOF + final refit.
- Comparison reporting: RMSECV variant A vs variant B.

---

## UC3 - Repetitions: several X for one Y

### 3.1 Business context

NIRS measurement of plant powder: for each physical sample (`sample_id`),
the technician acquires 3 spectra (positions A/B/C in the dish) and 1
chemical reference measurement (proteins). The Y exists at the sample level, not
of observation. The NIRS model must predict by observation, but aggregation
final is done at the sample level (mean or median), with exclusion of outliers.

### 3.2 Involved data

| Source       | Type            | Modality      | Granularity              | observation_id    | sample_id   | target_id | group_id  |
|--------------|-----------------|---------------|--------------------------|-------------------|-------------|-----------|-----------|
| `nir`        | `dense_signal`  | spectroscopy  | `per_sample_repeated`    | `nir_S001_A`...   | `S001`...   | `y_S001`  | `lot_42`  |
| `chem`       | `table`         | reference     | `per_sample`             | `chem_S001`       | `S001`      | `y_S001`  | `lot_42`  |

400 samples logiques, 1200 observations NIRS. 25 lots de production (15-20
samples/lot). `target_id == "y_" + sample_id`.

### 3.3 Desired DSL pipeline

```python
pipeline = [
    {"sources": ["nir"]},
    {"y_processing": "standardize"},
    [SNV(), SavitzkyGolay(window=11, deriv=2)],

    {"split_policy": SplitPolicy(
        split_unit="target",           # all observations sharing target_id together
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

### 3.4 Resulting DAG

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

### 3.5 Key ML invariants

- The 3 obs of an S001 sample go **all** in the same fold (train or val),
  never split.`split_unit="target"`guarantees this. - OOF:`partition="val"`predictions are produced at the observation level,
  the sample aggregation is done afterwards and stored separately with`aggregation_level="sample"`. -`keep_observation_predictions=True`=> DAG-ML stores two PredictionBlocks
  (one obs-level, one sample-level), sharing the same`fold_id`. - Reproducibility: even`SampleRelation`when predicted (verified via`schema_fingerprint`). - Outlier exclusion (0.95): only acts on aggregation, never on raw OFF.

### 3.6 Friction points

1. Which level to rank/select: RMSE obs-level (1200 preds) or
   sample-level (400 preds) ? The second is what matters, but the
   first reduces the estimation variance. 2. If a NIRS observation of a sample is very noisy, its biased inclusion
   the sample-mean. Do you need a`SpectralQualityFilter`upstream like`{"exclude": SpectralQualityFilter()}`before aggregation, or a filter on
   the predictions? 3.`vote`aggregation for multi-class classification: how to manage
   ties (3 votes for 3 different classes)? 4. Can`target_id`be automatically derived from`sample_id`when 1:1? Or should we always make it explicit in`SampleRelation`?

### 3.7 ML_DATA vs DAG-ML

| ML_DATA                                                        | DAG-ML                                          |
|----------------------------------------------------------------|-------------------------------------------------|
| Fournit `SampleRelation` complet (target_id, group_id)         | Choisit `split_unit="target"`                   |
| Materialize returns 1200 observations, not 400 samples         | Aggregator node consumes observation-level PredictionBlock |
| No aggregation: keeps the obs as is                | Implemente `robust_mean` + outlier exclusion    |
| Stores`observation_id`in each DataBlock                  | Persists the 2 levels of PredictionBlock       |

### 3.8 Expected result

- 2 PredictionBlocks per fold: obs-level (1200 OOF) and sample-level (400 OOF). - Bundle: 1 PLS (final refit) + adapters +`SampleRelation`replays predict. - Metric:`rmsecv_sample`(primary),`rmsecv_obs`(secondary).

---

## UC4 - Split unit by entity (patients / plots / lots)

### 4.1 Business context

Multi-patient clinical study: prediction of a biochemical score from
Raman spectra + clinical parameters. Each patient generates 5 to 15 measurements
over 2 years. The risk of temporal and inter-patient leakage requires a split at the`patient_id`level.

### 4.2 Involved data

| Source     | Type            | Granularity              | sample_id        | group_id      | N samples |
|------------|-----------------|--------------------------|------------------|---------------|-----------|
| `raman`    | `dense_signal`  | `per_sample_repeated`    | `meas_<n>`       | `patient_<p>` | 2400      |
| `clin`     | `table`         | `per_sample`             | `meas_<n>`       | `patient_<p>` | 2400      |
| target `y` | `table`         | `per_sample`             | `meas_<n>`       | -             | 2400      |

300 patients, 8 mesures moyennes/patient.

### 4.3 Desired DSL pipeline

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

### 4.4 Resulting DAG

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

Key nodes:

| Node id      | kind       | Notes                                            |
|--------------|------------|--------------------------------------------------|
| `rel:raman`  | sample_rel | returns `group_ids = patient_id`                |
| `split:sgkf` | split      | StratifiedGroupKFold, garantit no-patient-cross  |

### 4.5 Key ML invariants

- No patient appears simultaneously in train and valley of the same fold. - Stratification on quantiles of y (regression) or classes (classif). - OOF: predictions are OOF at the`meas_<n>`level, we can also aggregate
  patient-level via an optional`Aggregator`. - Reproducibility:`random_state=0`+`SeedContext`hierarchical. -`join="exact"`requires that raman and clin have exactly the same`sample_id`(twin measurements, no missing).

### 4.6 Friction points

1. If a patient only has 2 measurements, the fold val might have too few
   data. What minimum size per patient? Impose or warning? 2. Stratification + groups is non-trivial (`StratifiedGroupKFold`is greedy). Should we accept a distribution gap of y between folds? 3. Predict-time: if a new patient arrives with only 1 measurement, the bundle
   should he accept? Yes,`group_id`is only used in split. 4. How to expose the`patient_id -> [sample_ids]`mapping in the manifest
   without inflating the bundle size?

### 4.7 ML_DATA vs DAG-ML

| ML_DATA                                              | DAG-ML                                              |
|------------------------------------------------------|-----------------------------------------------------|
| Expose `SampleRelation.group_ids = patient_id`       | Selectionne `split_unit="group"`                    |
| AlignmentPolicy(exact) enforce sample-id parity      | Splitter `StratifiedGroupKFold` consomme groups     |
| No split, no knowledge of fold             | Persiste mapping patient -> folds dans manifest     |

### 4.8 Expected result

- PredictionBlocks by fold with`sample_ids=meas_*`and`group_ids=patient_*`. - Bundle: Ridge + adapters + partial`SampleRelation`(the patient_ids of the train). - Metric: rmsecv_meas + rmsecv_patient_mean.

---

## UC5 - Augmentation train-only avec OOF correct

### 5.1 Business context

Reduced NIRS dataset (180 samples). Application of augmentations (Gaussian noise,
shift wavelength, local mixup) to improve generalization, without contamination
inter-fold. Validation and testing must remain unaugmented; the OOFs
predictions must be made for the original fold val samples, not
for their copies generated at the train.

### 5.2 Involved data

| Source      | Type            | Granularity            | N originaux | apres aug train | sample_id  | origin_id   |
|-------------|-----------------|------------------------|-------------|------------------|------------|-------------|
| `nir`       | `dense_signal`  | `per_sample`           | 180         | +540 (3x)        | `aug_*`    | `S001`...   |
| target `y`  | `table`         | `per_sample`           | 180         | (heritage)       | `aug_*`    | -           |

`inherit_target=True`, `inherit_group=True`.

### 5.3 Desired DSL pipeline

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

### 5.4 Resulting DAG

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

### 5.5 Key ML invariants

-`forbid_validation_augmentation=True`: the increase turns on the view`partition="train"`of the fold, never`val`. -`origin_ids`is filled by`AugmentationAdapter`-> DAG-ML verifies that
  for any sample S of val_k, no line train_k has`origin_id=S`. Violation ->`LeakageError`. - OOF: 180 predictions (1 per original), not 720. Augmented copies do not
  never receive an OOF prediction (they do not have a validation Y). - Reproducibility:`SeedContext.child(node_id="aug:noise", fold_id=k)`derivative
  a deterministic seed; the bundle stores the`(seed, multiplier)`sequence. - Refit:`apply_to="train_only"`(default) means "increases all
  train partitions", which includes the CV train folds AND the final refit
  (180 -> 720). The bundle persists`AugmentationAdapter`but the predict does not
  does not play it again (the new samples are not increased). For a refit
  WITHOUT re-increase, use`apply_to="cv_only"`explicitly.

### 5.6 Friction points

1. Does Final Refit reapply the augmentation? Solved (Q18): default`apply_to="train_only"`= increase in CV AND at refit (the model ships
   match the OOF distribution). Explicit opt-out via`apply_to="cv_only"`. 2.`Mixup`crosses two train samples to create a new one. Which`origin_id`? Keeping both (`origin_ids = (id_a, id_b)`) requires a schema list
   for`origin_id`. Spec says single -> should we extend? 3. When a meta-model consumes OOF predictions (UC6), should they be
   produce at the original level or at the augmented level? Answer: origin only. 4. Augmentation is lossy by definition. How to mark the lineage so that
   predict doesn't apply it by accident?

### 5.7 ML_DATA vs DAG-ML

| ML_DATA                                                | DAG-ML                                              |
|--------------------------------------------------------|-----------------------------------------------------|
| `AugmentationAdapter.transform()` produces DataBlock + SampleRelation | Decide when to call it (only view train of a fold) |
| Stocke `origin_id` non-None dans SampleRelation         | Verifie `forbid_validation_augmentation` invariant  |
| Persiste `random_state` dans FittedAdapter             | Refuses to do OOF on augmented lines           |

### 5.8 Expected result

- 180 OOF predictions (originals). - Bundle: PLS + adapters + AugmentationAdapter (with seed) brand`phase=REFIT`. - Lineage: each AugmentedRow carries`lineage=("aug.noise", "aug.shift")`.

---

## UC6 - Stacking multi-niveau (3 preprocs/models + meta Ridge)

### 6.1 Business context

Estimate a chemical concentration from a NIRS spectrum, by combining
3 complementary preprocessing/model channels (SNV+PLS, MSC+RF, Detrend+SVR)
via a Ridge meta-model trains on OOF predictions.

### 6.2 Involved data

| Source     | Type             | Granularity   | N samples |
|------------|------------------|---------------|-----------|
| `nir`      | `dense_signal`   | `per_sample`  | 500       |
| target `y` | `table` (numeric)| `per_sample`  | 500       |

### 6.3 Desired DSL pipeline

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

### 6.4 Resulting DAG

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

### 6.5 Key ML invariants

- Same folds outer for the 3 branches (same`random_state`+ same`split:outer`). -`PredictionJoinNode`refuses by default any non-OOF`PredictionBlock`(`partition != "val"`). If a branch has leaked, raise. - The Ridge meta-model only consumes OOF -> size (500, 3). - Inner CV for Ridge may differ; don't change the base. - Refit: basic learners refit on full train. Meta Ridge refit on OOF
  produced by the inner CV pre-refit (not the base learners refit preds,
  otherwise train leak). -`allow_train_predictions_as_features=False`is explicit (default). See UC11 for opt-in.

### 6.6 Friction points

1. Should we stack raw OOF predictions or probabilities/quantiles? For regression: y_pred direct. For classification: y_proba. 2. Can Inner CV of the meta be`LeaveOneOut`when N=500? Expensive, but
   does not change the validity of the stacking. 3. If 2 branches are quasi-identical (correlation 0.99 between preds), the meta
   Ridge will surf these 2. Do we need an upstream decorrelator (drop, PCA)? 4. For the refit, do we need to refit the basic learners with exactly the same
   hyperparams as the best fold or with the agg of all folds?

### 6.7 ML_DATA vs DAG-ML

| ML_DATA                                                       | DAG-ML                                              |
|---------------------------------------------------------------|-----------------------------------------------------|
| Materialize nir + applique adapters (SNV, MSC, Detrend)       | Fork/join, OOF validation, refusal if non-OOF         |
| Provides FeatureTable of OOF preds with source_ids per col    | Refuse train preds par defaut (opt-in flag explicite) |
| No knowledge of the meta-model                            | Inner CV du meta, refit infrastructure              |

### 6.8 Expected result

- 1500 OOF preds base (500 x 3) + 500 OOF preds meta.
- Bundle: 3 base learners refit + Ridge meta refit + 3 caches OOF.
- Metrics: rmsecv par branche + rmsecv_meta + delta vs best single.

---

## UC7 - Generateurs + tuning bayesien

### 7.1 Business context

Large-scale hyperparameter search: crossing several preprocessings
candidates, several model families, several grid combinations, more
a Bayesian tuning on the continuous params (alpha, gamma). Lazy enumeration.

### 7.2 Involved data

| Source     | Type             | Granularity   | N samples |
|------------|------------------|---------------|-----------|
| `nir`      | `dense_signal`   | `per_sample`  | 800       |
| target `y` | `table` (numeric)| `per_sample`  | 800       |

### 7.3 Desired DSL pipeline

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

### 7.4 Resulting DAG

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

### 7.5 Key ML invariants

- Each trial respects the strict OOF: 5-fold CV. - Stateful adapters (SG) are fit fold-train only. - Reproducibility:`SeedContext.child(trial_id=k)`derives`random_state`. - Tuner separation:`TunerAdapter`does not see the val data, only the`rmsecv`score. - Bayesian pruning authorizes but must log`trial.state="pruned"`. - Refit: best variant only, not all trials.

### 7.6 Friction points

1. How to set`SearchSpace`for discrete mix (`_cartesian_`,`_grid_`)
   and continuous (`_sample_`)? Bayesian tuner expects continuous bounds. 2. Should we do`early_stopping`(Hyperband, BOHB) to avoid ending up
   trials clearly losing after 1 fold? 3. If a trial fails (operator incompatible), how to mark without corrupting
   tuner history?`TrialResult.state="error"`+ does not record score. 4. Transmission of`params`from the tuner to the`model`node and to the`adapt`node
   simultanement: format unifie ou per-node ?

### 7.7 ML_DATA vs DAG-ML

| ML_DATA                                                       | DAG-ML                                              |
|---------------------------------------------------------------|-----------------------------------------------------|
| Materialize nir + applies the trial adapter (SNV/MSC/...)   | Implemente `SearchSpace`, `TunerAdapter`            |
| No knowledge of trials                                | Enumeration lazy, scheduling, pruning              |
| Persiste `random_state` derive                                | Persist all`TrialResult`in manifest    |

### 7.8 Expected result

- 40`TrialResult`(params, score, time, pruned flag). - 1`SelectedGraph`corresponding to the best trial. - Refit bundle + complete trial log (useful for Bayes reporting).

---

## UC8 - Branches by metadata, merge concat

### 8.1 Business context

Multi-site NIRS dataset (3 production sites: A/B/C). Hypothesis: each site
has different instrument biases and a site-specific model is more
efficient than a global model. After training by site, concatenation of
predictions to produce a unified PredictionBlock (`sample_id`aligned).

### 8.2 Involved data

| Source     | Type             | Granularity   | Metadata `site` | N par site | Total |
|------------|------------------|---------------|-----------------|------------|-------|
| `nir`      | `dense_signal`   | `per_sample`  | "A","B","C"     | 300/250/200| 750   |
| `meta`     | `table`          | `per_sample`  | site, year, op  | 750        | 750   |
| target `y` | `table` (num)    | `per_sample`  | -               | -          | 750   |

### 8.3 Desired DSL pipeline

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

### 8.4 Resulting DAG

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

Key nodes:

| Node id           | kind        | Notes                                            |
|-------------------|-------------|--------------------------------------------------|
| `fork:by_meta`    | fork        | separation, disjoint sample groups                |
| `branch:A/B/C`    | subgraph    | independent CV + model per branch                |
| `merge:concat`    | prediction_join | reassemble sample order, no overlap            |

### 8.5 Key ML invariants

- Separation: the 3 branches are disjoint (empty intersection on sample_id). - Each branch makes its own CV (disjoint intra-branch folders). - Merge`concat`requires exact union = set of sample_ids -> error if
  a sample is not in any branch (metadata inconsistency). - OOF: each sample has 1 OOF prediction (its branch), not 3. - Refit: 3 models refit independently, bundle contains all 3. - Predict-time: new sample classified by`site`(lookup metadata), dispatch
  to the correct bundle model.

### 8.6 Friction points

1. A site with too few samples (ex: 50) does not support 5 folds. Fallback
   automatic towards 3 folds or error? 2. How to manage a new`D`site in predict? Strategies: error, fallback
   to the global model (if it exists), nearest neighbor site. 3. Should we always produce a “global” model as a reference? DSL option. 4. If we want to compare 3 separate models vs 1 global model, we create two
   variants. How can they coexist in the same run?

### 8.7 ML_DATA vs DAG-ML

| ML_DATA                                                | DAG-ML                                              |
|--------------------------------------------------------|-----------------------------------------------------|
| Materialize `meta` + expose colonne `site`             | ForkNode interprete `by_metadata="site"`            |
| Provide DataView with subset of sample_ids      | Gere CV par branche (folds independants)            |
| No separation by metadata side data               | Merge concat valide non-overlap                     |

### 8.8 Expected result

- 3 refit models + 1 concatenate PredictionBlock (750 OOF). - Bundle: dictate models per site +`site -> sample_ids`mapping. - Metric: global rmsecv + per site.

---

## UC9 - Refit + bundle + predict new heterogene

### 9.1 Business context

After UC1 training, deploy the bundle on 50 new samples received
6 months later. Check schema compatibility (fingerprint), manage
in the case where`photo_side`is missing for 5 samples, refuse if the`weather`arrives with a different diagram (change of sensor).

### 9.2 Involved data

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

### 9.4 Resulting DAG (predict)

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

### 9.5 Key ML invariants

-`schema_fingerprint`recalculates with the same sources + merge + adapt
  specs. Mismatch -> default refuses. - No fit to predict-time. All stateful adapters come from the bundle. -`photo_side`missing: the presence mask of the new data feeds the
  column`indicator__photo_side_present`(joiner train logic). -`weather`schema diverges -> clear error with payload structure {expected,
  got, diff}. - Reproducibility: same order of columns as train (lock by the FeatureJoiner). - Lineage: PredictionBlock carries`bundle_id`,`bundle_version`.

### 9.6 Friction points

1. Which fields to compare in the fingerprint? Sources (id+axes), fusion policy,
   adapt ids + versions. Not the sample_ids (which always change). 2. What to do if a train source is`optional=True`in`ModelInputSpec`and absent from the forecast? The spec says OK; but the RF learned with it, so
   possible degradation -> warning. 3. How to migrate between majors (image embedding v1 -> v2)? Re-fit or
   refusal?`PluginVersionError`+ explicit migration path. 4. The DataPlan stores`adapter_id`+`params`. If the user has redefined
   an adapter with the same id but different semantics, the bundle is
   silently compromised. Implementation hash required?

### 9.7 ML_DATA vs DAG-ML

| ML_DATA                                                       | DAG-ML                                              |
|---------------------------------------------------------------|-----------------------------------------------------|
| Calcule `schema_fingerprint(schema, fusion, adapters)`        | Compare fingerprint et decide accept/refuse         |
| `execute_transform(plan, fitted, view)` rejoue le DataPlan    | Charge le bundle, dispatch predict                  |
| Reject if missing plugin or out-of-range version            | Reject if schema_check fails                       |

### 9.8 Expected result

-`PredictionBlock(partition="predict", sample_ids=new50, y_pred=...)`. - No new artifacts (predict is pure read-only on the bundle). - If fingerprint mismatch:`SchemaFingerprintMismatch`with JSON payload
  detailing the difference. No silent prediction.

---

## UC10 - SubDAG reified as node (SubgraphNodeSpec)

### 10.1 Business context

A NIRS team developed a “NIR canonical preproc + PLS” sub-pipeline
generic reused in 5 projects. Package it like`SubgraphNodeSpec`version, then insert it as a node in a new, larger DAG (by
example, feed a meta-stacking with this sub-DAG as a “base learner”
among others). Choose`inline`vs`opaque`depending on the need for thin cache.

### 10.2 Involved data

| Source     | Type             | Granularity   | N samples |
|------------|------------------|---------------|-----------|
| `nir`      | `dense_signal`   | `per_sample`  | 1000      |
| `tab`      | `table`          | `per_sample`  | 1000      |
| target `y` | `table` (num)    | `per_sample`  | 1000      |

### 10.3 Desired DSL pipeline

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

### 10.4 Resulting DAG

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

### 10.5 Key ML invariants

-`GraphInterface`must be declared for a sub-DAG to be composable. - Compatibility check:`input_mapping["X"].representation`must be compatible
  with`PortSpec.representation`of sub-DAG. - OOF: whether inline or opaque, the sub-DAG respects its own invariant
  OOF (fold-train for stateful adapters). - Reproducibility:`SeedContext.child(node_id="branch:nir_canon", ...)`derive the seed from the subDAG (root_seed + subgraph_id + fold_id). - Refit: the sub-DAG is refit as a unit. - Versioning:`SerializableRef(version="1.2.0")`-> upgrade`1.3.0`must be
  explicit (the bundle stores the exact version).

### 10.6 Friction points

1. If an opaque sub-DAG has its own SearchSpace, should it appear in the
   parent tuner or not? Answer v1: no, opacity = closed scope. 2. For`inline`, should the planner refuse if two sub-DAGs have nodes
   same id? Solution: prefix with`subgraph_id`. 3. For`inline_policy="auto"`, what precise criterion triggers`inline`? Reuse
   detected by hash of the upstream IR? Utility score? 4. Can a sub-DAG be a DAG-ML of a different version from the parent? If
   yes, version-stable`ml_data.contract`contracts (v1 -> v1 ok), major
   different -> refusal.

### 10.7 ML_DATA vs DAG-ML

| ML_DATA                                            | DAG-ML                                              |
|----------------------------------------------------|-----------------------------------------------------|
| No concept of sub-DAG                          | Definit `SubgraphNodeSpec`, `GraphInterface`        |
| Provides the same data contracts to the sub-DAG         | Planner decide inline vs opaque                     |
| -                                                  | Cache key inclut `subgraph_id + version`            |

### 10.8 Expected result

- Parent bundle contains: 1 entry for the opaque sub-DAG (refit-as-unit),
  individual nodes for the inlined version, and the Ridge meta. - Lineage: each PredictionBlock carries complete`producer_node_id`including
  the path subgraph (ex:`branch:nir_canon/model:pls`).

---

## UC11 - Piege OOF: train predictions refusees par defaut

### 11.1 Business context

A user attempts to stack using train predictions (non-OOF)
base learners as features for the meta-model. It's a leak
classic data: the meta-model learns to correct overfitting
basic learners.

### 11.2 Involved data

| Source     | Type             | Granularity   | N samples |
|------------|------------------|---------------|-----------|
| `nir`      | `dense_signal`   | `per_sample`  | 500       |

### 11.3 Desired DSL pipeline (trap)

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

### 11.4 Resulting DAG + expected behavior

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

### 11.5 Key ML invariants

- DEFAULT: refuse train predictions; raise`OOFLeakageError`. - ESCAPE: single boolean`allow_train_predictions_as_features=True`on the`PredictionJoinNode`. The verbose name does the double-confirmation work. - PredictionBlock carries a`oof_safe: bool`field derived from partition + fold_id. -`PredictionJoinNode`validate:`all(block.oof_safe for block in inputs)`except
  if the opt-in flag is set. - Lineage: the PredictionBlock of the Ridge meta door`flags=["train_predictions_used"]`, and the joint carries`leakage_acknowledged=True`. Appears in the manifest, in the
  metrics, and in any report. - Selection:`RankingPolicy.exclude_leaky_variants=True`(default) excludes these
  variants of automatic top-k selection.

### 11.6 Friction points (3-4 questions)

1. Is a single flag with verbose name sufficient or do we need a second confirmation
   for the webapp UI? Decision v1: single flag. UI can impose a second
   confirmation over (modal), but the Python API remains single-flag. 2. Should we also reject predictions test (partition="test") as features? Yes, by default. Except in the case of transfer learning (to be discussed). 3. Should an OOF PredictionBlock but with a number of samples != train_full
   be accepted (case augmentation: origins only)? Yes, that's correct. 4. How can I present this error message in an understandable way? Payload JSON
   structure + i18n translation via webapp.

### 11.7 ML_DATA vs DAG-ML

| ML_DATA                                            | DAG-ML                                              |
|----------------------------------------------------|-----------------------------------------------------|
| No concept of OOF / partition                   | All OOF logic, OOFLeakageError              |
| Materialize data without knowing the use     | PredictionJoinNode valide partitions                |
| -                                                  | Definit champs `oof_safe`, `flags`, `leakage_acknowledged` |

### 11.8 Expected result

- In safe mode (default): refusal with`OOFLeakageError`+ remediation message. - In opt-in mode: pipeline runs but predictions and bundle carry
  a permanent flag`train_predictions_used`+`leakage_acknowledged=True`on the joint.

Expected error message (JSON extract):

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

### 12.1 Business context

A stacking where certain branches contribute their transformed features
directly (useful preprocessings as is, like PCA(20)) and others
branches contribute their OOF predictions (non-linear models like RF
or SVR). The meta-model consumes the union of the two.

### 12.2 Involved data

| Source     | Type             | Granularity   | N samples |
|------------|------------------|---------------|-----------|
| `nir`      | `dense_signal`   | `per_sample`  | 600       |
| target `y` | `table` (num)    | `per_sample`  | 600       |

### 12.3 Desired DSL pipeline

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

### 12.4 Resulting DAG

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

### 12.5 Key ML invariants

- Features of branch 0:`tabular.pca`fit on fold train -> transform train and
  val. For the join, only the val lines are used (otherwise mismatch with
  preds OOF of b1 and b2 which only contain val). - Predictions of branches 1, 2: strict OOF, refusal if train. - Alignment: the 3 blocks share`sample_ids`of the fold val. - Reproducibility: same fold (same seed) for the 3 branches. - Refit: PCA b0 refit on full train, base learners b1/b2 refit on full train,
  meta Ridge refit on OOF concat (cache of OOFs used for meta fit).

### 12.6 Friction points

1. How to unify the contract? Branch 0 product`FeatureBlock`, branch 1/2
   product`PredictionBlock`. The`MixedJoinNode`accepts a union with
   discriminator (`block_kind: "feature" | "prediction"`). 2. The PCA b0 must be fit fold-train and applied to the val samples. But in
   refit, we did on full train -> as predicted, no fold. So two artifacts
   PCA (CV vs refit). Cache of the OFA preserved only for the CV. 3. If branch 0 has a very large output dimension (PCA k=200), the
   features columns dominate the 2 predictions columns. Should we scale? 4. Cross-source validation: a feature of b0 could leak (PCA not stateful? we said stateful). Ensure that`fit_scope=fold_train`.

### 12.7 ML_DATA vs DAG-ML

| ML_DATA                                            | DAG-ML                                              |
|----------------------------------------------------|-----------------------------------------------------|
| Provides`FeatureTable`(b0) with stateful PCA      | MixedJoinNode unifie feat + pred                    |
| Fit `tabular.pca` avec fold_train scope            | Valide OOF pour pred blocks                         |
| No notion of partition                         | Select only val rows for join        |

### 12.8 Expected result

- 600 OOF preds meta + bundle (b0 PCA refit + b1 PLS refit + b2 RF refit + Ridge). - 22 meta features columns (20 PCA + 2 preds). - Lineage: each column carries its`source_id`=`branch:0/...`or`branch:1/model:pls`.

---

## Decisions design transverses

Summary of recurring friction issues in CUs. These questions
remain open and must be decided before implementation v1.

### D1 - Auto-resolution lossy vs explicit user choice

UC1 (image embedding), UC2 (resample), UC5 (increase), UC10 (sub-DAG): - When`find_path`finds a chain of lossy adapters (image embedding,
  PCA genotype, resampling with interpolation), should DAG-ML: (a) auto-resolve if`policy.allow_lossy_adapters=True`(default current)? (b) always escalate via`requires_user_choice`? (c) only escalate if multiple competing lossy chains?

Proposition: (c) par defaut. Plus un mode strict (a) opt-in via DSL.

### D2 - Ranking level and selection

UC3 (obs vs sample), UC4 (meas vs patient), UC6 (per-branch vs meta), UC8
(global vs per-site): - When a pipeline produces predictions at several levels (observation,
  sample, group, branch, meta), what level do we use to rank? Answer v1: most aggregated level (sample > group > observation), except
  override via`ranking_level`in`SelectPolicy`.

### D3 - Refit semantics: same hyperparams or aggregate?

UC6 (stacking refit), UC7 (tuner best refit), UC8 (per-branch refit): - At refit, should we: (a) reuse exactly the hyperparams of the best fold? (b) refit with an aggregate (for example median of the n_components of the 5 folds)? (c) re-tuner on full train?

Proposition: (a) par defaut, (c) opt-in.

### D4 - Schema fingerprint scope

UC9 (predict + diverge weather), UC10 (subgraph versioning): - Should the fingerprint include: - axis coordinates (exact wavelengths)? - exact feature names post-encoder? - the seeds used? - plugin versions?

Proposition: oui pour axes, feature names, plugin versions. Non pour seeds
(which change per run,`LineageRecord`wears them).

### D5 - Parallelism granularity and seeding

UC1 (5 folds + 6 sources), UC7 (40 trials), UC10 (inline subgraphs): - At what level to parallelize? - variant (joblib loky, default)
  - fold (useful for large models)
  - branch (for separation)
  - sub-DAG inline (rare)
- And how to guarantee that nested parallelization does not explode the
  threads (BLAS x joblib x torch)?

Proposition: parallelism budget unique au niveau `RunContext`, distribue
top-down by the scheduler. Only one parallel level active by default.

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

UC9 only traverses`COMPILE -> PLAN -> PREDICT`because it consumes an existing bundle. UC11 stops at`FIT_CV`when the OOF invariant is violated (run refusal).

---

## Appendix D - Artifacts produced by UC

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

## Appendix F - Order of phases by CPU (summary)

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
