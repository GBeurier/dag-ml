# DAG-ML depuis le runtime nirs4all

Note basee sur le code, pas sur les audits existants. Les chemins les plus structurants lus pour cette analyse sont:

- `nirs4all/pipeline/runner.py`
- `nirs4all/pipeline/execution/orchestrator.py`
- `nirs4all/pipeline/execution/executor.py`
- `nirs4all/pipeline/config/pipeline_config.py`
- `nirs4all/pipeline/config/_generator/`
- `nirs4all/pipeline/config/context.py`
- `nirs4all/pipeline/steps/step_runner.py`
- `nirs4all/pipeline/steps/parser.py`
- `nirs4all/pipeline/steps/router.py`
- `nirs4all/controllers/data/branch.py`
- `nirs4all/controllers/data/merge.py`
- `nirs4all/controllers/models/base_model.py`
- `nirs4all/controllers/models/sklearn_model.py`
- `nirs4all/controllers/models/stacking/reconstructor.py`
- `nirs4all/pipeline/execution/refit/`

## 1. Architecture concrete des pipelines nirs4all

### 1.1 Vue d'ensemble

Le moteur actuel execute deja un DAG, mais ce DAG n'est pas un objet explicite. Il est reconstruit dynamiquement a partir d'une liste de steps, d'un contexte mutable, de snapshots de branches, de folds de cross-validation et d'un store de predictions.

Le cycle reel est:

1. `PipelineRunner.run()` recoit `pipeline`, `dataset`, `refit`, store/cache et delegue a `PipelineOrchestrator`.
2. `PipelineOrchestrator.execute()` normalise la config, normalise les datasets, ouvre le run dans le store, cree `ArtifactRegistry`, `PipelineExecutor`, `Predictions` et execute chaque variante.
3. `PipelineConfigs` compile la syntaxe utilisateur: preprocessing des steps, expansion des generateurs (`_or_`, `_grid_`, `_cartesian_`, `_zip_`, etc.), generation des noms de variantes et conservation des `generator_choices`.
4. `PipelineExecutor.execute()` initialise `ExecutionContext`, ouvre un pipeline dans le store, cree une trace, boucle sur les steps et flush les predictions vers le store.
5. `StepRunner` parse chaque step, route vers un controller, execute le controller, puis normalise le retour en `StepResult`.
6. Les controllers font le vrai travail: split, transform, y-transform, tag/exclude, branch, merge, model, charts.
7. En fin de CV, `PipelineOrchestrator` lance le refit si demande: extraction de la meilleure config, reexecution sur train complet, ecriture de predictions `fold_id="final"`.

Ce n'est donc pas un simple `sklearn.Pipeline`: le pipeline nirs4all melange compilation de variantes, data routing, branches, CV, OOF, persistence, artifact lineage, selection et refit.

### 1.2 Generation: compile-time, pas runtime

`PipelineConfigs` est le compilateur actuel. Il accepte liste, dict, chemin ou string, puis:

- convertit une liste en `{"pipeline": [...]}`;
- fusionne les couples `xxx` / `xxx_params`;
- serialise les composants Python;
- detecte les generateurs;
- produit une liste materialisee de variantes (`self.steps`);
- attache a chaque variante ses `generator_choices`.

Point important: les generateurs dans une branche de duplication ne sont pas entierement traites par `PipelineConfigs`. Ils peuvent etre etendus plus tard par `BranchController`. Cela montre que la generation est aujourd'hui partagee entre le compilateur de config et certains controllers. Pour DAG-ML, il faut separer clairement:

- `SearchSpace`: decrit les choix;
- `Compiler`: transforme le DSL en IR;
- `Planner`: choisit une strategie d'enumeration lazy/materialisee;
- `Executor`: execute un plan deja resolu.

### 1.3 Execution: une ligne principale avec des graphes caches

`PipelineExecutor._execute_steps()` boucle lineairement sur `steps`, mais plusieurs steps creent des sous-graphes:

- `branch`: fan-out. Il cree plusieurs `branch_contexts`, chacun avec son `features_snapshot`, son `chain_snapshot`, son `branch_id`, son `branch_path`.
- `merge`: join. Il sort du mode branche et reconstitue des features ou predictions.
- `split`: cree les folds comme etat du dataset, pas comme noeuds DAG visibles.
- `model`: fork/join interne sur folds, puis predictions `fold`, `avg`, `w_avg`.
- `feature_augmentation`: execute un sous-pipeline sur plusieurs processings.
- sous-pipelines: `StepRunner` peut executer une liste de sous-steps comme une mini-pipeline.

Le DAG implicite ressemble a ceci:

```text
DSL pipeline
  -> PipelineConfigs/SearchSpace
  -> variantes materialisees
  -> execution CV par variante
       -> steps lineaires
       -> fan-out branch contexts
       -> fold subgraphs dans model
       -> prediction store
       -> merge OOF/features
       -> trace + chain + artifacts
  -> selection
  -> refit final
  -> predict/explain replay via artifacts
```

### 1.4 Controllers et operators

Le pattern actuel est sain:

- l'operator est le payload metier: objet sklearn, config de merge, splitter, filtre, transformer, modele;
- le controller est l'adapter d'execution: il sait comment fitter, transformer, predire, selectionner les samples, gerer train/predict/explain, sauver les artifacts;
- `StepParser` transforme la syntaxe utilisateur en `ParsedStep`;
- `ControllerRouter` parcourt `CONTROLLER_REGISTRY`, applique `matches()`, trie par priorite et instancie le controller.

Ce pattern doit etre le coeur de DAG-ML. Le nom pourrait changer (`OperatorAdapter`, `NodeHandler`, `ExecutorPlugin`), mais la separation operator/controller est la bonne abstraction.

Interface cible:

```python
class OperatorAdapter(Protocol):
    kind: NodeKind
    priority: int

    def matches(self, spec: NodeSpec, operator: object) -> bool: ...
    def declare_ports(self, spec: NodeSpec) -> PortSchema: ...
    def plan(self, spec: NodeSpec, ctx: PlanningContext) -> NodePlan: ...
    def execute(self, task: NodeTask, ctx: RunContext) -> NodeResult: ...
    def supports_phase(self, phase: Phase) -> bool: ...
    def cache_key(self, task: NodeTask, view: DataView) -> str | None: ...
```

### 1.5 OOF et stacking

La semantique OOF est un des vrais actifs de nirs4all.

Le flux actuel:

1. `CrossValidatorController` convertit un splitter sklearn-like en folds stockes en sample IDs absolus.
2. `BaseModelController` entraine un modele par fold.
3. Chaque fold ecrit des predictions `train`, `val`, `test`, avec `fold_id`, `sample_indices`, scores et metadata.
4. Pour plusieurs folds, le controller produit aussi `avg` et `w_avg`.
5. `TrainingSetReconstructor` reconstruit les meta-features d'entrainement a partir des predictions `partition="val"`, alignees par `sample_indices`. C'est le vrai OOF.
6. `MergeController` utilise ce reconstructeur pour transformer des predictions de branches en features de meta-modele. Par defaut, `unsafe=False`; `unsafe=True` prend des predictions train directes et le code marque explicitement cela comme fuite de donnees.

Contrat a extraire dans DAG-ML:

- Toute prediction utilisee comme feature d'un modele downstream en phase train doit etre OOF.
- L'alignement doit se faire par `sample_id`, jamais par position implicite seule.
- Le `PredictionBlock` doit contenir au minimum: `sample_ids`, `partition`, `fold_id`, `producer_node`, `target_space`, `y_pred`, `y_proba`, scores, branch lineage.
- Un `PredictionJoin` doit refuser par defaut un train block non OOF.

### 1.6 Merge et branches

`BranchController` gere deux familles:

- duplication: chaque branche voit les memes samples avec un traitement different;
- separation: les samples ou sources sont partitionnes par tag, metadata, filtre ou source.

`MergeController` gere ensuite:

- feature merge horizontal pour branches de duplication;
- merge vertical/reconstruction par sample ID pour branches disjointes;
- source merge (`concat`, `stack`, `dict`);
- prediction merge avec selection de modeles, agregation (`separate`, `mean`, `weighted_mean`, `proba_mean`) et OOF;
- mixed merge: features de certaines branches + predictions d'autres branches.

Dans DAG-ML, `branch` et `merge` ne doivent plus etre des effets de bord dans `context.custom`. Ils doivent devenir des noeuds formels:

- `ForkNode`: produit N `DataView` nommees;
- `MapNode`: applique un sous-graphe sur chaque branche;
- `FeatureJoinNode`: combine des `FeatureBlock`;
- `PredictionJoinNode`: combine des `PredictionBlock` avec validation OOF;
- `SourceJoinNode`: preserve ou concatene les sources.

### 1.7 Multi-source et layouts dans le code actuel

Le multi-source est aujourd'hui porte par `SpectroDataset`, mais c'est en realite une capacite DAG-ML generique.

Dans `nirs4all/data/features.py`, `Features` gere une liste de `FeatureSource`. Les sources sont alignees sur le meme axe samples, mais elles peuvent avoir des dimensions de features et des chaines de processings differentes. Chaque `FeatureSource` stocke ses donnees comme un tenseur 3D:

```text
(samples, processings, features)
```

Puis `LayoutTransformer` expose plusieurs vues:

- `2d`: `(samples, processings * features)`;
- `2d_interleaved`: `(samples, features * processings)`;
- `3d`: `(samples, processings, features)`;
- `3d_transpose`: `(samples, features, processings)`.

Quand `concat_source=True`, `Features.x()` concatene plusieurs sources sur le dernier axe. Pour des spectres numeriques, c'est acceptable: meme axe samples, tenseurs denses, processings compatibles ou flatten possible. Mais cette regle est trop pauvre pour DAG-ML general:

- images: concatener des pixels de cameras differentes n'a pas le meme sens qu'un concat de longueurs d'onde;
- graphes/arbres: il n'y a pas toujours d'axe features dense;
- sources heterogenes: une source peut etre un spectre, une image, un tableau clinique, un graphe molecular;
- multi-input models: certains modeles attendent un dict/list de tensors, pas un tableau concatene;
- sources manquantes: il faut choisir entre `inner`, `left`, `outer`, masque de presence, imputation ou erreur.

Les controllers actuels montrent deja le besoin:

- `TransformerMixinController` demande `layout="3d", concat_source=False`, boucle par source et par processing, puis persiste les artifacts avec `source_index`.
- Les controllers modeles declarent un layout prefere: sklearn -> `2d`, PyTorch -> `3d`, TensorFlow/JAX -> `3d_transpose`, avec `force_layout` pour forcer.
- `BaseModelController.get_xy()` demande `dataset.x(..., layout=layout)` avec le comportement par defaut `concat_source=True`, puis assert que le resultat est un `np.ndarray`. Donc le moteur suppose implicitement qu'un modele recoit un tenseur unique concatene, pas une entree multi-source generale.
- `MergeController` contient deja des modes `sources`, `dict`, `features`, `merge_sources`, `by_source`, mais ces semantics sont encodees dans le controller et dans `SpectroDataset`, pas dans un contrat DAG formel.
- `SampleLinker` dans `nirs4all/data/selection/sample_linker.py` gere deja l'alignement multi-source par cle (`inner`, `left`, `outer`). Cette logique appartient plutot au contrat ML_DATA qu'au moteur d'execution ML.

Donc: le multi-source ne doit pas etre "un detail dataset". Il doit devenir un type de port et d'edge DAG-ML.

### 1.8 Refit

Le refit actuel est une deuxieme phase apres selection:

- `extract_winning_config()` ou `extract_top_configs()` choisit les variantes;
- `execute_simple_refit()` deep-copy le dataset, remplace le splitter par un fold unique "train complet", injecte `best_params` et `refit_params`, puis reexecute le pipeline;
- les predictions refit sont relabelisees `fold_id="final"` et `refit_context="standalone"`;
- pour stacking/mixed merge/separation, le dispatcher part vers des strategies dediees;
- pour certaines branches, `BranchController` alimente `best_refit_chains`, ce qui permet de refitter chaque modele sur sa meilleure chaine sans requeter le store.

Dans DAG-ML, le refit doit etre un graphe de phase, pas un patch de steps:

```text
FitCVPlan -> SelectionPlan -> RefitPlan -> ExportPlan
```

Le refit doit recevoir une `SelectedGraph` contenant:

- graph variant ID;
- choix generateurs;
- best params;
- chemin de preprocessings;
- modele cible;
- strategie de refit;
- lien vers les predictions CV qui justifient la selection.

## 2. Proposition DAG-ML

### 2.1 Mission

DAG-ML devrait etre un moteur ML local/in-process, pas un orchestrateur de machines. Il formalise le coeur universel:

- compilation d'un DSL pipeline en DAG;
- generation de variantes;
- execution multi-phase fit/predict/explain/refit;
- data connectors;
- compatibilite explicite model/data et source/source;
- adapters operators/controllers;
- CV, OOF, stacking et refit;
- artifacts, cache, lineage, traces.

Il ne doit pas contenir de logique NIRS. `SpectroDataset`, wavelengths, sources spectrales et indexer actuel doivent etre branches via adapters depuis `nirs4all-data` / `nirs4all`.

### 2.2 Objets de base

```python
@dataclass(frozen=True)
class GraphSpec:
    nodes: dict[str, NodeSpec]
    edges: list[EdgeSpec]
    search_space: SearchSpace | None

@dataclass(frozen=True)
class NodeSpec:
    id: str
    kind: NodeKind
    operator: object
    params: dict[str, object]
    ports: PortSchema
    metadata: dict[str, object]

@dataclass(frozen=True)
class EdgeSpec:
    source: PortRef
    target: PortRef
    contract: EdgeContract

@dataclass
class RunContext:
    phase: Phase
    dataset: MLDataset
    artifact_store: ArtifactStore
    prediction_store: PredictionStore
    cache: CacheStore
    lineage: LineageRecorder
    resources: ResourceHints
```

Types importants:

- `MLDataset`: abstraction generique des samples, sources, targets, metadata et vues.
- `DataView`: selection immutable de samples/features/targets.
- `SourceDescriptor`: description stable d'une source, de sa modalite, de ses axes et de ses units.
- `DataBlock`: payload concret d'une source ou d'un merge, avec `sample_ids`, representation, axes et lineage.
- `FeatureBlock`: cas particulier dense de `DataBlock` pour features numeriques.
- `TargetBlock`: y + target transform lineage.
- `FoldSet`: folds en sample IDs stables.
- `PredictionBlock`: predictions OOF/test/final avec sample IDs.
- `ModelInputSpec`: contrat d'entree declare par un model adapter.
- `DataPlan`: conversion explicite entre blocs data et entree modele.
- `ArtifactRef`: pointeur versionne vers un modele/transformer fitted.
- `LineageRecord`: lien entre node, inputs, params, artifacts, outputs.

### 2.3 Phases

Les phases doivent etre explicites:

- `COMPILE`: DSL -> `GraphSpec`.
- `PLAN`: `GraphSpec` + choix generateurs -> `ExecutionPlan`.
- `FIT_CV`: fit des transformers/modeles par folds, emission de predictions OOF.
- `SELECT`: ranking par metrique, top-k, per-model selection, multi-criteria.
- `REFIT`: entrainement final sur train complet, emission `fold_id="final"`.
- `PREDICT`: replay minimal via artifacts.
- `EXPLAIN`: replay + hooks explainability.

Le moteur peut exposer:

```python
compiled = dagml.compile(pipeline_spec, registry=registry)
for variant in dagml.enumerate(compiled.search_space, lazy=True):
    plan = dagml.plan(compiled.graph, variant)
    cv_result = dagml.fit_cv(plan, dataset)

selected = dagml.select(cv_result, ranking="rmsecv", top_k=1)
final = dagml.refit(selected, dataset)
pred = dagml.predict(final.bundle, new_dataset)
```

### 2.4 DSL compatible nirs4all

La premiere version ne doit pas inventer une syntaxe neuve. Elle doit accepter la syntaxe nirs4all actuelle et la compiler en IR:

- `{"preprocessing": StandardScaler()} -> TransformNode`
- `{"split": KFold(...)} -> SplitNode`
- `{"model": PLSRegression(...)} -> ModelNode`
- `{"branch": [...]}` -> Fork/Map subgraph
- `{"merge": "predictions"}` -> PredictionJoinNode
- generateurs `_or_`, `_grid_`, `_cartesian_` -> `SearchSpace`

Ensuite seulement, on peut offrir une API builder plus typee.

### 2.5 Data connectors

Le point de decouplage principal est la donnee, mais il faut aller plus loin qu'un simple `dataset.x(layout="2d")`.

DAG-ML doit demander des donnees par contrat. ML_DATA doit repondre avec des schemas, des blocs types et un plan de conversion explicite. `nirs4all` fournirait alors seulement un `SpectroDatasetConnector`; une autre lib pourrait fournir tabular, images, timeseries, graphes ou gnom sans toucher au moteur DAG-ML.

Pour les spectres, `SpectroDatasetConnector` peut declarer:

- modalite `dense_signal`;
- representation native `(samples, processings, features)`;
- conversions lossless vers `flat_features`, `channels_first`, `channels_last`;
- concat source autorise si samples alignes et tenseurs denses;
- padding/truncation autorise seulement quand une policy explicite le demande.

Pour images/graphes/arbres/multimodal, il ne faut pas appliquer les regles spectrales par defaut. Le connecteur doit dire si une operation est possible, lossless, lossy ou interdite.

### 2.6 Contrat ML_DATA generique et extensible

Si la couche data devient une librairie separee et generique, DAG-ML ne doit plus importer `SpectroDataset`, `pandas`, `torch_geometric`, `PIL`, etc. DAG-ML doit seulement parler a un contrat stable. Les nouveaux types arrivent par plugins ML_DATA.

Architecture cible:

```text
DAG-ML
  - compile le graph
  - choisit les phases CV/refit/predict
  - demande un input compatible avec un modele
  - garantit no-leakage/OOF/folds

ML_DATA
  - decrit les sources
  - materialise les vues
  - aligne les samples
  - connait les types custom
  - expose les adapters de representation
  - produit des DataBlock / FeatureTable / Batch
```

#### 2.6.1 Schema minimal

```python
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, Sequence

SampleId = str | int
SourceId = str
RepresentationId = str
TypeId = str

AxisKind = Literal[
    "sample",
    "feature",
    "processing",
    "time",
    "height",
    "width",
    "channel",
    "node",
    "edge",
    "variant",
    "token",
    "target",
]

@dataclass(frozen=True)
class AxisSpec:
    name: str
    kind: AxisKind
    unit: str | None = None
    size: int | None = None
    variable: bool = False
    coordinates: Any | None = None

@dataclass(frozen=True)
class RepresentationSpec:
    id: RepresentationId
    type_id: TypeId                   # dense_signal, image_rgb, graph, table...
    rank: int | None                  # None for ragged/object containers
    axes: tuple[AxisSpec, ...]
    container: str                    # ndarray, dataframe, sparse, graph_batch, list, dict
    dtype: str | None = None
    sparse: bool = False
    ragged: bool = False

@dataclass(frozen=True)
class SourceDescriptor:
    id: SourceId
    name: str
    type_id: TypeId
    modality: str                     # spectroscopy, image, genotype, weather, metadata...
    native_representation: RepresentationSpec
    sample_key: str
    granularity: str                  # per_sample, per_sample_sequence, per_sample_set...
    schema: dict[str, Any] = field(default_factory=dict)
    tags: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class DatasetSchema:
    dataset_id: str
    sample_ids: tuple[SampleId, ...]
    sources: tuple[SourceDescriptor, ...]
    targets: dict[str, RepresentationSpec]
    metadata: dict[str, RepresentationSpec]
```

`SourceDescriptor` est le pivot. Ajouter un nouveau type custom veut dire: declarer un `type_id`, ses representations natives, ses axes, puis les adapters possibles.

#### 2.6.2 Blocks et vues

```python
@dataclass(frozen=True)
class DataView:
    sample_ids: tuple[SampleId, ...] | None = None
    partition: str | None = None
    fold_id: str | int | None = None
    source_ids: tuple[SourceId, ...] | None = None
    columns: tuple[str, ...] | None = None
    include_augmented: bool = True
    include_excluded: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class PresenceMask:
    sample_ids: tuple[SampleId, ...]
    source_id: SourceId
    present: tuple[bool, ...]

@dataclass(frozen=True)
class DataBlock:
    source_id: SourceId
    representation: RepresentationSpec
    sample_ids: tuple[SampleId, ...]
    data: Any
    axes: tuple[AxisSpec, ...]
    presence: PresenceMask | None = None
    feature_names: tuple[str, ...] | None = None
    lineage: Any | None = None

@dataclass(frozen=True)
class FeatureTable:
    sample_ids: tuple[SampleId, ...]
    X: Any                             # ndarray, scipy sparse, dataframe-like
    columns: tuple[str, ...]
    source_ids: tuple[SourceId, ...]
    presence: dict[SourceId, PresenceMask] = field(default_factory=dict)
    lineage: Any | None = None
```

`DataBlock` est general: image batch, graphe, serie, spectre, table. `FeatureTable` est seulement la representation tabulaire finale ou intermediaire.

#### 2.6.3 Interface dataset

```python
class MLDataset(Protocol):
    def schema(self) -> DatasetSchema:
        ...

    def view(self, selector: DataView) -> DataView:
        ...

    def materialize(
        self,
        source_id: SourceId,
        view: DataView,
        representation: RepresentationId | None = None,
    ) -> DataBlock:
        ...

    def target(
        self,
        name: str,
        view: DataView,
        representation: RepresentationId | None = None,
    ) -> DataBlock:
        ...

    def metadata(
        self,
        view: DataView,
        columns: Sequence[str] | None = None,
    ) -> DataBlock:
        ...

    def presence(
        self,
        source_id: SourceId,
        view: DataView,
    ) -> PresenceMask:
        ...
```

Cette interface ne fait pas de ML. Elle sait lire, filtrer et materialiser.

#### 2.6.4 Registre de types custom

```python
@dataclass(frozen=True)
class TypeCapability:
    type_id: TypeId
    native_representations: tuple[RepresentationId, ...]
    default_batching: str
    supports_missing: bool
    supports_sample_alignment: bool

class DataTypePlugin(Protocol):
    @property
    def type_id(self) -> TypeId:
        ...

    def infer_source(self, obj: Any, *, source_id: SourceId) -> SourceDescriptor:
        ...

    def validate(self, block: DataBlock) -> None:
        ...

    def capability(self) -> TypeCapability:
        ...

    def default_collator(self) -> "BatchCollator | None":
        ...

class DataTypeRegistry(Protocol):
    def register_type(self, plugin: DataTypePlugin) -> None:
        ...

    def get_type(self, type_id: TypeId) -> DataTypePlugin:
        ...

    def list_types(self) -> tuple[TypeId, ...]:
        ...

@dataclass(frozen=True)
class CollationPolicy:
    padding: Literal["none", "right", "left", "center"] = "none"
    truncate: bool = False
    batch_container: str | None = None
    emit_mask: bool = True

class BatchCollator(Protocol):
    def collate(
        self,
        blocks: Sequence[DataBlock],
        view: DataView,
        policy: "CollationPolicy",
    ) -> DataBlock:
        ...
```

Exemples de plugins:

- `DenseSignalType`: spectres, signaux 1D, arrays denses;
- `ImageRGBType`: images RGB;
- `GenotypeMatrixType`: variants genetiques;
- `TimeSeriesType`: sequences multivariables;
- `GraphType`: graphes/arbres;
- `TableType`: tabulaire/metadata.

#### 2.6.5 Adapters de representation

Les adapters sont le mecanisme qui rend les types composables.

```python
@dataclass(frozen=True)
class AdaptationPolicy:
    allow_lossy: bool = False
    allow_stateful: bool = True
    require_fit_on_train_only: bool = True
    max_output_features: int | None = None
    preferred_adapters: tuple[str, ...] = ()
    forbidden_adapters: tuple[str, ...] = ()

@dataclass(frozen=True)
class AdapterSpec:
    id: str
    input_type: TypeId
    input_representation: RepresentationId | None
    output_representation: RepresentationId
    output_type: TypeId
    supervised: bool = False
    stateful: bool = False
    lossy: bool = False
    fit_scope: Literal["none", "train_only", "fold_train"] = "none"
    cost_hint: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class AdapterContext:
    phase: str                         # fit_cv, refit, predict
    view: DataView
    fold_id: str | int | None
    random_state: int | None = None
    params: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class FittedAdapter:
    spec: AdapterSpec
    artifact: Any | None
    output_schema: RepresentationSpec
    feature_names: tuple[str, ...] | None = None

class RepresentationAdapter(Protocol):
    @property
    def spec(self) -> AdapterSpec:
        ...

    def can_adapt(
        self,
        source: SourceDescriptor,
        target: RepresentationSpec,
        policy: "AdaptationPolicy",
    ) -> bool:
        ...

    def fit(
        self,
        block: DataBlock,
        y: DataBlock | None,
        context: AdapterContext,
    ) -> FittedAdapter:
        ...

    def transform(
        self,
        block: DataBlock,
        fitted: FittedAdapter | None,
        context: AdapterContext,
    ) -> DataBlock:
        ...

    def fit_transform(
        self,
        block: DataBlock,
        y: DataBlock | None,
        context: AdapterContext,
    ) -> tuple[DataBlock, FittedAdapter | None]:
        ...

class AdapterRegistry(Protocol):
    def register_adapter(self, adapter: RepresentationAdapter) -> None:
        ...

    def adapters_from(
        self,
        source: SourceDescriptor,
        target: RepresentationSpec,
        policy: "AdaptationPolicy",
    ) -> tuple[RepresentationAdapter, ...]:
        ...

    def find_path(
        self,
        source: SourceDescriptor,
        target: RepresentationSpec,
        policy: "AdaptationPolicy",
    ) -> tuple[RepresentationAdapter, ...] | None:
        ...
```

Examples:

- `SpectraFlattenAdapter`: `(sample, processing, wavelength)` -> `tabular_numeric`;
- `ImageEmbeddingAdapter`: RGB image -> `tabular_numeric`;
- `ImageRawTensorAdapter`: RGB image -> `tensor_image`;
- `GenotypeDosageAdapter`: genotype -> `tabular_numeric`;
- `GenotypePCAAdapter`: genotype -> `tabular_numeric`, stateful, lossy, train-only;
- `WeatherAggregateAdapter`: sequence -> `tabular_numeric`;
- `WeatherSequenceAdapter`: sequence -> `sequence_tensor`;
- `TabularEncoderAdapter`: categories/numerics -> `tabular_numeric`.

DAG-ML decide quand `fit()` est appele. ML_DATA decide comment l'adapter transforme. C'est essentiel pour eviter les fuites: un encoder, une PCA ou un imputer se fit sur le fold train, pas sur tout le dataset.

#### 2.6.6 Fusion et alignement

```python
@dataclass(frozen=True)
class AlignmentPolicy:
    join: Literal["inner", "left", "outer", "exact"] = "inner"
    reference_source: SourceId | None = None
    on_missing_sample: Literal["error", "drop", "impute", "mask"] = "error"

@dataclass(frozen=True)
class FusionPolicy:
    mode: Literal["concat_features", "stack_channels", "dict_input", "list_input"]
    target_representation: RepresentationId
    alignment: AlignmentPolicy
    missing_source: Literal["error", "drop", "impute", "indicator", "mask"] = "error"
    namespace_columns: bool = True
    allow_lossy_adapters: bool = False
    max_output_features: int | None = None

@dataclass(frozen=True)
class AlignmentPlan:
    sample_ids: tuple[SampleId, ...]
    per_source_positions: dict[SourceId, tuple[int | None, ...]]
    presence: dict[SourceId, PresenceMask]

class FeatureJoiner(Protocol):
    def fit(
        self,
        tables: Sequence[FeatureTable],
        policy: FusionPolicy,
        context: AdapterContext,
    ) -> FittedAdapter:
        ...

    def transform(
        self,
        tables: Sequence[FeatureTable],
        fitted: FittedAdapter,
        context: AdapterContext,
    ) -> FeatureTable:
        ...
```

`FeatureJoiner` est un adapter comme les autres: il produit un schema stable au train, puis reapplique exactement ce schema au predict.

#### 2.6.7 Resolution pour un modele

```python
@dataclass(frozen=True)
class InputPortSpec:
    name: str
    accepted_representations: tuple[RepresentationId, ...]
    accepted_types: tuple[TypeId, ...]
    rank: int | None = None
    multi_source: bool = False
    optional: bool = False

@dataclass(frozen=True)
class ModelInputSpec:
    ports: tuple[InputPortSpec, ...]
    default_fusion: FusionPolicy | None = None

@dataclass(frozen=True)
class DataPlanStep:
    kind: Literal["materialize", "adapt", "align", "join", "collate"]
    inputs: tuple[str, ...]
    output: str
    adapter_id: str | None = None
    params: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class DataPlan:
    steps: tuple[DataPlanStep, ...]
    output_ports: dict[str, str]
    warnings: tuple[str, ...] = ()
    requires_user_choice: tuple[str, ...] = ()

class DataPlanner(Protocol):
    def resolve(
        self,
        dataset: MLDataset,
        sources: Sequence[SourceId],
        model_input: ModelInputSpec,
        policy: FusionPolicy,
    ) -> DataPlan:
        ...

    def execute_fit(
        self,
        plan: DataPlan,
        dataset: MLDataset,
        view: DataView,
        y: DataBlock | None,
        context: AdapterContext,
    ) -> tuple[dict[str, DataBlock], tuple[FittedAdapter, ...]]:
        ...

    def execute_transform(
        self,
        plan: DataPlan,
        dataset: MLDataset,
        view: DataView,
        fitted: Sequence[FittedAdapter],
        context: AdapterContext,
    ) -> dict[str, DataBlock]:
        ...
```

Le `DataPlanner` peut etre fourni par ML_DATA, mais DAG-ML garde le controle de la phase et des folds. Si `requires_user_choice` est non vide, DAG-ML doit refuser l'execution automatique ou demander une policy plus precise.

#### 2.6.8 Signature minimale cote DAG-ML

Un model adapter DAG-ML expose seulement son besoin:

```python
class ModelAdapter(Protocol):
    def input_spec(self, operator: Any) -> ModelInputSpec:
        ...

    def fit(
        self,
        operator: Any,
        inputs: dict[str, DataBlock],
        target: DataBlock,
        context: Any,
    ) -> Any:
        ...

    def predict(
        self,
        fitted_model: Any,
        inputs: dict[str, DataBlock],
        context: Any,
    ) -> Any:
        ...
```

Pour `RandomForest`, l'input spec est:

```python
ModelInputSpec(
    ports=(
        InputPortSpec(
            name="X",
            accepted_representations=("tabular_numeric",),
            accepted_types=("table",),
            rank=2,
            multi_source=True,
        ),
    ),
    default_fusion=FusionPolicy(
        mode="concat_features",
        target_representation="tabular_numeric",
        alignment=AlignmentPolicy(join="inner"),
        namespace_columns=True,
        allow_lossy_adapters=False,
    ),
)
```

Donc un nouveau type custom n'a besoin d'etre connu par DAG-ML que s'il existe un chemin dans ML_DATA:

```text
custom_type/native_representation -> ... -> tabular_numeric
```

Sans ce chemin, `RandomForest` est incompatible. Avec ce chemin, DAG-ML peut planifier l'adaptation sans connaitre le type lui-meme.

### 2.7 Compatibilite model/data et source/source

Le probleme central de DAG-ML est la compatibilite entre:

- ce que le modele accepte;
- ce que les sources peuvent fournir;
- ce que les edges de merge/split/concat declarent;
- ce que la phase courante autorise en termes de leakage, sample alignment et artifacts.

DAG-ML doit donc demander un plan de compatibilite avant execution. Dans le contrat propose, c'est la methode `DataPlanner.resolve()` cote ML_DATA, appelee par DAG-ML avec le `ModelInputSpec` du modele:

```python
plan = data_planner.resolve(
    dataset=dataset,
    sources=("nir", "photo_front", "photo_side", "genotype", "weather", "metadata"),
    model_input=random_forest_adapter.input_spec(RandomForestRegressor()),
    policy=fusion_policy,
)
```

Un `DataPlan` peut contenir:

- `direct`: le bloc est deja compatible;
- `flatten`: spectral/tabular dense vers `(samples, features)`;
- `transpose`: `channels_first` <-> `channels_last`;
- `concat_features`: concat horizontal de sources denses alignees;
- `stack_channels`: stack de sources compatibles comme canaux;
- `pad_or_truncate`: uniquement avec policy explicite;
- `dict_input` ou `list_input`: pour modeles multi-input;
- `featurize_required`: il faut un noeud amont qui convertit image/graphe/texte en features;
- `error`: aucune adaptation sure.

Regle importante: DAG-ML ne doit jamais silently flatten/concat une source complexe. Les conversions implicites doivent etre limitees aux cas denses simples, typiquement spectres/tableaux numeriques.

Exemples:

- sklearn classique: demande `flat_features`, rank 2, modalites denses. Pour des spectres multi-source, DAG-ML peut planifier `concat_features` si samples alignes.
- NN Conv1D spectral: demande rank 3. DAG-ML peut fournir `(samples, channels/processings, features)` ou `(samples, features, channels)` selon framework.
- modele multi-modal: demande `dict_input=True`; DAG-ML fournit `{nir: tensor, image: tensor, metadata: dataframe}` sans concatener.
- modele image: refuse un spectre brut sauf si un adapter explicite transforme le spectre en image/embedding.
- graph neural network: refuse les concat dense, demande un `GraphBatch` avec adjacency/edge features.

### 2.8 Ce qui appartient a DAG-ML vs ML_DATA

DAG-ML doit posseder:

- le graphe, les nodes, les ports et les edge contracts;
- l'orchestration du `DataPlanner` et le refus des plans qui violent les invariants ML;
- les phases `FIT_CV`, `SELECT`, `REFIT`, `PREDICT`, `EXPLAIN`;
- les invariants ML: OOF, no-leakage, fold alignment, refit provenance;
- la decision de planifier un `AdapterNode`, `JoinNode`, `SplitNode` ou de refuser;
- le cache, les traces, la lineage et les artifacts generiques;
- l'interface `OperatorAdapter` qui declare les inputs attendus.

ML_DATA doit posseder:

- le stockage concret des sources;
- l'alignement des samples entre sources (`inner`, `left`, `outer`, masques de presence);
- les descriptors de sources, axes, modalites, dtypes, units;
- les conversions de representation que la data sait faire sans perdre de sens;
- les policies de padding/collation/ragged batch;
- les vues samples/features/targets/metadata;
- les domain primitives: wavelengths NIRS, source names, signal types, image size, graph schema, time axis.

Le contrat partage est:

- `SourceDescriptor`;
- `DataBlock`;
- `DataView`;
- `ModelInputSpec`;
- `FusionPolicy`;
- `DataPlan`;
- `PresenceMask` pour sources manquantes;
- `AxisSpec` pour dire ce que signifie chaque dimension.

Cette separation evite deux erreurs:

1. mettre les semantics de merge multi-source dans chaque modele;
2. cacher des decisions de reshape/concat dans `Dataset.x()`.

### 2.9 Cas concret: early fusion heterogene vers RandomForest

Exemple cible:

```text
sample_id
  - 1 spectre NIRS
  - 2 photos RGB
  - 1 patrimoine genetique
  - 1 serie meteo multivariable
  - metadata
  -> RandomForest
```

Un `RandomForest` sklearn ne consomme pas "du multi-source". Il consomme un tableau numerique dense ou sparse de forme:

```text
(n_samples, n_features)
```

Donc l'early fusion n'est possible que si chaque source est transformee en `FeatureTable` alignee par `sample_id`, puis join horizontal:

```text
NIRS             -> SpectralVectorizer      -> FeatureTable[nir_*]
RGB front        -> ImageFeaturizer         -> FeatureTable[img_front_*]
RGB side         -> ImageFeaturizer         -> FeatureTable[img_side_*]
Genetics         -> GenotypeEncoder         -> FeatureTable[geno_*]
Weather series   -> TimeSeriesAggregator    -> FeatureTable[meteo_*]
Metadata         -> TabularEncoder          -> FeatureTable[meta_*]
                                                        |
                                                        v
                                FeatureJoin(on=sample_id, policy=...)
                                                        |
                                                        v
                                   TabularNumericBlock -> RandomForest
```

Qui declare quoi:

- le model adapter `RandomForestAdapter` declare `ModelInputSpec`: accepte `tabular_numeric`, rank 2, samples alignes, pas d'images/graphes/sequences brutes;
- `ML_DATA` declare chaque source: modalite, axes, units, cle sample, granularite, presence mask, representations natives;
- `ML_DATA` declare aussi les conversions disponibles vers `tabular_numeric`, mais ne choisit pas toujours seul;
- DAG-ML planifie `FeaturizerNode` / `EncoderNode` / `JoinNode` / `ImputerNode` pour satisfaire le contrat du modele;
- l'utilisateur ou une policy explicite choisit les conversions quand elles ne sont pas evidentes.

Manifest minimal cote `ML_DATA`:

```python
MLDataManifest(
    sample_key="sample_id",
    sources=[
        SourceDescriptor(
            id="nir",
            modality="dense_signal",
            granularity="per_sample",
            axes=["sample", "wavelength"],
            native_representation="array_2d",
            default_adapters={"tabular_numeric": "flatten_or_selected_preprocessings"},
        ),
        SourceDescriptor(
            id="photo_front",
            modality="image",
            granularity="per_sample",
            axes=["sample", "height", "width", "channel"],
            native_representation="rgb_image",
            default_adapters={"tabular_numeric": "image_embedding"},
        ),
        SourceDescriptor(
            id="photo_side",
            modality="image",
            granularity="per_sample",
            axes=["sample", "height", "width", "channel"],
            native_representation="rgb_image",
            default_adapters={"tabular_numeric": "image_embedding"},
        ),
        SourceDescriptor(
            id="genotype",
            modality="genotype",
            granularity="per_sample",
            axes=["sample", "variant"],
            native_representation="variant_matrix",
            default_adapters={"tabular_numeric": "dosage_or_pca"},
        ),
        SourceDescriptor(
            id="weather",
            modality="timeseries",
            granularity="per_sample_sequence",
            axes=["sample", "time", "variable"],
            native_representation="time_series",
            default_adapters={"tabular_numeric": "window_aggregates"},
        ),
        SourceDescriptor(
            id="metadata",
            modality="table",
            granularity="per_sample",
            axes=["sample", "column"],
            native_representation="dataframe",
            default_adapters={"tabular_numeric": "tabular_encoder"},
        ),
    ],
)
```

La fusion automatique n'est raisonnable que si tous les adapters vers `tabular_numeric` sont declares et si la policy l'autorise:

```python
EarlyFusionPolicy(
    target_representation="tabular_numeric",
    join="inner",                    # ou left/outer
    missing_sources="impute+indicator",
    namespace_columns=True,
    allow_lossy_adapters=False,       # image embedding ou PCA genetique deviennent explicites
    fit_adapters_on="train_only",
)
```

Si `allow_lossy_adapters=False`, DAG-ML doit s'arreter et demander un choix explicite pour:

- image brute -> embedding, pixels flatten, CNN fine-tune, features couleur/texture;
- genotype -> dosage brut, selection de variants, PCA, PRS, embedding;
- meteo sequence -> fenetres temporelles, lags, stats, modele sequence amont;
- metadata categorielle -> one-hot, target encoding, hashing, embeddings.

Ce n'est pas de la plomberie: ce sont des hypotheses de modelisation. DAG-ML peut automatiser l'assemblage, pas inventer silencieusement ces hypotheses.

Contrat de sortie d'un adapter vers RandomForest:

```python
@dataclass(frozen=True)
class FeatureTable:
    sample_ids: np.ndarray
    X: np.ndarray | scipy.sparse.spmatrix
    columns: list[str]
    source_id: str
    fitted_artifacts: list[ArtifactRef]
    missing_mask: np.ndarray | None
    lineage: LineageRef
```

Puis `FeatureJoin` valide:

- meme `sample_id` ou alignement explicite;
- pas de fuite: adapters fit sur train uniquement;
- schema stable train/predict;
- colonnes namespacees par source;
- NaN/missing geres selon policy;
- taille raisonnable ou warning avant explosion dimensionnelle.

Pour ce cas, `ML_DATA` doit donc fournir au minimum:

1. un catalogue de sources avec modalite, axes, units, granularite et sample key;
2. un moteur d'alignement sample/source avec `PresenceMask`;
3. des adapters declaratifs vers des representations cibles (`tabular_numeric`, `tensor_image`, `sequence`, `graph_batch`, `dict_input`);
4. des policies de missing/padding/collation par modalite;
5. une schema registry pour garantir que le predict reprend exactement les memes colonnes/adapters que le train;
6. des metadata de cout/dimension/lossiness pour que DAG-ML sache si une conversion peut etre automatique.

### 2.10 Points structurants a ne pas oublier

#### 2.10.1 Repetitions, plusieurs X pour un Y, et unite de split

Le modele de donnees ne doit pas confondre observation, sample logique et cible. En NIRS, on a souvent plusieurs spectres pour un meme echantillon et une seule valeur Y. En multimodal, on peut avoir deux images, plusieurs mesures meteo, plusieurs spectres, et un seul Y.

Il faut donc separer:

- `observation_id`: ligne physique dans une source;
- `sample_id`: unite logique utilisee pour aligner les sources;
- `target_id`: unite portant le Y;
- `group_id` ou `entity_id`: unite de leakage/split, par exemple plante, patient, lot, parcelle;
- `origin_id`: sample original quand une observation est augmentee.

Signatures:

```python
@dataclass(frozen=True)
class SampleRelation:
    source_id: SourceId
    observation_ids: tuple[str | int, ...]
    sample_ids: tuple[SampleId, ...]
    target_ids: tuple[str | int, ...]
    group_ids: tuple[str | int, ...] | None = None
    origin_ids: tuple[SampleId | None, ...] | None = None

@dataclass(frozen=True)
class AggregationPolicy:
    level: Literal["observation", "sample", "target", "group"]
    method: Literal["none", "mean", "median", "vote", "first", "custom"] = "none"
    custom_aggregator: str | None = None
    keep_observation_predictions: bool = True

@dataclass(frozen=True)
class SplitPolicy:
    split_unit: Literal["observation", "sample", "target", "group"] = "sample"
    group_key: str | None = None
    forbid_origin_cross_fold: bool = True
```

Regle DAG-ML: le split doit se faire au niveau qui evite la fuite. Si plusieurs X partagent un Y ou un `group_id`, ils doivent tomber dans le meme fold quand `split_unit="target"` ou `split_unit="group"`.

`PredictionBlock` doit pouvoir porter deux niveaux:

```python
@dataclass(frozen=True)
class PredictionBlock:
    prediction_ids: tuple[str, ...]
    sample_ids: tuple[SampleId, ...]
    observation_ids: tuple[str | int, ...] | None
    target_ids: tuple[str | int, ...] | None
    fold_id: str | int
    partition: str
    y_pred: Any
    y_true: Any | None
    aggregation_level: str = "observation"
```

Ensuite une prediction peut etre agregee en twin `sample` ou `group`, mais l'OOF brut reste tracable.

#### 2.10.2 Augmentation et OOF

L'augmentation est un adapter data, pas une mutation libre du dataset.

```python
@dataclass(frozen=True)
class AugmentationPolicy:
    apply_to: Literal["train_only", "train_and_refit", "all_partitions"] = "train_only"
    inherit_target: bool = True
    inherit_group: bool = True
    forbid_validation_augmentation: bool = True
    store_origin_mapping: bool = True
    seed_scope: Literal["run", "variant", "fold", "node"] = "fold"

class AugmentationAdapter(Protocol):
    def plan(
        self,
        block: DataBlock,
        policy: AugmentationPolicy,
        context: AdapterContext,
    ) -> "AugmentationPlan":
        ...

    def transform(
        self,
        block: DataBlock,
        plan: "AugmentationPlan",
        context: AdapterContext,
    ) -> tuple[DataBlock, SampleRelation]:
        ...
```

Regles OOF:

- les augmentations d'un sample train restent dans le fold train;
- aucune augmentation derivee d'un sample de validation ne peut entrer dans le train fold;
- les predictions OOF doivent etre produites pour les samples/origins de validation, pas pour des copies augmentees vues au train;
- si un meta-modele consomme des predictions, `PredictionJoin` doit utiliser les predictions OOF des origins, puis seulement eventuellement propager aux augmentations train;
- en refit final, l'augmentation peut etre reappliquee sur tout le train, mais les artifacts doivent etre marques `phase=REFIT`.

#### 2.10.3 Seeding et reproductibilite

Le seed doit etre hierarchique et serialise. Un simple `random_state=42` global ne suffit pas quand on a generateurs, folds, augmentation, tuning et parallelisme.

```python
@dataclass(frozen=True)
class SeedContext:
    root_seed: int | None
    run_id: str
    variant_id: str | None = None
    node_id: str | None = None
    fold_id: str | int | None = None
    trial_id: str | int | None = None

    def child(self, **labels: object) -> "SeedContext":
        ...

    def numpy_seed(self) -> int | None:
        ...

    def python_seed(self) -> int | None:
        ...
```

Regle: chaque node stateful recoit un seed derive de `(root_seed, run_id, variant_id, node_id, fold_id, trial_id)`. Le seed effectif est persiste dans le `LineageRecord`, pas seulement le root seed.

#### 2.10.4 Operateurs custom lies a la data

Les operators peuvent demander des informations auxiliaires aux sources: wavelengths, time coordinates, graph schema, image metadata, variant annotations. Dans nirs4all, certains transformers ont besoin de `wavelengths=`. Ce n'est pas special NIRS: c'est une dependance auxiliaire typee.

```python
@dataclass(frozen=True)
class AuxInputSpec:
    name: str
    kind: Literal["axis_coordinates", "source_metadata", "schema", "side_data"]
    axis: str | None = None
    required: bool = True

@dataclass(frozen=True)
class OperatorDataSpec:
    input: ModelInputSpec | None
    aux_inputs: tuple[AuxInputSpec, ...] = ()
    output_representation: RepresentationId | None = None

class DataAwareOperatorAdapter(Protocol):
    def data_spec(self, operator: Any) -> OperatorDataSpec:
        ...

    def fit(
        self,
        operator: Any,
        block: DataBlock,
        aux: dict[str, Any],
        y: DataBlock | None,
        context: AdapterContext,
    ) -> Any:
        ...

    def transform(
        self,
        fitted_operator: Any,
        block: DataBlock,
        aux: dict[str, Any],
        context: AdapterContext,
    ) -> DataBlock:
        ...
```

Exemple: un `SpectralTransformer` declare `AuxInputSpec(name="wavelengths", kind="axis_coordinates", axis="wavelength")`. ML_DATA fournit ces coordinates depuis `AxisSpec.coordinates`.

#### 2.10.5 Serialization et replay

Tout ce qui est necessaire au replay predict/refit doit etre serialisable. Il faut distinguer specs JSON et payloads binaires.

```python
@dataclass(frozen=True)
class SerializableRef:
    registry: str
    type_id: str
    version: str
    object_id: str

@dataclass(frozen=True)
class SerializedDataPlan:
    schema_fingerprint: str
    plan: DataPlan
    fitted_adapters: tuple[SerializableRef, ...]
    output_schema: RepresentationSpec

@dataclass(frozen=True)
class ExecutionBundle:
    graph_spec: GraphSpec
    data_schema: DatasetSchema
    data_plan: SerializedDataPlan
    artifacts: tuple[SerializableRef, ...]
    seeds: tuple[SeedContext, ...]
    plugin_versions: dict[str, str]
```

Regles:

- `GraphSpec`, `DataPlan`, `DatasetSchema`, `ModelInputSpec` doivent etre JSON/YAML serialisables;
- les objets fitted, modeles, encoders, PCA, imputers, CNN embeddings vont en artifact store;
- chaque plugin custom doit fournir `type_id`, `version`, et serializer/deserializer;
- le predict refuse si le schema courant ne matche pas le `schema_fingerprint`, sauf migration explicite;
- les colonnes tabulaires et leur ordre sont un artifact de schema, pas une convention implicite.

#### 2.10.6 Finetuning et recherche d'hyperparametres

Le finetuning doit etre un noeud DAG, pas un mode special cache dans le controller modele. Il peut tuner les parametres du modele, mais aussi ceux des adapters data.

```python
@dataclass(frozen=True)
class SearchSpace:
    params: dict[str, Any]

@dataclass(frozen=True)
class TrialResult:
    trial_id: str
    params: dict[str, Any]
    score: float
    metric: str
    artifacts: tuple[SerializableRef, ...] = ()

class TunerAdapter(Protocol):
    def suggest(self, trial_id: str, space: SearchSpace, seed: SeedContext) -> dict[str, Any]:
        ...

    def record(self, result: TrialResult) -> None:
        ...

@dataclass(frozen=True)
class TuningNodeSpec:
    search_space: SearchSpace
    objective_metric: str
    cv_policy: SplitPolicy
    max_trials: int
    tunes: tuple[str, ...]        # node ids or adapter ids
```

Regles:

- les trials s'executent avec les memes contraintes OOF que le training normal;
- un adapter stateful tune, par exemple PCA genotype ou image embedding, est fit uniquement sur le fold train;
- le meilleur `DataPlan` + `ModelParams` devient l'entree de refit;
- les trial seeds, params, scores et schemas de sortie sont persistes.

#### 2.10.7 DAG reifiable: un noeud peut etre un DAG

DAG-ML doit etre reifiable: un DAG est une valeur, serialisable, inspectable, versionnable et reutilisable comme un noeud dans un autre DAG.

```python
@dataclass(frozen=True)
class PortSpec:
    name: str
    kind: Literal["data", "target", "prediction", "artifact", "metric"]
    representation: RepresentationId | None = None

@dataclass(frozen=True)
class GraphInterface:
    inputs: tuple[PortSpec, ...]
    outputs: tuple[PortSpec, ...]

@dataclass(frozen=True)
class GraphSpec:
    id: str
    interface: GraphInterface
    nodes: dict[str, NodeSpec]
    edges: tuple[EdgeSpec, ...]
    search_space: SearchSpace | None = None

@dataclass(frozen=True)
class SubgraphNodeSpec:
    id: str
    graph: GraphSpec | SerializableRef
    input_mapping: dict[str, str]
    output_mapping: dict[str, str]
    inline_policy: Literal["inline", "opaque", "auto"] = "auto"
```

Semantique:

- `inline`: le sous-DAG est developpe dans le DAG parent pour scheduling/cache fin;
- `opaque`: le sous-DAG est execute comme une boite noire avec ses propres artifacts;
- `auto`: le planner choisit.

Cela permet:

- un pipeline preprocessing complet comme noeud;
- un modele ensemble comme sous-DAG;
- un featurizer image pre-entraine + pooling comme noeud;
- un stacker OOF comme noeud;
- une recette `nirs4all-methods` versionnee comme composant reifiable.

Un sous-DAG doit declarer son `GraphInterface`; sinon il n'est pas composable.

### 2.11 Performance

Les optimisations a formaliser:

- enumeration lazy des variantes pour eviter de materialiser de gros `_cartesian_`;
- `DataView` immutable + blocs copy-on-write plutot que deep copy de gros arrays;
- cache de node par hash `(operator, params, input lineage, phase)`;
- scheduler topologique avec barrieres pour `Join`, `Split`, `Refit`;
- `PredictionStore` colonne/array separe pour ne pas recopier les gros arrays dans la metadata;
- plans de compatibilite caches par `(source descriptors, model input spec, policy)`;
- collation/padding au plus tard possible, juste avant le modele;
- resource hints par node (`cpu`, `gpu`, `thread_safe`, `nested_parallelism`);
- execution parallele au niveau variant, branche ou fold, mais avec une regle unique de prevention du parallelisme imbrique.

### 2.12 Migration pragmatique

Ordre recommande:

1. Extraire les protocoles (`MLDataset`, `DataPlanner`, `PredictionStore`, `ArtifactStore`, `OperatorAdapter`) sans changer le runtime.
2. Ajouter un compilateur nirs4all DSL -> `GraphSpec` en lecture seule.
3. Extraire de `SpectroDataset` un `SpectroDatasetConnector` qui expose `SourceDescriptor`, `DataBlock` et les conversions `2d/3d/3d_transpose`.
4. Remplacer les appels implicites `dataset.x(layout=..., concat_source=...)` dans les controllers par `DataPlanner.resolve()` + `DataPlan`.
5. Remplacer progressivement `PipelineConfigs` par `SearchSpace` DAG-ML, en gardant la syntaxe.
6. Porter `StepParser`/`ControllerRouter` en registry DAG-ML.
7. Formaliser `FoldSet`, `PredictionBlock`, `OOFJoin` et branch/merge.
8. Deplacer selection/refit dans DAG-ML.
9. Garder les controllers NIRS-specifiques dans `nirs4all`, pas dans DAG-ML.

## 3. Alternative existante?

Il n'y a pas, a ma connaissance, d'alternative qui corresponde exactement.

- scikit-learn couvre bien `Pipeline`, `ColumnTransformer`, cache de transformers et stacking OOF via `StackingRegressor`/`StackingClassifier`, mais pas un DAG multi-branches avec data connectors, merge OOF arbitraire, artifact lineage, refit par topologie et compatibilite explicite multi-source/multi-modal.
- Kedro formalise des nodes, des inputs/outputs et une resolution topologique, mais c'est plutot un framework data pipeline. Il ne porte pas nativement les contrats ML fins: folds, OOF, leakage checks, refit final, replay predict/explain par artifacts.
- Dagster/Prefect sont forts pour orchestration, observabilite et assets/runs. Ils sont trop lourds et trop externes pour etre le coeur in-process d'un pipeline ML interactif.
- Hamilton est elegant pour generer un dataflow depuis des fonctions Python typees, mais son modele est moins adapte aux operators stateful sklearn-like, aux phases CV/refit et aux predictions OOF comme first-class edge.
- Dask/Ray peuvent servir de backend d'execution parallele, pas de modele semantique ML.

Conclusion: DAG-ML doit etre un moteur custom, mais il peut reprendre des idees:

- topological planning facon Kedro/Hamilton;
- assets/artifacts et lineage facon Dagster;
- scheduler local optionnel facon Dask;
- conventions sklearn pour fit/transform/predict, params et stacking;
- contrats OOF/refit propres a nirs4all.

References externes consultees:

- scikit-learn Pipeline/composite estimators: https://scikit-learn.org/stable/modules/compose.html
- scikit-learn stacking: https://scikit-learn.org/stable/modules/ensemble.html#stacking
- Kedro Pipeline object: https://docs.kedro.org/en/stable/build/pipeline_introduction/
- Dagster assets: https://docs.dagster.io/guides/build/assets
- Apache Hamilton functions/nodes/dataflow: https://hamilton.dagworks.io/en/latest/concepts/node/
- Dask delayed: https://docs.dask.org/en/stable/delayed.html
- Ray DAG API: https://docs.ray.io/en/latest/ray-core/ray-dag.html
- Prefect flows: https://docs.prefect.io/v3/concepts/flows
