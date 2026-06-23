# DAG-ML from the nirs4all runtime

Rating based on code, not existing audits. The most structuring paths read for this analysis are:

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

## 1. Concrete architecture of nirs4all pipelines

### 1.1 Overview

The current engine already executes a DAG, but this DAG is not an explicit object. It is dynamically reconstructed from a list of steps, a mutable context, branch snapshots, cross-validation folds and a prediction store.

The actual flow is:

1.`PipelineRunner.run()`receives`pipeline`,`dataset`,`refit`, store/cache and delegates to`PipelineOrchestrator`. 2.`PipelineOrchestrator.execute()`normalizes the config, normalizes the datasets, opens the run in the store, creates`ArtifactRegistry`,`PipelineExecutor`,`Predictions`and executes each variant. 3.`PipelineConfigs`compiles user syntax: preprocessing of steps, expansion of generators (`_or_`,`_grid_`,`_cartesian_`,`_zip_`, etc.), generation of variant names and conservation of`generator_choices`. 4.`PipelineExecutor.execute()`initializes`ExecutionContext`, opens a pipeline in the store, creates a trace, loops through the steps and flushes the predictions to the store. 5.`StepRunner`parses each step, routes to a controller, executes the controller, then normalizes the return to`StepResult`. 6. Controllers do the real work: split, transform, y-transform, tag/exclude, branch, merge, model, charts. 7. At the end of the CV,`PipelineOrchestrator`launches the refit if requested: extraction of the best config, reexecution on full train, writing of`fold_id="final"`predictions.

It is therefore not a simple`sklearn.Pipeline`: the nirs4all pipeline mixes compilation of variants, data routing, branches, CV, OOF, persistence, artifact lineage, selection and refit.

### 1.2 Generation: compile time, not runtime

`PipelineConfigs`is the current compiler. It accepts list, dict, path or string, then:

- converts a list to`{"pipeline": [...]}`; - merges the`xxx`/`xxx_params`couples; - serializes Python components; - detects generators; - produces a materialized list of variants (`self.steps`); - attaches its`generator_choices`to each variant.

Important point: generators in a duplication branch are not fully processed by`PipelineConfigs`. They can be extended later by`BranchController`. This shows that generation is today shared between the config compiler and certain controllers. For DAG-ML, it is necessary to clearly separate:

-`SearchSpace`: describes the choices; -`Compiler`: transforms DSL into IR; -`Planner`: chooses a lazy/materialized enumeration strategy; -`Executor`: executes an already resolved plan.

### 1.3 Execution: a main line with hidden graphs

`PipelineExecutor._execute_steps()`loops linearly over`steps`, but several steps create subgraphs:

-`branch`: fan-out. It creates several`branch_contexts`, each with its`features_snapshot`, its`chain_snapshot`, its`branch_id`, its`branch_path`. -`merge`: join. It exits branch mode and reconstructs features or predictions. -`split`: creates the folds as the state of the dataset, not as visible DAG nodes. -`model`: internal fork/join on folds, then predictions`fold`,`avg`,`w_avg`. -`feature_augmentation`: executes a sub-pipeline on several processings. - sub-pipelines:`StepRunner`can execute a list of sub-steps as a mini-pipeline.

The implicit DAG looks like this:

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

### 1.4 Controllers and operators

The current pattern is healthy:

- the operator is the business payload: sklearn object, merge config, splitter, filter, transform, model; - the controller is the execution adapter: it knows how to fit, transform, predict, select samples, manage train/predict/explain, save artifacts; -`StepParser`transforms user syntax into`ParsedStep`; -`ControllerRouter`iterates through`CONTROLLER_REGISTRY`, applies`matches()`, sorts by priority and instantiates the controller.

This pattern should be the heart of DAG-ML. The name might change (`OperatorAdapter`,`NodeHandler`,`ExecutorPlugin`), but the separation operator/controller is the correct abstraction.

Target interface:

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

### 1.5 OOF and stacking

The OOF semantics is one of the real assets of nirs4all.

The current feed:

1.`CrossValidatorController`converts a sklearn-like splitter into folds stored in absolute sample IDs. 2.`BaseModelController`trains one model by fold. 3. Each fold writes`train`,`val`,`test`predictions, with`fold_id`,`sample_indices`, scores and metadata. 4. For multiple folds, the controller also produces`avg`and`w_avg`. 5.`TrainingSetReconstructor`reconstructs the training meta-features from`partition="val"`predictions, aligned by`sample_indices`. This is the real OOF. 6.`MergeController`uses this reconstructor to transform branch predictions into meta-model features. Default,`unsafe=False`;`unsafe=True`takes direct train predictions and the code explicitly marks this as a data leak.

Contract to extract in DAG-ML:

- Any prediction used as a feature of a downstream model in the train phase must be OOF. - Alignment must be done by`sample_id`, never by implicit position alone. - The`PredictionBlock`must contain at least:`sample_ids`,`partition`,`fold_id`,`producer_node`,`target_space`,`y_pred`,`y_proba`, scores, branch lineage. - A`PredictionJoin`must refuse by default a non-OOF train block.

### 1.6 Merge and branches

`BranchController` handles two families:

- duplication: each branch sees the same samples with different processing; - separation: samples or sources are partitioned by tag, metadata, filter or source.

`MergeController` then handles:

- horizontal feature merge for duplication branches; - vertical merge/reconstruction by sample ID for disjoint branches; - source merge (`concat`,`stack`,`dict`); - merge prediction with model selection, aggregation (`separate`,`mean`,`weighted_mean`,`proba_mean`) and OOF; - mixed merge: features of certain branches + predictions of other branches.

In DAG-ML,`branch`and`merge`should no longer be side effects in`context.custom`. They must become formal nodes:

-`ForkNode`: product N`DataView`named; -`MapNode`: applies a subgraph to each branch; -`FeatureJoinNode`: combines`FeatureBlock`; -`PredictionJoinNode`: combines`PredictionBlock`with OOF validation; -`SourceJoinNode`: preserves or concatenates sources.

### 1.7 Multi-source and layouts in current code

Multi-source is today supported by`SpectroDataset`, but it is in reality a generic DAG-ML capability.

In`nirs4all/data/features.py`,`Features`manages a list of`FeatureSource`. The sources are aligned on the same sample axis, but they can have different feature dimensions and processing chains. Each`FeatureSource`stores its data as a 3D tensor:

```text
(samples, processings, features)
```

Then`LayoutTransformer`exposes several views:

- `2d`: `(samples, processings * features)`;
- `2d_interleaved`: `(samples, features * processings)`;
- `3d`: `(samples, processings, features)`;
- `3d_transpose`: `(samples, features, processings)`.

When`concat_source=True`,`Features.x()`concatenates several sources on the last axis. For digital spectra, it is acceptable: same axis samples, dense tensors, compatible or flattened processing possible. But this rule is too poor for general DAG-ML:

- images: concatenating pixels from different cameras does not have the same meaning as concatenating wavelengths; - graphs/trees: there is not always a dense features axis; - heterogeneous sources: a source can be a spectrum, an image, a clinical picture, a molecular graph; - multi-input models: some models expect a dict/list of tensors, not a concatenated array; - missing sources: you must choose between`inner`,`left`,`outer`, presence mask, imputation or error.

Current controllers already show the need:

-`TransformerMixinController`requests`layout="3d", concat_source=False`, loop by source and by processing, then persist the artifacts with`source_index`. - Model controllers declare a preferred layout: sklearn ->`2d`, PyTorch ->`3d`, TensorFlow/JAX ->`3d_transpose`, with`force_layout`to force. -`BaseModelController.get_xy()`requests`dataset.x(..., layout=layout)`with the default`concat_source=True`behavior, then asserts that the result is a`np.ndarray`. So the engine implicitly assumes that a model receives a single concatenate tensor, not a general multi-source input. -`MergeController`already contains`sources`,`dict`,`features`,`merge_sources`,`by_source`modes, but these semantics are encoded in the controller and in`SpectroDataset`, not in a formal DAG contract. -`SampleLinker`in`nirs4all/data/selection/sample_linker.py`already manages multi-source alignment by key (`inner`,`left`,`outer`). This logic belongs rather to the ML_DATA contract than to the ML execution engine.

Therefore, multi-source must not remain "a dataset detail." It must become a DAG-ML port and edge type.

### 1.8 Refit

The current refit is a second phase after selection:

-`extract_winning_config()`or`extract_top_configs()`chooses the variants; -`execute_simple_refit()`deep-copy the dataset, replaces the splitter with a single “full train” fold, injects`best_params`and`refit_params`, then re-executes the pipeline; - the refit predictions are relabeled`fold_id="final"`and`refit_context="standalone"`; - for stacking/mixed merge/separation, the dispatcher goes towards dedicated strategies; - for certain branches,`BranchController`powers`best_refit_chains`, which allows each model to be refitted on its best channel without requiring the blind.

In DAG-ML, the refit must be a phase graph, not a patch of steps:

```text
FitCVPlan -> SelectionPlan -> RefitPlan -> ExportPlan
```

The refit must receive a`SelectedGraph`containing:

- graph variant ID; - generator choices; - best params; - preprocessing path; - target model; - refit strategy; - link to the CV predictions which justify the selection.

## 2. Proposition DAG-ML

### 2.1 Mission

DAG-ML should be a local/in-process ML engine, not a machine orchestrator. It formalizes the universal heart:

- compilation of a DSL pipeline in DAG; - generation of variants; - multi-phase execution fit/predict/explain/refit; - data connectors; - explicit model/data and source/source compatibility; - adapter operators/controllers; - CV, OOF, stacking and refit; - artifacts, cache, lineage, traces.

It must not contain NIRS logic.`SpectroDataset`, wavelengths, spectral sources and current index must be connected via adapters from`nirs4all-data`/`nirs4all`.

### 2.2 Base objects

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

Important types:

-`MLDataset`: generic abstraction of samples, sources, targets, metadata and views. -`DataView`: immutable selection of samples/features/targets. -`SourceDescriptor`: stable description of a source, its modality, its axes and its units. -`DataBlock`: concrete payload of a source or a merge, with`sample_ids`, representation, axes and lineage. -`FeatureBlock`: dense special case of`DataBlock`for digital features. -`TargetBlock`: y + target transform lineage. -`FoldSet`: folds to stable sample IDs. -`PredictionBlock`: OOF/test/final predictions with sample IDs. -`ModelInputSpec`: entry contract declared by an adapted model. -`DataPlan`: explicit conversion between data blocks and model input. -`ArtifactRef`: versioned pointer to a fitted model/transformer. -`LineageRecord`: link between node, inputs, params, artifacts, outputs.

### 2.3 Phases

The phases must be explicit:

-`COMPILE`: DSL ->`GraphSpec`. -`PLAN`:`GraphSpec`+ generator choices ->`ExecutionPlan`. -`FIT_CV`: fit transformers/models by folds, emission of OOF predictions. -`SELECT`: ranking by metric, top-k, per-model selection, multi-criteria. -`REFIT`: final training on full train,`fold_id="final"`emission. -`PREDICT`: minimal replay via artifacts. -`EXPLAIN`: replay + hooks explainability.

The engine may exhibit:

```python
compiled = dagml.compile(pipeline_spec, registry=registry)
for variant in dagml.enumerate(compiled.search_space, lazy=True):
    plan = dagml.plan(compiled.graph, variant)
    cv_result = dagml.fit_cv(plan, dataset)

selected = dagml.select(cv_result, ranking="rmsecv", top_k=1)
final = dagml.refit(selected, dataset)
pred = dagml.predict(final.bundle, new_dataset)
```

### 2.4 nirs4all-compatible DSL

The first version should not invent new syntax. It should accept the current nirs4all syntax and compile it to IR:

- `{"preprocessing": StandardScaler()} -> TransformNode`
- `{"split": KFold(...)} -> SplitNode`
- `{"model": PLSRegression(...)} -> ModelNode`
- `{"branch": [...]}` -> Fork/Map subgraph
- `{"merge": "predictions"}` -> PredictionJoinNode
- generateurs `_or_`, `_grid_`, `_cartesian_` -> `SearchSpace`

Only then can we offer a more typical API builder.

### 2.5 Data connectors

The main decoupling point is data, but we need to go further than a simple`dataset.x(layout="2d")`.

DAG-ML must request data by contract. ML_DATA must respond with schemas, standard blocks and an explicit conversion plan.`nirs4all`would then only provide one`SpectroDatasetConnector`; another lib could provide tabular, images, timeseries, graphs or gnom without touching the DAG-ML engine.

For spectra,`SpectroDatasetConnector`can declare:

-`dense_signal`mode; - native`(samples, processings, features)`representation; - lossless conversions to`flat_features`,`channels_first`,`channels_last`; - concat source allows if aligned samples and dense tensors; - padding/truncation only allows when an explicit policy requests it.

For images/graphs/trees/multimodal, the default spectral rules should not be applied. The connector must say whether an operation is possible, lossless, lossy or prohibited.

### 2.6 Generic and extensible ML_DATA contract

If the data layer becomes a separate and generic library, DAG-ML should no longer import`SpectroDataset`,`pandas`,`torch_geometric`,`PIL`, etc. DAG-ML only needs to talk to a stable contract. New types arrive via ML_DATA plugins.

Target architecture:

```text
DAG-ML
  - compile le graph
  - chooses the CV/refit/predict phases
  - requests an input compatible with a model
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

`SourceDescriptor`is the pivot. Adding a new custom type means: declaring a`type_id`, its native representations, its axes, then possible adaptations.

#### 2.6.2 Blocks and views

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

`DataBlock`is general: batch image, graph, series, spectrum, table.`FeatureTable`is only the final or intermediate tabular representation.

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

This interface does not do ML. She knows how to read, filter and materialize.

#### 2.6.4 Custom type register

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

-`DenseSignalType`: spectra, 1D signals, dense arrays; -`ImageRGBType`: RGB images; -`GenotypeMatrixType`: genetic variants; -`TimeSeriesType`: multivariable sequences; -`GraphType`: graphs/trees; -`TableType`: tabular/metadata.

#### 2.6.5 Adapters de representation

Adapters are the mechanism that makes types composable.

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

DAG-ML decides when`fit()`is called. ML_DATA decides how to adapt the transform. This is essential to avoid leaks: an encoder, a PCA or an imputer is done on the fold train, not on the entire dataset.

#### 2.6.6 Fusion and alignment

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

`FeatureJoiner`is an adapter like any other: it produces a stable pattern to the train, then exactly reapplies this pattern to the predict.

#### 2.6.7 Resolution for a model

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

The`DataPlanner`can be provided by ML_DATA, but DAG-ML retains control of the phase and folds. If`requires_user_choice`is non-empty, DAG-ML must refuse automatic execution or request a more precise policy.

#### 2.6.8 Signature minimale cote DAG-ML

A DAG-ML model adapter only exposes its need:

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

For`RandomForest`, the input spec is:

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

So a new custom type only needs to be known by DAG-ML if there is a path in ML_DATA:

```text
custom_type/native_representation -> ... -> tabular_numeric
```

Without this path,`RandomForest`is incompatible. With this path, DAG-ML can plan the adaptation without knowing the type itself.

### 2.7 Model/data and source/source compatibility

The central problem of DAG-ML is the compatibility between:

- what the model accepts; - what the sources can provide; - what the merge/split/concat edges declare; - what the current phase allows in terms of leakage, sample alignment and artifacts.

DAG-ML must therefore request a compatibility plan before execution. In the proposed contract, it is the method`DataPlanner.resolve()`side ML_DATA, called by DAG-ML with the`ModelInputSpec`of the model:

```python
plan = data_planner.resolve(
    dataset=dataset,
    sources=("nir", "photo_front", "photo_side", "genotype", "weather", "metadata"),
    model_input=random_forest_adapter.input_spec(RandomForestRegressor()),
    policy=fusion_policy,
)
```

A`DataPlan`may contain:

-`direct`: the block is already compatible; -`flatten`: dense spectral/tabular to`(samples, features)`; -`transpose`:`channels_first`<->`channels_last`; -`concat_features`: horizontal concate of aligned dense sources; -`stack_channels`: stack of compatible sources as channels; -`pad_or_truncate`: only with explicit policy; -`dict_input`or`list_input`: for multi-input models; -`featurize_required`: you need an upstream node which converts image/graph/text into features; -`error`: no sure adaptation.

Important rule: DAG-ML must never silently flatten/concat a complex source. Implicit conversions should be limited to simple dense cases, typically digital spectra/arrays.

Exemples:

- classic sklearn:`flat_features`request, rank 2, dense modalities. For multi-source spectra, DAG-ML can plan`concat_features`if samples are aligned. - NN Conv1D spectral: request rank 3. DAG-ML can provide`(samples, channels/processings, features)`or`(samples, features, channels)`depending on framework. - multi-modal model:`dict_input=True`request; DAG-ML provides`{nir: tensor, image: tensor, metadata: dataframe}`without concatenation. - image model: refuses a raw spectrum unless an explicit adapter transforms the spectrum into an image/embedding. - graph neural network: refuses dense concat, requests a`GraphBatch`with adjacency/edge features.

### 2.8 Ce qui appartient a DAG-ML vs ML_DATA

DAG-ML doit posseder:

- the graph, nodes, ports and edge contracts; - the orchestration of`DataPlanner`and the refusal of plans that violate ML invariants; - phases`FIT_CV`,`SELECT`,`REFIT`,`PREDICT`,`EXPLAIN`; - ML invariants: OOF, no-leakage, fold alignment, refit provenance; - the decision to plan a`AdapterNode`,`JoinNode`,`SplitNode`or to refuse; - cache, traces, lineage and generic artifacts; - the`OperatorAdapter`interface which declares the expected inputs.

ML_DATA doit posseder:

- concrete storage of sources; - the alignment of samples between sources (`inner`,`left`,`outer`, presence masks); - descriptors of sources, axes, modalities, dtypes, units; - the representation conversions that the data can do without losing meaning; - padding/snack/ragged batch policies; - the samples/features/targets/metadata views; - domain primitives: NIRS wavelengths, source names, signal types, image size, graph schema, time axis.

The sharing contract is:

-`SourceDescriptor`; -`DataBlock`; -`DataView`; -`ModelInputSpec`; -`FusionPolicy`; -`DataPlan`; -`PresenceMask`for missing sources; -`AxisSpec`to tell what each dimension means.

Cette separation evite deux erreurs:

1. put multi-source merge semantics in each model; 2. hide reshape/concat decisions in`Dataset.x()`.

### 2.9 Concrete case: heterogeneous early fusion towards RandomForest

Target example:

```text
sample_id
  - 1 spectre NIRS
  - 2 photos RGB
  - 1 patrimoine genetique
  - 1 serie meteo multivariable
  - metadata
  -> RandomForest
```

A`RandomForest`sklearn does not consume “multi-source”. It consumes a dense or sparse numerical array of the form:

```text
(n_samples, n_features)
```

So early fusion is only possible if each source is transformed into`FeatureTable`aligned by`sample_id`, then join horizontally:

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

- the model adapter`RandomForestAdapter`declares`ModelInputSpec`: accepts`tabular_numeric`, rank 2, aligned samples, no raw images/graphs/sequences; -`ML_DATA`declares each source: modality, axes, units, sample key, granularity, presence mask, native representations; -`ML_DATA`also declares available conversions to`tabular_numeric`, but does not always choose alone; - DAG-ML plans`FeaturizerNode`/`EncoderNode`/`JoinNode`/`ImputerNode`to satisfy the model contract; - the user or an explicit policy chooses conversions when they are not obvious.

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

Automatic merging is only reasonable if all adapters to`tabular_numeric`are declared and if the policy allows it:

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

If`allow_lossy_adapters=False`, DAG-ML must stop and ask for an explicit choice for:

- raw image -> embedding, flattened pixels, CNN fine-tune, color/texture features; - genotype -> raw dosage, selection of variants, PCA, PRS, embedding; - weather sequence -> time windows, lags, stats, upstream sequence model; - categorical metadata -> one-hot, target encoding, hashing, embeddings.

This is not plumbing: these are modeling hypotheses. DAG-ML can automate assembly, not silently invent these hypotheses.

Release contract from an adapter to RandomForest:

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

- same`sample_id`or explicit alignment; - no leaks: adapters fit on train only; - stable train/predict scheme; - columns namespaced by source; - NaN/missing managed according to policy; - reasonable size or warning before dimensional explosion.

For this case,`ML_DATA`must therefore provide at least:

1. a catalog of sources with modality, axes, units, granularity and sample key; 2. a sample/source alignment engine with`PresenceMask`; 3. declarative adapters towards target representations (`tabular_numeric`,`tensor_image`,`sequence`,`graph_batch`,`dict_input`); 4. missing/padding/snack policies by modality; 5. a schema registry to guarantee that the predict uses exactly the same columns/adaptors as the train; 6. cost/dimension/lossiness metadata so that DAG-ML knows if a conversion can be automatic.

### 2.10 Structural points not to forget

#### 2.10.1 Repetitions, several X for one Y, and split unit

The data model must not confuse observation, logical sample and target. In NIRS, we often have several spectra for the same sample and a single Y value. In multimodal, we can have two images, several weather measurements, several spectra, and a single Y.

It is therefore necessary to separate:

-`observation_id`: physical line in a source; -`sample_id`: logical unit used to align the sources; -`target_id`: unit bearing the Y; -`group_id`or`entity_id`: leakage/split unit, for example plant, patient, batch, plot; -`origin_id`: original sample when an observation is increased.

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

DAG-ML rule: the split must be done at the level which avoids leaking. If multiple Xs share a Y or`group_id`, they must fall into the same fold when`split_unit="target"`or`split_unit="group"`.

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

Then a prediction can be aggregated into twin`sample`or`group`, but the raw OOF remains traceable.

#### 2.10.2 Augmentation and OOF

The augmentation is a data adapter, not a free mutation of the dataset.

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

- increases in a sample train remain in the fold train; - no increase derived from a validation sample can enter the fold train; - OOF predictions must be produced for validation samples/origins, not for augmented copies seen in the stream; - if a meta-model consumes predictions,`PredictionJoin`must use the OOF predictions of the origins, then only optionally propagate to the train augmentations; - in final refit, the increase can be reapplied to the entire train, but the artifacts must be marked`phase=REFIT`.

#### 2.10.3 Seeding and reproducibility

The seed must be hierarchical and serialized. A simple global`random_state=42`is not enough when we have generators, folds, augmentation, tuning and parallelism.

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

Rule: each stateful node receives a seed derived from `(root_seed, run_id, variant_id, node_id, fold_id, trial_id)`. The effective seed is persisted in the `LineageRecord`, not just the root seed.

#### 2.10.4 Operateurs custom lies a la data

Operators can request auxiliary information from sources: wavelengths, time coordinates, graph schema, image metadata, variant annotations. In nirs4all, some transformers need`wavelengths=`. This is not NIRS special: it is a typical auxiliary dependency.

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

#### 2.10.5 Serialization and replay

Everything needed for replay predict/refit must be serializable. It is necessary to distinguish between JSON specs and binary payloads.

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

-`GraphSpec`,`DataPlan`,`DatasetSchema`,`ModelInputSpec`must be JSON/YAML serializable; - fitted objects, models, encoders, PCA, imputers, CNN embeddings go to the artifact store; - each custom plugin must provide`type_id`,`version`, and serializer/deserializer; - the predict refuses if the current schema does not match the`schema_fingerprint`, unless explicit migration; - tabular columns and their order are a schema artifact, not an implicit convention.

#### 2.10.6 Finetuning and hyperparameter search

Finetuning should be a DAG node, not a special mode hidden in the controller model. It can tune the model parameters, but also those of the data adapters.

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

- the trials are executed with the same OOF constraints as normal training; - a stateful tune adapter, for example PCA genotype or image embedding, is only fit on the fold train; - the best`DataPlan`+`ModelParams`becomes the refit entry; - trial seeds, params, scores and output patterns are persisted.

#### 2.10.7 Reifiable DAG: a node can be a DAG

DAG-ML must be reifiable: a DAG is a value, serializable, inspectable, versionable and reusable as a node in another DAG.

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

-`inline`: the sub-DAG is expanded in the parent DAG for fine scheduling/cache; -`opaque`: the sub-DAG is executed as a black box with its own artifacts; -`auto`: the planner chooses.

Cela permet:

- a complete preprocessing pipeline as a node; - an ensemble model as a sub-DAG; - a pre-trained image featurizer + pooling as a node; - an OOF stacker as node; - a`nirs4all-methods`recipe versioned as a reifiable component.

A sub-DAG must declare its`GraphInterface`; otherwise it is not composable.

### 2.11 Performance

Les optimisations a formaliser:

- lazy enumeration of variants to avoid materializing large`_cartesian_`; -`DataView`immutable + copy-on-write blocks rather than deep copy of large arrays; - node cache by`(operator, params, input lineage, phase)`hash; - topological scheduler with barriers for`Join`,`Split`,`Refit`; -`PredictionStore`separate column/array so as not to copy large arrays into the metadata; - compatibility plans hidden by`(source descriptors, model input spec, policy)`; - snack/padding as late as possible, just before the model; - resource hints by node (`cpu`,`gpu`,`thread_safe`,`nested_parallelism`); - parallel execution at the variant, branch or fold level, but with a unique rule for preventing nested parallelism.

### 2.12 Migration pragmatique

Ordre recommande:

1. Extract the protocols (`MLDataset`,`DataPlanner`,`PredictionStore`,`ArtifactStore`,`OperatorAdapter`) without changing the runtime. 2. Add a read-only nirs4all DSL ->`GraphSpec`compiler. 3. Extract from`SpectroDataset`a`SpectroDatasetConnector`which exposes`SourceDescriptor`,`DataBlock`and`2d/3d/3d_transpose`conversions. 4. Replace implicit`dataset.x(layout=..., concat_source=...)`calls in controllers with`DataPlanner.resolve()`+`DataPlan`. 5. Gradually replace`PipelineConfigs`with`SearchSpace`DAG-ML, keeping the syntax. 6. Bring`StepParser`/`ControllerRouter`to DAG-ML registry. 7. Formalize`FoldSet`,`PredictionBlock`,`OOFJoin`and branch/merge. 8. Move selection/refit in DAG-ML. 9. Keep NIRS-specific controllers in`nirs4all`, not in DAG-ML.

## 3. Alternative existante?

To my knowledge, there is no alternative that matches exactly.

- scikit-learn covers`Pipeline`,`ColumnTransformer`, transformer cache and OOF stacking via`StackingRegressor`/`StackingClassifier`, but not a multi-branch DAG with data connectors, arbitrary OOF merge, artifact lineage, refit by topology and explicit multi-source/multi-modal compatibility. - Kedro formalizes nodes, inputs/outputs and topological resolution, but it is more of a data pipeline framework. It does not natively support fine ML contracts: folds, OOF, leakage checks, final refit, replay predict/explain by artifacts. - Dagster/Prefect are strong for orchestration, observability and assets/runs. They are too heavy and too external to be the in-process heart of an interactive ML pipeline. - Hamilton is elegant for generating a dataflow from typed Python functions, but its model is less suited to stateful sklearn-like operators, CV/refit phases and OOF predictions like first-class edge. - Dask/Ray can serve as a parallel execution backend, not a semantic ML model.

Conclusion: DAG-ML doit etre un moteur custom, mais il peut reprendre des idees:

- topological planning Kedro/Hamilton style; - assets/artifacts and lineage Dagster style; - optional local scheduler like Dask; - sklearn conventions for fit/transform/predict, params and stacking; - OOF/refit contracts specific to nirs4all.

References externes consultees:

- scikit-learn Pipeline/composite estimators:https://scikit-learn.org/stable/modules/compose.html- scikit-learn stacking:https://scikit-learn.org/stable/modules/ensemble.html#stacking- Kedro Pipeline object:https://docs.kedro.org/en/stable/build/pipeline_introduction/- Dagster assets:https://docs.dagster.io/guides/build/assets- Apache Hamilton functions/nodes/dataflow:https://hamilton.dagworks.io/en/latest/concepts/node/- Dask delayed:https://docs.dask.org/en/stable/delayed.html- Ray DAG API:https://docs.ray.io/en/latest/ray-core/ray-dag.html- Prefect flows:https://docs.prefect.io/v3/concepts/flows
