# DAG-ML Specification v1

Status: design v1, ready for implementation.
Companion library: ML_DATA (see `ml_data_specification_v1.md`). DAG-ML owns the
execution graph, ML phases and invariants; ML_DATA owns the data layer.

The document is self-sufficient: every type, protocol and algorithm referenced
in a section is defined here or in `ml_data.contract` (the shared module).
Pseudocode appears whenever an algorithm has multiple branches.

---

## 1. Mission, perimeter, frontier

### 1.1 Mission

DAG-ML is a local, in-process ML engine that formalises:

- compilation of a DSL pipeline into an explicit DAG;
- enumeration of variants from a search space;
- multi-phase execution (`COMPILE`, `PLAN`, `FIT_CV`, `SELECT`, `REFIT`, `PREDICT`, `EXPLAIN`);
- CV / OOF / stacking with leakage-free invariants;
- refit on full data using a `SelectedGraph`;
- predict / explain replay from an `ExecutionBundle`;
- artifact / cache / lineage / trace stores;
- node-level parallelism (variant, branch, fold) via pluggable schedulers;
- a stable adapter contract for operators (sklearn, PyTorch, TF, Keras, LightGBM, XGBoost, custom);
- multi-source / multi-modal compatibility through the ML_DATA contract.

DAG-ML is extracted and generalised from the `nirs4all` runtime. It contains no
NIRS-specific logic, no `SpectroDataset`, no wavelengths primitives. Domain
knowledge enters via plugins.

### 1.2 Non-perimeter

DAG-ML explicitly **does not** own:

- source storage, file formats, parsing, schema inference;
- modality descriptors, axis units, conversions between representations;
- sample alignment between sources (`inner` / `left` / `outer`);
- raw-data caching, file-backed materialisation, ragged collation.

All of the above belong to ML_DATA. DAG-ML interacts via:
`MLDataset`, `DataPlanner`, `SourceDescriptor`, `DataBlock`, `FeatureTable`,
`TargetBlock`, `DataView`, `DataPlan`, `ModelInputSpec`, `FusionPolicy`,
`AlignmentPolicy`, `AdapterContext`, `FittedAdapter`, `SampleRelation`,
`AuxInputSpec`, `SerializableRef`.

### 1.3 Frontier diagram

```text
+------------------------- Application -----------------------------+
| nirs4all, tabular-ml, image-ml, multimodal-ml ...                 |
| - registers operator adapters                                     |
| - registers DataTypePlugin / RepresentationAdapter (ML_DATA side) |
| - builds a DSL pipeline, calls dagml.run / dagml.predict          |
+-------------------------------------------------------------------+
       |                                              |
       | dsl pipeline / config                        | dataset path / handle
       v                                              v
+--------------------- DAG-ML ----------------------+ +------- ML_DATA ----+
| Compiler   GraphSpec / NodeSpec / EdgeSpec        | | DatasetSchema      |
| SearchSpace, Enumerator, Planner                  | | SourceDescriptor   |
| Phases: COMPILE PLAN FIT_CV SELECT REFIT PREDICT  | | DataBlock,         |
|         EXPLAIN                                   | | DataView,          |
| OperatorRegistry, ModelAdapter, TunerAdapter      | | DataPlanner,       |
| Scheduler (sequential / loky / ray)               | | AdapterRegistry,   |
| FoldSet, PredictionStore, OOFJoin                 | | FeatureJoiner,     |
| ArtifactStore, CacheStore, LineageRecorder        | | BatchCollator      |
| Selection, Refit, ExecutionBundle, Replay         | |                    |
+---------------------------------------------------+ +--------------------+
       ^                                              ^
       |  ml_data.contract types only --------------> |
       |                                              |
       +----------------------------------------------+
```

What crosses the frontier (DAG-ML -> ML_DATA): `DataView`, `AdapterContext`,
`FusionPolicy`, `ModelInputSpec`, `AuxInputSpec`, `SerializableRef`.
What crosses the other way: `DataBlock`, `FeatureTable`, `TargetBlock`,
`DataPlan`, `FittedAdapter`, `SampleRelation`, `PresenceMask`.

What never crosses: DAG-ML never passes a `FoldSet`, `PredictionBlock`, fold
metadata, `LineageRecord`, `CacheKey` or `ArtifactRef` to ML_DATA.

---

## 2. Graph model

### 2.1 NodeKind

```python
from enum import Enum

class NodeKind(str, Enum):
    TRANSFORM        = "transform"          # X-side stateless or stateful transform
    Y_TRANSFORM      = "y_transform"        # y-side transform with invert at predict
    SPLIT            = "split"              # CV / holdout splitter -> FoldSet
    MODEL            = "model"              # supervised model (fit / predict / explain)
    FORK             = "fork"               # branch fan-out (duplication or separation)
    MAP              = "map"                # apply a subgraph to each branch
    FEATURE_JOIN     = "feature_join"       # horizontal join of FeatureTables
    PREDICTION_JOIN  = "prediction_join"    # OOF prediction merge (stacking)
    MIXED_JOIN       = "mixed_join"         # mixed features + predictions join (UC12)
    SOURCE_JOIN      = "source_join"        # multi-source fusion (delegates to ML_DATA)
    TAG              = "tag"                # mark samples (non-removal)
    EXCLUDE          = "exclude"            # remove samples from training only
    AUGMENTATION     = "augmentation"       # data augmentation node
    ADAPTER          = "adapter"            # ML_DATA RepresentationAdapter wrapper
    AGGREGATOR       = "aggregator"         # observation -> sample/group aggregation
    RESTRUCTURE      = "restructure"        # repetition -> sources / preprocessings
    TUNER            = "tuner"              # hyperparameter search node
    SUBGRAPH         = "subgraph"           # reusable sub-DAG
    CHART            = "chart"              # visualisation / side-effect, no edges out
```

### 2.2 Port and edge primitives

```python
from dataclasses import dataclass, field
from typing import Any, Literal

PortKind = Literal["data", "target", "prediction", "artifact", "metric", "control"]

@dataclass(frozen=True)
class PortSpec:
    name: str
    kind: PortKind
    representation: str | None = None       # RepresentationId from ML_DATA
    cardinality: Literal["one", "many", "optional"] = "one"
    description: str = ""

@dataclass(frozen=True)
class PortSchema:
    inputs: tuple[PortSpec, ...] = ()
    outputs: tuple[PortSpec, ...] = ()

@dataclass(frozen=True)
class PortRef:
    node_id: str
    port_name: str

@dataclass(frozen=True)
class EdgeContract:
    kind: PortKind
    representation: str | None = None
    requires_oof: bool = False              # only meaningful for prediction edges
    requires_fold_alignment: bool = False
    propagates_lineage: bool = True

@dataclass(frozen=True)
class EdgeSpec:
    source: PortRef
    target: PortRef
    contract: EdgeContract
```

### 2.3 NodeSpec and GraphSpec

```python
@dataclass(frozen=True)
class NodeSpec:
    id: str
    kind: NodeKind
    operator: Any | None                    # opaque payload (sklearn estimator, dict spec, ...)
    params: dict[str, Any] = field(default_factory=dict)
    ports: PortSchema = field(default_factory=PortSchema)
    metadata: dict[str, Any] = field(default_factory=dict)
    resource_hints: "ResourceHints | None" = None
    aux_inputs: tuple["AuxInputSpec", ...] = ()
    seed_label: str | None = None           # used to derive a deterministic seed

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
    search_space: "SearchSpace | None" = None
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class SubgraphNodeSpec:
    id: str
    graph: GraphSpec | "SerializableRef"
    input_mapping: dict[str, str]           # parent_port -> sub_port
    output_mapping: dict[str, str]
    inline_policy: Literal["inline", "opaque", "auto"] = "auto"
```

Invariants:

- `NodeSpec.id` is unique inside its `GraphSpec`.
- For every `EdgeSpec(source=PortRef(a, p), target=PortRef(b, q))`, the
  source node owns an output port named `p`, the target node owns an input
  port named `q`, and their declared representations are compatible (equal or
  bridgeable via ML_DATA adapters, decided at PLAN).
- Cycles are forbidden in `edges` (acyclicity is checked by the validator;
  recursion is achieved through `SubgraphNodeSpec`).
- A `GraphSpec` with a non-empty `search_space` is a *family*; concrete
  variants are produced by enumeration.

### 2.4 Resource hints

```python
@dataclass(frozen=True)
class ResourceHints:
    cpu: int | None = None                  # logical cores requested by the node
    gpu: int | None = None                  # number of gpus (0 = explicitly no gpu)
    gpu_memory_mb: int | None = None
    memory_mb: int | None = None
    thread_safe: bool = True
    nested_parallelism: Literal["forbid", "allow", "auto"] = "auto"
    timeout_seconds: float | None = None
```

Resource hints are read by the scheduler; nodes with conflicting hints (e.g.
`thread_safe=False`) cannot share a thread.

---

## 3. Phases

### 3.1 Phase enum and table

```python
class Phase(str, Enum):
    COMPILE = "compile"
    PLAN    = "plan"
    FIT_CV  = "fit_cv"
    SELECT  = "select"
    REFIT   = "refit"
    PREDICT = "predict"
    EXPLAIN = "explain"
```

| Phase    | Inputs                                                       | Outputs                                  | Invariants                                                                 | Side effects                                                  |
|----------|--------------------------------------------------------------|------------------------------------------|----------------------------------------------------------------------------|---------------------------------------------------------------|
| COMPILE  | DSL pipeline + registry                                      | `GraphSpec`, `SearchSpace`               | acyclic, port arity checked, no operator instantiation                     | none (pure)                                                   |
| PLAN     | `GraphSpec`, variant choice, `MLDataset` schema, policies    | `ExecutionPlan`                          | all edge contracts solvable, all DataPlans resolvable or refused           | none (pure; may consult `DataPlanner.resolve`)                |
| FIT_CV   | `ExecutionPlan`, `MLDataset`, `FoldSet`                      | fitted artifacts, OOF `PredictionBlock`s | OOF, no leakage, fold alignment, augmentation rules                        | writes `ArtifactStore`, `PredictionStore`, `LineageRecorder`  |
| SELECT   | CV result, `RankingPolicy`                                   | `SelectedGraph`                          | scores are computed only from OOF / val predictions                        | writes a `SelectionRecord`                                    |
| REFIT    | `SelectedGraph`, `MLDataset`                                 | `RefitArtifacts`, `PredictionBlock`s     | uses full train; produces `fold_id="final"`; same data contract as CV      | writes `ArtifactStore`, `PredictionStore`, `LineageRecorder`  |
| PREDICT  | `ExecutionBundle`, new `MLDataset`                           | `PredictionBlock`s                       | schema fingerprint matches; no fit calls; uses stored fitted adapters      | reads artifacts; may write to a derived store                 |
| EXPLAIN  | `ExecutionBundle`, new `MLDataset`, `ExplainerAdapter`       | explanation tensors                      | replays PREDICT path; calls explainer; no fit calls                        | reads artifacts; writes explanation outputs                   |

### 3.2 Phase flow diagram

```text
DSL pipeline
    |
    v
[COMPILE]  GraphSpec + SearchSpace ----------------------+
    |                                                    |
    v                                                    |
[enumerate variants]                                     |
    |                                                    |
    v                                                    |
[PLAN]  ExecutionPlan_v (per variant)                    |
    |                                                    |
    v                                                    |
[FIT_CV]  fit folds -> Predictions(OOF) + Artifacts      |
    |                                                    |
    +--------------------+-------------------------------+
                         v
                     [SELECT]  SelectedGraph
                         |
                         v
                     [REFIT]   fit on full train -> RefitArtifacts
                         |
                         v
                     [export]  ExecutionBundle (graph + plan + artifacts + schema)
                         |
                         v
            new dataset -> [PREDICT] / [EXPLAIN] (replay)
```

### 3.3 Phase contract for nodes

Every `OperatorAdapter` declares `supports_phase(phase)`. A node that supports
`PREDICT` must reproduce its training-time transformation deterministically
from a `FittedArtifact`. A node that does not support `PREDICT` (e.g.
`AugmentationNode` with `apply_to="train_only"`) is **skipped** at PREDICT.

---

## 4. DSL compilation

### 4.1 Accepted DSL

The compiler accepts the nirs4all DSL surface (list of dicts and bare
operators) and produces a `GraphSpec`. The DSL surface:

```python
pipeline = [
    StandardScaler(),                                     # bare operator -> TransformNode
    {"y_processing": MinMaxScaler()},                     # YTransformNode
    {"tag": YOutlierFilter(method="iqr")},                # TagNode
    {"exclude": [YOutlierFilter(), XOutlierFilter()],
     "mode": "any"},                                      # ExcludeNode
    ShuffleSplit(n_splits=5, test_size=0.2),              # SplitNode -> FoldSet
    {"sample_augmentation": GaussianAdditiveNoise(),
     "policy": {"apply_to": "train_only"}},               # AugmentationNode
    {"branch": [                                          # ForkNode + MapNode
        [SNV(), PLSRegression(10)],
        [MSC(), {"_or_": [PLSRegression(15), Ridge()]}],
    ]},
    {"merge": "predictions"},                             # PredictionJoinNode
    {"model": Ridge()},                                   # ModelNode (meta-stacker)
    {"_cartesian_": [
        {"_or_": [SNV(), MSC(), Detrend()]},
        {"_or_": [PLSRegression(10), Ridge()]},
    ]},                                                   # SearchSpace
    {"_grid_": {"n_components": [5, 10, 20]}},            # SearchSpace
]
```

### 4.2 Compiler stages

| Stage     | Input                      | Output                                  | Notes                                                                            |
|-----------|----------------------------|-----------------------------------------|----------------------------------------------------------------------------------|
| Parser    | raw DSL                    | `ParsedStep[]` (typed)                  | normalises bare operators, dict keys, splits aliases (e.g. `xxx_params`)         |
| Validator | `ParsedStep[]`              | `ParsedStep[]`                          | rejects unknown keywords; checks arity; merges `xxx`/`xxx_params`                |
| Lowering  | `ParsedStep[]`              | `NodeSpec[]` + `EdgeSpec[]`             | maps keywords to `NodeKind`; assigns ids; resolves operator -> adapter           |
| SearchSpace extractor | `NodeSpec[]` + lowered  | `SearchSpace`                           | collects `_or_` / `_grid_` / `_cartesian_` / etc. into slots                     |
| Assembler | nodes + edges + space      | `GraphSpec`                             | computes `GraphInterface`, runs acyclicity / port-arity checks                   |

### 4.3 Lowering rules (per DSL keyword)

| DSL keyword                | Produces                                                  | Notes                                                            |
|----------------------------|-----------------------------------------------------------|------------------------------------------------------------------|
| bare operator              | `TransformNode` (kind=TRANSFORM)                          | adapter resolved via registry                                    |
| `{"preprocessing": op}`    | `TransformNode`                                           | explicit form of bare operator                                   |
| `{"y_processing": op}`     | `YTransformNode` (kind=Y_TRANSFORM)                       | invertible at PREDICT (`inverse_transform`)                      |
| `{"tag": filter}`          | `TagNode`                                                 | adds tag column to dataset view                                  |
| `{"exclude": [...] }`      | `ExcludeNode` with `mode in {any, all}`                   | applies only on FIT_CV / REFIT views                             |
| `Splitter()`               | `SplitNode` -> `FoldSet`                                  | sample_ids in folds, never positions                             |
| `{"sample_augmentation"}`  | `AugmentationNode` with `AugmentationPolicy`              | leakage rules in section 9                                       |
| `{"feature_augmentation"}` | sub-pipeline over multiple `processings` with `concat`    | lowered into `AdapterNode`s + `FeatureJoinNode`                  |
| `{"concat_transform": ...}`| `FeatureJoinNode` of transformed branches                 |                                                                  |
| `{"rep_to_sources"}`       | `RestructureNode` (kind=RESTRUCTURE, mode="to_sources")   | groups repetitions into named sources; emits a `SampleRelation` artifact for replay |
| `{"rep_to_pp"}`            | `RestructureNode` (kind=RESTRUCTURE, mode="to_processings")| groups repetitions into preprocessing channels of one source                          |
| `{"aggregate": {...}}`     | `AggregatorNode` (kind=AGGREGATOR)                        | observation -> sample/group reduction; `method`, `level`, `keep_observation_predictions` |
| `{"branch": [...]}`        | `ForkNode` (duplication) + `MapNode`                      | one `MapNode` per branch path; subgraph per branch               |
| `{"branch": {"by_X": ...}}`| `ForkNode` (separation, mode=by_metadata/by_tag/by_source)| disjoint sample subsets; merge typically `concat`                |
| `{"merge": "predictions"}` | `PredictionJoinNode`                                      | requires OOF (section 8)                                         |
| `{"merge": "features"}`    | `FeatureJoinNode`                                         | horizontal concat of FeatureTables                               |
| `{"merge": "concat"}`      | `FeatureJoinNode` in separation mode (reassembly)         |                                                                  |
| `{"merge": {"sources":..}}`| `SourceJoinNode`                                          | delegates to ML_DATA fusion                                      |
| `{"model": op}`            | `ModelNode` (kind=MODEL)                                  | OOF emitted per fold                                             |
| `{"_or_": [...]}`          | adds a `Slot(choice, n alternatives)` to `SearchSpace`    | one slot per occurrence                                          |
| `{"_range_": [a,b,n]}`     | `Slot(linear, n values)`                                  |                                                                  |
| `{"_log_range_": [...]}`   | `Slot(log, n values)`                                     |                                                                  |
| `{"_grid_": {...}}`        | `Slot(grid, list_of_combinations)`                        |                                                                  |
| `{"_cartesian_": [...]}`   | `Slot(cartesian, list_of_pipelines)`                      | each pipeline is itself a sub-DSL                                |
| `{"_zip_": {...}}`         | `Slot(zip, list_of_paired_dicts)`                         | preserves ordering, refuses unequal lengths                      |
| `{"_chain_": [...]}`       | `Slot(chain, list_of_configs)`                            | order-preserving alternatives                                    |
| `{"_sample_": {...}}`      | `Slot(sample, distribution_spec)`                         | random sampling at enumeration time                              |
| `Chart(...)` or `{"chart": ...}` | `ChartNode`                                       | side-effect node, no outgoing edges                              |

### 4.4 SearchSpace

```python
SlotKind = Literal[
    "choice", "linear", "log", "grid", "cartesian", "zip", "chain", "sample",
]

@dataclass(frozen=True)
class SlotSpec:
    id: str                                # path identifier, e.g. "node:branch:0:model"
    kind: SlotKind
    values: tuple[Any, ...] | None = None  # for choice / linear / log / chain
    params: dict[str, Any] = field(default_factory=dict)
                                           # grid_dict / cartesian_paths / zip_dict / sample_spec

@dataclass(frozen=True)
class SearchSpace:
    slots: tuple[SlotSpec, ...] = ()
    priority: tuple[str, ...] = ()         # slot ids ordered by enumeration priority
    seed: int | None = None
    max_variants: int | None = None
```

### 4.5 Variant enumeration (lazy)

```text
def enumerate(space: SearchSpace, *, lazy: bool = True) -> Iterator[Variant]:
    # 1. lower each slot into an iterable of choices
    slot_iters: dict[str, Callable[[], Iterator[Any]]] = {}
    for slot in space.slots:
        slot_iters[slot.id] = _iter_for_slot(slot, seed=space.seed)

    # 2. compose
    if space.slots and any(s.kind == "cartesian" for s in space.slots):
        # cartesian slots can themselves carry sub-spaces; expand recursively
        composite = _compose_cartesian(slot_iters, space.slots)
        gen = composite()
    else:
        # default: cartesian over independent slots
        gen = _independent_product(slot_iters, space.priority or [s.id for s in space.slots])

    # 3. respect max_variants
    seen = 0
    for choice_map in gen:
        if space.max_variants and seen >= space.max_variants:
            return
        yield Variant(slot_choices=choice_map)
        seen += 1

def _iter_for_slot(slot, seed):
    if slot.kind == "choice":
        return iter(slot.values)
    if slot.kind == "linear":
        a, b, n = slot.params["from"], slot.params["to"], slot.params["num"]
        return iter(np.linspace(a, b, n))
    if slot.kind == "log":
        a, b, n = slot.params["from"], slot.params["to"], slot.params["num"]
        return iter(np.logspace(np.log10(a), np.log10(b), n))
    if slot.kind == "grid":
        return iter(_dict_product(slot.params["grid"]))
    if slot.kind == "zip":
        return iter(_zip_dict(slot.params["zip"]))
    if slot.kind == "chain":
        return iter(slot.values)
    if slot.kind == "sample":
        return _sample_iter(slot.params, seed)
```

Lazy enumeration is critical when `_cartesian_` explodes (typical real cases:
10k+ variants).

---

## 5. Planning

### 5.1 PlanningContext

```python
@dataclass(frozen=True)
class PlanningContext:
    variant: "Variant"
    dataset_schema: "DatasetSchema"        # ml_data.contract
    policy: "PlanningPolicy"
    registry: "OperatorRegistry"
    data_planner: "DataPlanner"
    cache: "PlanCacheStore | None" = None

@dataclass(frozen=True)
class PlanningPolicy:
    fusion: "FusionPolicy"
    alignment: "AlignmentPolicy"
    adaptation: "AdaptationPolicy"
    augmentation: "AugmentationPolicy | None" = None
    split: "SplitPolicy | None" = None
    ranking: "RankingPolicy | None" = None
    seed: int | None = None
```

### 5.2 NodePlan and ExecutionPlan

```python
@dataclass(frozen=True)
class NodePlan:
    node_id: str
    operator: Any                          # concretised operator (after slot substitution)
    adapter_id: str                        # OperatorAdapter id
    phases: tuple[Phase, ...]              # phases where this node runs
    data_plan: "DataPlan | None" = None    # if the node consumes data (ml_data)
    aux_inputs: tuple[tuple["AuxInputSpec", str], ...] = ()
                                           # (spec, source_id)
    seed_label: str | None = None
    resource_hints: ResourceHints | None = None
    fingerprint: str = ""                  # SHA256 of (operator_id, params_canonical)

@dataclass(frozen=True)
class ExecutionPlan:
    variant_id: str
    graph: GraphSpec
    node_plans: dict[str, NodePlan]
    topological_order: tuple[str, ...]
    schema_fingerprint: str
    plugin_versions: dict[str, str] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
```

### 5.3 Planning algorithm

```text
def plan(graph: GraphSpec, ctx: PlanningContext) -> ExecutionPlan:
    cache_key = (graph.id, ctx.variant.fingerprint, ctx.dataset_schema.fingerprint,
                 canonical_json(ctx.policy))
    if ctx.cache and ctx.cache.has(cache_key):
        return ctx.cache.get(cache_key)

    # 1. substitute slot choices into nodes
    nodes_concrete = _substitute_slots(graph.nodes, ctx.variant)

    # 2. topological sort
    topo = topological_sort(nodes_concrete, graph.edges)

    # 3. per-node planning
    node_plans: dict[str, NodePlan] = {}
    for nid in topo:
        node = nodes_concrete[nid]
        adapter = ctx.registry.resolve(node)
        if adapter is None:
            raise PlanningError(f"no adapter for {nid} ({node.kind}, {type(node.operator)})")

        # ports compatibility (DAG side)
        _validate_edge_contracts(graph, nid, node_plans)

        # data plan (ML_DATA side) -- only if node declares a ModelInputSpec
        dp = None
        if hasattr(adapter, "input_spec"):
            model_input = adapter.input_spec(node.operator)
            sources = _sources_for_node(nid, graph, ctx.dataset_schema)
            try:
                dp = ctx.data_planner.resolve_from_schema(
                    schema=ctx.dataset_schema,
                    sources=sources,
                    model_input=model_input,
                    policy=ctx.policy.fusion,
                )
            except DatasetRequiredForPlanning:
                # An adapter declared it cannot be picked without inspecting
                # the data. Defer planning to FIT_CV: leave `dp=None` here
                # and rerun `data_planner.resolve(dataset, ...)` inside the
                # fold scope before fit.
                dp = None
            if dp is not None and dp.requires_user_choice:
                raise PlanningError(f"unsafe auto-plan for {nid}: {dp.requires_user_choice}")

        aux = _resolve_aux_sources(node.aux_inputs, ctx.dataset_schema)

        node_plans[nid] = NodePlan(
            node_id=nid,
            operator=node.operator,
            adapter_id=adapter.spec.id,
            phases=adapter.supported_phases(),
            data_plan=dp,
            aux_inputs=aux,
            seed_label=node.seed_label or nid,
            resource_hints=node.resource_hints,
            fingerprint=_fingerprint(node.operator, node.params),
        )

    schema_fp = _schema_fingerprint(ctx.dataset_schema, ctx.policy, node_plans)
    plan = ExecutionPlan(
        variant_id=ctx.variant.id,
        graph=graph,
        node_plans=node_plans,
        topological_order=tuple(topo),
        schema_fingerprint=schema_fp,
        plugin_versions=_collect_plugin_versions(node_plans),
    )
    if ctx.cache:
        ctx.cache.put(cache_key, plan)
    return plan
```

### 5.4 Plan cache

```python
class PlanCacheStore(Protocol):
    def has(self, key: tuple[str, ...]) -> bool: ...
    def get(self, key: tuple[str, ...]) -> ExecutionPlan: ...
    def put(self, key: tuple[str, ...], plan: ExecutionPlan) -> None: ...
```

Implementation: in-memory LRU keyed by `(graph_id, variant_fingerprint, schema_fingerprint, policy_hash)`.

---

## 6. Operator adapters

### 6.1 Core protocol

```python
class OperatorAdapter(Protocol):
    spec: "AdapterSpec"                    # not the ML_DATA one; DAG-ML AdapterSpec

    @classmethod
    def matches(cls, node: NodeSpec, operator: Any) -> bool: ...

    def supported_phases(self) -> tuple[Phase, ...]: ...

    def declare_ports(self, node: NodeSpec) -> PortSchema: ...

    def execute(
        self,
        task: "NodeTask",
        ctx: "RunContext",
    ) -> "NodeResult": ...

    def cache_key(self, task: "NodeTask") -> str | None: ...

@dataclass(frozen=True)
class AdapterSpec:
    id: str                                # globally unique adapter id (e.g. "sklearn.transformer")
    kind: NodeKind
    priority: int                          # lower = higher priority in the matches() race
    version: str = "1.0.0"
    capabilities: frozenset[str] = field(default_factory=frozenset)

@dataclass(frozen=True)
class NodeTask:
    node_id: str
    phase: Phase
    plan: NodePlan
    inputs: dict[str, Any]                 # incoming port name -> value
    view: "DataView"
    fold_id: str | int | None
    seed: "SeedContext"

@dataclass(frozen=True)
class NodeResult:
    outputs: dict[str, Any]                # outgoing port name -> value
    artifacts: tuple["ArtifactRef", ...] = ()
    metrics: dict[str, float] = field(default_factory=dict)
    lineage: "LineageRecord | None" = None
```

### 6.2 DataAwareOperatorAdapter

```python
class DataAwareOperatorAdapter(OperatorAdapter, Protocol):
    def input_spec(self, operator: Any) -> "ModelInputSpec": ...
    def aux_inputs(self, operator: Any) -> tuple["AuxInputSpec", ...]: ...
```

A `DataAwareOperatorAdapter` participates in the DataPlanner contract: the
planner resolves a `DataPlan` from the `ModelInputSpec` and the requested
sources.

### 6.3 ModelAdapter

```python
class ModelAdapter(DataAwareOperatorAdapter, Protocol):
    def fit(
        self,
        operator: Any,
        inputs: dict[str, "DataBlock"],
        target: "TargetBlock",
        ctx: "RunContext",
    ) -> Any: ...

    def predict(
        self,
        fitted: Any,
        inputs: dict[str, "DataBlock"],
        ctx: "RunContext",
    ) -> "PredictionPayload": ...

    def predict_proba(
        self,
        fitted: Any,
        inputs: dict[str, "DataBlock"],
        ctx: "RunContext",
    ) -> "PredictionPayload | None": ...

    def feature_importance(self, fitted: Any) -> dict[str, float] | None: ...

    def explain_hooks(self, fitted: Any) -> "ExplainHooks | None": ...
```

### 6.4 OperatorRegistry

```python
class OperatorRegistry(Protocol):
    def register(self, adapter: type[OperatorAdapter]) -> None: ...
    def resolve(self, node: NodeSpec) -> OperatorAdapter | None: ...
    def list_for_kind(self, kind: NodeKind) -> tuple[type[OperatorAdapter], ...]: ...
```

Resolution algorithm:

```text
def resolve(node):
    candidates = [a for a in registry.list_for_kind(node.kind)
                  if a.matches(node, node.operator)]
    if not candidates: return None
    candidates.sort(key=lambda a: a.spec.priority)
    return candidates[0]()
```

### 6.5 Core adapters shipped with DAG-ML

| Adapter id                       | NodeKind        | Operator type accepted                              | Phases supported                  | Notes                                                  |
|----------------------------------|-----------------|-----------------------------------------------------|-----------------------------------|--------------------------------------------------------|
| `sklearn.estimator`              | MODEL           | `sklearn.base.BaseEstimator` with `fit/predict`     | FIT_CV, REFIT, PREDICT            | tabular_numeric input, regression / classification     |
| `sklearn.transformer`            | TRANSFORM       | `sklearn.base.TransformerMixin`                     | FIT_CV, REFIT, PREDICT            | stateful per fold                                      |
| `sklearn.ytransformer`           | Y_TRANSFORM     | scaler / encoder on `y`                              | FIT_CV, REFIT, PREDICT            | inverse-transform at PREDICT                            |
| `sklearn.cv_splitter`            | SPLIT           | `sklearn.model_selection.BaseCrossValidator`        | FIT_CV                            | produces `FoldSet`                                     |
| `pytorch.module`                 | MODEL           | `torch.nn.Module` + adapter config                  | FIT_CV, REFIT, PREDICT, EXPLAIN   | preferred rank=3 input; uses `BatchCollator`           |
| `tensorflow.model`               | MODEL           | `tf.keras.Model` or `tf.Module`                     | FIT_CV, REFIT, PREDICT            | rank=3 default                                          |
| `keras.model`                    | MODEL           | `keras.Model`                                       | FIT_CV, REFIT, PREDICT            | wraps `keras.Model.fit`                                |
| `lightgbm.estimator`             | MODEL           | `lgbm.LGBMRegressor` / `lgbm.LGBMClassifier`        | FIT_CV, REFIT, PREDICT            | tabular_numeric, native categorical hints              |
| `xgboost.estimator`              | MODEL           | `xgb.XGBRegressor` / `xgb.XGBClassifier`            | FIT_CV, REFIT, PREDICT            | idem                                                   |
| `ml_data.adapter`                | ADAPTER         | wraps a `RepresentationAdapter`                     | FIT_CV, REFIT, PREDICT            | bridges from a `DataBlock` to another representation   |
| `dagml.fork.duplication`         | FORK            | `{"branch": [list of subpipelines]}`                | FIT_CV, REFIT, PREDICT            | fan-out                                                |
| `dagml.fork.separation`          | FORK            | `{"branch": {"by_X": ...}}`                         | FIT_CV, REFIT, PREDICT            | disjoint sample subsets                                |
| `dagml.map`                      | MAP             | (internal)                                          | FIT_CV, REFIT, PREDICT            | applies a subgraph per branch                          |
| `dagml.feature_join`             | FEATURE_JOIN    | (internal)                                          | FIT_CV, REFIT, PREDICT            | horizontal table concat                                |
| `dagml.prediction_join`          | PREDICTION_JOIN | (internal)                                          | FIT_CV, REFIT, PREDICT            | OOF stack of `PredictionBlock`s                        |
| `dagml.source_join`              | SOURCE_JOIN     | (internal)                                          | PLAN-time + FIT_CV runtime        | delegates to `DataPlanner`                             |
| `dagml.augmentation`             | AUGMENTATION    | `AugmentationAdapter` from ML_DATA registry         | FIT_CV (train), REFIT (full train)| no-op at PREDICT/EXPLAIN                               |
| `dagml.tag`                      | TAG             | filter expression / function                         | FIT_CV, REFIT, PREDICT            | tags samples, no removal                               |
| `dagml.exclude`                  | EXCLUDE         | filter expression / function                         | FIT_CV, REFIT (train only)        | no-op at PREDICT                                       |
| `dagml.tuner.optuna`             | TUNER           | `OptunaTunerConfig`                                 | FIT_CV                            | optional dep; emits nested fits per trial              |
| `dagml.tuner.ray`                | TUNER           | `RayTuneConfig`                                     | FIT_CV                            | optional dep                                           |
| `dagml.subgraph`                 | SUBGRAPH        | `SubgraphNodeSpec`                                  | inherits                          | inline / opaque                                        |
| `dagml.chart`                    | CHART           | chart config / callable                              | any                               | side-effect only                                       |

---

## 7. Folds and splits

### 7.1 Fold types

```python
SampleIdT = str

@dataclass(frozen=True)
class Fold:
    fold_id: str                           # canonical: "0", "1", ..., "final"
    train_samples: tuple[SampleIdT, ...]
    val_samples: tuple[SampleIdT, ...]
    test_samples: tuple[SampleIdT, ...] = ()
    seed: int | None = None

@dataclass(frozen=True)
class FoldSet:
    folds: tuple[Fold, ...]
    split_unit: Literal["observation", "sample", "target", "group"] = "sample"
    group_key: str | None = None
    origin_aware: bool = True              # if True, origin_ids of augmented rows
                                           # are kept on the same fold side as their origin
```

### 7.2 SplitPolicy

```python
@dataclass(frozen=True)
class SplitPolicy:
    split_unit: Literal["observation", "sample", "target", "group"] = "sample"
    group_key: str | None = None           # column in metadata or SampleRelation
    forbid_origin_cross_fold: bool = True
    seed: int | None = None
    stratify_by: str | None = None
    shuffle: bool = True
```

### 7.3 Key rules

- Fold IDs and sample IDs are **absolute strings**; folds never reference array
  positions. This is the most important invariant for leakage-free CV.
- `FoldSet.split_unit` decides which `SampleRelation` column is used to enforce
  same-fold membership: `sample` (default), `target_ids`, `group_ids`,
  `observation_ids`.
- If `forbid_origin_cross_fold=True`, an augmented observation whose
  `origin_id == s` and a validation row with `sample_id == s` cannot coexist
  across train/val of the same fold. This rule is enforced by DAG-ML at fold
  construction.
- Splitters return `Fold(train_samples, val_samples)` for every fold and
  optionally a global `test_samples` set.
- At REFIT, the executor synthesises a single `Fold("final", train_samples=all,
  val_samples=())` from the union of original train+val (test stays out).
- When `SampleRelation` indicates `granularity="per_sample_repeated"`, every
  observation of the same sample inherits the fold of that sample. The
  `FoldSet` only carries sample-level labels and the executor broadcasts to
  observations at materialisation time.

---

## 8. OOF and stacking (core invariant)

### 8.1 PredictionBlock and PredictionStore

```python
@dataclass(frozen=True)
class PredictionBlock:
    prediction_id: str
    producer_node: str                     # GraphSpec node id
    sample_ids: tuple[SampleIdT, ...]
    observation_ids: tuple[str, ...] | None = None
    target_ids: tuple[str, ...] | None = None
    fold_id: str                           # "0".."K-1" or "final"
    partition: Literal["train", "val", "test", "final"]
    y_pred: Any                            # ndarray (regression) or int array (classif)
    y_proba: Any | None = None
    y_true: Any | None = None
    target_space: str                      # "raw", "scaled", "log", ...
    metrics: dict[str, float] = field(default_factory=dict)
    aggregation_level: Literal["observation", "sample", "target", "group"] = "observation"
    branch_path: tuple[str, ...] = ()
    seed: int | None = None
    artifact_ref: "ArtifactRef | None" = None
    lineage: "LineageRecord | None" = None

class PredictionStore(Protocol):
    def append(self, block: PredictionBlock) -> None: ...
    def find(
        self,
        *,
        producer_node: str | None = None,
        partition: str | None = None,
        fold_id: str | None = None,
        branch_path: tuple[str, ...] | None = None,
        sample_ids: tuple[SampleIdT, ...] | None = None,
    ) -> tuple[PredictionBlock, ...]: ...
    def delete(self, prediction_id: str) -> None: ...
    def flush(self) -> None: ...

@dataclass(frozen=True)
class PredictionPayload:
    y_pred: Any
    y_proba: Any | None = None
    target_space: str = "raw"
    extras: dict[str, Any] = field(default_factory=dict)
```

### 8.2 OOFJoin algorithm

This is the rigour core of DAG-ML.

```text
def oof_join(producers: list[node_id], folds: FoldSet, store: PredictionStore,
             policy: AggregationPolicy, *, partition: str = "val",
             branch_path: tuple[str, ...]) -> FeatureTable:
    """
    Build a meta-feature FeatureTable from the OOF predictions of `producers`.
    Each (sample_id, fold_id) appears exactly once: the prediction comes from
    the fold where the sample was in VAL (not seen at train time).
    """
    # 1. Validate that every producer covers the same sample universe at val
    sample_universe = None
    fold_count_per_producer = {}
    for p in producers:
        blocks = store.find(producer_node=p, partition=partition,
                            branch_path=branch_path)
        if not blocks:
            raise OOFError(f"producer {p} has no '{partition}' predictions")
        sids_p = sorted(set(s for b in blocks for s in b.sample_ids))
        if sample_universe is None:
            sample_universe = sids_p
        else:
            mismatch = set(sample_universe) ^ set(sids_p)
            if mismatch:
                if policy.coverage == "drop_incomplete":
                    sample_universe = sorted(set(sample_universe) & set(sids_p))
                elif policy.coverage == "error":
                    raise OOFCoverageError(producer=p, missing=mismatch)
                elif policy.coverage == "impute":
                    pass  # impute later
        fold_count_per_producer[p] = sorted({b.fold_id for b in blocks})

    # 2. Cross-producer fold alignment
    fc = set(tuple(v) for v in fold_count_per_producer.values())
    if len(fc) > 1:
        if policy.fold_mismatch == "warn":
            warn("producers have different fold structures")
        elif policy.fold_mismatch == "error":
            raise OOFFoldMisalignError(fold_count_per_producer)

    # 3. Build (sample_id -> producer -> y_pred) map
    cell: dict[tuple[str, str], np.ndarray] = {}
    proba_dim: dict[str, int] = {}
    for p in producers:
        blocks = store.find(producer_node=p, partition=partition,
                            branch_path=branch_path)
        for b in blocks:
            arr = b.y_proba if (policy.use_proba and b.y_proba is not None) else b.y_pred
            arr = np.asarray(arr)
            for i, sid in enumerate(b.sample_ids):
                key = (sid, p)
                if key in cell:
                    # duplicate: same sample appears in val of two folds.
                    if policy.duplicate_resolution == "error":
                        raise OOFDuplicateError(sample_id=sid, producer=p,
                                                fold_ids=[b.fold_id, "previous"])
                    if policy.duplicate_resolution == "mean":
                        cell[key] = (cell[key] + arr[i]) / 2
                    elif policy.duplicate_resolution == "last":
                        cell[key] = arr[i]
                else:
                    cell[key] = arr[i]
                if arr.ndim > 1:
                    proba_dim[p] = arr.shape[-1]

    # 4. Build the FeatureTable
    columns: list[str] = []
    source_ids: list[str] = []
    X_cols: list[np.ndarray] = []
    for p in producers:
        if p in proba_dim:
            for k in range(proba_dim[p]):
                columns.append(f"{p}.proba_{k}")
                source_ids.append(p)
        else:
            columns.append(f"{p}.pred")
            source_ids.append(p)

    out_rows: list[list[float]] = []
    kept_samples: list[str] = []
    for sid in sample_universe:
        row: list[float] = []
        complete = True
        for p in producers:
            arr = cell.get((sid, p))
            if arr is None:
                if policy.missing_value == "drop":
                    complete = False
                    break
                elif policy.missing_value == "impute_zero":
                    row.extend([0.0] * (proba_dim.get(p, 1)))
                elif policy.missing_value == "error":
                    raise OOFMissingError(sample_id=sid, producer=p)
            else:
                if arr.ndim == 0:
                    row.append(float(arr))
                else:
                    row.extend(arr.tolist())
        if complete:
            out_rows.append(row)
            kept_samples.append(sid)

    X = np.asarray(out_rows, dtype=np.float32)
    return FeatureTable(
        sample_ids=tuple(kept_samples),
        X=X,
        columns=tuple(columns),
        source_ids=tuple(source_ids),
    )
```

### 8.3 AggregationPolicy

```python
@dataclass(frozen=True)
class AggregationPolicy:
    use_proba: bool = False                # use y_proba when available
    coverage: Literal["error", "drop_incomplete", "impute"] = "drop_incomplete"
    missing_value: Literal["error", "drop", "impute_zero"] = "drop"
    duplicate_resolution: Literal["error", "mean", "last"] = "error"
    fold_mismatch: Literal["error", "warn"] = "error"
    aggregation_level: Literal["observation", "sample", "target", "group"] = "sample"
    method: Literal["none", "mean", "median", "vote", "robust_mean"] = "none"
    outlier_threshold: float | None = None
```

### 8.4 PredictionJoinNode execution

`PredictionJoinNode` is the DAG-ML node form of OOF stacking. At FIT_CV:

1. Resolve `producers` (set of upstream `MODEL` node ids whose predictions are
   merged).
2. Call `oof_join(producers, folds=runtime.folds, store=runtime.predictions,
   policy=node.params["aggregation"], partition="val",
   branch_path=runtime.branch_path)` -> `FeatureTable`.
3. Emit a `DataBlock` with `representation.id="tabular_numeric"` and
   `source_ids` equal to producer node ids (so the downstream
   `ModelNode.fit` sees clearly-namespaced columns).

At PREDICT:

1. For each producer, call its `predict` once on the new dataset (no folds),
   producing a single `PredictionBlock(partition="final")`.
2. Stack them horizontally in the same `columns` order recorded at FIT_CV in
   the join's fitted artifact (column order is part of the schema fingerprint).

### 8.5 The leakage opt-in flag

DAG-ML refuses by default to consume `partition="train"` predictions in a
`PredictionJoin`. The opt-in flag is a single, deliberately verbose boolean:

```python
PredictionJoinNode(..., allow_train_predictions_as_features=True)
```

When set:

- the node accepts `partition in {"train", "val"}` predictions as input;
- a `WARNING`-level entry is emitted to `MetricsLogger`;
- the `LineageRecord` for the join carries `leakage_acknowledged=True`;
- every `PredictionBlock` produced downstream carries the flag
  `"train_predictions_used"` in `flags`;
- `SELECT` may filter or rank these variants differently (see
  `RankingPolicy.exclude_leaky_variants`).

The verbose name is the design: it survives code review and grep without
requiring a second confirmation boolean. Violations raise `OOFLeakageError`.

---

## 9. Augmentation, branches, merge as first-class nodes

### 9.1 AugmentationNode

```python
@dataclass(frozen=True)
class AugmentationPolicy:
    apply_to: Literal["train_only", "cv_only", "all_partitions"] = "train_only"
    inherit_target: bool = True
    inherit_group: bool = True
    forbid_validation_augmentation: bool = True
    store_origin_mapping: bool = True
    seed_scope: Literal["run", "variant", "fold", "node"] = "fold"
    multiplier: int | None = None
```

Execution (FIT_CV, fold f):

```text
def execute_augmentation(node, task, ctx):
    if node.policy.apply_to == "train_only" and task.view.partition != "train":
        return passthrough(task)
    adapter = ml_data.adapter_registry.get(node.adapter_id)   # AugmentationAdapter
    block_in = ctx.dataset.materialize(node.source_id, task.view)
    plan = adapter.plan(block_in, node.policy, adapter_context_from(task, ctx))
    block_out, relation = adapter.transform(block_in, plan, ctx_adapter)
    # Append to local view; the executor records origin_ids so downstream
    # OOF join knows that observation X with origin_id=s should be treated as
    # 'same fold as s'.
    ctx.runtime.augmented_relations[node.id] = relation
    return NodeResult(outputs={"out": block_out}, artifacts=(), lineage=...)
```

Invariants:

- An augmented row may never sit on the validation side of a fold.
- When `forbid_validation_augmentation=True`, an `AugmentationNode` is skipped
  for views whose partition is `val` (a `passthrough` is emitted).
- `apply_to` semantics (resolves Q18):

  | Value             | CV train folds | REFIT (full train) | val / test |
  |-------------------|----------------|--------------------|------------|
  | `train_only` (default) | augmented   | augmented          | skipped    |
  | `cv_only`         | augmented      | skipped            | skipped    |
  | `all_partitions`  | augmented      | augmented          | augmented (rare; debug only) |

  Default = `train_only` because the model shipped via REFIT must match the
  augmented distribution that justified its selection in CV. Users wanting
  strict same-samples between OOF and refit set `cv_only` explicitly.

### 9.2 ForkNode

Two modes:

```python
@dataclass(frozen=True)
class ForkPolicy:
    mode: Literal["duplication", "separation"]
    # duplication: same samples, different downstream subgraphs
    # separation: disjoint sample subsets

    # only used for separation:
    by: Literal["metadata", "tag", "filter", "source"] | None = None
    metadata_column: str | None = None
    tag_name: str | None = None
    filter: Any | None = None
    by_source_steps: dict[str, list] | None = None
    values_map: dict[str, Any] | None = None   # {"clean": False, "outliers": True}
```

Execution: emits N named branches, each as a `DataView` with branch-specific
sample subsets (separation) or a copy of the same view (duplication). The
runtime keeps `branch_path: tuple[str, ...]` and passes it to all subsequent
nodes for lineage isolation.

### 9.3 MapNode

A `MapNode` applies a subgraph to each branch produced by an upstream
`ForkNode`. Equivalent to "for each branch b: execute subgraph(b)".

```python
@dataclass(frozen=True)
class MapPolicy:
    parallel: bool = True                  # run branches in parallel if the scheduler allows
    isolate_artifacts: bool = True         # branch_path prefixes artifact ids
    on_branch_failure: Literal["error", "skip", "continue"] = "error"
```

### 9.4 FeatureJoinNode

Horizontal concat of `FeatureTable`s (one per branch). The join wraps an
ML_DATA `FeatureJoiner` with a fixed schema captured at FIT_CV; PREDICT
reapplies the same column order and missing-value handling.

```python
@dataclass(frozen=True)
class FeatureJoinPolicy:
    namespace_columns: bool = True
    on_missing_branch: Literal["error", "drop_branch", "impute_zero"] = "error"
    allow_schema_drift: bool = False
```

### 9.5 PredictionJoinNode

Already described in section 8. Wraps `oof_join` and persists the resulting
`FeatureTable` as a fitted artifact (columns + producer ids + dtypes).

### 9.6 SourceJoinNode

`SourceJoinNode` is the DAG-ML wrapper for ML_DATA fusion. It does not
implement fusion logic; it forwards to `DataPlanner.execute_fit` /
`DataPlanner.execute_transform` with the chosen `FusionPolicy`. It exists so
that fusion is **explicit in the graph** rather than hidden in
`dataset.x(layout=...)`.

### 9.7 Contract: OOF with augmentation + branches

The meta-stacker receives **OOF predictions of origins**, never of augmented
copies. Concretely:

- Producers emit `PredictionBlock`s with `aggregation_level="observation"` and
  carry `sample_ids` *and* `observation_ids`. When `observation_id` was an
  augmented row in train, the producer does **not** emit a `partition="val"`
  block for it (augmented rows are train-only by rule 9.1).
- At `oof_join` time, the join builds the meta-table indexed by `sample_id`
  (the canonical sample axis); each cell is one prediction per producer.
- If a producer has multiple observations per sample on val (e.g.
  `per_sample_repeated` source), the join aggregates them per
  `policy.aggregation_level` (default `sample`) using `policy.method`.

Pseudocode for the augmentation-aware aggregation:

```text
def aggregate_to_sample(blocks, policy):
    by_sample: dict[str, list[np.ndarray]] = defaultdict(list)
    for b in blocks:
        for i, sid in enumerate(b.sample_ids):
            # skip augmented observations: their origin_id maps elsewhere
            obs_origin = lookup_origin(b.observation_ids[i] if b.observation_ids else None)
            if obs_origin is not None:
                continue  # augmented row -> never in val (invariant)
            by_sample[sid].append(b.y_pred[i])
    out = {}
    for sid, preds in by_sample.items():
        if policy.method == "mean":
            out[sid] = np.mean(preds, axis=0)
        elif policy.method == "median":
            out[sid] = np.median(preds, axis=0)
        elif policy.method == "robust_mean":
            out[sid] = robust_mean(preds, threshold=policy.outlier_threshold)
        elif policy.method == "vote":
            out[sid] = majority_vote(preds)
        else:
            out[sid] = preds[0]
    return out
```

---

## 10. Search space, generators, tuning

### 10.1 SearchSpace (see section 4.4)

`SearchSpace.slots` is the canonical representation. Slots are extracted from
the DSL at COMPILE; enumeration is performed lazily at PLAN.

### 10.2 Variant

```python
@dataclass(frozen=True)
class Variant:
    id: str                                # canonical: stable hash of slot_choices
    slot_choices: dict[str, Any]
    fingerprint: str                       # = id (kept for clarity)
    priority: int | None = None
```

### 10.3 Enumerator strategies

| Strategy id        | Slots used                                  | Order                                    | Lazy? |
|--------------------|---------------------------------------------|------------------------------------------|-------|
| `independent`      | one of each slot                            | priority order, cartesian over slots     | yes   |
| `cartesian_nested` | `_cartesian_` + nested sub-spaces           | depth-first over cartesian path          | yes   |
| `materialised`     | small spaces (`<= max_eager`)               | sorted by `priority`                     | no    |
| `prioritised`      | priority queue over (estimated cost, id)    | best-cost first                          | yes   |

### 10.4 TrialResult and TunerAdapter

```python
@dataclass(frozen=True)
class TrialResult:
    trial_id: str
    params: dict[str, Any]
    score: float
    metric_name: str
    ascending: bool                        # False = higher is better
    artifacts: tuple["ArtifactRef", ...] = ()
    extras: dict[str, Any] = field(default_factory=dict)

class TunerAdapter(Protocol):
    spec: AdapterSpec

    def suggest(
        self,
        trial_id: str,
        space: SearchSpace,
        seed: "SeedContext",
        history: tuple[TrialResult, ...],
    ) -> dict[str, Any]: ...

    def record(self, result: TrialResult) -> None: ...

    def best(self, k: int = 1) -> tuple[TrialResult, ...]: ...
```

### 10.5 TuningNodeSpec

```python
@dataclass(frozen=True)
class TuningNodeSpec:
    id: str
    target_node_ids: tuple[str, ...]       # which nodes the tuner controls
    search_space: SearchSpace
    objective_metric: str
    ascending: bool = False
    cv_policy: SplitPolicy
    max_trials: int = 50
    tuner_adapter_id: str = "dagml.tuner.optuna"
    early_stopping: dict[str, Any] = field(default_factory=dict)
```

Tuning runs nested CV: for each trial, a fresh `FoldSet` is built (or reused)
and the tuned subgraph is executed with the same OOF rules. The best trial's
params become the operator params used at REFIT.

### 10.6 Integration with Optuna / Ray

Optuna and Ray Tune are optional dependencies. DAG-ML defines `TunerAdapter`;
adapter packages live in companion modules:

```python
# pyproject.toml extras
[project.optional-dependencies]
optuna = ["optuna>=3.0"]
ray    = ["ray[tune]>=2.0"]
```

The adapter wraps the external tuner's API. DAG-ML never imports
`optuna`/`ray` at the core level.

---

## 11. Selection and refit

### 11.1 RankingPolicy

```python
@dataclass(frozen=True)
class RankingPolicy:
    metric: str                            # "rmsecv", "r2", "f1", "accuracy", ...
    ascending: bool = False                # default: higher is better; rmsecv ascending=True
    top_k: int = 1
    per_model: bool = False                # rank within each model class
    tie_breaker: Literal["first", "fingerprint", "complexity"] = "first"
    aggregate_over_folds: Literal["mean", "median", "min", "max", "p90"] = "mean"
```

### 11.2 SelectedGraph

```python
@dataclass(frozen=True)
class SelectedGraph:
    variant_id: str
    graph: GraphSpec
    plan: ExecutionPlan
    best_params: dict[str, dict[str, Any]]     # node_id -> params (post-tuning)
    cv_scores: dict[str, float]
    metric: str
    ascending: bool
    branch_chains: dict[str, tuple[str, ...]] = field(default_factory=dict)
                                                # branch_id -> winning node chain
    refit_strategy: Literal["standalone", "stacking", "mixed",
                            "separation", "competing_branches"] = "standalone"
```

### 11.3 RefitPlan

```python
@dataclass(frozen=True)
class RefitPlan:
    selected: SelectedGraph
    fold_set: FoldSet                      # single fold "final" with full train
    refit_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    keep_oof_predictions: bool = True      # do not delete CV predictions at refit
```

### 11.4 Refit dispatch

```text
def refit(selected: SelectedGraph, dataset: MLDataset, ctx: RunContext) -> RefitArtifacts:
    if selected.refit_strategy == "standalone":
        return refit_standalone(selected, dataset, ctx)
    if selected.refit_strategy == "stacking":
        return refit_stacking(selected, dataset, ctx)
    if selected.refit_strategy == "mixed":
        return refit_mixed(selected, dataset, ctx)
    if selected.refit_strategy == "separation":
        return refit_separation(selected, dataset, ctx)
    if selected.refit_strategy == "competing_branches":
        return refit_competing_branches(selected, dataset, ctx)
    raise RefitError(selected.refit_strategy)
```

### 11.5 Strategy summaries (condensed pseudocode)

#### Standalone

```text
def refit_standalone(selected, dataset, ctx):
    fold_set = FoldSet(folds=(Fold(fold_id="final",
                                   train_samples=dataset.train_universe(),
                                   val_samples=()),), split_unit=...)
    plan = re_plan(selected.graph, variant=Variant(slot_choices=selected.best_params))
    return execute(plan, dataset, fold_set, ctx, phase=Phase.REFIT)
```

#### Stacking (meta-stacker + branches)

```text
def refit_stacking(selected, dataset, ctx):
    # Phase A: refit each branch leaf independently on full train.
    leaves = leaf_models_of(selected.graph)
    branch_artifacts = {}
    for leaf in leaves:
        chain = selected.branch_chains[leaf.branch_id]
        sub_plan = isolate_chain(selected.plan, chain)
        a = execute(sub_plan, dataset, fold_set_final(), ctx, phase=Phase.REFIT)
        branch_artifacts[leaf.id] = a

    # Phase B: regenerate OOF predictions of leaves on the (already-fitted) CV folds.
    #          (Re-use stored CV predictions; do not refit folds.)
    meta_features = oof_join(producers=[l.id for l in leaves],
                             folds=ctx.cv_folds,
                             store=ctx.predictions,
                             policy=AggregationPolicy(),
                             partition="val", branch_path=())

    # Phase C: fit meta-stacker on (meta_features, y) once.
    meta_node = meta_stacker_of(selected.graph)
    fitted_meta = execute_single(meta_node, inputs={"X": meta_features},
                                 target=dataset.target(...), ctx, phase=Phase.REFIT)

    return RefitArtifacts(branch_artifacts | {"meta": fitted_meta})
```

#### Mixed merge (features + predictions)

```text
def refit_mixed(selected, dataset, ctx):
    # Build a FeatureTable that mixes raw branch features and OOF predictions
    # of selected branches, then refit the meta-model.
    parts = []
    for branch in branches(selected.graph):
        if branch.merge_mode == "features":
            parts.append(execute_branch_full_train(branch, dataset, ctx))
        elif branch.merge_mode == "predictions":
            parts.append(oof_join(producers=[branch.leaf_id], ...))
    fused = feature_join(parts, FeatureJoinPolicy(...))
    return execute_single(meta_node, inputs={"X": fused}, ...)
```

#### Separation branches (disjoint subsets)

```text
def refit_separation(selected, dataset, ctx):
    artifacts = {}
    for branch in selected.graph.separation_branches():
        sub_view = view_for_subset(branch.subset_filter)
        sub_artifacts = execute(plan_for(branch), dataset, fold_set_final(sub_view),
                                ctx, phase=Phase.REFIT)
        artifacts[branch.id] = sub_artifacts
    return RefitArtifacts(artifacts)
```

#### Competing branches (best chain per leaf)

```text
def refit_competing_branches(selected, dataset, ctx):
    artifacts = {}
    for leaf, chain in selected.branch_chains.items():
        sub_plan = build_chain_plan(chain)
        artifacts[leaf] = execute(sub_plan, dataset, fold_set_final(),
                                  ctx, phase=Phase.REFIT)
    return RefitArtifacts(artifacts)
```

---

## 12. Reproducibility

### 12.1 SeedContext

```python
@dataclass(frozen=True)
class SeedContext:
    root_seed: int | None
    run_id: str
    variant_id: str | None = None
    node_id: str | None = None
    fold_id: str | int | None = None
    trial_id: str | int | None = None
    branch_path: tuple[str, ...] = ()
    aug_index: int | None = None

    def child(self, **labels: Any) -> "SeedContext": ...
    def numpy_seed(self) -> int | None: ...
    def python_seed(self) -> int | None: ...
    def torch_seed(self) -> int | None: ...
    def derived(self) -> int | None: ...
        # SHA256(root_seed || canonical_path_labels) modulo 2**32
```

### 12.2 Rules

- Every stateful node receives a `SeedContext` derived from the **full path**:
  `(root_seed, run_id, variant_id, node_id, fold_id, trial_id, branch_path, aug_index)`.
- `derived()` is deterministic: same labels => same integer.
- Persisted in `LineageRecord` and reused at PREDICT / EXPLAIN (for nodes that
  need a seed at inference, e.g. dropout-based MC sampling).
- A node MUST NOT call `numpy.random.seed(...)` globally; it MUST use its
  derived seed locally (e.g. `np.random.default_rng(seed=ctx.derived())`).

### 12.3 Cross-run determinism

A full run from `(root_seed, dsl, dataset_schema, policies)` is deterministic
when:

1. ML_DATA materialises deterministically (canonical sort).
2. All operators' RNGs are derived from `SeedContext`.
3. The scheduler does not change result ordering (the reducer over folds is
   commutative or sorts by `fold_id`).
4. No floating-point non-determinism (BLAS thread count fixed when required).

---

## 13. Artifacts, lineage, cache

### 13.1 ArtifactRef and ArtifactStore

```python
@dataclass(frozen=True)
class ArtifactRef:
    id: str                                # content-addressed (sha256 of payload)
    kind: Literal["model", "transformer", "tuner_state", "fitted_adapter",
                  "ml_data_fitted_adapter", "metadata"]
    backend: Literal["joblib", "torch", "tensorflow", "onnx", "json", "raw"]
    size_bytes: int
    plugin: str | None = None
    plugin_version: str | None = None

class ArtifactStore(Protocol):
    def put(self, payload: Any, *, kind: str, backend: str) -> ArtifactRef: ...
    def get(self, ref: ArtifactRef) -> Any: ...
    def exists(self, ref: ArtifactRef) -> bool: ...
    def delete(self, ref: ArtifactRef) -> None: ...
    def iter_refs(self) -> Iterator[ArtifactRef]: ...
```

Implementations:

- `InMemoryArtifactStore`: dict keyed by id, useful in tests.
- `FilesystemArtifactStore`: writes `<root>/artifacts/<sha256>.<backend>`.
- `ContentAddressedArtifactStore`: dedupe-by-hash, used in workspaces.
- `SQLiteArtifactStore`: metadata in SQLite, payload on disk.

### 13.2 LineageRecord and LineageRecorder

```python
@dataclass(frozen=True)
class LineageRecord:
    record_id: str
    node_id: str
    phase: Phase
    variant_id: str
    fold_id: str | None
    branch_path: tuple[str, ...]
    inputs: dict[str, str]                 # port -> upstream record_id
    params_fingerprint: str
    artifact_refs: tuple[ArtifactRef, ...]
    metrics: dict[str, float] = field(default_factory=dict)
    seed: int | None = None
    plugin_versions: dict[str, str] = field(default_factory=dict)
    started_at: float = 0.0                # epoch seconds
    ended_at: float = 0.0
    parent_refs: tuple[str, ...] = ()      # for sub-DAG nesting
    unsafe_flags: tuple[str, ...] = ()     # e.g. ("unsafe_leakage",)

class LineageRecorder(Protocol):
    def record(self, rec: LineageRecord) -> None: ...
    def query(self, **filters: Any) -> Iterator[LineageRecord]: ...
    def export_graph(self, run_id: str) -> dict[str, Any]: ...
```

### 13.3 CacheStore

```python
@dataclass(frozen=True)
class CacheKey:
    operator_id: str
    params_fingerprint: str
    input_lineage: tuple[str, ...]         # parent record_ids (sorted)
    phase: Phase
    seed: int | None
    schema_fingerprint: str

class CacheStore(Protocol):
    def has(self, key: CacheKey) -> bool: ...
    def get(self, key: CacheKey) -> tuple[Any, LineageRecord]: ...
    def put(self, key: CacheKey, value: Any, rec: LineageRecord) -> None: ...
    def evict(self, key: CacheKey) -> None: ...
    def stats(self) -> dict[str, int]: ...
```

Cache lookup happens in the executor between scheduling and node execution:

```text
def maybe_cached(task, cache):
    key = CacheKey(operator_id=task.plan.adapter_id,
                   params_fingerprint=task.plan.fingerprint,
                   input_lineage=tuple(sorted(rec.record_id
                                              for rec in task.inputs_lineage)),
                   phase=task.phase,
                   seed=task.seed.derived(),
                   schema_fingerprint=ctx.schema_fingerprint)
    if cache.has(key):
        value, rec = cache.get(key)
        ctx.lineage.record(rec.with_replay(True))
        return value, rec
    return None
```

### 13.4 Default policies

| Concern             | Default                                                             |
|---------------------|---------------------------------------------------------------------|
| Cache TTL           | none; eviction is LRU by total size                                 |
| Cache size budget   | configurable via `CacheConfig.step_cache_max_mb`                    |
| Artifact retention  | per-run; orphans GC'd at next run start                             |
| Lineage retention   | persistent (SQLite); never deleted by a single run                  |
| Unsafe operations   | recorded in lineage; never silently elided                          |

---

## 14. Execution

### 14.1 RunContext and ExecutionContext

```python
@dataclass
class RunContext:
    run_id: str
    phase: Phase
    dataset: "MLDataset"
    data_planner: "DataPlanner"
    artifact_store: ArtifactStore
    prediction_store: PredictionStore
    cache: CacheStore
    lineage: LineageRecorder
    resources: ResourceHints
    seed: SeedContext
    cv_folds: FoldSet | None = None
    branch_path: tuple[str, ...] = ()
    metrics_logger: "MetricsLogger | None" = None

@dataclass
class ExecutionContext:
    plan: ExecutionPlan
    run: RunContext
    scheduler: "Scheduler"
    schema_fingerprint: str
    runtime: dict[str, Any] = field(default_factory=dict)
                                         # node_id -> NodeResult (kept until consumed)
```

### 14.2 Scheduler protocol

```python
class Scheduler(Protocol):
    def submit(
        self,
        tasks: tuple["ScheduledTask", ...],
        ctx: ExecutionContext,
    ) -> tuple["NodeResult", ...]: ...

    def supports_parallelism(self, kind: Literal["variant", "branch", "fold"]) -> bool: ...

@dataclass(frozen=True)
class ScheduledTask:
    task: NodeTask
    dependencies: tuple[str, ...]          # node ids
    fan_in: int
```

### 14.3 Scheduler implementations

| Id                    | Description                                                      | Notes                                              |
|-----------------------|------------------------------------------------------------------|----------------------------------------------------|
| `dagml.scheduler.sequential`  | Topological; one task at a time                          | for debugging                                      |
| `dagml.scheduler.loky`        | `joblib.Parallel(backend="loky")` for variants/folds      | process-based, default for `n_jobs > 1`            |
| `dagml.scheduler.threadpool`  | `concurrent.futures.ThreadPoolExecutor`                   | only with `thread_safe=True` nodes                 |
| `dagml.scheduler.ray`         | Ray Actor pool                                            | optional dependency                                |

### 14.4 Parallelism rules

- Variant-level: independent variants run on separate workers; each worker gets
  its own `PredictionStore` (e.g. `store=None` for in-memory) to avoid DB
  lock contention. The orchestrator merges back at the end.
- Branch-level: a `MapNode` with `parallel=True` and a parallel scheduler
  spawns one task per branch.
- Fold-level: a `ModelNode` with `n_jobs_folds > 1` may parallelise folds.
- Nested parallelism is forbidden by default
  (`ResourceHints.nested_parallelism="forbid"`). A node nested inside a
  parallel context runs single-threaded.

### 14.5 Topological execution loop

```text
def execute(plan: ExecutionPlan, ctx: ExecutionContext):
    ready = [nid for nid in plan.topological_order
             if not plan.graph.predecessors(nid)]
    remaining = set(plan.topological_order)
    while remaining:
        batch = [n for n in ready if all_deps_done(n, plan)]
        if not batch:
            raise SchedulerError("deadlock")
        tasks = [build_task(n, ctx) for n in batch]
        results = ctx.scheduler.submit(tasks, ctx)
        for nid, res in zip(batch, results):
            ctx.runtime[nid] = res
            ctx.lineage.record(res.lineage)
            for art in res.artifacts:
                ctx.run.artifact_store  # already persisted by node
            remaining.discard(nid)
            ready = [n for n in plan.topological_order
                     if n in remaining and all_deps_done(n, plan)]
    return collect_terminal_outputs(plan, ctx)
```

---

### 14.6 Error taxonomy

```python
class DagMLError(Exception): ...

class CompileError(DagMLError): ...
class PlanningError(DagMLError): ...
class NoAdapterError(PlanningError): ...
class PortContractError(PlanningError): ...
class CyclicGraphError(CompileError): ...
class DataPlanIncompatibility(PlanningError): ...

class FoldError(DagMLError): ...
class SplitPolicyError(FoldError): ...

class OOFError(DagMLError): ...
class OOFCoverageError(OOFError): ...
class OOFFoldMisalignError(OOFError): ...
class OOFDuplicateError(OOFError): ...
class OOFMissingError(OOFError): ...
class OOFLeakageError(OOFError): ...      # opt-in violation or augmentation cross-fold

class RefitError(DagMLError): ...
class SelectionError(DagMLError): ...

class SchemaFingerprintMismatch(DagMLError): ...
class PluginVersionError(DagMLError): ...
class ArtifactMissingError(DagMLError): ...
class MissingSourceError(DagMLError): ...

class SchedulerError(DagMLError): ...
class ResourceContentionError(SchedulerError): ...
```

Every exception carries a structured `payload: dict` (JSON-serialisable) so
that downstream tooling (UI, CLI, webapp backend) can render it without
parsing strings:

```python
@dataclass(frozen=True)
class ErrorPayload:
    code: str                              # short stable code, e.g. "OOF_MISSING"
    message: str
    node_id: str | None = None
    fold_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
```

### 14.7 Runtime invariants checked by the executor

Before each `NodeTask` is dispatched, the executor verifies:

| Invariant                                                          | Failure mode                                  |
|--------------------------------------------------------------------|-----------------------------------------------|
| All required input ports have a value in `ctx.runtime`             | `PortContractError`                           |
| Input representation matches edge contract (or 0-hop adaptation)   | `PortContractError`                           |
| `phase in adapter.supported_phases()`                              | passthrough (skip) or `PlanningError`         |
| `task.seed` is derived from `ctx.run.seed.child(node_id=...)`      | implementation invariant; tested in CI        |
| Fold sample_ids subset of dataset sample_ids                       | `FoldError`                                   |
| Augmented observation never lands in val of its origin's fold      | `OOFError` (caught during fold validation)    |
| Producers of a `PredictionJoin` share the upstream `SplitNode`     | `OOFFoldMisalignError`                        |
| Cache hit lineage record refers to a still-valid `ArtifactRef`     | re-execute on stale cache hit                 |

### 14.8 MetricsLogger

```python
class MetricsLogger(Protocol):
    def log_scalar(self, name: str, value: float, *, step: int | None = None,
                   node_id: str | None = None, fold_id: str | None = None,
                   variant_id: str | None = None) -> None: ...
    def log_histogram(self, name: str, values: Any, *,
                      node_id: str | None = None) -> None: ...
    def flush(self) -> None: ...
```

Implementations: `NoopMetricsLogger`, `StdoutMetricsLogger`,
`SQLiteMetricsLogger` (workspace-backed). Optional adapters (Weights & Biases,
MLflow) live in companion packages.

---

## 15. Bundle, replay, predict / explain

### 15.1 ExecutionBundle

```python
@dataclass(frozen=True)
class SerializedDataPlan:
    schema_fingerprint: str
    plan: "DataPlan"
    fitted_adapter_refs: tuple["SerializableRef", ...]
    output_schema: "RepresentationSpec"

@dataclass(frozen=True)
class ExecutionBundle:
    bundle_id: str
    graph_spec: GraphSpec
    selected: SelectedGraph
    plan: ExecutionPlan
    data_schema: "DatasetSchema"
    data_schema_fingerprint: str                   # canonical SHA256 of (schema, policy, adapter_specs)
    policy: PlanningPolicy                         # frozen at refit; replay must use same fusion/alignment/adaptation
    data_plans: dict[str, SerializedDataPlan]      # node_id -> serialized data plan
    artifact_refs: dict[str, tuple[ArtifactRef, ...]]
                                                   # node_id -> artifacts (refit ones)
    seed_root: SeedContext
    plugin_versions: dict[str, str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def verify_fingerprint(self) -> bool:
        """
        Recompute the canonical fingerprint from `data_schema`, `policy`, and
        the adapter specs referenced by `data_plans`. Returns True iff it
        matches the stored `data_schema_fingerprint`. Used to detect a corrupt
        or hand-edited bundle.
        """
        recomputed = schema_fingerprint(
            self.data_schema,
            self.policy.fusion,
            tuple(sp.plan for sp in self.data_plans.values()),
        )
        return recomputed == self.data_schema_fingerprint
```

The fingerprint is **stored**, not recomputed at replay. Reasons:

- The canonicalisation algorithm may evolve between library versions; storing
  the fingerprint pins the value used at refit and prevents silent breakage
  of older bundles.
- Replay does a single string comparison instead of a re-canonicalisation.
- `verify_fingerprint()` is available for paranoid callers (audit, restore
  from disk).

Bundle is JSON + binary side-payloads:

- `bundle.json` carries every spec (DataPlan, GraphSpec, ArtifactRef indexes).
- `artifacts/<sha256>.<backend>` carries the binary fitted objects.
- `predictions/<dataset>.parquet` may carry the original CV predictions
  (optional).

### 15.2 PREDICT replay

```text
def predict(bundle: ExecutionBundle, new_dataset: MLDataset, *,
            target_partition: str = "final") -> tuple[PredictionBlock, ...]:
    # 1. Schema verification
    new_fp = schema_fingerprint(new_dataset.schema())
    if new_fp != bundle.data_schema_fingerprint:
        # not always fatal: only the parts used by the plan must match.
        diff = diff_schema(bundle.data_schema, new_dataset.schema())
        if diff.breaks_data_plans(bundle.data_plans):
            raise SchemaFingerprintMismatch(diff)

    # 2. Plugin version check
    available = current_plugin_versions()
    require = bundle.plugin_versions
    if not satisfies_compat(available, require):
        raise PluginVersionError(available, require)

    # 3. Build a RunContext for PREDICT phase
    ctx = RunContext(
        run_id=new_run_id(),
        phase=Phase.PREDICT,
        dataset=new_dataset,
        data_planner=bundle_data_planner(bundle),
        artifact_store=open_artifact_store(bundle),
        prediction_store=InMemoryPredictionStore(),
        cache=NoCache(),
        lineage=in_memory_lineage(),
        resources=default_resources(),
        seed=bundle.seed_root,
    )

    # 4. Topological walk with each node delegating to its adapter's predict path.
    for nid in bundle.plan.topological_order:
        adapter = registry.resolve(bundle.graph_spec.nodes[nid])
        if Phase.PREDICT not in adapter.supported_phases():
            # e.g. AugmentationNode -> passthrough
            ctx.runtime[nid] = passthrough(ctx.runtime[parent(nid)])
            continue
        if nid in bundle.artifact_refs:
            fitted = ctx.artifact_store.get(bundle.artifact_refs[nid][0])
        else:
            fitted = None
        task = NodeTask(node_id=nid, phase=Phase.PREDICT,
                        plan=bundle.plan.node_plans[nid],
                        inputs=collect_inputs(nid, ctx),
                        view=ctx.dataset.view(DataView(partition="final")),
                        fold_id="final", seed=ctx.seed)
        result = adapter.execute(task, ctx)
        ctx.runtime[nid] = result
        ctx.prediction_store.append_if_prediction(result)

    return ctx.prediction_store.find(partition=target_partition)
```

### 15.3 Failure modes

| Failure                                | Where detected            | Behaviour                                                                         |
|----------------------------------------|---------------------------|------------------------------------------------------------------------------------|
| schema fingerprint mismatch (used part)| PREDICT init               | raise `SchemaFingerprintMismatch`, attach diff                                     |
| schema fingerprint mismatch (unused)   | PREDICT init               | warn, proceed                                                                       |
| missing source for a node              | PREDICT exec               | raise `MissingSourceError(node, source_id)`                                        |
| plugin missing or out-of-range         | PREDICT init               | raise `PluginVersionError(required, available)`                                    |
| fitted artifact missing                | PREDICT exec               | raise `ArtifactMissingError(node_id)`                                              |
| data plan unsolvable on new schema     | PREDICT init               | raise `DataPlanIncompatibility(node, plan, schema)`                                |
| node not predict-capable but reached   | PREDICT exec               | adapter returns passthrough; lineage records `phase_skipped="predict"`             |
| value out of range in y_transform      | PREDICT exec               | depends on YTransform adapter (raise / clip / extrapolate)                          |

### 15.4 EXPLAIN replay

```python
class ExplainHooks(Protocol):
    def shap_values(self, X: Any) -> Any: ...
    def feature_importance(self) -> dict[str, float] | None: ...

class ExplainerAdapter(Protocol):
    spec: AdapterSpec
    def explain(
        self,
        fitted: Any,
        inputs: dict[str, "DataBlock"],
        ctx: RunContext,
    ) -> "ExplainResult": ...

@dataclass(frozen=True)
class ExplainResult:
    node_id: str
    method: str                            # "shap.tree", "shap.linear", "permutation", ...
    values: Any
    feature_names: tuple[str, ...] | None
    sample_ids: tuple[SampleIdT, ...] | None
    metadata: dict[str, Any] = field(default_factory=dict)
```

EXPLAIN walks the same path as PREDICT up to the explained node, then calls
the `ExplainerAdapter.explain()` instead of (or in addition to) the model's
`predict()`. The bundle does not need to be re-fitted.

---

## 16. Public API

```python
# nirs4all-agnostic; lives in package "dag_ml"

def compile(
    pipeline: Any,
    *,
    registry: OperatorRegistry,
) -> GraphSpec: ...

def enumerate(
    space: SearchSpace,
    *,
    lazy: bool = True,
    seed: int | None = None,
    max_variants: int | None = None,
) -> Iterator[Variant]: ...

def plan(
    graph: GraphSpec,
    variant: Variant,
    *,
    dataset_schema: "DatasetSchema",
    policy: "PlanningPolicy",
    data_planner: "DataPlanner",
    cache: "PlanCacheStore | None" = None,
) -> ExecutionPlan: ...

def fit_cv(
    plan: ExecutionPlan,
    dataset: "MLDataset",
    *,
    fold_set: FoldSet,
    artifact_store: ArtifactStore,
    prediction_store: PredictionStore,
    cache: CacheStore | None = None,
    lineage: LineageRecorder | None = None,
    scheduler: Scheduler | None = None,
    seed: SeedContext | None = None,
) -> "CVResult": ...

def select(
    cv_result: "CVResult",
    *,
    ranking: RankingPolicy,
) -> SelectedGraph: ...

def refit(
    selected: SelectedGraph,
    dataset: "MLDataset",
    *,
    artifact_store: ArtifactStore,
    prediction_store: PredictionStore,
    lineage: LineageRecorder | None = None,
    scheduler: Scheduler | None = None,
) -> "RefitResult": ...

def export(
    selected: SelectedGraph,
    refit_result: "RefitResult",
    *,
    path: str | None = None,
) -> ExecutionBundle: ...

def predict(
    bundle: ExecutionBundle | str,
    dataset: "MLDataset",
    *,
    target_partition: str = "final",
) -> tuple[PredictionBlock, ...]: ...

def explain(
    bundle: ExecutionBundle | str,
    dataset: "MLDataset",
    *,
    method: str = "shap.auto",
    node_id: str | None = None,
) -> ExplainResult: ...

def run(
    pipeline: Any,
    dataset: "MLDataset",
    *,
    registry: OperatorRegistry,
    policy: "PlanningPolicy",
    fold_set: FoldSet | None = None,
    ranking: RankingPolicy | None = None,
    artifact_store: ArtifactStore | None = None,
    n_jobs: int = 1,
    seed: int | None = None,
) -> "RunResult": ...
```

---

## 17. Use cases

### UC1: Single-source NIRS + PLS + KFold + refit (baseline)

```text
DSL pipeline = [
    SNV(),
    {"y_processing": MinMaxScaler()},
    KFold(n_splits=5, shuffle=True, random_state=42),
    {"model": PLSRegression(n_components=10)},
]
```

Graph:

```text
[src: nir]      <-- MLDataset materialises native signal_with_processings
     |
     v
[SNV]      kind=TRANSFORM   adapter=sklearn.transformer
     |
     v
[YScaler]  kind=Y_TRANSFORM adapter=sklearn.ytransformer
     |
     v
[KFold]    kind=SPLIT       adapter=sklearn.cv_splitter
     |                       -> emits FoldSet (5 folds, sample-unit)
     v
[PLS]      kind=MODEL       adapter=sklearn.estimator
                            DataPlan: signal_with_processings -> tabular_numeric
                                       via spectra.flatten
```

Phase walk:

| Phase   | Effect                                                                                   |
|---------|------------------------------------------------------------------------------------------|
| COMPILE | builds 4-node GraphSpec, empty SearchSpace                                               |
| PLAN    | one variant; DataPlanner resolves DataPlan(materialize -> spectra.flatten) for PLS       |
| FIT_CV  | 5 folds; PLS emits 5 val PredictionBlocks (fold_id="0".."4") + 5 train                   |
| SELECT  | ranking="rmsecv"; top_k=1; single variant -> returned as SelectedGraph                   |
| REFIT   | one fold "final" = all train samples; PLS refit; emits PredictionBlock(fold_id="final") |
| export  | ExecutionBundle with SNV, YScaler, PLS artifacts                                         |
| PREDICT | replay: SNV.transform -> YScaler keeps inverse_transform -> PLS.predict                 |

### UC2: Multi-source heterogeneous (NIRS + photo + genotype + meteo + metadata) -> RandomForest

```text
DSL pipeline = [
    {"source_join": {
        "sources": ["nir", "photo_front", "photo_side",
                    "genotype", "weather", "metadata"],
        "policy": {"mode": "concat_features",
                   "target_representation": "tabular_numeric",
                   "alignment": {"join": "inner"},
                   "missing_source": "indicator",
                   "namespace_columns": True,
                   "allow_lossy_adapters": True}}},
    ShuffleSplit(n_splits=3, test_size=0.2),
    {"model": RandomForestRegressor(n_estimators=500)},
]
```

Graph:

```text
            +-- src:nir ---+ ---> [adapt: spectra.flatten] --+
            |              |                                |
            +-- src:photo_front -> [adapt: image.embedding]-+
            |                                               |
            +-- src:photo_side  -> [adapt: image.embedding]-+
[SourceJoin] -- src:genotype -> [adapt: genotype.pca] -----+--> [align: inner]
            |                                               |
            +-- src:weather -> [adapt: weather.aggregate] --+
            |                                               |
            +-- src:metadata -> [adapt: tabular.encoder] ---+
                                                            |
                                                            v
                                              [join: fusion.feature_joiner]
                                                            |
                                                            v
                                                       [split: ShuffleSplit]
                                                            |
                                                            v
                                                          [RF]
```

Notes:

- `SourceJoin` is a single DAG-ML node that delegates to ML_DATA's
  `DataPlanner.resolve` with a `FusionPolicy(mode="concat_features", ...)`.
- DataPlan contains `materialize` (x6) + `adapt` (x6) + `align` + `join`.
- `image.embedding` and `genotype.pca` are stateful and lossy: they fit on
  fold-train (FIT_CV) and refit on full train (REFIT). Their fitted artifacts
  are saved as `FittedAdapter` in `ml_data_fitted_adapter` blobs.
- `allow_lossy_adapters=True` is **explicit** in the policy: the user
  acknowledges the modelling assumption.
- `missing_source="indicator"` keeps a presence column per source so RF can
  exploit missingness signals.
- At PREDICT, the data plan is replayed with the same fitted adapters; the
  RF artifact ID is fetched and `predict` is called once on the joined matrix.

### UC3: Stacking (branch + merge predictions) -> Ridge meta-model

```text
DSL pipeline = [
    {"branch": [
        [SNV(), PLSRegression(15)],
        [MSC(), RandomForestRegressor(n_estimators=300)],
        [Detrend(), GradientBoostingRegressor()],
    ]},
    {"merge": "predictions",
     "policy": {"aggregation_level": "sample",
                "method": "mean",
                "use_proba": False}},
    KFold(n_splits=5),
    {"model": Ridge(alpha=1.0)},
]
```

Graph:

```text
[src: nir] -> [Fork: duplication]
                |        |        |
                v        v        v
              [SNV]    [MSC]   [Detrend]
                |        |        |
                v        v        v
              [PLS]    [RF]    [GBR]
                |        |        |
                +--------+--------+
                         |
                         v
                 [PredictionJoin]
                         |
                         v
                    [KFold]   (for meta-stacker)
                         |
                         v
                      [Ridge]
```

Invariants enforced by DAG-ML:

- The three leaf models (`PLS`, `RF`, `GBR`) are fit on the *same* outer CV
  folds (they share the upstream `KFold` driving the OOF generation).
- `PredictionJoin` calls `oof_join(producers=[PLS, RF, GBR], folds=outer,
  store, policy)` and produces a `FeatureTable` (3 columns: pred per model,
  per sample, OOF).
- The meta-stacker's `KFold` is **nested inside** the outer CV; the executor
  inserts a synthetic inner-CV unit when needed (this is one of the points
  where ml_data DataPlanner is not involved at all -- it is pure DAG-ML).
- If `allow_train_predictions_as_features=True` were set on the
  `PredictionJoin`, the lineage record carries `leakage_acknowledged=True`
  and `dagml.run` emits a warning summary.
- Refit (`refit_stacking`): refit each leaf on full train; refit Ridge on the
  meta-feature matrix derived from the stored CV OOF (not re-trained on
  refit predictions).

Refusal scenarios:

| Scenario                                                        | DAG-ML response                                              |
|------------------------------------------------------------------|--------------------------------------------------------------|
| One producer has only `partition="train"` predictions             | `OOFError` at PredictionJoin                                  |
| Producers used different `KFold` (different fold structures)      | `OOFFoldMisalignError` (unless `policy.fold_mismatch="warn"`) |
| Sample missing from one producer's val universe                   | `policy.coverage="drop_incomplete"` drops it (default)        |
| `allow_train_predictions_as_features=True` not set, train preds requested | refuse: `OOFLeakageError`                                     |

### UC4: Repetitions + group-aware split + sample-level aggregation

Scenario: NIRS with 3 scans per leaf; chemistry Y at the leaf level;
plants partition for leakage control.

```text
DSL pipeline = [
    SNV(),
    {"y_processing": StandardScaler()},
    {"sample_relation": {"split_unit": "group", "group_key": "plant_id"}},
    GroupKFold(n_splits=5),
    {"model": PLSRegression(n_components=12)},
    {"aggregate": {"level": "sample", "method": "median"}},
]
```

Graph:

```text
[src: nir]   granularity=per_sample_repeated (3 obs / sample)
   |
   v
[SNV]        per-observation
   |
   v
[YScaler]    per-sample
   |
   v
[GroupKFold] fold_set with split_unit="group", group_key="plant_id"
             -- ML_DATA's SampleRelation.group_ids drives splitter
   |
   v
[PLS]        fits at observation level
             emits PredictionBlock with aggregation_level="observation"
   |
   v
[Aggregator] node that reduces observation -> sample using median
             produces PredictionBlock(aggregation_level="sample")
```

Key points:

- `MLDataset.materialize("nir", view)` returns 3 rows per sample. The
  `SampleRelation` is consulted by `GroupKFold` to ensure all 3 observations
  of `S001` plus all 3 of `S002` from the same plant end up in the same fold.
- The PLS model fits at the *observation* level (no aggregation inside the
  model). It emits one prediction per observation.
- The `Aggregator` is a DAG-ML node (kind=MODEL with a degenerate fit) that
  groups by `sample_id`. It is part of the graph, not hidden in a controller.
- At PREDICT, the same path replays: 3 obs per sample -> PLS -> median.
- Lineage: `PredictionBlock.aggregation_level` allows downstream metrics to
  pick the right level (e.g. RMSE per sample, not per observation).

Refusal scenario: if `split_unit="group"` is requested but
`SampleRelation.group_ids is None`, the splitter raises immediately:

```text
SplitPolicyError("split_unit='group' requires SampleRelation.group_ids,
                 but source 'nir' has no group annotation in the schema")
```

---

## 18. Performance

| Optimisation                                       | Where                                | Effect                                                                 |
|----------------------------------------------------|--------------------------------------|------------------------------------------------------------------------|
| Lazy variant enumeration                           | `enumerate(space, lazy=True)`        | avoid materialising 10k+ variants for `_cartesian_`                    |
| Copy-on-write DataView                             | DataView -> ML_DATA materialise      | branch snapshots are slice-views, not deep copies                      |
| Plan cache `(graph_id, variant_fp, schema_fp)`     | `PlanCacheStore`                     | reuse plans across variants with identical concrete graphs             |
| Step cache `(op, params, lineage, phase, seed)`    | `CacheStore`                         | reuse fitted transformers between variants that share a prefix         |
| Topological scheduler with batched submit          | `Scheduler.submit()`                 | parallelise sibling tasks (branches, folds)                            |
| Column-store PredictionStore                       | Parquet-backed `PredictionStore`     | predictions array stays out of metadata; metadata stays in SQLite      |
| Late collation                                     | DataPlan `collate` step is last      | tensor padding only at the model boundary                              |
| Resource hints                                     | `ResourceHints` per node             | scheduler avoids overcommit and nested parallelism                     |
| Variant-level parallelism                          | `loky` / `ray` schedulers            | each worker isolates its stores; orchestrator merges                   |
| Branch-level parallelism                           | `MapNode.parallel=True`              | sibling branches execute concurrently                                  |
| Fold-level parallelism                             | `ModelNode.n_jobs_folds`             | each fold of CV runs on a worker                                       |
| Operator caching by hash                           | `OperatorAdapter.cache_key`          | reuse `(transformer, params, fold)` instead of refit                   |
| `find_path` Dijkstra cache                         | inside ML_DATA `AdapterRegistry`     | constant-time path lookup for repeated schemas                         |
| Lineage write batching                             | `LineageRecorder.record()` flush     | amortise DB writes                                                     |
| Artifact content-addressing                        | `ArtifactStore.put` -> sha256        | deduplicate identical fitted objects across variants                   |
| Frozen ndarrays                                    | DataBlock immutability               | no defensive copies in consumer nodes                                  |

---

## 19. Migration from nirs4all

Steps (extracted and condensed from `dag_ml_externalization_from_code.md` §2.12):

1. Extract `ml_data.contract` (shared types). Vendor a stub package; freeze the
   API. No runtime change to nirs4all yet.
2. Introduce `OperatorRegistry`, `ModelAdapter`, `DataAwareOperatorAdapter`,
   `OperatorAdapter`. Map the existing `CONTROLLER_REGISTRY` entries to
   adapters one-to-one; the runtime keeps calling controllers, but adapter
   metadata is now declared.
3. Externalise `SpectroDataset` -> `SpectroDatasetConnector` implementing
   `MLDataset`. Replace `dataset.x(layout=..., concat_source=...)` calls with
   `DataPlanner.resolve` + per-node `DataPlan`.
4. Replace `PipelineConfigs` with `Compiler` + `SearchSpace` (keep DSL syntax).
   `Variant` becomes the unit of work for the orchestrator.
5. Reify `FoldSet`, `PredictionBlock`, `PredictionStore`, `OOFJoin`. Migrate
   `TrainingSetReconstructor` to call `oof_join` and remove duplicated logic.
6. Reify `ForkNode`, `MapNode`, `FeatureJoinNode`, `PredictionJoinNode`,
   `SourceJoinNode`. Eliminate `context.custom` side-effects.
7. Move `selection` and `refit` into DAG-ML phases. The five refit strategies
   become explicit dispatch in `refit()`.
8. Move `ExecutionBundle` to DAG-ML; rename `.n4a` to a DAG-ML bundle format
   (keep an alias for nirs4all read compatibility).
9. Drop NIRS-specific controllers (Savitzky-Golay derivative, OSC, EPO) into
   `nirs4all_operators` adapter modules; they register via
   `OperatorRegistry`.
10. Optional: pre-publish DAG-ML 0.x and consume it from nirs4all 0.9.

---

## 20. Non-goals

DAG-ML explicitly refuses to do the following.

| Concern                                            | DAG-ML behaviour                                                                |
|----------------------------------------------------|---------------------------------------------------------------------------------|
| Source storage / file formats                      | not modelled; delegated to ML_DATA                                              |
| Sample alignment between sources                   | delegated to `AlignmentPolicy` + ML_DATA                                        |
| Representation conversions                         | delegated to ML_DATA `RepresentationAdapter`s and `find_path`                   |
| Distributed clusters / remote schedulers           | only local schedulers in core; Ray is optional                                  |
| Persistent process management                      | no daemon; engine is in-process                                                 |
| Web UI / dashboards                                | none; metrics are logged via `MetricsLogger`                                    |
| Dataset versioning                                 | none beyond `schema_fingerprint`                                                |
| Mutable graph (runtime add / remove nodes)         | not supported; graph is frozen after COMPILE                                    |
| Automatic feature engineering beyond DataPlanner   | not modelled                                                                    |
| Domain primitives (wavelengths, image size, etc.)  | exposed only via `AuxInputSpec` from ML_DATA                                    |
| Splitting strategy invention                       | DAG-ML wraps existing splitters; it does not invent new CV schemes              |
| OOF auto-fix on misalignment                       | refuses (`OOFError`) unless `AggregationPolicy` opts in                         |
| Implicit unsafe predictions in stacking            | refused by default                                                              |
| Plan auto-relaxation                               | refused; user must opt-in via policy                                            |

---

## 21. Questions ouvertes

1. **Streaming datasets**: should `MLDataset.materialize` support a lazy /
   chunked `DataBlock` (e.g. backed by `dask`/`zarr`) for out-of-core
   workloads? DAG-ML phases assume eager materialisation today. The fold
   abstraction would need streaming-friendly checkpointing.
2. **Cross-source adapters (joint adapters)**: should DAG-ML define a
   `JointAdapter` first-class node (e.g. CCA between NIRS and image
   embeddings) or always express it as a sub-graph + custom adapter? Current
   answer: sub-graph for v1.
3. **Multi-target models**: should DAG-ML carry `TargetBlock`s with rank-2
   `y` as a primary contract, or always wrap them in multiple `TargetBlock`s
   joined at the model node? Current answer: wrap; revisit when adding the
   `MultiTargetAdapter`.
4. **Online / incremental learning**: should `ModelAdapter` declare a
   `partial_fit` capability and DAG-ML model nodes accept streaming folds?
   Current answer: out of scope; revisit in v1.1.
5. **Soft constraints in tuning**: should `TuningNodeSpec` carry hard / soft
   constraints (e.g. max parameter count, max wall time per trial) at the
   spec level, or as `TunerAdapter`-specific extras? Current answer: extras
   for v1; promote to spec if needed.
