# DAG-ML — Cœur polyglotte : document de travail

Statut : design en cours (discussion). Compagnon de `dag_ml_specification_v1.md`
(le moteur ML : graphe, phases, invariants OOF) et `ml_data_specification_v1.md`
(la couche données). Ce document explore une **variante d'implémentation** : un
cœur natif (Rust/C++) orchestrant des controllers et des données qui restent
dans le langage hôte (Python, R, ou natif), via une ABI stable.

Objectif du document : (1) restituer **l'intégralité** de la délibération
Python vs Rust/C++ (les deux options, leurs pros/cons/verrous, la voie médiane,
les frontières de marshalling, le portage cross-langage) ; (2) figer l'ABI de
controller ; (3) détailler trois mécanismes critiques — liveness des handles,
format du blob `describe`, trace d'exécution d'un stacking ; (4) discuter de la
possession du RNG par le cœur.

Note de lecture : la Partie I n'est pas une recommandation tranchée — c'est le
**compte rendu du débat**. Elle expose la voie « Python pur + kernels Rust
ciblés » *et* la voie « cœur natif polyglotte » à parité. Le choix dépend d'un
seul critère, énoncé en §I.13. Les Parties II→VII détaillent la voie polyglotte
parce que c'est celle qui demande une conception nouvelle ; ce n'est pas un
verdict contre la voie médiane.

---

## Partie I — Le débat Python vs Rust/C++ et la direction retenue

### I.1 Le constat fondateur

DAG-ML, tel que spécifié, **n'est pas un noyau de calcul — c'est un plan de
contrôle**. Il compile un DSL en DAG, énumère des variants, planifie, ordonnance
les phases (`COMPILE → PLAN → FIT_CV → SELECT → REFIT → PREDICT → EXPLAIN`), fait
les jointures OOF, gère lineage/cache/artifacts. Le calcul lourd (`fit`/`predict`/
`transform`) est **délégué** aux opérateurs qui enveloppent sklearn, PyTorch, TF,
Keras, LightGBM, XGBoost (§6.5 de la spec). Ces opérateurs *sont* des objets du
langage hôte ; leurs hot loops sont déjà en natif (BLAS, CUDA, C++).

Conséquence centrale : **DAG-ML ne fait quasiment aucun FLOP lui-même.** Les
seuls calculs qu'il possède en propre sont la jointure OOF (§8.2), l'agrégation
(§9.7), l'énumération de variants (§4.5), le hashing/fingerprint, le RNG de
contrôle, et le store de prédictions. Tout le reste est de l'I/O et de
l'orchestration. **On n'optimise pas la couche qui ne calcule pas** — mais on
peut vouloir la **porter** ailleurs que Python. C'est ce double constat (peu de
compute propre / désir de portage) qui structure tout le débat.

### I.2 Option A — Python pur

| | |
|---|---|
| **Pros** | • La spec **est déjà du Python** : dataclasses, `Protocol`, `Any`, duck typing (`matches()`). Implémenter = transcrire. Time-to-MVP minimal.<br>• Intégration zéro-friction avec tout l'écosystème opérateurs + Optuna/Ray (§10.6). Le contrat d'adapter *est* l'API Python de ces libs.<br>• Migration depuis nirs4all = refactor incrémental (§19), pas une réécriture.<br>• numpy/pyarrow/joblib/SQLite donnent gratuitement les stores colonne, la sérialisation d'artifacts, le Parquet.<br>• Équipe Python, webapp FastAPI → nirs4all : cohérence org + bassin de contributeurs.<br>• Débogabilité, introspection, pdb, payloads d'erreur riches : naturels.<br>• GIL pas un verrou réel : unités parallèles grossières (variant/fold), en process séparés (loky §14.3) ou natif qui relâche le GIL. |
| **Cons** | • L'overhead d'orchestration pur-Python (boucle topo, dicts dans l'OOF join, dataclasses `frozen` par millions) devient visible **à l'extrême échelle** : 10k+ variants × folds × millions de prédictions.<br>• Le scheduler loky impose le **pickle** des tasks/résultats → coût de sérialisation + duplication mémoire des gros `DataBlock` (atténué par CoW + Parquet §18).<br>• Aucune garantie compile-time sur les invariants : la correction repose sur checks runtime (§14.7) + tests.<br>• Empreinte mémoire des objets Python (PredictionBlock/LineageRecord). |
| **Verrous** | • **OOF join (§8.2) et agrégation (§9.7)** écrits en boucles Python par échantillon → à grand volume, **vectoriser** (numpy/pandas/pyarrow groupby). Verrou réel mais résolu *dans* Python.<br>• **Déterminisme** sous loky (§12.3) : fixer threads BLAS, trier le reducer par `fold_id`. Indépendant du langage.<br>• **Picklabilité** de `ExecutionPlan`/`NodePlan` (d'où `SerializableRef`). Gérable. |

### I.3 Option B — Cœur Rust/C++ + bindings Python

| | |
|---|---|
| **Pros** | • Les structures du plan de contrôle (GraphSpec, tri topo, énumération `_cartesian_`, fingerprint, graphe de lineage) : rapides, compactes, **vrais threads** (pas de GIL) pour l'orchestration.<br>• Enums exhaustifs Rust → certains invariants encodés **au compile-time** (NodeKind, contrats de port).<br>• Kernel OOF-join / agrégation sur **Arrow** (arrow-rs/polars) : zero-copy avec pyarrow, plus rapide que numpy vectorisé.<br>• Empreinte mémoire des millions de records lineage/prediction divisée.<br>• ABI propre pour plusieurs front-ends. |
| **Cons** | • **Les opérateurs sont en Python.** Tout cœur Rust doit rappeler Python (pyo3) pour exécuter **chaque** nœud MODEL/TRANSFORM → **réacquisition du GIL + traversée FFI à chaque nœud**. Le bénéfice "no-GIL" s'évapore là où le travail se fait.<br>• Architecture hybride : Rust tient le graphe, Python tient opérateurs + arrays. Chaque `NodeTask` fait transiter ndarrays + objets opérateurs (`Py<PyAny>`). Complexité énorme pour **zéro gain de compute**.<br>• Explosion build/CI : maturin/cibuildwheel, wheels par plateforme (linux/mac/win × CPython), manylinux, compat ABI numpy/pyarrow. Taxe lourde sur une CI ruff/mypy/pytest.<br>• La spec est *Python-shaped* (`operator: Any`, résolution dynamique par `matches()`) → la réexprimer en types Rust **combat le design**.<br>• Skillset équipe réduit. Time-to-value MVP ≈ **5–10×** pour une couche dont le runtime est dominé par du code qu'on ne possède pas. |
| **Verrous** | • **FFI sur opérateurs opaques** : tenir/invoquer des estimateurs Python arbitraires depuis Rust (`Py<PyAny>`, GIL token dans l'exécuteur) → exécuteur GIL-bound à chaque nœud.<br>• **Échange zero-copy** : exigerait Arrow de bout en bout, mais les opérateurs attendent numpy/torch → **reconversion/copie** à chaque frontière modèle.<br>• **Sérialisation d'artifacts** : joblib/torch/tf Python-only → `ArtifactStore` reste Python.<br>• **RNG/SeedContext** : seeds consommés par numpy/torch → côté Python.<br>• **Optuna/Ray** : Python-only. |

### I.4 Le verrou commun, fatal pour B en *mono-langage Python*

Un cœur Rust/C++ ne peut pas exécuter le pipeline sans embarquer un interpréteur
Python, parce que **toute la valeur exécutable (opérateurs, sérialisation,
tuners, RNG) vit dans l'orbite Python**. En déploiement Python-seul, on obtient
un orchestrateur Rust qui passe son temps à reprendre le GIL — le pire des deux
mondes : complexité Rust *plus* contention GIL.

Corollaire : **la spec a déjà choisi Python implicitement.** Chaque type est une
dataclass/Protocol ; joblib/SQLite/Parquet/loky/optuna/ray y sont nommés. « Faire
du Rust » signifie *réécrire la spec*, pas l'implémenter. Ce point n'invalide pas
B — il invalide B **si l'unique cible est Python**. Le retournement vient en §I.8.

### I.5 La voie médiane : Python + kernels Rust ciblés

Entre A et B, une troisième voie — celle des projets `polars` / `pydantic-core` /
`tokenizers` : surface et orchestration en Python, **noyau chaud en Rust pour 3-4
kernels mesurés comme chauds**, jamais un cœur monolithique.

1. **Implémenter DAG-ML en Python** maintenant (orchestration au-dessus de ML
   Python, spec Python, migration incrémentale §19).
2. **Garder l'archi FFI-friendly.** Les seuls vrais kernels que DAG-ML possède
   sont isolables derrière un `Protocol` :
   - `oof_join` + agrégation (§8.2 / §9.7) — candidat n°1, sur Arrow
   - énumération de très grands espaces `_cartesian_` (§4.5)
   - fingerprint/hashing canonique (SHA256 récurrent)
   - éventuellement le `PredictionStore` colonne (§8.1, déjà Parquet)
3. **Si — et seulement si — le profilage** montre que ces kernels dominent à
   grande échelle, les remplacer par une extension Rust (pyo3 + Arrow zero-copy),
   *derrière le même Protocol Python*.

Cette voie capte l'essentiel du gain stockage **sans aucun Rust** : il suffit de
backer `PredictionStore`/`FeatureTable` sur Arrow/Polars au lieu de
dataclasses+numpy. Rust n'ajoute de valeur que (a) pour les kernels de calcul sur
ces données Arrow et (b) **pour les bindings hors Python** — ce qui mène à §I.8.

### I.6 Les deux frontières à ne jamais confondre

Le coût de marshalling Rust/Python qui inquiète n'est pas uniforme. Il faut
séparer :

- **Frontière données** (arrays, tables de prédictions, feature tables) : le
  marshalling coûteux **n'est pas une fatalité**. Structures à partage direct
  (zero-copy) — §I.7.
- **Frontière comportement** (opérateurs : estimateurs sklearn, modules torch) :
  le coût d'appel reste, et aucune structure zero-copy ne le supprime, car
  exécuter du code hôte exige le runtime hôte.

**Les données sont partageables sans copie ; le comportement non.** DAG-ML fait
traverser les deux : les arrays peuvent rouler sur Arrow/DLPack quasi gratis ;
l'appel d'opérateur reste un call GIL + dispatch Python par nœud.

### I.7 Structures zero-copy disponibles, et où la copie revient

| Mécanisme | Usage | Zero-copy ? |
|---|---|---|
| **Apache Arrow C Data Interface** | tables colonnes entre langages | Oui — deux structs (`ArrowSchema`, `ArrowArray`), passage de pointeurs + release callback ; ABI C stable, sans dépendance de build. `pyarrow` ↔ `arrow-rs` ↔ R `arrow` ↔ JS. |
| **rust-numpy / buffer protocol (PEP 3118)** | ndarray numpy ↔ `ndarray::ArrayView` | Oui si **C-contigu** ; copie sinon |
| **DLPack** | tenseurs torch/tf/jax/cupy, **GPU compris** | Oui (`__dlpack__`) |

Donc pour un `PredictionStore` colonne, une `FeatureTable`, les arrays OOF : le
transfert est O(1), par pointeur. Le marshalling cher (pickle, JSON) ne concerne
que les frontières où l'on n'a *pas* standardisé sur Arrow.

**Trois fuites où la copie revient quand même :**

1. **Colonne (Arrow) vs row-major dense (sklearn).** Une matrice
   `(n_samples, n_features)` stockée en N colonnes Arrow → gather/transpose pour
   sklearn. Solvable (stocker en `FixedSizeList`/tensor extension = un buffer),
   mais choix délibéré.
2. **GPU pour sklearn** (CPU only ; DLPack ne règle que torch/tf).
3. **Les objets fittés ne traversent jamais en zero-copy** — état hôte opaque, le
   cœur n'en tient qu'un *handle*. Au predict, le cœur rappelle Python sous le
   GIL.

### I.8 L'argument qui change tout : le portage R / JS

C'est ici que l'Option B cesse d'être perdante. Le vrai moteur de B n'est pas la
perf — c'est **la portée multi-langage**, et le mécanisme est exactement Arrow :

- **Arrow C Data Interface est la lingua franca polyglotte** : cœur Rust ↔ R
  (package `arrow` / `extendr`), ↔ JS (`apache-arrow` / WASM), ↔ Python
  (`pyarrow`). C'est ainsi qu'on bâtit un cœur multi-langage (modèle Polars,
  DataFusion).

Mais il faut nommer le piège : **un plan de contrôle portable ne donne pas un
écosystème d'opérateurs portable.**

- **Portable** (et c'est le *cœur de rigueur* de la spec) : compilation DSL →
  GraphSpec, FoldSet, invariants OOF/leakage, jointure OOF, lineage,
  déterminisme, scheduler. Un cœur Rust porte tout cela vers R et JS.
- **Non portable** : sklearn, torch, lightgbm. En R → tidymodels/mlr3 ; en JS →
  tfjs/onnx ; ou réembarquer Python. DAG-ML orchestre des opérateurs ; le
  squelette est portable, **les choses orchestrées ne le sont pas**.

Bénéfice sous-estimé : le problème GIL **disparaît hors Python** — en R/JS les
opérateurs sont natifs, donc le call structurel n'a pas de GIL à reprendre. Le
surcoût FFI-par-nœud que dénonce le §I.4 est *spécifique à Python*.

### I.9 La vision polyglotte

Le cœur natif **ne possède ni les données ni les opérateurs**. Il possède : le
graphe, les phases, les folds, la jointure OOF, le lineage, le scheduler, le
search space, le RNG de contrôle — **les invariants**. Il manipule des **handles
opaques** (u64) résolus côté hôte, et ne voit en clair que l'**identité** et les
**prédictions** (en Arrow).

```
        dagml-core (Rust)                      bindings minces
  ┌────────────────────────────┐        ┌──────────────────────────┐
  │ Compiler / GraphSpec        │  pyo3  │ Python: controllers       │
  │ FoldSet, SearchSpace        │◄──────►│   sklearn / torch (handles)│
  │ OOFJoin / PredictionStore   │ extendr│ R: controllers mlr3       │
  │ Lineage / Cache / RNG       │◄──────►│                           │
  │ Scheduler                   │ napi/  │ JS/WASM: tfjs / onnx      │
  │  ── identité+preds = Arrow ─│  wasm  │ natif: nirs4all-methods   │
  └────────────────────────────┘        └──────────────────────────┘
   données   : Arrow C Data Interface (zero-copy)
   modèles   : ONNX / safetensors (inférence cross-langage)
   contrôle  : déterminisme cœur (folds, OOF, RNG)
```

Différence clé avec Polars : Polars *possède* les données ; ici le cœur en est
**aveugle** (sauf identité+prédictions). C'est ce qui le rend portable sans
réimplémenter un écosystème ML par langage. Les `DataBlock`/`FeatureTable`/
`PredictionBlock` traversent en zero-copy ; seuls les handles d'opérateurs restent
opaques côté langage hôte.

### I.10 Le GIL : persiste, mais pas goulot ; l'asymétrie R

- **Structurellement** : tout appel à un controller Python exige le GIL (PyO3
  impose un token `Python<'py>`). Pas d'échappatoire en CPython standard.
- **En débit** : le cœur est en Rust et ne touche jamais le GIL (traversée
  graphe, OOF, lineage, scheduling = threads Rust libres). Les controllers
  lourds **relâchent** le GIL pendant BLAS/CUDA. GIL tenu par `fit` ≈ µs de
  dispatch ; compute ≈ ms–s GIL-free. Ratio < 0,1 % → scaling threads
  quasi-linéaire. Le régime où le GIL sérialise (nœuds ~100 µs) est précisément
  celui où l'on n'a pas besoin de paralléliser. **C'est strictement supérieur**
  au loky/multiprocess de la spec (pas de pickle, pas de duplication mémoire).

**Asymétrie R** : R n'a pas de GIL mais **n'est pas thread-safe** et ne relâche
rien pendant le calcul. Parallélisme R → **processus obligatoires** (fork/PSOCK,
cf. `parallel`/`future`).

| Controllers | Parallélisme intra-process | Mécanisme |
|---|---|---|
| Python (torch/sklearn) | OK si relâche le GIL | threads + GIL bref |
| R (mlr3) | **non** | processus |
| Natif Rust (nirs4all-methods) | **libre, total** | threads, aucun verrou |

Le build full-natif est le **seul** sans contrainte de concurrence du langage
hôte — c'est le plafond de perf et le différenciateur. Les bindings Python/R sont
des modes "reach/compat" avec les contraintes de leur hôte.

### I.11 Trois canaux de transport, trois rôles

- **Arrow** = transport des *données* (cross-langage, zero-copy).
- **ONNX / safetensors** = transport des *modèles fittés* (inférence
  cross-langage). Déjà prévu : `ArtifactRef.backend` liste `"onnx"` (§13.1). Ne
  pas demander à Arrow de porter la repro des modèles.
- **Cœur natif + RNG de contrôle** = transport de la *rigueur* (folds, OOF,
  lineage — déterministe, et cross-langage pour le plan de contrôle ; Partie VI).

### I.12 Les deux principes que l'ABI grave

1. **Visibilité.** Le cœur voit en Arrow exactement trois choses : l'**identité**
   (sample/observation/target/group/origin ids), les **prédictions**
   (y_pred/y_proba), et **y_true** pour le scoring. Jamais X ni les features.
   Tout le reste = handles.
2. **Ownership.** Le cœur possède le **cycle de vie des handles** ; l'hôte
   possède l'**objet sous-jacent** et le libère sur `release`. Arrow porte son
   propre release callback (ownership auto-décrit).

Violer (1) réintroduit le marshalling du gros. Violer (2) = use-after-free
cross-langage ou fuite pilotée par le GC hôte.

### I.13 La décision, reformulée

Ce n'est **pas** "Python vs Rust pour la vitesse". C'est : **veux-tu un plan de
contrôle polyglotte ?**

- **Opérateurs essentiellement Python** → **voie médiane (§I.5)** : Python pur,
  Arrow comme format interne pour le gain stockage, kernels Rust chirurgicaux si
  le profilage le justifie. Un cœur Rust ici coûterait le GIL à chaque nœud
  modèle pour un gain de compute nul.
- **Portée R/JS visée** → **cœur natif polyglotte (§I.9)** : le marshalling
  données n'est pas l'obstacle (Arrow zero-copy) ; le GIL disparaît hors Python ;
  on obtient le **même moteur de rigueur ML en Python, R et JS**. Coût : CI
  multi-plateforme, ABI de plugin à figer, et on réécrit la spec au lieu de
  l'implémenter.

Le reste du document (Parties II→VII) détaille la voie polyglotte.

---

## Partie II — L'ABI

### 8. Substrat

- **Vtable C** (`#[repr(C)]`, pointeurs de fonction) — PGCD que PyO3, extendr et
  Rust-natif remplissent à l'identique.
- **Arrow C Data Interface** pour tout ce que le cœur lit.
- **Blobs canoniques versionnés** (`Bytes`) pour le riche-mais-évolutif (params,
  `ModelInputSpec`, descripteurs, erreurs). Jamais de struct C pour ça → l'ABI
  ne bouge pas quand la spec évolue.

### 9. `ControllerVTable`

```rust
// ───────────────────────── types de frontière ─────────────────────────
#[repr(C)] pub struct Bytes { ptr: *const u8, len: usize }      // annotation d'owner ci-dessous
#[repr(u8)] pub enum Phase { Compile, Plan, FitCv, Select, Refit, Predict, Explain }
#[repr(u8)] pub enum Status { Ok, Skip, Error }
#[repr(u8)] pub enum Backend { Joblib, Torch, Onnx, Safetensors, Json, Raw } // = ArtifactRef.backend §13.1
pub type HandleId = u64;                                         // 0 = null

#[repr(C)] pub struct ArrowSchema { /* Arrow CDI */ }
#[repr(C)] pub struct ArrowArray  { /* Arrow CDI ; porte son propre release */ }
#[repr(C)] pub struct ArrowOut { schema: *mut ArrowSchema, array: *mut ArrowArray } // [owned→core]
#[repr(C)] pub struct NamedHandle { port: Bytes, handle: HandleId }                  // entrée multi-source

// capabilities (bitset) — fusionne AdapterSpec.capabilities §6.1 + ResourceHints §2.4
pub const SUPPORTS_PREDICT:      u64 = 1<<0;
pub const SUPPORTS_PROBA:        u64 = 1<<1;
pub const SUPPORTS_EXPLAIN:      u64 = 1<<2;
pub const STATEFUL:              u64 = 1<<3;   // fit -> fitted handle ; sinon stateless
pub const INVERTIBLE:            u64 = 1<<4;   // y_transform
pub const GIL_FREE_COMPUTE:      u64 = 1<<5;   // relâche le GIL pendant le compute -> thread-schedulable
pub const THREAD_SAFE:           u64 = 1<<6;   // instances concurrentes OK
pub const DETERMINISTIC:         u64 = 1<<7;   // (seed/stream) reproductible -> cache valide
pub const REQUIRES_DATASET_PLAN: u64 = 1<<8;   // planification différée à FIT_CV §5.3
pub const EMITS_RELATION:        u64 = 1<<9;   // change l'ensemble des samples (augmentation)
pub const RNG_FROM_CORE:         u64 = 1<<10;  // tire tout son aléa du RNG du cœur (Tier 1, Partie VI)

// ───────────────────────── contexte d'appel (cœur → hôte) ─────────────────────────
#[repr(C)] pub struct CallContext {
    phase: Phase,
    run_id: Bytes, variant_id: Bytes, node_id: Bytes,           // [borrowed]
    fold_id: Bytes,                                             // "0".."K-1"|"final"|""
    branch_path: Bytes,                                         // tuple canonique
    rng_stream: u64,                                            // stream RNG dérivé par le cœur (Partie VI)
    rng_seed_legacy: u64,                                       // graine entière dérivée (Tier 2, frameworks)
    callbacks: *const CoreCallbacks,
}
#[repr(C)] pub struct CoreCallbacks {                           // upcalls hôte → cœur (minimisés)
    ctx: *mut c_void,
    log_metric:      extern "C" fn(*mut c_void, name: Bytes, value: f64),
    report_progress: extern "C" fn(*mut c_void, done: u64, total: u64),
    check_cancel:    extern "C" fn(*mut c_void) -> bool,
    rng_fill_f64:    extern "C" fn(*mut c_void, stream: u64, n: u64) -> ArrowOut,   // tirage cœur (Tier 1)
    rng_permutation: extern "C" fn(*mut c_void, stream: u64, n: u64) -> ArrowOut,   // permutation cœur
}

// ───────────────────────── enveloppes de résultat ─────────────────────────
#[repr(C)] pub struct FitResult {
    status: Status,
    fitted: HandleId,        // [handle, owned→host] 0 si erreur/skip
    metrics: ArrowOut,       // [owned→core] nullable
    error: Bytes,            // [owned→host] ErrorPayload §14.6 si Error
}
#[repr(C)] pub struct PredictResult {
    status: Status,
    predictions: ArrowOut,   // [owned→core] y_pred (+ proba en colonnes), schéma auto-décrit
    target_space: Bytes,     // [owned→host] "raw"|"scaled"|...
    metrics: ArrowOut,       // [owned→core] nullable
    error: Bytes,
}
#[repr(C)] pub struct TransformResult {
    status: Status,
    fitted: HandleId,        // [handle] 0 si stateless
    output: HandleId,        // [handle, owned→host] X transformé — JAMAIS lu par le cœur
    relation: ArrowOut,      // [owned→core] nullable ; non-null si EMITS_RELATION (origin_id…)
    error: Bytes,
}

// ───────────────────────── la vtable ─────────────────────────
#[repr(C)] pub struct ControllerVTable {
    abi_version: u32,
    state: *mut c_void,            // instance controller / env de closure hôte
    kind: u8,                      // NodeKind servi (§2.1)
    capabilities: u64,

    // — construction lazy : le cœur pilote QUELS params (search space), l'hôte COMMENT —
    clone_with: extern "C" fn(state: *mut c_void, operator: HandleId, params: Bytes) -> HandleId,

    // — introspection PLAN (sans données) —
    describe: extern "C" fn(state: *mut c_void, operator: HandleId) -> Bytes,   // [owned→host] cf. §13
    matches:  Option<extern "C" fn(state: *mut c_void, kind: u8, operator: HandleId) -> bool>,

    // — SPLIT (identité ; data handle optionnel pour splitters feature-based, ex. KS/SPXY) —
    split: Option<extern "C" fn(state: *mut c_void, operator: HandleId,
                                identity: *mut ArrowArray, identity_schema: *mut ArrowSchema, // [borrowed]
                                y: *mut ArrowArray, y_schema: *mut ArrowSchema,                // nullable
                                data: HandleId,                                                // 0 sauf feature-based
                                ctx: *const CallContext) -> ArrowOut>,  // (sample_id, fold_id, partition)

    // — FIT-like — (target = handle : résolu hôte-side, jamais marshalé dans le controller) —
    fit: Option<extern "C" fn(state: *mut c_void, operator: HandleId,
                              inputs: *const NamedHandle, n_inputs: usize, target: HandleId,
                              ctx: *const CallContext) -> FitResult>,
    transform_fit: Option<extern "C" fn(state: *mut c_void, operator: HandleId,
                                        input: HandleId, ctx: *const CallContext) -> TransformResult>,

    // — APPLY-like —
    predict: Option<extern "C" fn(state: *mut c_void, fitted: HandleId,
                                  inputs: *const NamedHandle, n_inputs: usize, want_proba: bool,
                                  ctx: *const CallContext) -> PredictResult>,
    transform_apply: Option<extern "C" fn(state: *mut c_void, fitted: HandleId,
                                          input: HandleId, ctx: *const CallContext) -> TransformResult>,
    invert: Option<extern "C" fn(state: *mut c_void, fitted: HandleId,
                                 y_in: *mut ArrowArray, y_schema: *mut ArrowSchema) -> ArrowOut>,

    // — EXPLAIN (feuille ; sortie opaque-au-cœur) —
    explain: Option<extern "C" fn(state: *mut c_void, fitted: HandleId,
                                  inputs: *const NamedHandle, n_inputs: usize,
                                  cfg: Bytes, ctx: *const CallContext) -> ArrowOut>,

    // — persistance / inférence cross-langage —
    serialize:   extern "C" fn(state: *mut c_void, fitted: HandleId, backend: Backend) -> Bytes,
    deserialize: extern "C" fn(state: *mut c_void, blob: Bytes, backend: Backend) -> HandleId,

    // — cache (override optionnel ; sinon le cœur calcule la clé §13.3) —
    cache_key: Option<extern "C" fn(state: *mut c_void, operator: HandleId,
                                    inputs: *const NamedHandle, n_inputs: usize,
                                    ctx: *const CallContext) -> Bytes>,

    // — cycle de vie —
    release:    extern "C" fn(state: *mut c_void, handle: HandleId),
    free_bytes: extern "C" fn(state: *mut c_void, b: Bytes),
    destroy:    extern "C" fn(state: *mut c_void),
}
```

Mapping vers la spec §6 : `describe` ⊃ `input_spec`+`aux_inputs`+`declare_ports`+
`supported_phases` ; `fit`/`predict`/`predict_proba` → `fit`/`predict(want_proba)` ;
`transform_*` → `TransformerMixin` ; `invert` → `inverse_transform` ;
`cache_key` → idem. `clone_with` et `split` sont des raffinements ABI nécessaires
révélés par la trace (Partie V).

### 10. `DataVTable` (contre-pas obligatoire)

La signature controller est vide de sens sans définir qui fabrique/résout les
handles et qui expose l'identité au cœur.

```rust
#[repr(C)] pub struct DataVTable {
    abi_version: u32, state: *mut c_void,
    materialize:   extern "C" fn(state, source_id: Bytes, ctx: *const CallContext) -> HandleId,
    view_identity: extern "C" fn(state, h: HandleId) -> ArrowOut,  // sample/obs/target/group/origin ids
    make_view:     extern "C" fn(state, h: HandleId,               // slice par sample-ids (folds §7.3)
                                 ids: *mut ArrowArray, sch: *mut ArrowSchema, partition: u8) -> HandleId,
    target_arrow:  extern "C" fn(state, h: HandleId) -> ArrowOut,  // y_true pour scoring (cœur)
    ingest_arrow:  extern "C" fn(state, *mut ArrowArray, *mut ArrowSchema) -> HandleId, // OOF -> handle
    handle_nbytes: extern "C" fn(state, h: HandleId) -> u64,       // budget step-cache §13.4
    schema_fingerprint: extern "C" fn(state, h: HandleId) -> Bytes,
    release: extern "C" fn(state, h: HandleId), destroy: extern "C" fn(state),
}
```

Point décisif : `make_view` slice **par sample-ids**, jamais par positions —
l'invariant anti-leakage (§7.3) est porté *par l'ABI elle-même*. Le controller
reçoit un handle déjà tranché ; il ne peut pas regarder hors du fold.

### 11. Les trois implémentations remplissent la **même** struct

- **Python (PyO3)** : chaque fn acquiert le GIL, résout `handle→objet` dans une
  registry hôte, appelle `estimator.fit(X, y)`, ré-enregistre le fitted, rend un
  handle. `clone_with` = `sklearn.clone` + `set_params`. `serialize(Onnx)` =
  `skl2onnx`/`torch.onnx`.
- **R (extendr)** : idem ; évaluateur R non thread-safe → `THREAD_SAFE=0`,
  `GIL_FREE_COMPUTE=0` pour tous → scheduler route en **processus**.
- **Rust natif (nirs4all-methods)** : les fns *sont* du Rust ; handle = index
  dans une slab ; `GIL_FREE_COMPUTE=1`, `THREAD_SAFE=1` ; zéro marshalling, zéro
  indirection au-delà du pointeur de vtable.

---

## Partie III — (a) Protocole de liveness des handles

Le verrou n°1. L'ABI rend l'ownership explicite (`release` + release Arrow), mais
la **correction du suivi de liveness** à travers branches/folds/sous-DAG est à la
charge du cœur. Bug = use-after-free (release trop tôt) ou fuite (jamais de
release, pilotée par le GC hôte).

### a.1 Les deux approches naïves

| Approche | Principe | + | − |
|---|---|---|---|
| **Refcount par arête** | refcount = nb de consommateurs ; décrément à chaque consommateur fini ; release à 0 | release prompt, mémoire minimale | branches/folds/map multiplient les consommateurs ; cache complique ; skips à gérer |
| **Sweep par phase/scope** | tout handle d'une génération vit jusqu'à la fin du scope, puis release en bloc | simple, robuste | mémoire = pic du scope entier |

### a.2 La synthèse retenue : arènes scopées + refcount sur les échappés

Chaque exécution de fold/branche/variant est une **arène**. Règle :

- Un handle créé dans une arène et **non échappé** est libéré en bloc à la
  fermeture de l'arène (sweep, simple et infaillible).
- Un handle qui **échappe** l'arène (ex. un modèle de base qui survit jusqu'au
  REFIT ; une prédiction OOF qui alimente le join ; un handle mis en cache) est
  **refcounté** et promu au scope parent.

**Invariant de refcount d'un handle échappé :**

```
refcount(h) = (# nœuds consommateurs non encore exécutés dans le plan, h en entrée)
            + (1 si h est référencé par le CacheStore)
            + (1 si h est promu vers le bundle / REFIT)
```

Quand `refcount(h)` atteint 0 → `vtable.release(state, h)`.

### a.3 Points de déclenchement

| Évènement | Action |
|---|---|
| Nœud terminé | pour chaque handle en entrée : si échappé, décrément ; sinon ne touche pas (l'arène s'en charge) |
| Nœud *skipped* (phase non supportée) ou en erreur | décrémente quand même les refs de ses entrées (sinon fuite) |
| Fermeture d'arène (fin de fold/branche) | sweep : release de tous les handles locaux non échappés |
| Mise en cache d'un handle | +1 ref (cache) → survit à l'arène |
| Éviction LRU (budget `step_cache_max_mb` via `handle_nbytes`) | −1 ref (cache) ; release si 0 |
| Promotion au bundle (REFIT) | +1 ref ; libérée à l'export du bundle |

### a.4 Connaître le nombre de consommateurs

Depuis `ExecutionPlan.topological_order` + les arêtes sortantes par port. Pour le
fan-out dynamique (`MapNode` sur branches), le compte est connu **après expansion
du fork**. Pour les folds, chaque fold est une instance de consommateur distincte.

### a.5 Cross-process

Les refcounts sont **par process**. Un worker (variant) ouvre une arène,
l'exécute, et au retour : `destroy` de ses vtables → bulk-free de tout. Les
résultats remontés à l'orchestrateur sont de l'**Arrow** (déjà owned→core), donc
la liveness cross-process est triviale (bornée à la vie du worker). Le step-cache
est alors **local au worker** — utile seulement pour des batches de variants
assignés au même worker (dégradation acceptable).

### a.6 Micro-exemple (un fold de UC6, branche b0)

```
arène fold0 ouverte
  Hd:train0  = make_view(Hd:nir, ids_train0, train)      # local arène
  Hd:val0    = make_view(Hd:nir, ids_val0,   val)         # local arène
  Hd:Xt_tr   = transform_fit(SNV, Hd:train0).output       # local (SNV stateless)
  Hd:Xt_val  = transform_apply(SNV, Hd:val0).output        # local
  Hf:pls0    = fit(PLS, [X:Hd:Xt_tr], y_tr).fitted         # ÉCHAPPE -> refcount=2 (REFIT? non; predict0 + ?)
  A:pred_val = predict(Hf:pls0, [X:Hd:Xt_val])             # Arrow -> PredictionStore (owned core)
  # Hf:pls0 : consommé par predict0 -> décrément ; pas d'autre consommateur en CV -> release
  #   (au REFIT, un NOUVEAU Hf:pls_final sera fitté sur full train ; le pls0 du fold ne survit pas)
fermeture arène fold0 -> sweep: release Hd:train0, Hd:val0, Hd:Xt_tr, Hd:Xt_val
```

Seul l'`A:pred_val` (Arrow, petit) survit, dans le `PredictionStore`. Tous les
handles X du fold sont balayés. Pic mémoire = un fold à la fois.

---

## Partie IV — (b) Format du blob `describe`

`describe(operator) -> Bytes` est le **contrat de PLAN**. Encodage : JSON
canonique (clés triées, UTF-8) préfixé d'un tag de version
`"dagml.describe/1"`. Choix JSON pour v1 : débogable, stable, présent dans tout
langage. (Msgpack possible si compacité requise ; le schéma logique est
identique.)

### b.1 Schéma logique

```jsonc
{
  "v": "dagml.describe/1",
  "adapter": { "id": "sklearn.estimator", "version": "1.0.0", "kind": "model" },
  "phases": ["fit_cv", "refit", "predict"],
  "capabilities": ["supports_predict", "stateful", "deterministic",
                   "gil_free_compute", "thread_safe"],
  "ports": {
    "inputs":  [{ "name": "X", "kind": "data",
                  "representation": "tabular_numeric", "cardinality": "one" }],
    "outputs": [{ "name": "y_pred", "kind": "prediction",
                  "representation": "tabular_numeric", "cardinality": "one" }]
  },
  "input_spec": {                       // = ModelInputSpec (ml_data.contract)
    "representation": "tabular_numeric",
    "rank": 2, "dtype": "float32", "layout": "row_major",
    "required_sources": ["*"],          // "*" = toute source fusionnée ; sinon noms
    "accepts_missing": false,
    "max_features": null
  },
  "aux_inputs": [],                     // ex. [{"name":"wavelengths","representation":"axis_coords","required":false}]
  "planning": {
    "requires_dataset_at_plan": false,  // true -> data_plan différé à FIT_CV (§5.3)
    "allow_lossy": false,
    "fit_scope": "fold_train"           // fold_train | train_once | stateless
  },
  "identity_params": ["n_components"],  // params qui entrent dans le fingerprint/cache
  "resources": {                         // = ResourceHints (§2.4)
    "cpu": 1, "gpu": 0, "memory_mb": null,
    "thread_safe": true, "nested_parallelism": "forbid", "timeout_seconds": null
  }
}
```

### b.2 Exemple — CNN PyTorch (contraste)

```jsonc
{
  "v": "dagml.describe/1",
  "adapter": { "id": "pytorch.module", "version": "1.0.0", "kind": "model" },
  "phases": ["fit_cv", "refit", "predict", "explain"],
  "capabilities": ["supports_predict", "supports_explain", "stateful"],   // PAS gil_free pendant le fit Python pur
  "ports": { "inputs":  [{ "name": "X", "kind": "data",
                           "representation": "signal_with_processings", "cardinality": "one" }],
             "outputs": [{ "name": "y_pred", "kind": "prediction",
                           "representation": "tabular_numeric", "cardinality": "one" }] },
  "input_spec": { "representation": "signal_with_processings",
                  "rank": 3, "dtype": "float32", "layout": "row_major",
                  "required_sources": ["*"], "accepts_missing": false, "max_features": null },
  "aux_inputs": [],
  "planning": { "requires_dataset_at_plan": true,    // l'archi dépend de la dim d'entrée -> différé
                "allow_lossy": false, "fit_scope": "fold_train" },
  "identity_params": ["arch", "lr", "epochs", "batch_size"],
  "resources": { "cpu": 4, "gpu": 1, "gpu_memory_mb": 4096,
                 "thread_safe": false, "nested_parallelism": "forbid", "timeout_seconds": 3600 }
}
```

### b.3 Champs qui pilotent une décision du cœur

| Champ | Décision cœur |
|---|---|
| `requires_dataset_at_plan` | si `true` → `data_plan=None` au PLAN, re-résolution dans le scope du fold avant fit (§5.3) |
| `fit_scope` | `fold_train` = refit par fold (OOF correct, cher) ; `train_once` = fit unique sur train complet figé pour la CV (leakage léger, cas UC1 friction #4) ; `stateless` = pas de fitted |
| `capabilities: gil_free_compute / thread_safe` | choix threads vs processus (§I.10) |
| `identity_params` | entrée du fingerprint + clé de cache (§13.3) |
| `resources` | scheduler : overcommit, parallélisme imbriqué, timeout |
| `allow_lossy` | refus / escalation `requires_user_choice` au PLAN (UC1) |

---

## Partie V — (c) Trace d'exécution : UC6 (stacking 3 voies + méta Ridge)

Pipeline (rappel) : `nir(500) → y standardize → KFold(5,rs42) outer → branch[
SNV+PLS(12) | MSC+RF(300) | Detrend+SVR(rbf,C10) ] → merge predictions(validate
OOF) → KFold(5,rs42) inner → Ridge(1.0)`.

Notation : `[Ho:x]` handle opérateur · `[Hd:x]` handle data · `[Hf:x]` handle
fitted · `[A:x→core]` Arrow possédé par le cœur · `vt.X` = appel vtable.

### COMPILE (front hôte → cœur)

```
HÔTE: parse DSL, enregistre les opérateurs dans la registry hôte, émet une ProtoGraph neutre:
      SNV→[Ho:1] PLS12→[Ho:2] MSC→[Ho:3] RF300→[Ho:4] Detrend→[Ho:5] SVR→[Ho:6]
      Ridge→[Ho:7] yStd→[Ho:8] KFouter→[Ho:9] KFinner→[Ho:10]
CŒUR: GraphSpec, contrôle acyclicité + arités de ports. Pas de search space ici -> 1 variant.
```

### PLAN

```
CŒUR: pour chaque opérateur data-aware:
      vt.describe([Ho:2]) -> {model, tabular_numeric, rank2, fit_scope=fold_train, identity_params=[n_components]}
      vt.describe([Ho:4]) -> {model, ..., capabilities sans rng_from_core (RF bootstrap = Tier 2)}
      vt.describe([Ho:6]) -> {model, ..., deterministic (SVR sans RNG)}
      KFold (identité only) -> splitter NATIF du cœur, pas de vt.split (Tier 1, voir Partie VI)
      résout DataPlans (tabular_numeric), schema_fingerprint.
```

### FIT_CV — niveau 0 (bases)

```
CŒUR: data.materialize("nir") -> [Hd:nir]   ; data.view_identity([Hd:nir]) -> [A:ids 500]
CŒUR: split outer NATIF avec RNG cœur(stream dérivé de path "split:outer") -> [A:folds_outer]   # Tier 1
```

Pour **fold0** (folds 1–4 identiques, chacun avec son sous-stream de path) :

```
arène fold0:
  data.make_view([Hd:nir], ids_train0, train) -> [Hd:tr0]
  data.make_view([Hd:nir], ids_val0,   val)   -> [Hd:val0]

  # y standardize, fit sur train (per-fold, anti-leakage), invert gardé pour le scoring
  vt.transform_fit([Ho:8], target_handle(tr0)) -> [Hf:ystd0], y_scaled_train (host-side)

  # ── branche b0 : SNV + PLS ──
  vt.transform_fit([Ho:1], [Hd:tr0])  -> ([Hf:0 stateless], [Hd:Xt_tr_b0])      # SNV stateless
  vt.transform_apply([Ho:1], [Hd:val0]) -> [Hd:Xt_val_b0]
  vt.fit([Ho:2], inputs=[(X,[Hd:Xt_tr_b0])], target=y_scaled_train, ctx{fold0}) -> [Hf:pls0]
  vt.predict([Hf:pls0], inputs=[(X,[Hd:Xt_val_b0])], want_proba=false) -> [A:pred_b0_scaled→core]
  vt.invert([Hf:ystd0], [A:pred_b0_scaled]) -> [A:pred_b0_raw→core]
  CŒUR: PredictionStore.append(producer=b0, fold=0, partition=val, ids=ids_val0, y_pred=pred_b0_raw)

  # ── branche b1 : MSC (stateful) + RF (bootstrap = Tier 2) ──
  vt.transform_fit([Ho:3], [Hd:tr0]) -> ([Hf:msc0], [Hd:Xt_tr_b1])              # MSC apprend le spectre moyen
  vt.transform_apply([Hf:msc0], [Hd:val0]) -> [Hd:Xt_val_b1]
  vt.fit([Ho:4], [(X,[Hd:Xt_tr_b1])], y_scaled_train, ctx{rng_seed_legacy})     # RF: random_state = seed Tier 2
       -> [Hf:rf0]
  vt.predict([Hf:rf0], [(X,[Hd:Xt_val_b1])]) -> [A:pred_b1_scaled→core]
  vt.invert([Hf:ystd0], [A:pred_b1_scaled]) -> [A:pred_b1_raw→core] ; append(b1,fold0,val)

  # ── branche b2 : Detrend + SVR (déterministe) ──
  vt.transform_fit([Ho:5], [Hd:tr0]) -> ([Hf:0], [Hd:Xt_tr_b2])
  vt.transform_apply([Ho:5], [Hd:val0]) -> [Hd:Xt_val_b2]
  vt.fit([Ho:6], [(X,[Hd:Xt_tr_b2])], y_scaled_train) -> [Hf:svr0]
  vt.predict([Hf:svr0], [(X,[Hd:Xt_val_b2])]) -> [A:pred_b2_scaled→core]
  vt.invert([Hf:ystd0], [A:pred_b2_scaled]) -> [A:pred_b2_raw→core] ; append(b2,fold0,val)

fermeture arène fold0 -> sweep: release [Hd:tr0],[Hd:val0],[Hd:Xt_*],[Hf:pls0],[Hf:rf0],[Hf:svr0],
                                        [Hf:msc0],[Hf:ystd0]   # rien n'échappe en CV niveau 0
```

Après les 5 folds : `PredictionStore` contient 3 producteurs × 500 OOF (chaque
sample prédit une fois, dans son fold de validation).

### join:pred — `oof_join` (100 % cœur, Rust, sur Arrow)

```
CŒUR (aucun appel vtable):
  pour b0,b1,b2: vérifie couverture des 500 ids en partition=val
                 vérifie ABSENCE de partition=train  -> sinon OOFLeakageError (allow_train_predictions=false)
  construit [A:meta_features→core] : colonnes (b0_pls,b1_rf,b2_svr), 500 lignes, indexées sample_id
  data.ingest_arrow([A:meta_features]) -> [Hd:meta]      # l'OOF ré-entre comme donnée native
```

### FIT_CV — niveau 1 (méta)

```
CŒUR: split inner NATIF, RNG cœur(path "split:inner") -> [A:folds_inner]     # Tier 1, mêmes 500 samples
pour chaque inner fold j:
  arène inner_j:
    data.make_view([Hd:meta], ids_tr_j, train) -> [Hd:meta_tr_j]
    data.make_view([Hd:meta], ids_val_j, val)  -> [Hd:meta_val_j]
    vt.fit([Ho:7], [(X,[Hd:meta_tr_j])], target=y_train_j) -> [Hf:ridge_j]
    vt.predict([Hf:ridge_j], [(X,[Hd:meta_val_j])]) -> [A:meta_pred_j→core]
    PredictionStore.append(producer=ridge, fold=j, val)
  fermeture -> sweep
```

### SELECT

```
CŒUR: rmsecv_meta depuis meta OOF (500) + y_true (data.target_arrow). 1 variant -> sélectionné. SelectionRecord.
```

### REFIT (fold_id="final")

```
CŒUR: ouvre arène "final"
  # bases refit sur FULL train (500)
  pour b in {b0,b1,b2}:
     vt.transform_fit(preproc_b, [Hd:nir_fulltrain]) -> Xt_full_b
     vt.fit(model_b_op, [(X,Xt_full_b)], y_full) -> [Hf:base_b_final]   # ÉCHAPPE -> promu bundle (ref+1)
  # méta refit sur les OOF (PAS sur les preds des bases refit -> sinon train leak)
  vt.fit([Ho:7], [(X,[Hd:meta])], y_full) -> [Hf:ridge_final]            # ÉCHAPPE -> promu bundle
  # sérialisation portable
  pour h in {base0,base1,base2,ridge}_final: vt.serialize(h, Onnx|Joblib) -> bytes -> ArtifactStore
  export bundle: graph + plan + 4 artifacts + 3 caches OOF + schema_fingerprint
  release des handles promus après export
```

### Lecture de la trace

- **Handles côté hôte** : tout X transformé, tout fitted. Ne traversent jamais.
- **Arrow → cœur** : folds, prédictions, y_true, table OOF, identité. Tout petit.
- **L'unique matérialisation lourde** : `ingest_arrow(meta_features)` — mais
  c'est 500×3, négligeable.
- **Validation de leakage** : 100 % cœur, sans toucher l'hôte (Rust sur Arrow).
- **Deux tiers de RNG visibles** : KFold via RNG cœur (Tier 1) ; bootstrap RF via
  `rng_seed_legacy` (Tier 2). Voir Partie VI.
- **Refit méta sur OOF, pas sur preds refit** : invariant anti-leak porté par le
  cœur, pas par le controller.

---

## Partie VI — Discussion : le RNG dans la lib (le cœur)

Question : peut-on faire **posséder le RNG par le cœur** plutôt que de déléguer à
la RNG du langage hôte (numpy/R/torch) ? Enjeu : la reproductibilité
cross-langage, qui échouait jusqu'ici parce que `numpy.random ≠ RNG de R ≠ torch`.

### VI.1 Qu'est-ce qui est aléatoire dans un pipeline ML ?

| Source d'aléa | Espace | Possédable par le cœur ? |
|---|---|---|
| Splits / shuffles de folds | indices / identité | **Oui** — c'est déjà côté cœur |
| Échantillonnage du tuner (random/Bayes) | params | **Oui** — déjà côté cœur (search space) |
| Bootstrap (RF) : tirage d'indices | indices | **Oui en principe** (indices), **non en pratique** (sklearn ne laisse pas injecter un flux externe) |
| Augmentation : *sélection* (quels samples, paires mixup, coefficients) | indices / scalaires | **Oui** — petit, pré-tirable |
| Augmentation : *tenseur de bruit* (shape features) | feature-space | **Non** sans voir les dims (gros) |
| Init de poids NN / masques dropout | paramètres framework | **Non** — internes torch/tf, irredirigeables |

**Clé** : le cœur peut posséder le RNG de tout le **plan de contrôle**
(indices/identité/params) — précisément là où la rigueur ML l'exige. Il ne peut
pas posséder le RNG **interne aux frameworks** (init de poids).

### VI.2 La techno habilitante : PRNG compteur, splittable

Un RNG **counter-based** (Philox, Threefry) ou **splittable** (SplitMix, le PRNG
à la JAX) est :

- **portable et reproductible cross-langage par construction** — arithmétique
  entière pure, rounds/constantes définis : même clé+compteur → mêmes bits
  partout (contrairement aux Mersenne Twister dont le *seeding* diffère). C'est
  exactement pourquoi JAX utilise Threefry et pourquoi numpy a ajouté Philox/PCG.
- **splittable** : chaque chemin (run, variant, node, fold, branch, aug_index)
  dérive une sous-clé indépendante, déterministe, **sans coordination**.
- **O(1) en accès** (counter-based) → trivialement parallèle.

### VI.3 Réécriture de `SeedContext` en arbre de PRNG splittable

Aujourd'hui `SeedContext.derived() = SHA256(root || path) mod 2³²` (§12) — perd
de l'entropie, n'est pas un flux. Proposition :

```
key(path)  = SHA256(canonical_utf8(path_labels))[:16]      # 128-bit, portable
stream     = Philox(key=key(path), counter=0)              # flux indépendant par chemin
```

`SeedContext.child(...)` devient un *split* du PRNG. Le `CallContext` transporte
un `rng_stream: u64` (id de flux résolu par le cœur) et, pour les frameworks
irredirigeables, un `rng_seed_legacy: u64` (entier dérivé du même flux).

### VI.4 Deux tiers de reproductibilité (à assumer explicitement)

- **Tier 1 — RNG de contrôle, possédé par le cœur, cross-langage bit-identique.**
  Splits, échantillonnage tuner, *sélection* d'augmentation, bootstrap *quand*
  l'opérateur accepte un flux externe. Le controller déclare `RNG_FROM_CORE` et
  tire son aléa via les upcalls `rng_fill_f64` / `rng_permutation`, **ou** le
  cœur **pré-tire** les valeurs et les passe en Arrow (cas des splits : le cœur
  fait le split nativement et passe `[A:folds]`).

- **Tier 2 — RNG interne au framework, possédé par l'hôte, reproductible
  intra-lib seulement.** Init de poids, dropout, bootstrap sklearn. Le cœur passe
  `rng_seed_legacy` ; reproductible uniquement à lib/version/plateforme égales.

Le couple `RNG_FROM_CORE + DETERMINISTIC` ⇒ bit-identité cross-langage. Son
absence ⇒ Tier 2.

### VI.5 Mécanisme de passage (efficacité)

- **Petits tirages** (permutations, sélections, coefficients) : le cœur
  **pré-tire en Arrow**. Toujours bit-identique, l'hôte n'a besoin d'aucune RNG.
  Couvre splits, tuner, sélection d'augmentation, indices de bootstrap.
- **Tirages feature-shaped** (gros tenseur de bruit) : soit upcall `rng_fill`
  par chunks (Tier 1, coût d'appel), soit accepté en Tier 2. v1 : **contrôle =
  Arrow pré-tiré (Tier 1)** ; **interne framework = seed (Tier 2)**.

### VI.6 Le gain inattendu : déterminisme indépendant du scheduling

Comme un PRNG splittable détermine le flux **par le chemin, pas par l'ordre
d'exécution**, le résultat est identique que les folds tournent en séquentiel, en
threads, en loky ou en Ray. Le point §12.3.3 de la spec (« le scheduler ne change
pas l'ordre des résultats ») devient **gratuit** pour tout l'aléa de contrôle.
C'est un argument fort en plus de la portabilité.

### VI.7 Cas des splitters

- **Splitters identité-only** (KFold, ShuffleSplit, GroupKFold, Stratified : ne
  lisent que ids/y/groups) → **natifs au cœur**, RNG cœur, Tier 1, cross-langage.
  Pas d'appel `vt.split`.
- **Splitters feature-based** (KennardStone, SPXY : distances spectrales → ont
  besoin de X) → controllers hôtes via `vt.split(..., data=handle)`. Ils sont
  **déterministes** (greedy, sans RNG) ; leur reproductibilité est un problème de
  déterminisme numérique (ties de distances), pas de RNG.

### VI.8 Limites honnêtes

- Les internes des frameworks (init torch, bootstrap sklearn) restent Tier 2 :
  **frontière fondamentale, pas un échec de design**.
- Chaque controller Tier 1 doit accepter son aléa en entrée (Arrow) plutôt que
  d'appeler sa RNG native — **adaptation requise**, surtout pour l'augmentation.
  Les splitters identité sont gratuits (le cœur les fait).
- Deux tiers à documenter, et un drapeau (`RNG_FROM_CORE`) à tenir honnête sous
  peine de fausse promesse de reproductibilité.

### VI.9 Décision proposée

Mettre le **PRNG splittable counter-based dans le cœur** (Philox/Threefry) et
réécrire `SeedContext` en arbre de flux. Couvrir le plan de contrôle en Tier 1
(Arrow pré-tiré). Conserver `rng_seed_legacy` pour le Tier 2. Documenter les deux
tiers comme contrat de reproductibilité.

### VI.10 Conséquence : reproductibilité cross-langage avec nirs4all-methods

Question concrète : avec les méthodes C++ portables validées (**nirs4all-methods**),
peut-on avoir un pipeline Python — auquel on pourrait *en plus* greffer un modèle
pur Python (torch, sklearn) — et un pipeline R, qui donneraient **exactement les
mêmes résultats**, à l'exception des modèles propres à chaque langage ?

**Oui** — et l'atout décisif est précisément de **posséder nirs4all-methods en
C++**. Décomposition par type de nœud :

| Type de nœud | Python | R | Identique entre les deux ? |
|---|---|---|---|
| Preprocessing / modèle **nirs4all-methods (C++)** | même binaire C++ | même binaire C++ | **Oui, bit-identique** |
| Plan de contrôle (folds, OOF, sélection, aléa) | cœur + RNG Tier 1 | cœur + RNG Tier 1 | **Oui, bit-identique** |
| Modèle **pur langage** (torch/sklearn / mlr3) | présent | absent ou différent | **Non — c'est l'exception** |

Pourquoi c'est solide : c'est *ton* code. Le même binaire C++, appelé depuis le
binding Python ou R, sur les **mêmes données** (nirs4all-io C++) et le **même aléa
de contrôle** (RNG du cœur, Tier 1), produit des résultats **bit-identiques** —
une garantie que numpy/torch/R-natif ne donnent jamais, faute de contrôler leur
compilation.

**Localisation de la divergence.** Elle est confinée aux modèles spécifiques au
langage. Un modèle torch en Python n'a pas de jumeau R → à ce nœud, les deux
pipelines calculent des choses différentes. Ce n'est pas de la
non-reproductibilité : ce sont **littéralement deux pipelines différents à ce
nœud** (Tier 2). Concrètement :

- Python `[C++ methods only]` ≡ R `[mêmes C++ methods only]` → **identiques bout
  en bout**.
- Python `[C++ methods + torch]` vs R `[mêmes C++ methods + mlr3]` →
  **bit-identiques jusqu'au nœud modèle**, puis divergence (par design).
- Pour rendre le nœud modèle *aussi* identique cross-langage : (a) un modèle natif
  **nirs4all-methods** (C++, partagé), ou (b) exporter le fitté en **ONNX** et
  rejouer l'**inférence** (déterministe) dans l'autre langage — mais ce sont alors
  les *mêmes poids entraînés*, pas un modèle réentraîné indépendamment.

**Les 5 conditions du mot « exactement ».** Le bit-identique des nœuds C++ tient à
un déterminisme flottant que tu es, justement, le seul à pouvoir garantir :

1. **Compilation cohérente des deux bindings** — idéalement la *même* lib
   partagée, sinon même toolchain/flags. Pas de `-ffast-math` divergent entre la
   wheel Python et le package R.
2. **Réductions à ordre déterministe** — le piège classique : une somme parallèle
   réordonne les flottants → différences sur les derniers ULPs, amplifiées dans
   les algos itératifs (NIPALS/SVD d'un PLS). Réductions mono-thread ou à ordre
   fixe.
3. **Algèbre linéaire maîtrisée** — si les méthodes appellent un BLAS/LAPACK
   *système*, Python et R peuvent en lier deux différents (OpenBLAS vs MKL,
   threads différents) → divergence. Algèbre interne ou Eigen (header-only,
   déterministe à compilation égale) supprime le risque.
4. **Aléa de contrôle exclusivement du cœur** (`RNG_FROM_CORE` + pré-tirage
   Arrow) — aucun controller ne tire avec sa propre RNG pour une décision Tier 1.
5. **Même ingestion** (nirs4all-io C++) et **même dtype** des prédictions dans le
   schéma Arrow.

Ces conditions sont **sous ton contrôle**, contrairement à l'écosystème Python.
Remplies, elles donnent : un pipeline Python et un pipeline R **bit-identiques sur
tout nœud partagé (C++ + contrôle)**, la seule divergence étant — exactement — les
modèles propres à chaque langage. C'est le contrat que la séparation Tier 1 /
Tier 2 et le cœur natif rendent atteignable.

---

## Partie VII — Confrontation aux objectifs & verrous résiduels

| Objectif | Verdict | Note |
|---|---|---|
| Cœur aveugle aux données sauf identité+prédictions | ✅ | handles pour X/features/fitted ; Arrow pour identité, prédictions, y_true, relation |
| Même signature Python/R/natif | ✅ | vtable C ; seules les implémentations diffèrent |
| GIL non-bloquant en débit | ✅ conditionnel | knob `GIL_FREE_COMPUTE` route thread vs process ; build natif court-circuite |
| « fit handle → handle + preds Arrow » | ✅ | `fit→FitResult{fitted}` puis `predict→PredictResult{predictions}` |
| Reproductibilité | ⚠️ cadrée → ✅ Tier 1 | RNG cœur rend le **plan de contrôle** cross-langage bit-identique ; internes framework = Tier 2 ; inférence = ONNX |
| Batteries natives nirs4all | ✅ | vtable en Rust : zéro GIL, zéro marshalling, parallélisme libre = plafond perf |
| Stabilité ABI vs évolution spec | ✅ | riche/évolutif en `Bytes` versionnés ; vtable extensible par bits + fns optionnelles |

### Verrous résiduels (ordre de risque)

1. **GC des handles à travers le DAG** (Partie III) — la ligne de faille réelle.
   Le protocole arènes+refcount le rend tractable, mais sa correction
   (branches/folds/cache/skip/erreur) est l'ingénierie dure.
2. **Marshalling Arrow forcé** pour controllers non-threadables (Python
   GIL-bound, **tout** R) → Arrow IPC/mémoire partagée. Irréductible mais borné à
   ces controllers ; le step-cache devient local au worker.
3. **EXPLAIN feature-space** — sorties de la dimension des features, opaques au
   cœur (stockées/transmises, non interprétées). Léger accroc au principe de
   visibilité, borné car non interprété.
4. **Adaptation Tier 1 des controllers d'augmentation** (accepter l'aléa en
   entrée) — coût de portage, pas un blocage.

### Questions ouvertes

- Format du blob `describe` : JSON canonique (v1, débogable) vs msgpack
  (compact) — trancher à l'implémentation.
- `rng_fill` par upcall pour les tirages feature-shaped : Tier 1 chunké ou Tier 2
  assumé ? (v1 : Tier 2.)
- Granularité d'arène : par fold, par branche, ou par variant ? (Impacte le pic
  mémoire vs la fréquence de sweep.)
- Streaming / out-of-core (cf. spec §21 Q1) : incompatible avec la
  matérialisation eager supposée ici ; hors v1.

---

## Partie VIII — Langage du cœur : Rust vs C++ (décision ouverte)

**Statut : non tranché — à revisiter.** Cette partie consigne les deux voies à
parité et les critères qui les départageront. Elle n'est pas un verdict.

Contexte de fait :
- **nirs4all-io est déjà en Rust** (les loaders).
- **nirs4all-methods est en C++** (validé, portable).
- Dans les **deux** options, les méthodes C++ **restent C++**, derrière l'ABI C
  (controllers natifs). Aucune réécriture des méthodes n'est en jeu ici ; la
  question porte uniquement sur le **langage du cœur** (graphe, phases, folds,
  OOF, lineage, scheduler, RNG, ABI).

### VIII.1 Le recadrage spécifique à cette architecture

Le débat classique Rust vs C++ est ici biaisé par un fait de conception :
**l'architecture impose déjà un ABI C entre le cœur et *tous* les controllers,
méthodes natives comprises** (Partie II). nirs4all-methods entre comme un
controller qui remplit la `ControllerVTable` avec des fns `extern "C"`. Le cœur
appelle les méthodes C++ via la même vtable que les controllers Python : appel
C ABI, **coût nul**.

Conséquence : le pro historique de C++ (« même langage que les méthodes, donc
intégration native ») est **largement neutralisé** — et coupler le cœur aux
méthodes par des types C++ partagés serait un **anti-pattern** (ça casserait la
symétrie polyglotte et la frontière ABI propre). Le débat se réduit donc aux
qualités *intrinsèques* du langage pour un **plan de contrôle concurrent à
liveness cross-FFI**, plus l'outillage de binding.

### VIII.2 Rust (pour le cœur)

| | |
|---|---|
| **Pros** | • **Sécurité mémoire sans GC, exactement sur le verrou n°1** : la bookkeeping de liveness des handles (refcounts, arènes, promotion — Partie III) est logique *interne* au cœur → en Rust safe, le compilateur empêche l'UAF. Un UAF de handle = crash dans le process Python/R de l'utilisateur, le pire à débugger.<br>• **Concurrence sans data race** : `Send`/`Sync` protègent le scheduler au compile-time.<br>• **Stack de binding/Arrow = template prouvé** : pyo3+maturin, extendr, wasm-bindgen/napi, arrow-rs. C'est l'architecture textuelle de Polars, DataFusion, pydantic-core, tokenizers, ruff.<br>• **Cargo** : build reproductible, deps, cross-compilation des wheels sans CMake/vcpkg.<br>• **nirs4all-io déjà en Rust** → ramp-up fait, pont Rust↔C++ déjà pratiqué, io ↔ cœur sans shim. |
| **Cons** | • **Pont Rust↔C++** pour nirs4all-methods : linker la lib C++ (build.rs + `cc`, ou `cxx`). Standard, mais étape de build (déjà résolue côté io).<br>• Le FFI lui-même reste `unsafe` des deux côtés — Rust sécurise la bookkeeping complexe, pas le passage de pointeur brut.<br>• Écosystème numérique plus jeune (peu pertinent : le cœur ne calcule pas). |

### VIII.3 C++ (pour le cœur)

| | |
|---|---|
| **Pros** | • **Intégration native avec nirs4all-methods** (mais §VIII.1 : gain largement illusoire vu l'ABI C, et coupler serait un anti-pattern).<br>• **Arrow a son implé de référence en C++** ; Eigen/BLAS/LAPACK directs.<br>• pybind11 (Python), Rcpp (R) matures ; contrôle direct ABI/layout. |
| **Cons** | • **Sécurité mémoire manuelle, pile sur le verrou n°1** : UAF/double-free/fuites de handles à travers le FFI = attrapés au runtime (ASan/UBSan + discipline), pas prévenus à la compilation.<br>• **Concurrence manuelle** : data races du scheduler à la charge du dev.<br>• **Binding/build bas niveau** : wheels à la main (scikit-build/cibuildwheel), WASM via emscripten, pas de gestionnaire de paquets, dependency hell CMake.<br>• **Incohérence avec nirs4all-io (Rust)** : deux langages systèmes dans la même base, deux toolchains.<br>• Aucun template récent « cœur C++ + bindings multi-langage + Arrow » comme nouveau projet. |

### VIII.4 Topologie résultante (identique dans les deux cas)

```
<langage du cœur>                      C++                         par langage
┌──────────────────────────┐          ┌─────────────────┐        ┌────────────────┐
│ dagml-core (contrôle)     │  ABI C   │ nirs4all-methods│  pyo3  │ sklearn/torch  │
│  graph, folds, OOF,       │◄────────►│  (controllers   │◄──────►│ (par langage)  │
│  lineage, scheduler, RNG  │ extern"C"│   validés C++)  │ extendr│ mlr3 (R)       │
│ nirs4all-io (Rust)        │          └─────────────────┘        └────────────────┘
└──────────────────────────┘
```

Le choix Rust vs C++ ne change que le bloc de gauche. nirs4all-io est en Rust ;
si le cœur est en Rust, io ↔ cœur est sans shim ; si le cœur est en C++, io
(Rust) devient lui aussi un controller/lib derrière une frontière C.

### VIII.5 Les faits qui tranchent, et les critères pour plus tard

Trois faits orientent (sans trancher) :

1. **nirs4all-io déjà en Rust** → cohérence + ramp-up fait → penche Rust.
2. **Le verrou le plus dur est la liveness des handles + le scheduler
   concurrent** → le point faible du projet est la force de Rust → penche Rust.
3. **nirs4all-methods en C++ validé** → mais neutralisé par l'ABI C (§VIII.1) →
   n'oriente quasiment plus.

Critères à évaluer au moment de trancher :

- L'équipe s'engage-t-elle sur Rust **au-delà de io** (cœur compris) ? (io en
  Rust suggère oui.)
- Le plan de contrôle devient-il un goulot à grande échelle (millions de
  records lineage/prediction) → favorise un cœur compilé compact (Rust ou C++).
- Coût réel ressenti du pont Rust↔C++ pour les méthodes (cf. shim `extern "C"`,
  à prototyper).
- Besoin WASM/JS sérieux → favorise nettement Rust (wasm-bindgen).

### VIII.6 Sous-question : langage des nouvelles méthodes

Indépendant du langage du cœur, mais lié :

- Méthodes **existantes** (validées) → restent **C++**, derrière l'ABI. Pas de
  re-validation.
- Méthodes **nouvelles** → si le cœur est Rust, les écrire en Rust les fait vivre
  dans le cœur natif **sans même le shim** ; sinon C++. La frontière ABI rend les
  deux interchangeables, donc pas d'urgence à uniformiser.
```
