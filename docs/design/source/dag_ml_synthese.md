# DAG-ML — Synthèse : objectif, choix techniques, feuille de route

Document d'entrée. À lire en premier. Pour le détail des mécanismes, voir le doc
de travail `dag_ml_polyglot_core_design.md` ; pour le contrat ML complet, les
specs `dag_ml_specification_v1.md`, `dag_ml_use_cases.md`,
`ml_data_specification_v1.md`.

Langage du cœur : **Rust** (acté). Méthodes existantes : **C++** (nirs4all-methods,
conservées). Loaders : **Rust** (nirs4all-io, en cours).

---

## 1. Objectif

DAG-ML est un **moteur ML local, in-process**, extrait et généralisé de nirs4all,
qui formalise : compilation d'un DSL pipeline en DAG, énumération de variants,
exécution multi-phases (`COMPILE → PLAN → FIT_CV → SELECT → REFIT → PREDICT →
EXPLAIN`), CV/OOF/stacking **sans leakage**, refit, predict/explain par replay,
stores artifacts/cache/lineage, parallélisme, et un contrat d'opérateur stable.
Aucune logique NIRS dedans : le domaine entre par plugins.

Le **but de l'architecture polyglotte** : porter le **moteur de rigueur ML**
(graphe, OOF, folds, lineage, déterminisme) vers plusieurs langages
(Python, R, JS, natif) **sans réécrire l'écosystème d'opérateurs**. On veut, à
terme, le même moteur — donc les mêmes garanties — utilisable depuis Python
(sklearn/torch), R (mlr3), ou en build full-natif.

---

## 2. L'idée directrice

**Le cœur est un plan de contrôle, pas un noyau de calcul.** Il ne fait quasiment
aucun FLOP : le calcul lourd vit dans les opérateurs (sklearn/torch/C++), déjà
natifs. Le cœur possède le **graphe, les phases, les folds, l'OOF, le lineage,
le scheduler, le RNG** — les invariants.

Deux principes gravés dans l'ABI :

1. **Visibilité** — le cœur voit en Arrow exactement trois choses : l'**identité**
   (sample/observation/target/group/origin ids), les **prédictions**, et
   **y_true** (scoring). **Jamais X ni les features.** Tout le reste = **handles
   opaques** (u64) résolus côté hôte.
2. **Ownership** — le cœur possède le **cycle de vie des handles** ; l'hôte
   possède l'**objet sous-jacent** et le libère sur `release`. Arrow porte son
   propre release callback.

Violer (1) = on remarshalle le gros. Violer (2) = use-after-free cross-FFI.

---

## 3. Architecture

```
RUST                                   C++                         par langage
┌──────────────────────────┐          ┌─────────────────┐        ┌────────────────┐
│ dagml-core (contrôle)     │  ABI C   │ nirs4all-methods│  pyo3  │ sklearn/torch  │
│  graph, folds, OOF,       │◄────────►│  (controllers   │◄──────►│ (par langage)  │
│  lineage, scheduler, RNG  │ extern"C"│   validés C++)  │ extendr│ mlr3 (R)       │
│ nirs4all-io (loaders)     │          └─────────────────┘ napi/  │ tfjs/onnx (JS) │
│  ── identité+preds=Arrow ─│                              wasm    └────────────────┘
└──────────────────────────┘
  données  : Arrow C Data Interface (zero-copy)  /  DLPack (tenseurs, GPU)
  modèles  : ONNX / safetensors (inférence cross-langage)
  contrôle : déterminisme cœur (folds, OOF, RNG)
```

**Trois strates, trois statuts de langage** (vrai pour données *et* opérateurs) :

| Strate | Contenu | Statut |
|---|---|---|
| Métadonnée / plan | schema, axes, représentations, GraphSpec, DataPlan, ModelInputSpec, fingerprint | **neutre** (cœur) |
| Raisonnement | find_path, planification, OOF join, scheduling, RNG de contrôle | **neutre** (cœur Rust) |
| Buffers + exécution | DataBlock, fit/predict/transform, collation | **par langage** (controllers) |

---

## 4. Décisions techniques

| Décision | Justification (1 ligne) |
|---|---|
| **Cœur en Rust** | Le point faible du projet (liveness des handles cross-FFI + scheduler concurrent) est la force de Rust ; stack binding/Arrow prouvée (Polars, pydantic-core, tokenizers) ; cohérent avec nirs4all-io déjà en Rust. |
| **Méthodes en C++ derrière l'ABI C** | Code validé/portable, zéro réécriture ; appelées comme controllers via vtable `extern "C"`, coût d'appel nul. |
| **ABI C entre cœur et TOUS les controllers** | Une seule frontière, symétrique Python/R/natif ; neutralise le besoin de coupler cœur et méthodes. |
| **Données via handles, jamais en clair** | Le buffer reste hôte ; seules identité + prédictions traversent (Arrow). Marshalling du gros = nul intra-process. |
| **Arrow C Data Interface** pour ce que le cœur lit | Zero-copy, ABI stable, lingua franca polyglotte (py/R/JS). |
| **DLPack** pour les tenseurs denses | Zero-copy CPU/GPU (torch/tf/jax). |
| **ONNX/safetensors** pour les modèles fittés | Inférence cross-langage (≠ Arrow, qui porte les données). |
| **RNG counter-based splittable dans le cœur** | Déterminisme cross-langage **et** indépendant du scheduling pour tout le plan de contrôle. |
| **pyo3/maturin** binding primaire ; extendr (R), wasm-bindgen (JS) ensuite | Outillage mûr, wheels sans CMake. |
| **Splitters d'identité natifs au cœur** (KFold, GroupKFold…) | Pas de données nécessaires → Tier 1 RNG, cross-langage identique. Feature-based (KS/SPXY) = controllers. |

---

## 5. Les contrats de frontière (ABI)

Deux vtables C `#[repr(C)]` (détail complet : doc de travail §9-10). L'essentiel :

**`ControllerVTable`** (un opérateur sklearn/torch/mlr3/C++) :
- `clone_with(op, params)` — construction lazy d'un variant (le cœur pilote *quels*
  params, l'hôte *comment*).
- `describe(op) → blob` — contrat de PLAN (ModelInputSpec, ports, phases, flags ;
  format JSON canonique versionné).
- `fit / transform_fit` → handle fitté ; `predict` → **prédictions Arrow** ;
  `transform_apply` → handle de données ; `invert` (y-transform).
- `split` (identité + data optionnelle pour KS/SPXY) → table de folds Arrow.
- `serialize / deserialize` (joblib/onnx/safetensors), `cache_key`,
  `release / free_bytes / destroy`.
- `capabilities` (bitset) : `GIL_FREE_COMPUTE`, `THREAD_SAFE`, `STATEFUL`,
  `INVERTIBLE`, `DETERMINISTIC`, `RNG_FROM_CORE`, `REQUIRES_DATASET_PLAN`,
  `EMITS_RELATION`…

**`DataVTable`** (la couche données par langage) :
- `materialize`, `make_view` (slice **par sample-ids**, jamais par positions →
  anti-leakage porté par l'ABI), `view_identity`, `target_arrow`, `feature_arrow`,
  `ingest_arrow` (OOF → handle), `handle_nbytes`, `schema_fingerprint`,
  `release / destroy`.

**Ce qui traverse** : entrée = handles (u64) + scalaires + blobs ; sortie =
handles + Arrow (identité, prédictions, relations). Le buffer lourd ne traverse
jamais intra-process.

---

## 6. Données & dimensions

Une dimension n'est pas un entier : c'est un **axe sémantique** (`AxisSpec{kind,
unit, size, coordinates}`). La compatibilité modèle est une **recherche de chemin**
(`find_path`, Dijkstra) du `native_representation` d'une source vers une
représentation acceptée du `ModelInputSpec`, produisant un `DataPlan`
(`materialize → adapt* → align → join → collate`). Décidée au PLAN sur le schéma
seul ; refusable ; escaladable si ambigu.

- **Bloc par type de données** = `DataTypePlugin` (+ adapters + collator) : signal,
  image, génotype, série temporelle, graphe, table, texte. Unité d'extension.
- **Collation = en dernier** : padding/batch/ordre canal seulement au bord du
  modèle (l'impédance Arrow-colonne ↔ tenseur dense est isolée là).
- **Impact langage** : l'algèbre de forme (schema, find_path, DataPlan) est
  **neutre** (cœur) ; les buffers + l'exécution d'adapter + la collation sont
  **par langage**. nirs4all-io (Rust) rend l'**ingestion identique** cross-langage.

---

## 7. RNG & reproductibilité

PRNG **counter-based splittable** (Philox/Threefry) dans le cœur ; `SeedContext`
= arbre de flux dérivés du chemin (`SHA256(path)[:16] → clé`). Deux tiers :

- **Tier 1 — aléa de contrôle** (splits, sampling tuner, sélection d'augmentation) :
  possédé par le cœur, **cross-langage bit-identique** et indépendant du
  scheduling. Pré-tiré en Arrow ou via upcall.
- **Tier 2 — aléa interne framework** (init NN, bootstrap sklearn) : graine passée,
  reproductible **intra-lib** seulement.

**Reproductibilité cross-langage** (Python ≡ R) sur tout nœud partagé, sous 5
conditions, toutes sous ton contrôle : (1) compilation cohérente des bindings ;
(2) réductions à ordre déterministe ; (3) algèbre linéaire maîtrisée (pas de BLAS
système divergent) ; (4) aléa de contrôle exclusivement du cœur ; (5) même
ingestion (nirs4all-io) + même dtype. Divergence confinée aux **modèles propres au
langage** (torch/sklearn vs mlr3). Modèle identique cross-langage = méthode native
C++ partagée, ou même artefact rejoué via ONNX.

---

## 8. Feuille de route

Ordre de construction proposé, chaque phase livrant quelque chose de vérifiable.

**Phase 0 — Geler les contrats.**
- Types neutres `ml_data.contract` (schema, axes, représentations, ModelInputSpec,
  DataPlan, FusionPolicy, SampleRelation…) — déjà spécifiés.
- L'ABI C : `ControllerVTable`, `DataVTable`, format du blob `describe`, conventions
  Arrow/handle/ownership. *DoD : header C + crate Rust de types, versionnés.*

**Phase 1 — `dagml-core` (Rust) + chemin Python minimal.**
- Cœur : GraphSpec, SearchSpace + énumérateur lazy, planner (`find_path`), FoldSet +
  splitters d'identité, scheduler séquentiel, PredictionStore (Arrow/Parquet) +
  `oof_join` + agrégation, LineageRecorder, CacheStore, **gestionnaire de liveness
  des handles** (arènes + refcount), **RNG** splittable.
- Binding pyo3 + maturin ; couche données Python (registry numpy) ; controller
  sklearn. *DoD : UC6 (stacking) bout-en-bout en Python, OOF correct, reproductible.*

**Phase 2 — Blocs natifs.**
- Brancher nirs4all-io (Rust) comme loaders ; nirs4all-methods (C++) comme
  controllers via shim `extern "C"`. Build full-natif (sans Python).
  *DoD : pipeline natif = pipeline Python bit-identique sur nœuds partagés (§7).*

**Phase 3 — Parallélisme.**
- Scheduler threads (controllers `GIL_FREE_COMPUTE`) ; workers processus (Python
  GIL-bound, R) avec Arrow IPC. *DoD : scaling sur folds/variants, déterminisme
  conservé.*

**Phase 4 — Binding R.**
- extendr + couche données R + controllers mlr3 (isolés en processus).
  *DoD : pipeline R = pipeline Python bit-identique sur nœuds C++/contrôle partagés.*

**Phase 5 — Persistance & extras.**
- Bundle (graph + plan + artifacts + schema_fingerprint), PREDICT/EXPLAIN par
  replay, export ONNX, TunerAdapter (Optuna). *DoD : train→export→predict sur
  nouvelles données, cross-langage en inférence.*

**Plus tard** — JS/WASM (wasm-bindgen) ; migration incrémentale depuis nirs4all
(cf. spec §19) ; streaming/out-of-core.

---

## 9. Verrous à surveiller

| # | Verrou | Parade |
|---|---|---|
| 1 | **GC des handles à travers le DAG** (le plus dur) | Arènes scopées + refcount sur les échappés (doc de travail Partie III) ; Rust safe sur la bookkeeping. |
| 2 | **Marshalling cross-process** (controllers GIL-bound / R) | Arrow IPC ; borné à ces controllers ; cache local au worker. |
| 3 | **Impédance Arrow par type** (graphes/ragged) | Convention figée, ou garder host-local (handle) sans traverser. |
| 4 | **Déterminisme FP des méthodes C++** | Les 5 conditions du §7 ; algèbre interne/Eigen, réductions à ordre fixe. |
| 5 | **EXPLAIN feature-space** | Sorties opaques-au-cœur, stockées/transmises non interprétées. |

---

## 10. Renvois

- Détail ABI, liveness, blob describe, trace UC6, RNG, confrontation, comparatif
  Rust/C++ : `dag_ml_polyglot_core_design.md`.
- Contrat moteur complet : `dag_ml_specification_v1.md`.
- 12 cas d'usage matérialisés : `dag_ml_use_cases.md`.
- Contrat données complet : `ml_data_specification_v1.md`.
