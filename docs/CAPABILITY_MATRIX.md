# Capability Matrix

The long-term target is to replace the current `nirs4all` core pipeline engine
with a lower-level, reproducible OOF campaign engine. The matrix below is the
scope guard: every capability must either be represented in core contracts or
explicitly delegated to `dag-ml-data`/controllers without weakening OOF and
leakage guarantees.

## Pipeline Surface

| Capability | `dag-ml` responsibility | `dag-ml-data` / controller responsibility | OOF / leakage invariant |
|---|---|---|---|
| Multisource | graph/source join nodes, phase control | source descriptors, alignment, presence masks, fusion plans | sample ids are canonical across sources; missing-source policy is explicit |
| Repetitions | split unit, aggregation decisions and custom aggregation-controller task/result validation | `SampleRelation` observation/sample/target mapping; optional external custom reducers | no observation from the same leakage unit crosses train/validation, and custom aggregation outputs must preserve requested sample/unit coverage |
| Grouped samples | group-aware split validation | expose group ids in sample relations | group id cannot appear in both train and validation for a fold |
| Augmentation | train-only phase gating, origin checks | expose `origin_id`, augmentation adapter declarations | validation origins cannot be augmented into train leakage or vice versa |
| Processings | node lineage and fit scope | representation adapters and fitted adapter refs | stateful processing fits only on fold train during CV |
| Splitters | identity fold generation and validation | group/origin/sample identity inputs | folds are sample-id based, deterministic and replayable |
| Models | controller ABI, fit/predict phase ordering | host controller implementation | downstream training may consume only validation OOF predictions by default |
| Refit | selected graph replay and final-fit phase | replay data plans and fitted adapter refs | refit artifacts cannot be used to manufacture training meta-features |
| Branching | fork/map/subgraph semantics | branch-local views and materialization | branch outputs preserve lineage and fold identity |
| Merging | feature/prediction/source join nodes | feature/source alignment and concatenation | prediction merges validate OOF; feature/source merges validate identity alignment |
| Concatenation | declare merge intent and downstream contracts | feature joiner and namespace policy | row order is canonical by sample id; no positional join |
| Finetuning | phase/fold control and leakage flags | stateful controller/adapter fit implementation | any learned transform/model is fitted on fold train only during CV |
| Generation | search-space expansion, variant fingerprints, typed node-param override lowering | adapter/model params as serializable JSON | each variant has deterministic seeds, fingerprints, effective params and lineage |
| Tuning | tuner node phase control and nested split policy | tuner/controller execution | tuner observations respect nested CV boundaries |
| Prediction replay | bundle validation and phase restrictions | schema fingerprint and data plan replay | predict never reuses CV validation labels/features in training mode |
| Explainability | replay hooks and opaque outputs | controller-specific explanations | explanation payloads do not alter fit/predict lineage |

## Non-Negotiable Rules

1. All train/validation/test/final semantics are keyed by stable sample ids.
2. Fold construction and OOF joins must never use row position as identity.
3. Stateful processing, finetuning and supervised adapters are fitted inside the
   current training boundary only.
4. Refit artifacts are final inference artifacts, not meta-training features.
5. Any unsafe train-prediction-as-feature path must be explicit, searchable and
   permanently marked in lineage.
6. Generated variants carry deterministic fingerprints, seed contexts and
   parameter choices.
7. `dag-ml-data` may describe relations and fit scopes, but `dag-ml` enforces
   ML invariants.
8. A training-phase edge marked `requires_oof` must be backed by validation
   predictions in the core `PredictionStore`; raw upstream handles are not
   forwarded across that edge.

## MVP To Full Replacement Path

| Stage | Coverage |
|---|---|
| MVP | UC6 stacking success and UC11 train-prediction refusal |
| Next | group-aware folds, source alignment, stateful processing fit scopes |
| Replacement spike | multisource + repetitions + augmentation + branch/merge from current `nirs4all` fixtures |
| Hardening | generated variants, nested tuning, refit/replay bundles, process/thread scheduling |
