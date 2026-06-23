# Cible — répartition nirs4all ↔ dag-ml ↔ dag-ml-data

> **Directive maintainer (2026-06-23, renforcée) :** dag-ml doit fournir **TOUTES** les
> fonctionnalités génériques de nirs4all **nativement en Rust/C-ABI** — y compris l'**agrégation
> CV** (par-fold + ensembles `avg`/`w_avg` + refit `final`), le **scoring/metrics**, la
> **sélection**, et la **persistance prédictions/scores**. Parité **exacte et native** (ne PAS
> ré-implémenter l'agrégation côté host Python). État final : **nirs4all entièrement
> cross-language ; seuls les operators/controllers restent par-langage.** Persistance
> prédictions/scores : projet natif léger envisagé **entre `nirs4all-io` et `dag-ml-data`**
> (rapport d'impact/bien-fondé en cours — `NATIVE_PERSISTENCE_LAYER_REPORT.md`).


> **État *final* visé** du chantier core→dag-ml. À la fin : `nirs4all` ne garde que ce qui
> **calcule sur des spectres** (opérateurs/contrôleurs/modèles) + la **matérialisation des
> données** + HPO + explainabilité + l'**API/UX** ; `dag-ml` devient **tout l'orchestrateur** ;
> `dag-ml-data` porte les **contrats de données**. Synthèse — pas un inventaire exhaustif.

## Qui possède quoi

| Capacité | nirs4all (Python natif) | dag-ml | dag-ml-data |
|---|:---:|:---:|:---:|
| Opérateurs preprocessing (SNV, MSC, SavGol, Detrend, OSC/EPO/CARS…) | ✓ code numérique | — | — |
| Modèles (PLS*, sklearn, PyTorch/TF/JAX) — `fit`/`predict` | ✓ | — | — |
| Contrôleurs (host controllers `fit`/`transform`/`predict`) | ✓ parlent le protocole host dag-ml | invoque (NodeTask) | — |
| Splitters (KennardStone, SPXY, KFold, ShuffleSplit…) | ✓ calcule les indices | orchestre les folds | contrat `FoldSet` |
| **Matérialisation données** (X/y, identité `sample_id`, multi-source) | ✓ `SpectroDataset` + resolver | **jamais les matrices** (identity-keyed) | schémas/axes/représentations/envelope |
| Lecture fichiers (~58 formats) | ✓ nirs4all-io / nirs4all-formats | — | — |
| Hyperparamétrisation (`finetune_params`/Optuna) | ✓ exécute la recherche | génère+sélectionne les variantes (`_or_`/`_grid_`…) | — |
| Explainabilité (SHAP) | ✓ calcule les valeurs | orchestre la phase EXPLAIN + lineage | — |
| Compile / Plan / Scheduling | — | ✓ | — |
| CV · OOF (join par `sample_id`) · anti-fuite | — | ✓ | contrat fold/relations |
| Sélection de modèle · fingerprints | — | ✓ | — |
| Replay · lineage · provenance (RO-Crate/PROV/OpenLineage) | — | ✓ | — |
| Façade API (`run`/`predict`/`explain`/`retrain`/`session`/`generate`) | ✓ fine, délègue | moteur | — |
| Workspace (SQLite+Parquet) · bundles `.n4a` | ✓ contrat utilisateur/UX | alimenté par bundles+lineage dag-ml | — |
| Visualisation · charts · analyzers | ✓ | — | — |

## En trois phrases

- **nirs4all** = la lib de **numérique NIRS** : opérateurs + modèles + contrôleurs, **+ matérialisation
  des données**, + HPO, + explainabilité, + API/UX/workspace. *Tout ce qui touche des spectres.*
- **dag-ml** = le **moteur d'orchestration** : `compile→plan→fit_cv→select→refit→predict→explain`,
  OOF/anti-fuite, variantes, fingerprints, replay/lineage. *Tout ce qui coordonne — sans jamais voir une matrice.*
- **dag-ml-data** = les **contrats de données** sample-aligned : schémas, représentations, envelope,
  `FoldSet`, vtable host-provider.

## La frontière qui décide tout (vérifiée empiriquement)

Le cœur dag-ml **ne voit jamais X/y** : les données circulent par `sample_id`, et c'est nirs4all qui
les résout en vraies lignes `SpectroDataset` (le `MaterializationResolver`, livré en 2b-i). C'est
pourquoi **la matérialisation des données reste irréductiblement Python** — ce n'est pas *seulement*
les opérateurs qui restent côté nirs4all. Tout le reste de la coordination part vers dag-ml.

> Nuance HPO/sélection : la *recherche* d'hyperparamètres (entraîner/évaluer chaque candidat) reste
> Python (Optuna), mais la *génération déterministe des variantes* et la *sélection finale* (avec
> fingerprints + anti-fuite) sont des responsabilités dag-ml. La ligne exacte se fige à l'étape HPO.
