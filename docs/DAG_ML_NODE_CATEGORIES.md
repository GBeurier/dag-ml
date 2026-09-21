# Catégories de nœuds DAG-ML et DSL pour un éditeur de pipelines

État observé le 18 septembre 2026. Document de réflexion pour un éditeur inspiré
des chaînes de plugins VST, avec la possibilité de déplier un DAG plus complexe
à la manière de ComfyUI.

DAG-ML expose vingt `NodeKind` génériques. Les opérateurs concrets — `SNV`,
`Ridge`, `SMOTE`, etc. — sont des payloads opaques exécutés par des contrôleurs
externes. Le DSL fournit une vue plus ergonomique et compile ses structures vers
le DAG.

La catégorisation ci-dessous est une proposition de palette d'éditeur, pas une
hiérarchie officielle du format. Les exemples illustrent les usages : leur
exécution dépend des contrôleurs disponibles dans l'environnement. Un type
déclaré dans le DAG ne garantit pas une implémentation native de chaque exemple.

## Nœuds métier et techniques

| Catégorie UI | Sous-catégorie / type DAG-ML | Disponibilité | Exemples d'instances | Entrées → sorties usuelles |
|---|---|---|---|---|
| Données | `transform` | DAG + DSL direct | `SNV`, `StandardScaler`, `PCA` | `data:x` → `data:x_out` |
| Cibles | `y_transform` | DAG + DSL direct | `StandardScaler(y)`, encodage de classes, transformation logarithmique | `target:y` → `target:y_out` |
| Annotation | `tag` | DAG + DSL direct | marquer un groupe, ajouter un label de qualité, tagger une source | `data:x` → `data:x_out` |
| Filtrage | `exclude` | DAG + DSL sous `exclude`, `filter`, `sample_filter` | `YOutlierFilter`, filtre de qualité, exclusion par métadonnée | `data:x` → `data:x_out`, avec changement possible de population/relations |
| Augmentation | `augmentation` | DAG + DSL sous `augmentation`, `feature_augmentation`, `sample_augmentation` | bruit gaussien, jitter spectral, création de répétitions | `data:x` → `data:x_out`, avec identité/origine et shape potentiellement modifiées |
| Génération runtime | `generator` | DAG + DSL sous `data_generation` ou alias `generation` | `SMOTE`, génération synthétique, génération de features | `data:x` → `data:x_out` |
| Modélisation | `model` | DAG + DSL direct | `PLSRegression`, `Ridge`, `RandomForestRegressor` | `data:x` → `prediction:oof` |
| Optimisation | `tuner` | DAG + DSL sous `tuner` ou `finetune` | `OptunaTuner`, recherche d'hyperparamètres, fine-tuning réseau | `data:x` → `prediction:oof` |
| Fusion de features | `feature_join` | DAG ; produit par `concat_transform` ou `merge` | concaténation PCA+dérivée, fusion multi-vues, fusion de sources | plusieurs `data:*` → `data:x_out` |
| Fusion de prédictions | `prediction_join` | DAG ; produit par `merge` | fusion OOF, sélection de prédictions, empilement de prédictions en features | plusieurs `prediction:*` → `prediction:prediction` ou `data:x_out` |
| Fusion mixte | `mixed_join` | DAG ; produit par `merge` | prédictions + données originales, features + OOF, stacking enrichi | plusieurs `prediction:*` + `data:*` → `data:x_out` |
| Fusion de sources | `source_join` | DAG ; produit par `merge output_as=sources` | NIR + métadonnées, plusieurs instruments, plusieurs tables alignées | plusieurs `data:*` → `data:x_out` |
| Méta-modèle | `model` produit par `merge_model` | DSL direct | `RidgeMetaStacker`, PLS de second niveau, classifieur de stacking | plusieurs `prediction:*` + éventuellement `data:x_original` → `prediction:oof` |
| Agrégation | `aggregator` | DAG brut ; pas d'étape dédiée dans le DSL canonique | moyenne des répétitions, vote de modèles, agrégation par groupe | ports définis par le contrôleur ; typiquement `prediction` → `prediction` |
| Adaptation | `adapter` | DAG brut ; pas d'étape dédiée dans le DSL canonique | dense→sparse, dataframe→tensor, adaptation de schéma | ports définis par le contrôleur ; typiquement `data` → `data` |
| Restructuration | `restructure` | DAG brut ; pas d'étape dédiée dans le DSL canonique | reshape, pivot/unpivot, regroupement d'axes | ports définis par le contrôleur ; typiquement `data` → `data` |
| Encapsulation | `subgraph` | DAG brut ; pas d'étape dédiée dans le DSL canonique | chaîne réutilisable, preprocessing complet, bloc encodeur | ports définis par le contrat du nœud |
| Visualisation | `chart` | DAG + DSL direct | aperçu spectral, projection PCA, diagnostic de distribution | dans le DSL actuel : `data:x` → `data:x_out` ; rendu selon le contrôleur |
| Routage bas niveau | `fork` | DAG brut ; pas d'étape dédiée dans le DSL canonique | duplication vers deux modèles, branche diagnostic, plusieurs chemins de preprocessing | ports définis par le contrôleur ; conceptuellement une entrée → plusieurs sorties |
| Routage bas niveau | `map` | DAG brut ; pas d'étape dédiée dans le DSL canonique | application par source, par cible, par groupe | ports et comportement définis par le contrôleur |

Les conventions précises du compilateur DSL sont dans
[compiler.rs](../crates/dag-ml-core/src/dsl/compiler.rs). Les types de nœuds et de
ports sont définis dans [graph.rs](../crates/dag-ml-core/src/graph.rs) et dans le
[schéma GraphSpec](contracts/graph_spec.schema.json).

## Nœuds de contrôle et structures du DSL

Ces éléments sont importants pour une UI « chaîne VST qui peut s'ouvrir en DAG »,
mais tous ne deviennent pas un nœud runtime.

| Catégorie UI | Construction | Résultat de compilation / statut | Exemples | Flux logique |
|---|---|---|---|---|
| Validation | `split_invocation` | Élément du plan de campagne | `KFold`, `GroupKFold`, `SPXY`/`KS` externe | identités + groupes + seed, données si nécessaire via contrôleur → `FoldSet` |
| Validation legacy | `NodeKind::Split` | Type déclaré ; éventuel nœud de contrôle compatible | splitter historique importé | contrôle uniquement ; aucune sortie de features |
| Conteneur | `sequential` | Aucun nœud propre ; ses enfants sont compilés dans la séquence | preprocessing, chaîne transform→model, bloc nommé | flux des étapes enfants |
| Branchement | `branch` | Plusieurs sous-graphes partageant une entrée ou utilisant des vues sélectionnées | chemins PLS/RF, un modèle par instrument, un chemin par population | `data:x` → N branches de `data` et/ou `prediction` |
| Branchement — variantes | `duplication`, `separation`, `by_source`, `by_metadata`, `by_tag`, `by_filter` | Modes de branche ; plans de vues et métadonnées selon le mode | duplication du même X, séparation par source, sélection par tag | une vue commune ou sélectionnée → N vues |
| Concaténation | `concat_transform` | N chaînes de `transform` + un `feature_join` | PCA brut + PCA dérivée, SNV + MSC, plusieurs fenêtres spectrales | `data:x` dupliqué → N `data` → `data:x_out` |
| Fusion | `merge` | `feature_join`, `prediction_join`, `mixed_join` ou `source_join` selon la configuration | concat features, fusion OOF, prédictions + X original | N flux produits en amont → un flux `data` ou `prediction` |
| Stacking | `merge_model` | Un `model` avec plusieurs entrées OOF | Ridge stacker, PLS stacker, modèle de blending | N `prediction:oof` + X optionnel → `prediction:oof` |
| Génération de variantes | `generator` (`or`/`cartesian`) | Expansion de sous-séquences ; distinct du `NodeKind::Generator` runtime | choisir SNV ou MSC, produit preprocessing×modèle, pick/arrange | une définition → plusieurs séquences candidates |
| Génération de paramètres | `variants`, `or`, `range`, `log_range`, `grid`, `pick`, `arrange` | Dimensions du `GenerationSpec` | `n_components=[5,10]`, plage d'alpha, grille de paramètres | paramètres → variantes, sans flux de données |

Un splitter produit un `FoldSet` au niveau campagne. Le contrat précise qu'il ne
doit pas être traité comme une transformation de données : voir
[COORDINATOR_SPEC.md — Splitters](COORDINATOR_SPEC.md#splitters).

Les constructions canoniques sont recensées dans
[types.rs](../crates/dag-ml-core/src/dsl/types.rs) et le
[schéma du DSL](contracts/pipeline_dsl.schema.json). Le frontend de compatibilité
nirs4all accepte également des syntaxes compactes, notamment `_or_`,
`_cartesian_`, `_chain_`, `_grid_`, `_range_`, `_log_range_`, `_zip_` et `_sample_`.

Attention à la sémantique d'une séquence : un modèle produit des prédictions mais
ne remplace pas le flux de features courant. Deux modèles consécutifs peuvent
donc consommer le même X. Le passage des prédictions vers un modèle aval se fait
explicitement via les constructions de fusion/stacking.

## Types de ports disponibles

Tous les nœuds du DAG brut déclarent leurs ports. Le `NodeKind` ne fixe donc pas
à lui seul leur signature. Les signatures nommées dans les tableaux précédents
correspondent au compilateur DSL ; les autres sont des usages illustratifs.

| Type de port | Contenu logique |
|---|---|
| `data` | Features ou vue de données ; en pratique souvent un handle host |
| `target` | Cibles `y` et leur espace de représentation |
| `prediction` | Prédictions, notamment OOF : prédictions hors fold d'apprentissage |
| `artifact` | Modèle ajusté, état ou référence d'artefact |
| `metric` | Score ou mesure |
| `control` | Information de contrôle selon le contrat du contrôleur |

Chaque port possède une cardinalité `one`, `many` ou `optional`, ainsi que des
champs de représentation, de niveau d'unité et de clé d'alignement. Les arêtes
peuvent imposer l'alignement par fold, le caractère OOF, la propagation de lineage
et une politique de données manquantes.

Le DSL standard génère principalement des ports `data`, `target` et `prediction`.
Les ports `artifact`, `metric` et `control` sont disponibles dans le DAG brut et
les contrats contrôleur.

Un `model` DSL reçoit `data:x` comme port graphique. Les cibles, folds et vues
d'apprentissage lui arrivent via le contexte d'exécution et le plan de données,
pas par une arête `target` ordinaire générée par ce compilateur. Un `y_transform`
déclare bien `y` et `y_out`, sans être câblé dans la chaîne X.

Les arêtes de prédiction produites par `merge` et `merge_model` portent
`requires_oof=true` et `requires_fold_alignment=true`, pour sécuriser notamment
l'utilisation de prédictions comme features d'apprentissage.

## Conséquences pour l'éditeur

Une vue façon VST peut présenter une chaîne compacte de transformations,
augmentations et modèles, avec `branch`, `concat_transform`, `merge` et
`generator` comme blocs dépliables. Les types techniques `adapter`,
`restructure`, `fork`, `map`, `subgraph` et les ports avancés peuvent apparaître
dans un mode expert.

Les splits, la CV interne, les variantes, la sélection et le refit relèvent du
plan de campagne. L'éditeur peut les rendre visibles à côté de la chaîne et
indiquer leur portée sur les étapes concernées.

Il faut distinguer visuellement trois actions : exécuter plusieurs branches,
explorer plusieurs variantes, et répartir les échantillons en folds. Elles
peuvent toutes ressembler à un embranchement à l'écran, mais ont des effets
différents sur l'exécution et la sécurité OOF.

## Exemples du dépôt

- [Chaînes, augmentation, concaténation et stacking](../examples/pipeline_dsl_nirs4all_parity.json)
- [Branches et méta-modèle](../examples/pipeline_dsl_branch_merge.json)
- [Générateurs de séquences](../examples/pipeline_dsl_nirs4all_generator_parity.json)
- [Générateurs de paramètres](../examples/pipeline_dsl_compact_generation.json)
- [Génération runtime de données](../examples/pipeline_dsl_runtime_generation_executable.json)
- [Tuner](../examples/pipeline_dsl_tuner_executable.json)
