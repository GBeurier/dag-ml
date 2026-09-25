# Binding R : parité nirs4all et portabilité interlangage

> Réaudit au 24–25 septembre 2026. Ce document décrit les sources présentes
> dans le workspace, pas une promesse de release. Les niveaux « supporté »,
> « conformance », « expérimental » et « backlog » gardent le sens défini dans
> [Supported surface](SUPPORTED.md).

## Résumé décisionnel

Les briques d'un nirs4all R existent, mais elles ne forment pas encore un
produit équivalent au nirs4all Python :

- `nirs4all-core` publie déjà un package R nommé `nirs4all`. C'est aujourd'hui
  la façade agrégée correcte, mais son runner est limité à
  Kennard–Stone/SNV/Savitzky–Golay/PLS et réorchestre ce cas directement en R ;
- `dagml` expose désormais en R un registre local de pertes/métriques, une
  phase planifiée, HPO par processus, CV→refit→predict, refit initial sans CV
  et replay de bundle. Ces chemins pilotent le CLI natif avec des fichiers JSON
  et un adaptateur exécutable ; ce n'est pas encore une API de produit R ;
- `dagmldata` expose déjà le provider en mémoire, les vues, cibles, features et
  collations. L'interface est cependant JSON-in/JSON-out et n'est pas encore
  reliée à un chargement R de données numériques de bout en bout ;
- `n4m` est un binding R natif substantiel (138 exports et 22 méthodes S3 avant
  le présent correctif). L'import/export N4MM sur `raw()` et ses tests locaux
  R/Python ont été ajoutés ; la parité de tous les pipelines reste à démontrer ;
- `nirs4allformats` est déjà une bonne brique R de lecture ; `nirs4allio`
  charge en R une synthèse structurelle, pas encore les buffers numériques
  attendus par le provider pour une exécution complète ;
- les adaptateurs R `prospectr` et `mdatools` de DAG-ML sont des adaptateurs de
  conformance, pas une couche R de production générale.

La trajectoire recommandée est donc :

1. compléter d'abord le binding R de DAG-ML et le chemin IO/Data sans déplacer
   l'orchestration hors du cœur natif ;
2. rendre tous les nœuds n4m éligibles pilotables par un contrôleur Methods
   générique, puis qualifier un profil de pipelines « n4m-only » en R ;
3. ajouter des contrôleurs R pour les moteurs classiques, puis `torch` et
   `keras3`, en laissant toujours DAG-ML posséder CV, OOF, sélection et HPO ;
4. transporter les modèles n4m par leurs octets N4MM ; pour les autres moteurs,
   séparer strictement recette réentraînable, inférence portable et artefact
   hôte opaque ;
5. développer le dépôt produit local `nirs4all-r` pour les contrôleurs et
   l'expérience utilisateur R, tout en gardant Core comme façade portable
   basse. La propriété du nom R public `nirs4all` doit être migrée une seule fois.

## 1. Périmètre et définitions de la parité

« Parité » recouvre trois propriétés différentes. Elles doivent être annoncées
et testées séparément.

| Niveau | Question | Résultat attendu |
|---|---|---|
| Recette | Le même DAG et les mêmes intentions peuvent-ils être interprétés ? | Le pipeline peut être recompilé et réentraîné dans l'autre langage, avec les écarts sémantiques déclarés. |
| Exécution | Les mêmes contrats de données, folds, OOF, sélection, refit et scores sont-ils appliqués ? | DAG-ML produit les mêmes décisions et la même provenance, sous tolérances publiées. |
| Modèle ajusté | Le même état appris peut-il être rechargé sans `fit()` ? | Les prédictions sont reproduites depuis un artefact portable ou depuis le runtime hôte déclaré. |

La parité R doit viser les contrats stables de DAG-ML, et non réimplémenter
tous les détails historiques du moteur legacy Python. Le Python reste l'oracle
fonctionnel pendant la migration, mais la coordination générique doit vivre
dans DAG-ML afin d'être identique dans tous les langages.

Un pipeline « n4m-only » désigne ici un pipeline dont tous les opérateurs
calculatoires sont fournis par `nirs4all-methods`. Les splitters, transforms,
augmentations, filtres, sélecteurs, modèles et diagnostics concernés doivent
être explicitement inscrits dans un profil portable versionné. « Tous les
pipelines » ne peut pas signifier tester le produit cartésien infini de toutes
les compositions ; la preuve doit combiner :

- la qualification individuelle de chaque nœud et de son cycle de vie ;
- des compositions représentatives couvrant les changements de forme et
  d'identité ;
- des tests génératifs de chaînes compatibles ;
- des refus explicites pour les compositions ou paramètres non supportés.

## 2. Répartition cible des responsabilités

| Couche | Propriétaire | Responsabilité dans le produit R |
|---|---|---|
| Orchestration ML | `dag-ml` | Graphe, plan, folds, OOF, anti-fuite, RNG de contrôle, sélection, refit, prédictions/scores, lineage, replay. |
| Données alignées | `dag-ml-data` | Identités, relations, représentations, vues et collations ; les buffers lourds restent chez l'hôte. |
| Lecture et assemblage | `nirs4all-formats`, `nirs4all-io` | Décodage des formats, construction du dataset et émission du contrat Data. |
| Méthodes NIRS | `nirs4all-methods` | Calculs n4m, états de modèles et sérialisation N4MM/N4MOPT. |
| Contrôleurs R | binding/package R | Adaptation de `prospectr`, `mdatools`, `ranger`, `glmnet`, `xgboost`, `torch`, `keras3`, etc. |
| Façade portable | `nirs4all-core` | Agrégation, verrou de versions, découverte des capacités, API publique mince et gates interlangages. |
| Produit Python | `nirs4all` | Contrôleurs Python et expérience Python ; oracle temporaire de migration. |

Cette répartition suit la directive d'architecture de DAG-ML : les binaires de
modèles non portables appartiennent au langage hôte ; tout le contrôle et les
résultats génériques appartiennent au cœur natif. n4m est l'exception favorable :
le même moteur et le même binaire de modèle peuvent être partagés entre R,
Python et les autres bindings.

## 3. État actuel du binding R

### 3.1 Vue d'ensemble

| Composant | État observé | Ce qui fonctionne | Limite structurante |
|---|---|---|---|
| `dagml` R 0.3.26 | Intermédiaire | Registre loss/metric, phase planifiée, HPO, CV→refit→predict, refit initial et replay de bundle via CLI/adaptateur. | Entrées/sorties par fichiers JSON/processus, pas de frontend R ni de contrôleur Methods complet sur données réelles. |
| `dagmldata` R 0.2.11 | Réel mais bas niveau | Provider en mémoire, materialize/view/release, identités, cibles, features, collation. | JSON-in/JSON-out ; pas encore intégré au parcours public `nirs4all` R ni optimisé comme chemin de matrices volumineuses. |
| `n4m` R 1.0.21 | Substantiel | PLS et variantes, sélection, diagnostics, SNV, SG, Kennard–Stone, formule/S3 ; N4MM `raw()` ajouté et testé localement. | N4MOPT et couverture catalogue/pipelines non qualifiés ; N4MM non raccordé aux bundles DAG-ML. |
| `nirs4allformats` R 0.2.10 (branche `fix/r-flat-dataset-identity`) | Avancé | Lecture native, probe, records, dataset, parcours, bytes et sidecars. `nirs4all-r` convertit son dataset homogène en matrice/target/IDs/axe pour fit local et campagne DAG native. La branche ajoute au dataset plat l'axe `kind`, la provenance par enregistrement et le refus des mélanges d'unités/types. | Ce chemin passe par un sidecar RDS et un adaptateur process, pas encore par des data bindings/provider DAG-ML natifs. La branche s'installe depuis un checkout propre avec Cargo ; le tarball CRAN autoportant reste à revalider. |
| `nirs4allio` R 0.2.0 | Partiel | `nio_to_spec`, `nio_infer`, `nio_load`, validation et résumé assemblé. | `nio_load()` retourne une synthèse structurelle sans les matrices ; l'émission `dag-ml-data` reste côté Rust/CLI. |
| `nirs4all` R 0.3.30 dans Core | Façade bornée | Registre des six domaines, parsing JSON/YAML, runner KS/SNV/SG/PLS validé sur un oracle Python. | Pas un exécuteur DAG-ML général, pas de prédiction depuis modèle sérialisé, pas de contrôleurs R généraux. |
| Adaptateurs `prospectr`/`mdatools` | Conformance | Quelques transforms, PCA, PLS et PLS-DA via protocole processus. | Données synthétiques dans les smokes, couverture réduite, persistance et états ajustés incomplets. |

### 3.2 Écart entre les bindings DAG-ML Python et R

Le package R `dagml` exporte actuellement sept points d'entrée :

- `dagml_local_implementation_registry()` ;
- `dagml_execute_execution_plan_phase()` ;
- `dagml_host_hpo_search()` ;
- `dagml_cv_refit_predict()` ;
- `dagml_initial_full_refit()` et `dagml_initial_full_refit_predict()` ;
- `dagml_replay_bundle()`.

C'est un progrès réel depuis le précédent audit : CV, refit et replay sont
accessibles sans réécrire la coordination en R. Mais l'API R prend des chemins
de fichiers DSL/manifests/envelopes et un exécutable JSONL, puis renvoie un
JSON décodé. Le binding Python expose en plus validation/compilation/planification,
provider, configuration Methods, objets de résultats typés et conformal. Le
parcours R actuel n'accepte pas directement un `data.frame`, une matrice, une
formule ou un learner R avec sauvegarde automatique.

Le wrapper HPO R accepte désormais plusieurs essais parallèles : le CLI isole
les processus opérateurs. Le test Rust `r_hpo_ridge` exerce un vrai opérateur R
sur les folds physiques, le pruning, le checkpoint/reprise, un sidecar RDS et
un PREDICT dans un processus neuf. C'est une preuve d'intégration ciblée, pas
une qualification d'un produit R général.

Indépendamment du binding R, le contrôleur Methods intégré au produit DAG-ML
ne donne aujourd'hui accès qu'à PLS et à Ridge sous forme linéaire importée ;
le HPO Methods natif est limité à PLS. La richesse du package `n4m` R n'est
donc pas encore atteignable à travers les contrôleurs DAG-ML actuels.

### 3.3 Couverture nirs4all-methods en R

Le binding `n4m` exposait 138 fonctions R et 22 méthodes S3 à la révision
auditée ; le correctif N4MM porte ce nombre à 141. La surface comprend
notamment :

- PLS/PCR/OPLS et interfaces compatibles `pls`/`mdatools` ;
- sparse/weighted/robust/ridge/continuum/multiblock/kernel/recursive PLS ;
- plusieurs modèles supervisés spécialisés et ensembles ;
- 28 familles de sélection exposées directement ou par alias de rôle ;
- diagnostics, monitoring, PRESS approximé et règle one-SE ;
- AOM/POP, SNV, Savitzky–Golay et Kennard–Stone.

Cette liste ne prouve pas que tout le catalogue n4m est utilisable dans un DAG
R. Le catalogue couvre aussi preprocessing, augmentation, splitters, filters,
utilities, models, selection, diagnostics et AOM/POP. Certains appels passent
par le dispatcher générique, certaines API Python sont de style sklearn, et le
catalogue signale encore des symboles en cours de réconciliation. La métrique
utile n'est donc pas « 138 fonctions R contre N fonctions Python », mais une
matrice générée par `method_id` :

| Champ minimum | Exemple de valeur |
|---|---|
| Méthode et rôle | `models.regularized.ridge`, `model` |
| ABI native | symbole, ABI minimale, CPU/GPU |
| Binding R | fonction, forme des entrées/sorties, lifecycle |
| Descripteur DAG-ML | `semantic_id`, schéma de paramètres, ports |
| Phases | FIT_CV, REFIT, PREDICT, éventuellement EXPLAIN |
| État | stateless, fit-state, modèle N4MM, autre payload |
| Gates | test unitaire R, cross-binding, pipeline, replay |
| Niveau | metadata, plan, execute-local, parity-validated |

Le test de parité R principal de Methods ajuste actuellement un SIMPLS et
compare ses prédictions à une fixture native avec une tolérance stricte. Les
tests `testthat` couvrent davantage de méthodes et d'interfaces, mais ne testent
pas les invariants DAG-ML : folds, OOF, refit, sélection, handles, archive et
replay. Le test N4MM ajouté prouve un aller-retour R et un import de modèle
linéaire exporté depuis Python ; il ne qualifie pas tous les modèles.

### 3.4 Adaptateurs de frameworks R déjà présents

L'adaptateur `prospectr` couvre des transforms stateless comme binning,
continuum removal, gap derivative, Savitzky–Golay et SNV. MSC est omis à raison :
sa référence moyenne doit être ajustée sur le train de chaque fold, persistée,
puis réutilisée en validation/prédiction. L'appliquer indépendamment à chaque
batch créerait une divergence et potentiellement une fuite.

L'adaptateur `mdatools` couvre PCA, PLS et PLS-DA et désactive la CV interne du
framework afin de laisser DAG-ML la posséder. Il sauvegarde les objets avec
`saveRDS()` et les recharge avec `readRDS()`. Le contrat natif accepte
maintenant `backend = "rds"` : l'ancien défaut de schéma est corrigé. Il reste
à brancher cet adaptateur sur un vrai provider R, à qualifier la sécurité et
la compatibilité des sidecars ainsi que le replay sur données réelles. Un
smoke sur features synthétiques ne suffit pas.

## 4. État et rôle de nirs4all-core

### 4.1 Ce qu'est Core aujourd'hui

`nirs4all-core` 0.3.30 est la distribution agrégée portable. Elle enregistre et
verrouille six domaines (`formats`, `io`, `datasets`, `methods`, `dag_ml`,
`dag_ml_data`), publie les façades par langage, porte les manifests de capacité,
les contrats d'artefacts, la release et les tests de parité interlangages. Elle
n'est ni un septième moteur numérique ni un second ordonnanceur.

Son package R est déjà nommé `nirs4all`. Il déclare les packages amont
`nirs4allformats`, `nirs4allio`, `nirs4alldatasets`, `n4m`, `dagml` et
`dagmldata` comme dépendances suggérées et les atteint paresseusement. Ce choix
fait de Core le point d'entrée naturel pour un premier nirs4all R portable.

### 4.2 La limite du runner R actuel

`nirs4all_run_portable_pipeline()` parse une liste/JSON/YAML puis réalise en R :

1. un split optionnel Kennard–Stone ;
2. SNV et/ou Savitzky–Golay ;
3. plusieurs fits PLS selon `n_components` ;
4. RMSE, sélection du minimum et mise en forme du résultat.

Les calculs sont bien délégués à `n4m`, mais la boucle de contrôle est codée
dans Core. Ce cas borné est utile comme oracle de bootstrap ; l'étendre à la CV,
aux branches, à l'OOF, au HPO et à d'autres modèles créerait précisément une
seconde orchestration que l'architecture interdit. Il existe aussi un écart
entre la description du package (« aucune logique pipeline réimplémentée ») et
ce runner manuel.

La cible est de conserver la signature publique si elle est utile, mais de la
faire déléguer à `dagml`/`dagmldata` et aux contrôleurs. Une fois le nouveau
chemin qualifié, l'implémentation R manuelle doit être dépréciée puis retirée.

### 4.3 Rôle cible de Core

Core doit rester :

- la façade R de découverte, installation et composition ;
- le verrou compatible des versions amont ;
- le lieu des profils portables et des déclarations honnêtes de capacité ;
- le gate d'intégration qui vérifie qu'un pipeline réellement exécuté par les
  amonts reproduit l'oracle ;
- éventuellement l'API ergonomique `pipeline()`, `run()`, `predict()` et
  `load()` si ces fonctions ne contiennent que traduction et délégation.

Core ne doit pas contenir : kernels, split/CV, scoring, sélection, stockage de
prédictions, implémentation d'un modèle ou adaptateur substantiel de framework.

## 5. Ce qui manque pour la parité Python

### 5.1 Fondations obligatoires

| Priorité | Manque | Critère de sortie |
|---|---|---|
| P0 | Surface R complète des contrats DAG-ML | Valider/compiler le DSL, construire un plan et exposer les résultats typés sans passer par Python. |
| P0 | Session native CV → SELECT → REFIT → PREDICT | Handles R possédés par `externalptr`/finalizers, libération vérifiée et replay dans un processus neuf. |
| P0 | Chemin IO → Data → DAG | Charger de vraies matrices et identités depuis R, les présenter au provider et exécuter sans reconstruction positionnelle. |
| P0 | Contrat d'artefact R | Ajouter/cadrer RDS ou un codec hôte générique, SHA-256, version runtime, politique de confiance et confinement d'URI. |
| P0 | Contrôleur Methods générique | Résoudre les descripteurs n4m par catalogue plutôt que coder PLS/Ridge au cas par cas. |
| P0 | Sérialisation n4m en R | Inspecter, exporter et importer N4MM ; sauver/reprendre N4MOPT lorsqu'un HPO natif l'utilise. |
| P0 | Matrice de capacité générée | Toute revendication R est dérivée du catalogue, de l'ABI, du NAMESPACE et de tests exécutables. |
| P0 | Harness cross-langage | Python→R et R→Python pour plan, fit, predict, archive, erreurs et provenance. |

Le provider R de `dag-ml-data` existe déjà. Le travail Data P0 consiste donc
moins à créer un package qu'à intégrer ses handles au binding `dagml`, éviter
les gros allers-retours JSON et raccorder `nirs4allio` à un payload/provider
réel. Les matrices R sont column-major ; le contrat de tensor doit porter
shape, dtype et strides, et copier explicitement seulement lorsque le cœur ou
le contrôleur exige un layout différent.

### 5.2 Capacités de pipeline à qualifier

Pour atteindre l'équivalent du socle Python, les gates R doivent couvrir :

- jeux de données multi-source, cibles multiples, répétitions, groupes,
  partitions indépendantes et identités non ordonnées ;
- splitters simples et groupés, CV partagée, OOF, agrégation par fold,
  sélection, refit final et prédiction sur une cohorte distincte ;
- preprocessing X et y stateless/stateful, filtres, sélection de variables,
  augmentation train-only et conservation des origines ;
- branches, générateurs de variantes, concaténation/fusion et stacking avec
  uniquement des prédictions OOF comme features d'entraînement ;
- tuning avec espace ordonné, reprise, pruning et provenance, sans déléguer la
  propriété de la CV au framework hôte ;
- scoring, résultats/predictions persistés, workspace, export, chargement et
  replay ;
- classification, régression, multioutput et formes de prédictions déclarées ;
- pertes/métriques R locales, explainability et conformal aux niveaux
  effectivement supportés par les contrats natifs.

Les graphiques, SHAP, synthèse de données et fonctions très spécifiques au
produit Python peuvent venir après le socle d'exécution. Ils ne doivent pas
bloquer la qualification « pipeline portable n4m-only », mais leur absence
doit rester visible dans la matrice de capacités.

### 5.3 Définition de terminé pour le profil n4m-only

Une méthode n4m n'est déclarée `parity-validated` en R que si :

1. son `method_id`, ses paramètres, ports et contraintes sont versionnés ;
2. le même descripteur compile en un plan canonique en Python et en R ;
3. FIT_CV, REFIT et PREDICT utilisent le contrôleur Methods natif, sans calcul
   numérique alternatif en R ;
4. tout état appris par fold est distinct et le refit possède son propre état ;
5. l'artefact est inspectable/rechargeable, avec ABI et empreinte vérifiées ;
6. les identités, folds, OOF, scores et sélection sont identiques, et les
   sorties numériques respectent une tolérance publiée ;
7. un test Python-fit→R-predict et un test R-fit→Python-predict passent lorsque
   le type de modèle est sérialisable ;
8. paramètres invalides, payloads corrompus, ABI incompatible et composition
   illégale échouent avant l'exécution.

## 6. Architecture technique proposée

### 6.1 Une seule orchestration, plusieurs contrôleurs

Le package R public construit ou lit une recette déclarative. `dagml` la
compile et exécute le plan. `dagmldata` présente les vues de données. Pour
chaque tâche, le contrôleur concerné reçoit les handles et retourne les
prédictions, artefacts et attestations. Le package `nirs4all` ne boucle pas sur
les folds et ne choisit pas le gagnant lui-même.

La surface R cible peut rester idiomatique : listes nommées, formules et
data.frames en entrée ; objets S3/R6 de résultat en sortie. Ces objets sont des
vues d'un résultat natif immuable et non une seconde implémentation du contrat.

### 6.2 Transfert des méthodes n4m

Le C ABI fournit déjà :

- `n4m_model_export_size` et `n4m_model_export_to_buffer` ;
- `n4m_model_import_from_buffer` ;
- `n4m_serialization_inspect`, `inspect_model_v1` et `inspect_pipeline_v1` ;
- le format distinct N4MOPT pour la reprise d'optimiseur.

Le travail R recommandé est :

1. exposer `n4m_model_export()`, `n4m_model_import()` et
   `n4m_model_inspect()` sur des `raw()` R (fait dans le workspace, à qualifier
   sur les plateformes) ;
2. garder le handle importé dans un `externalptr` avec finalizer idempotent ;
3. refuser avant allocation les formats, tailles, checksums et ABI non permis ;
4. attacher le même payload N4MM aux artefacts DAG-ML, avec SHA-256, descripteur
   du modèle, ABI minimale et provenance du nœud ;
5. tester les deux directions R/Python et le rechargement dans un processus
   sans workspace d'entraînement ;
6. ajouter de la même manière `n4m_optimizer_save_raw()`/
   `n4m_optimizer_load_raw()` si la reprise HPO est exposée en R.

N4MM format 1 transporte l'état ajusté d'un modèle. Le format 2 ne représente
qu'un pipeline borné `SNV(ddof=0) → Savitzky–Golay(interp, deriv=0, delta=1) →
PLS`. N4MM n'est donc ni un graphe DAG-ML général, ni une archive nirs4all
complète. Il faut conserver le graphe, le schéma de données, les décisions,
scores et empreintes dans l'archive DAG-ML/Core.

Pour généraliser les pipelines n4m, chaque nœud doit être sérialisé selon sa
nature : descripteur seul pour un opérateur réellement stateless, payload
versionné d'état pour un transform ajusté, N4MM pour un modèle supporté. Une
archive de pipeline universelle codée en parallèle dans n4m recréerait le rôle
de DAG-ML et n'est pas recommandée.

### 6.3 Portabilité hors n4m : niveaux explicites

| Niveau | Mécanisme | Promesse |
|---|---|---|
| P0 | Même payload natif portable, par exemple N4MM | Même état appris et prédiction cross-langage sous tolérance. |
| P1 | Graphe d'inférence neutre, principalement ONNX | Même inférence si tous les opérateurs, dtypes et opsets sont qualifiés ; pas de réentraînement. |
| P2 | Convertisseur d'état spécifique à une famille | Même modèle ajusté seulement pour le sous-ensemble explicitement couvert. |
| P3 | Dictionnaire de recette | Pipeline sémantiquement proche, réentraîné dans le langage cible ; nouvelle lineage et nouveaux résultats. |
| P4 | Artefact hôte opaque (`joblib`, RDS, `.keras`, etc.) | Replay dans le runtime d'origine verrouillé ; aucune portabilité interlangage. |

Le choix doit être enregistré par nœud. Un export peut mélanger les niveaux :
par exemple preprocessing n4m portable, réseau ONNX pour l'inférence, et objet
RDS conservé uniquement pour reprendre l'entraînement en R.

### 6.4 Dictionnaire de traduction de recettes

Le dictionnaire doit être déclaratif, versionné et résolu avant l'exécution.
Une entrée minimale contient :

```yaml
semantic_id: tree.random_forest.regression.v1
source:
  runtime: sklearn
  class: sklearn.ensemble.RandomForestRegressor
targets:
  - runtime: r
    engine: ranger
    task: regression
    portability: recipe_retrain
    parameter_map:
      n_estimators: num.trees
      max_depth: max.depth
      min_samples_leaf: min.node.size
    unsupported:
      - monotonic_cst
    semantic_deltas:
      - categorical_handling
      - random_number_generator
      - split_and_tie_breaking
artifact_codecs: [onnx, host_native]
```

Il faut aussi décrire : valeurs par défaut figées, plages et types, traitement
des NA/catégories, encodage des classes, shape des sorties, seeds, versions des
packages, capacités fit/predict/export, tolérances et fixtures de référence.
Toute option inconnue ou non mappable doit faire échouer la traduction, sauf si
l'utilisateur accepte explicitement une dégradation enregistrée.

Un `RandomForestRegressor` sklearn ne doit donc pas devenir silencieusement un
`randomForest` ou un `ranger` R. Les défauts, le bootstrap, `max_features`/`mtry`,
les critères, catégories, valeurs manquantes, RNG et détails de croissance des
arbres diffèrent. Deux chemins honnêtes existent :

- convertir le modèle ajusté en ONNX et exécuter ce graphe en R ;
- traduire la recette vers `ranger`, réentraîner, et enregistrer un nouveau
  modèle avec `translation_kind = recipe_retrain` et ses deltas sémantiques.

Le dictionnaire ne doit pas tenter d'inventer un convertisseur d'état sklearn
→ ranger tant qu'un format commun des arbres et des tests exhaustifs ne le
justifient pas.

### 6.5 Rôle d'ONNX et autres formats

ONNX est adapté à l'inférence portable, pas à la portabilité générale de
l'entraînement. `skl2onnx` couvre notamment PLSRegression, Ridge, Random Forest,
Extra Trees, Gradient Boosting, SVM, MLP et de nombreux preprocessings, mais la
couverture n'est pas totale et certains pipelines exigent des convertisseurs
enregistrés. Chaque export doit être préflighté puis comparé sortie par sortie
au modèle source.

La matrice d'installation officielle d'ONNX Runtime ne publie pas aujourd'hui
de package R de première classe. Un contrôleur R de production doit donc soit
lier l'API C/C++ d'ONNX Runtime dans un package possédé par le projet, soit
utiliser un worker isolé. Une dépendance R tierce ne doit pas être considérée
supportée sans gate de version et de plateforme.

Autres options, plus spécialisées :

- PMML pour un sous-ensemble de modèles classiques, sans en faire le format
  d'archive principal ;
- Treelite pour certains arbres, sous réserve de couverture des producteurs et
  d'un runtime R maîtrisé ;
- `.keras` pour sauvegarde/reprise Keras dans son écosystème ;
- poids `safetensors` plus architecture déclarative pour certains modèles DL ;
- Arrow pour les données et prédictions, jamais comme format de modèle.

### 6.6 Contenu de l'archive interlangage

Une archive complète doit au minimum contenir :

```text
pipeline.json                 DAG canonique et versions de schéma
execution-plan.json           plan et manifests de contrôleurs
operator-resolution.json      semantic_id, moteur choisi, deltas et niveau P0–P4
data-contract.json            schémas, identités, relations et empreintes
artifacts/<node>/...          N4MM, ONNX ou artefact hôte selon le descripteur
predictions/...               prédictions OOF/refit/test avec identités
scores/...                    métriques, agrégations et sélection
lineage/...                   seeds, versions, folds, attestations et provenance
environment/...               dépendances, plateformes et politique de chargement
```

Chaque artefact référence son codec, runtime, versions, taille, SHA-256,
capacités et politique de confiance. `joblib`/pickle et RDS peuvent reconstruire
des objets exécutables : ils sont refusés par défaut pour une archive non fiable,
chargés seulement dans un environnement isolé et ne deviennent jamais un
format portable par simple changement d'extension.

## 7. Frameworks R à binder

| Candidat | Usage recommandé | Règle d'intégration |
|---|---|---|
| n4m | Socle NIRS portable prioritaire | Contrôleur natif générique et artefacts N4MM. |
| `prospectr`, `mdatools` | Preprocessing/chimiométrie R | Promouvoir les adaptateurs existants après gestion des états par fold et artefacts. |
| `ranger`, `glmnet`, `xgboost` | Premiers moteurs ML classiques | Contrôleurs directs pour figer exactement paramètres et défauts. |
| `parsnip`/tidymodels | Frontend R ergonomique et résolution d'engine | Compiler la spécification vers un descripteur ; ne pas lui déléguer CV/HPO. |
| `mlr3` | Catalogue de learners et enveloppe train/predict | Utiliser les learners, pas les resamplings/tuners lorsque DAG-ML orchestre. |
| `torch` R | Modèles DL R-native | Worker/contrôleur possédant tensors, GPU et état ; artefact hôte versionné, ONNX optionnel. |
| `keras3` | Réseaux Keras multi-backend | Contrôleur possédant fit/predict ; `.keras` pour replay hôte, ONNX seulement après qualification. |

`parsnip` est utile comme vocabulaire d'interface et sélection d'engine, mais
ses défauts doivent être matérialisés avant fingerprinting. `mlr3` expose un
cycle learner train/predict adapté à un contrôleur, mais ses mécanismes de
resampling/tuning entreraient en concurrence avec DAG-ML. Les documentations
officielles de `torch` R avertissent que `torch_save()` n'est pas un format de
stockage long terme garanti ; le bundle doit donc verrouiller version et
architecture. Keras 3 fournit un format `.keras` comprenant configuration,
poids et état d'optimiseur, utile pour le replay dans le même écosystème.

## 8. Coût estimatif

Les chiffres ci-dessous sont des ordres de grandeur en semaines-ingénieur pour
une personne familière du workspace. Ils incluent code, fixtures, tests Linux,
documentation et refus négatifs, mais pas le délai CRAN, les runners GPU ni la
qualification Windows/macOS complète.

| Lot | Charge | Livrable |
|---|---:|---|
| Inventaire généré et profils de capacité | 2–3 | Matrice catalogue/ABI/Python/R/DAG et définition n4m-only. |
| Binding `dagml` R complet | 6–10 | Compile/plan/session CV-refit-predict, résultats et replay. |
| Intégration IO/Data et tenseurs R | 3–5 | Vraies données, identités, handles et chemin volumineux. |
| Contrôleur n4m générique + N4MM/N4MOPT R | 6–10 | Méthodes éligibles, artefacts et cycles de vie. |
| Harness n4m-only cross-langage | 4–6 | Compositions générées, Python↔R, archives et erreurs. |
| Premiers contrôleurs ML R classiques | 6–10 | ranger/glmnet/xgboost plus promotion prospectr/mdatools. |
| Dictionnaire : schéma, resolver, audit | 3–5 | Traduction fail-closed, deltas et provenance. |
| Cinq premières familles de dictionnaire | 6–10 | Linéaire/Ridge, PLS, RF/ExtraTrees, boosting, SVM par exemple. |
| Runtime ONNX côté R | 3–6 | API C ou worker, packaging et parité d'inférence. |
| Contrôleurs `torch` + `keras3` | 10–16 | Fit/predict/replay/GPU, seeds et artefacts ; ONNX séparé. |
| Façade, documentation et release R | 2–4 | API publique, erreurs, vignettes et matrice installée. |

Un MVP crédible « pipelines n4m-only complets en R » représente donc environ
21–34 semaines-ingénieur. Le dictionnaire lui-même coûte environ 9–15 semaines
pour l'infrastructure et cinq grandes familles ; une famille simple ajoute
typiquement 1–2 semaines, une famille complexe 2–4 semaines. La maintenance
des changements de défauts et de versions de frameworks demande ensuite
environ 0,1–0,25 ETP selon le nombre de couples de runtimes supportés.

Ajouter ML classique, dictionnaire et ONNX porte le programme autour de 35–55
semaines-ingénieur. Une expérience R large avec DL, packaging multi-OS et les
fonctions de haut niveau du Python est plutôt un programme de 50–80 semaines,
parallélisable entre cœur/bindings, Methods, données et contrôleurs.

## 9. Faut-il un dépôt nirs4all-r dédié ?

### 9.1 Options

| Option | Avantages | Risques |
|---|---|---|
| Étendre `nirs4all-core/bindings/r` | Façade et package existent ; cohérence du lock multi-langage ; coût de gouvernance minimal. | Core grossit ou viole ses frontières si des contrôleurs/UX R substantiels y entrent. |
| Créer `nirs4all-r` | Ownership R clair, cadence CRAN indépendante, contrôleurs/recettes/vignettes et CI R isolés ; symétrie avec le produit Python. | Nouveau repo/release, risque de duplication et collision immédiate avec le package `nirs4all` déjà publié par Core. |
| Créer plusieurs packages d'adaptateurs | Dépendances optionnelles et releases ciblées. | Expérience fragmentée, matrice de compatibilité et installation plus difficiles. |

### 9.2 Recommandation par étapes

**Décision validée le 25 septembre 2026 : dépôt produit R dédié.** Le dépôt
public [GBeurier/nirs4all-r](https://github.com/GBeurier/nirs4all-r) a été
initialisé sous AGPL-3.0-or-later avec une API fit/predict sur matrices R,
des contrôleurs `n4m` PLS, `stats::lm`, `ranger`, `glmnet` et un MLP CPU
`torch` R. Son modèle n4m est sauvegardé en octets N4MM, tandis que le
module torch utilise son sérialiseur natif dans le bundle RDS. Depuis le
25 septembre, une voie native **mono-modèle** construit les contrats R,
demande à DAG-ML FIT_CV/OOF/REFIT/PREDICT et fait tourner ces cinq familles
de contrôleurs sur les lignes de folds prescrites. Les prétraitements restent
des paramètres internes au nœud modèle : les graphes à branches, HPO et
prédiction sur cohorte externe ne sont pas encore exposés. La demande d'une API
R de bout en bout, de contrôleurs `ranger`/`parsnip`/`mlr3`/`torch` et d'une
cadence CRAN indépendante atteint désormais le seuil où ces composants ne
doivent pas entrer dans Core. `nirs4all-core/bindings/r` reste une façade
portable pendant la transition ; le runner manuel ne doit pas être généralisé.
La logique d'exécution reste dans `dagml`, les données dans `dagmldata`/
`nirs4allio` et les calculs portables dans `n4m`.

**Justification de l'extraction :** plusieurs conditions sont maintenant
explicitement recherchées par le produit :

- plusieurs contrôleurs ML/DL R ont leur propre cycle de release ;
- une couche de compilation formulas/recipes/parsnip/mlr3 devient substantielle ;
- l'API R ajoute explainability, visualisation ou expérience interactive
  propres à R ;
- les dépendances et la cadence CRAN ne peuvent plus suivre la release Core ;
- l'équipe R a un ownership et une roadmap indépendants.

La solution propre est de **migrer**, et non dupliquer, le package
public `nirs4all` vers le dépôt `nirs4all-r` lorsque les premiers contrôleurs
R réels et l'API de façade sont qualifiés. Le dépôt Core peut alors ne plus
publier de package R de haut niveau, ou publier un éventuel package technique
`nirs4allcore` si une façade basse est réellement nécessaire. Cette décision
est un changement de gouvernance et de nommage à versionner explicitement.

Publier simultanément deux dépôts avec un package nommé `nirs4all` est exclu.
Un dépôt `nirs4all-r` peut être créé comme développement non publié pendant la
transition, mais le `packages.json` R-universe et les métadonnées CRAN ne
doivent changer de propriétaire qu'une fois le nouveau package testé. Le nom
alternatif `nirs4allR` éviterait la collision mais fragmenterait l'API.

Les packages bas niveau (`dagml`, `dagmldata`, `n4m`, `nirs4allio`,
`nirs4allformats`) restent dans leurs dépôts propriétaires dans tous les cas.

## 10. Plan de réalisation proposé

### Phase R0 — vérité des capacités et contrats

- générer la matrice Methods par `method_id` ;
- figer les profils `n4m-only-r-v1` et `r-host-v1` ;
- qualifier de bout en bout le backend `rds` désormais accepté par le contrat ;
- définir les codecs, politiques de confiance et tolérances ;
- ajouter les gates qui empêchent Core et les bindings de sur-déclarer.

### Phase R1 — binding natif DAG/Data complet

- porter en R les fonctions de validation, compilation, planification et
  résultats actuellement accessibles en Python ;
- exposer une session native possédant controller/data/artifact stores ;
- raccorder `dagmldata` et un transport de tenseurs sans gros JSON ;
- prouver folds, OOF, sélection, refit et replay sur contrôleur factice R.

### Phase R2 — profil n4m-only

- exploiter N4MM désormais exposé en R ; ajouter inspection et N4MOPT si requis ;
- remplacer les contrôleurs PLS/Ridge codés au cas par cas par une résolution
  générée depuis le catalogue ;
- qualifier transforms, selectors, augmentations, modèles et diagnostics
  éligibles ;
- exécuter Python-fit→R-predict, R-fit→Python-predict et archives corrompues ;
- faire déléguer `nirs4all_run_portable_pipeline()` au nouveau chemin.

### Phase R3 — moteurs R classiques et dictionnaire

- promouvoir `prospectr`/`mdatools` ;
- ajouter ranger/glmnet/xgboost, puis un frontend parsnip optionnel ;
- publier le dictionnaire v1 avec des deltas explicites ;
- intégrer ONNX pour l'inférence des familles effectivement convertibles.

### Phase R4 — DL et migration du nom public

- ajouter les contrôleurs `torch` et `keras3` avec workers GPU isolables ;
- qualifier artefacts hôtes, export ONNX et reprise ;
- qualifier les contrôleurs ML/DL R et la distribution autoportante ;
- transférer le package public `nirs4all` de Core à `nirs4all-r` sans double
  publication, une fois les gates de la section 11 passés.

## 11. Gates de validation minimales

Une release ne doit annoncer la parité R que si les gates suivants passent :

- même DSL → même graphe/plan/fingerprints en Python et R ;
- mêmes folds et refus de fuite pour groupes, répétitions et augmentations ;
- mêmes OOF, décisions de sélection et agrégations ;
- mêmes prédictions n4m dans les deux sens sous tolérance par méthode ;
- replay dans un processus neuf sans accès au dossier d'entraînement ;
- import refusé pour checksum, ABI, schéma, codec ou runtime incompatibles ;
- test de chaque entrée du dictionnaire avec paramètres par défaut et limites ;
- différence source/ONNX mesurée sur un corpus de qualification ;
- artefacts RDS/joblib non fiables refusés par défaut ;
- matrice de capacité installée identique à celle publiée dans la documentation.

### Vérifications effectivement exécutées pendant ce réaudit

| Gate | Résultat local | Ce que cela ne prouve pas |
|---|---|---|
| `n4m` R : suite `testthat` | Réussite sur R 4.6.0/Linux, un test de comparaison externe `pls` sauté selon sa règle CRAN. | Pas les centaines de méthodes du catalogue ni Windows/macOS. |
| `n4m` R : fixture native | `rmse_rel = 1,267×10⁻¹⁶` avec le tarball vendored 1.0.21 (ABI 2.6.0). | Une seule fixture SIMPLS, pas toutes les méthodes. |
| `n4m` source vendored : `R CMD check --as-cran --no-manual` | 0 erreur, 0 warning, 2 NOTEs sur R 4.6.0/Linux : nouvelle soumission et `-march=nocona` injecté par R conda-forge. | Autres systèmes et R-devel non testés. |
| `dagml` : `R CMD check --no-manual` | 0 erreur, 0 warning, 0 NOTE sur R 4.6.0/Linux avec bibliothèque native explicitement fournie. | Le CLI et la bibliothèque sont encore requis hors du tarball R. |
| `dagmldata` : `R CMD check --no-manual` | 0 erreur, 0 warning, 0 NOTE sur R 4.6.0/Linux après documentation de `InMemoryProvider`. | Provider bas niveau, pas de parcours nirs4all R intégré. |
| `nirs4alldatasets` : `R CMD check --no-manual` | 0 erreur, 0 warning, 0 NOTE sur R 4.6.0/Linux. | Source ~24 Mo ; taille et multi-OS à qualifier pour CRAN. |
| `nirs4allio` : `R CMD check --no-manual` | 0 erreur, 0 warning, 0 NOTE sur R 4.6.0/Linux après déclaration de GNU make et retrait d'une copie de licence AGPL déjà présente sous `inst/LICENSES/`. | Le binding R ne fournit encore qu'une synthèse structurelle, pas les matrices d'entraînement. |
| `nirs4allformats` : `R CMD check --no-manual` | 0 erreur, 0 warning, 0 NOTE sur R 4.6.0/Linux après vendoring du tarball et déclaration de GNU make. | Le tarball brut du sous-dossier échoue : le hook vendoring doit être exécuté avant `R CMD build`. Source ~13 Mo. |
| N4MM R | Aller-retour `fit → raw → import → predict` et refus d'entrées malformées réussis. R-fit→Python-predict : écart max 0 sur la fixture SIMPLS ; Python→R : modèle affine importé et prédictions exactes. | Toutes les familles de modèles et archives DAG-ML non qualifiées ; l'essai interlangage initial utilisait le `libn4m` de développement 1.0.20, le gate n4m/Core a ensuite été repassé avec le tarball 1.0.21. |
| Core R portable | Quatre fixtures Python-oracle KS/SNV/SG/PLS passent avec `NIRS4ALL_CORE_REQUIRE_METHODS_PARITY=1`, y compris contre le tarball n4m 1.0.21 autoportant. | Pas un pipeline DAG général ni un jeu de test indépendant. |
| Nouveau produit public `nirs4all-r` (`5fc12fb`) | `R CMD check --as-cran --no-manual` : tests avec `ranger` 0.18.0, `glmnet` 5.1, `torch` R 0.17.0 CPU, replay dans un processus neuf et quatre cas Python-oracle vendorizés obligatoires. Écarts max des prédictions sélectionnées : `6,66×10⁻¹⁵` (SNV/PLS), `5,00×10⁻¹⁵` (SG/PLS), `2,89×10⁻¹⁵` (KS/SNV/PLS), `2,79×10⁻¹⁴` (KS/SNV/SG/PLS avec cinq variantes). 0 erreur, 1 warning incoming : nouvelle soumission, version de développement et dépendance forte `n4m` hors CRAN. | Ces tests utilisent les fonctions n4m et la façade produit, pas un DAG natif. Ils ne qualifient ni folds/OOF/HPO, ni l'ensemble des méthodes n4m, ni GPU/multi-OS. Le module torch et le bundle RDS sont R-spécifiques. |
| Pont DAG-ML R mono-modèle (`nirs4all-r` `b4e3ca5`) | Test strict `NIRS4ALL_REQUIRE_DAG_PARITY=1` sous Linux/R 4.6.0 : `dagml` + CLI exécutent FIT_CV, OOF, REFIT et replay pour PLS/SNV/SG, `lm`, `ranger`, `glmnet` et MLP `torch` CPU sur matrices R réelles. Chaque OOF est comparé à un ajustement manuel limité au train de son fold ; chaque replay au refit complet. Le contrôle d'empreinte rejette une cible RDS altérée. `R CMD check --as-cran --no-manual` : 0 erreur, 1 warning incoming (nouvelle soumission, version dev, `n4m`/`dagml` hors CRAN). | Le CLI et la bibliothèque `dagml` ne sont pas distribués par ce tarball. Pas encore de data bindings natifs explicites, ni de branches/merge, HPO, cohorte externe, GPU ou multi-OS. `dagml` reste absent du dépôt R-universe public courant. |
| Méthodes `n4m` supplémentaires (`nirs4all-r` `ecb2977`, `nirs4all-methods` `2b870100`) | Huit régressions linéaires MethodResult exposées par contrôleur R ; six vérifiées contre un oracle Python figé, écarts max ≤ `5×10⁻¹⁴`. CPPLS et ridge-PLS R utilisent le solveur NIPALS natif ; la régression continuum Python est explicitement dirigée vers le chemin canonique Stone–Brooks. Tests R/Python amont et test DAG natif ridge/CPPLS passent. | Les coefficients MethodResult sont conservés en RDS, pas en modèle N4MM ; ceci ne couvre pas les méthodes n4m non linéaires, sélecteurs, augmentations ou prétraitements manquants du binding R. |
| Intégration `nirs4all-formats` → `nirs4all-r` (`364d8f4`) | Fixture CSV spectrale décodée par le binding R Rust : 12 échantillons, 8 longueurs d'onde, cible numérique, IDs uniques. Fit/prédiction locaux et FIT_CV→OOF→REFIT→replay DAG natifs passent ; altération d'unité/type d'axe et IDs dupliqués refusées. Sur la fixture amont réelle de 50×200, le fit local SNV/PLS passe. La branche formats `ed7dccb` (0.2.10) a passé `R CMD INSTALL` depuis un checkout propre avec Cargo ; son `test-io.R` : 33 assertions, 0 échec. Le produit R a ensuite passé les tests formats et DAG sur ce package réellement installé. Contrôle strict `R CMD check --as-cran --no-manual` du produit R : 0 erreur, 1 warning incoming. | L'archive Cargo vendorizée du vieux checkout local reste périmée (`bytes` 1.12.0 contre lock 1.12.1) ; cela n'affecte pas l'installation du checkout propre mais le tarball CRAN autoportant doit être revalidé. Axes hétérogènes, cubes et chargement multi-fichiers restent hors du convertisseur. Pas encore de data bindings DAG natifs. |
| `dag-ml-cli` + opérateur R Ridge | Test `r_hpo_ridge` réussi : folds, HPO parallèle, reprise, sidecar et replay. | Pas un contrôleur `ranger`, `torch` ou Methods général. |
| `R CMD check` de `nirs4all` Core | 0 erreur, 0 warning, 0 NOTE après retrait de la notice `LICENSE` redondante, avec `Suggests` manquants non forcés. | Check CRAN complet avec toutes dépendances et plateformes non effectué. |

La publication du produit `nirs4all` R n'est donc **pas qualifiée**. En outre,
le tarball `dagml` ne fournit ni le CLI ni la bibliothèque C ABI qu'exercent
ses tests R : le check vert ci-dessus les prend dans le build Rust local. Il
faut un mode d'installation autoportant ou une dépendance système explicitement
distribuée avant d'annoncer un package R-universe/CRAN installable hors de ce
workspace. Le dépôt R-universe listait `nirs4all` depuis Core mais omettait
`dagml`, pourtant suggéré par Core. L'entrée a été ajoutée localement, sans
push. R-universe construirait de toute façon les révisions GitHub publiées,
pas les correctifs non poussés du workspace. Une
mise en ligne immédiate annoncerait une parité qui n'existe pas.

Pour CRAN, un tarball source est soumis **par package**, dans l'ordre des
dépendances ; une archive composite de tous les tarballs n'est qu'un kit de
transmission, pas un package CRAN soumettable. Le kit de release doit inclure
les tarballs source autoportants, leurs SHA-256, les versions/commits, les
licences, le journal `R CMD check --as-cran` par plateforme, les résultats de
parité, les commentaires au mainteneur et la déclaration des dépendances non
présentes sur CRAN. Aucun texte ne doit prétendre à un check vert non exécuté.
Voir le [dossier de soumission R](R_CRAN_SUBMISSION_READINESS.md) pour les
textes de formulaires et les bloqueurs précis.

## 12. Sources de l'audit

Snapshot local inspecté :

| Dépôt | Révision | Sources principales |
|---|---|---|
| `dag-ml` | `e90706f` | `bindings/r`, `crates/dag-ml-py`, `crates/dag-ml-core`, `docs/SUPPORTED.md`, adaptateurs R. |
| `dag-ml-data` | `ffe5337` | `crates/dag-ml-data-r`, provider et contrats. |
| `nirs4all-core` | `57fee01` | `bindings/r`, `docs/ARCHITECTURE.md`, `docs/CAPABILITIES.md`. |
| `nirs4all-methods` | `55779328` | `bindings/r/n4m`, `catalog`, ABI et documentation N4MM. |
| `nirs4all` | `3d902618` | API/pipelines Python et intégration DAG-ML. |
| `nirs4all-io` | `d2fddd0` | binding R, C ABI et bridge `nirs4all-io-dagml`. |
| `nirs4all-formats` | `9ddb8f5` | package R `nirs4allformats`. |

Références externes consultées pour les choix de frameworks et de formats :

- [parsnip `rand_forest`](https://parsnip.tidymodels.org/reference/rand_forest.html) ;
- [mlr3 : interface learners et prédiction](https://mlr3book.mlr-org.com/chapters/chapter15/predsets_valid_inttune.html) ;
- [`torch` R : sérialisation](https://torch.mlverse.org/docs/articles/serialization) et
  [`torch_save`](https://torch.mlverse.org/docs/reference/torch_save.html) ;
- [Keras 3 R : sauvegarde et sérialisation](https://keras3.posit.co/articles/serialization_and_saving.html) ;
- [matrice d'installation ONNX Runtime](https://onnxruntime.ai/docs/install/) ;
- [modèles sklearn couverts par `skl2onnx`](https://onnx.ai/sklearn-onnx/supported.html)
  et [limites de conversion des pipelines](https://onnx.ai/sklearn-onnx/pipeline.html).
