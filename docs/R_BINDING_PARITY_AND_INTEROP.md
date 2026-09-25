---
orphan: true
---

# Binding R : parité nirs4all et portabilité interlangage

> Réaudit au 24–25 septembre 2026, mis à jour après la décision d'extraire le
> paquet R du core. Ce document décrit les sources présentes
> dans le workspace, pas une promesse de release. Les niveaux « supporté »,
> « conformance », « expérimental » et « backlog » gardent le sens défini dans
> [Supported surface](SUPPORTED.md).

## Résumé décisionnel

**Décision de propriété au 25 septembre 2026.** Le paquet R public s'appelle
`nirs4all` et est construit depuis `GBeurier/nirs4all-r`. L'ancien
`nirs4all-core/bindings/r` a été retiré de Core `main` ;
ses fixtures JSON/YAML de référence restent sous
`nirs4all-core/tests/parity/fixtures` et sont copiées avec empreinte vérifiée
dans le produit R. Les bindings Python, JS/WASM et MATLAB du Core ne sont pas
retirés dans cette étape. Les parties ci-dessous qui décrivent Core comme
propriétaire du paquet R doivent être lues comme **état historique antérieur
au transfert**, et non comme cible de publication.

### Niveaux de portabilité demandés et priorité de cette livraison

| Niveau | Contrat visé | Statut / gate nécessaire |
|---|---|---|
| 1 — recette native | Le même JSON/YAML désigne des opérations n4m et des primitives DAG-ML/Data par identifiants sémantiques ; chaque hôte les résout vers son binding, avec refus explicite des nœuds inconnus. | **Partiel.** Les quatre exemples KS/SNV/SG/PLS passent dans le lecteur R. `nirs4all-r` 0.4.0.9019 exporte JSON/YAML pour le profil plat SNV/SG/PLS avec les identifiants `n4m.*`, reconstruit les branches `branch`→`merge: features` de Python et développe `_or_`/`_cartesian_` puis les variantes DAG. Son lecteur accepte aussi `n4m.LSNV`, `n4m.RNV`, `n4m.AreaNormalization`, `n4m.Detrend`, `n4m.MSC` et `n4m.EMSC`, avec parité numérique Python n4m testée, mais **leur export interlangage n'est pas encore qualifié**. Core #15 reconnaît seulement le premier profil SNV/SG/PLS dans les parseurs Python/Rust/JS/MATLAB. Le catalogue n4m complet, les modificateurs de générateurs et les graphes DAG arbitraires ne passent pas encore. |
| 2 — état natif entraîné | Une archive de pipeline n4m/DAG-ML restitue son état pour PREDICT et conserve une recette/lineage permettant un nouveau FIT dans un autre hôte. | **Partiel, deux profils natifs exacts.** `nirs4all` R 0.4.0.9019 échange le PLS seul en N4MM format 1 et `n4m` R 1.0.21.9002 produit/inspecte N4MM format 2 avec SNV→SG embarqué. L'état est importé contre une recette JSON/YAML séparée ; solveur, composantes, largeur et prétraitement embarqué sont vérifiés. Des processus Python distincts prouvent R→Python et Python→R à `1e-10` dans les deux profils. Le RDS et les sidecars DAG ne sont pas interlangages. Il reste à raccorder l'Archive V2/V3 validée par Rust, identités/manifestes/lineage et recette de réentraînement, puis à élargir les profils/familles. |
| 3 — poids PyTorch | Inférence R à partir de poids entraînés en Python, sous architecture, dtypes et opérateurs explicitement qualifiés ; WASM à étudier. | Différé, sauf chemin déjà vérifiable. Un RDS `torch` R n'est pas ce contrat. |
| 4 — alias de recettes ML | Traduire par exemple sklearn Random Forest en `ranger` pour **réentraîner** depuis JSON/YAML, avec différences sémantiques enregistrées ; aucun transfert du binaire appris. | Différé ; dictionnaire limité et versionné possible ensuite. |
| 5 — ONNX | Inférence commune pour les opérateurs et runtimes effectivement couverts, sans supposer la reprise d'entraînement. | Description seulement à ce stade. |
| 6 — pipeline mixte | DAG-ML coordonne les nœuds exécutés dans leurs langages ; les bindings assurent le marshalling typé et l'attestation. | Description seulement à ce stade. |

Les niveaux ne sont pas interchangeables : un fichier de recette réentraînable
ne rend pas ses poids portables ; N4MM transporte un modèle natif mais pas à lui
seul le graphe, les folds, les données ni les états de toutes les transformations.
Pour cette livraison, ne publier la revendication « niveaux 1–2 complets »
qu'une fois les gates ci-dessus verts, y compris les erreurs de paramètres,
les données alignées et le replay hors du processus d'entraînement.

Les briques d'un nirs4all R existent, mais elles ne forment pas encore un
produit équivalent au nirs4all Python :

- `nirs4all-r` est désormais le produit R nommé `nirs4all`. Il possède les
  contrôleurs R, le chemin DAG-ML sur matrices et formats, les prétraitements
  n4m ajustés sur le train et le lecteur JSON/YAML portable. Le profil plat
  exportable entre hôtes reste SNV/SG/PLS ; son RDS n'est pas interlangage ;
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
5. maintenir la propriété du nom R public dans `nirs4all-r`, avec la
   distribution R de Core retirée et ses autres bindings encore présents et
   les fixtures partagées, sans introduire un second paquet R `nirs4allcore`.

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
| Agrégat portable (transition) | `nirs4all-core` | Rust agrège les contrats natifs, verrouille les versions et porte les gates interlangages. Les bindings Python/JS-WASM/MATLAB y restent provisoirement ; le produit R et sa publication n'y résident plus. |
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
| `nirs4all` R 0.4.0.9020 dans `nirs4all-r` | Produit R en développement | Régression n4m/PLS, `lm`, `ranger`, `glmnet`, `parsnip`, `mlr3`, `torch` CPU ; classification `ranger`, `parsnip`, `mlr3` et `torch` CPU avec facteurs et probabilités ; données matrices et formats avec cibles numériques ou catégorielles explicites ; CV/OOF/refit DAG-ML avec groupes, variantes, chaînes de nœuds n4m et concaténation bornée de branches ; recettes JSON/YAML avec huit familles de prétraitement n4m et expansion `_or_`/`_cartesian_` bornée, transfert N4MM format 1 PLS seul / format 2 SNV→SG→SIMPLS. | Pas encore d'Archive V2/V3 interlangage, de tous les nœuds n4m ni de graphe DAG arbitraire. R-universe sert 0.4.0.9018 depuis le dépôt R dédié, mais 0.4.0.9020 attend sa synchronisation. Seul SNV/SG/PLS est qualifié pour l'export de recette interlangage ; les modèles ML/DL R restent des sidecars RDS. |
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
continuum removal, gap derivative, Savitzky–Golay et SNV. MSC reste omis de cet
adaptateur `prospectr` : sa référence moyenne doit être ajustée sur le train de
chaque fold, persistée, puis réutilisée en validation/prédiction. Ce chemin est
désormais couvert séparément par `n4m` et le produit `nirs4all` R ; l'appliquer
indépendamment à chaque batch créerait une divergence et potentiellement une fuite.

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

Historiquement, Core distribuait aussi un paquet R nommé `nirs4all`. Cette
distribution est transférée vers `nirs4all-r` ; la branche de retrait supprime
le binding R et son workflow du Core, mais conserve les fixtures et oracles
communs. Python, JS/WASM et MATLAB restent dans Core pour ce cycle, conformément
au transfert progressif convenu entre agents.

### 4.2 La limite du runner R actuel

L'ancien `nirs4all_run_portable_pipeline()` de Core parse une liste/JSON/YAML
puis réalise en R :

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

La signature de compatibilité est reprise dans `nirs4all-r` et traduit ces
quatre cas vers les prétraitements et le PLS du produit. Étendre cette boucle
manuelle aux branches/HPO serait une erreur : ces décisions doivent passer par
les contrats et l'ordonnanceur natifs de DAG-ML. L'ancien runner Core disparaît
avec le paquet R qu'il hébergeait.

### 4.3 Rôle cible de Core

Core doit rester, à terme, l'agrégat Rust et les contrats portables avant
bindings. Pendant le transfert par étapes, il conserve ses distributions
Python/JS/MATLAB existantes. En particulier, il garde :

- le verrou compatible des versions amont pour ses propres artefacts ;
- le lieu des profils portables et des déclarations honnêtes de capacité ;
- le gate d'intégration et les fixtures JSON/YAML temporaires qui vérifient
  qu'un pipeline réellement exécuté par les amonts reproduit l'oracle.

La façade ergonomique R (`pipeline()`, `run()`, `predict()`, `load()`) appartient
à `nirs4all-r`. Core ne doit pas contenir kernels, split/CV, scoring,
sélection, stockage de prédictions ni adaptateur substantiel de framework.

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

Pour le niveau 2 R, la voie de production proposée est un **pont natif possédé
par `nirs4all-r`** vers les fonctions Rust d'Archive V2/V3 de Core : validation
de l'archive et de l'inventaire *avant* extraction, lecture des descripteurs et
payloads, écriture atomique, puis exécution Methods/DAG-ML sans rappel Python.
L'interface R ne doit exposer que des objets R typés et des erreurs stables ;
elle ne doit pas recopier le validateur ZIP/JSON en R, ni réintroduire un paquet
public `nirs4allcore` concurrent. Un CLI natif peut servir de banc de
conformance provisoire, mais pas de dépendance implicite d'un paquet CRAN.
Le choix `.Call`/C ABI et la distribution des sources Rust vendoriées doivent
être qualifiés séparément sur Linux, macOS et Windows avant soumission CRAN.

L'archive doit distinguer explicitement **PREDICT** (graphe, états ajustés,
N4MM, ordre et identité des features, ABI, empreintes) et **RETRAIN** (recette,
paramètres, seeds et provenance des données, avec nouvel identifiant de run).
Les prédictions d'une archive importée doivent être comparées en processus
neuf dans les deux sens Python↔R ; un simple aller-retour RDS ou N4MM isolé ne
suffit pas. Chaque opérateur n4m sans sérialisation d'état qualifiée doit être
refusé à l'export de niveau 2 plutôt que silencieusement recalculé.
En particulier, le dispatcher R `n4m_method()` renvoie des tableaux
`MethodResult` pour 33 familles de modèles, et non le handle sérialisable
accepté par `n4m_model_export()` ; ces familles ne gagnent donc pas la
portabilité de niveau 2 du seul fait qu'elles sont calculables en R. Leur
qualification exige un codec d'état n4m stable et un test de prédiction
Python↔R par famille, ou une interdiction explicite d'export entraîné.

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
| `parsnip`/tidymodels | Adaptateur de régression R optionnel, `fit_xy`/`predict` avec moteur explicitement choisi ; CV/OOF/refit exécutés par DAG-ML. | Étendre le contrat aux recettes/formulas, classification et sérialisation interlangage ; ne pas lui déléguer CV/HPO. |
| `mlr3` | Contrôleur optionnel de régression `LearnerRegr` présent dans `nirs4all` R ; `regr.rpart` testé en fit/prédiction, CV/OOF/refit/replay et processus neuf. | Cloner le learner vierge à chaque fit ; utiliser les learners, pas les resamplings/tuners lorsque DAG-ML orchestre. Étendre la couverture moteurs et classification séparément. |
| `torch` R | Contrôleurs CPU MLP et module `nn_module` pour régression et classification, avec facteurs/probabilités `N×K`, fit/prédiction, RDS/fresh-process et CV/OOF/refit/replay DAG-ML testés. | Étendre au GPU, autres pertes et architectures ; le builder et les poids restent R-spécifiques, ONNX/Python→R séparés. |
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
des contrôleurs `n4m` PLS, `stats::lm`, `ranger`, `glmnet`, `parsnip`/`mlr3` régression et un MLP CPU
`torch` R. Son modèle n4m est sauvegardé en octets N4MM, tandis que le
module torch utilise son sérialiseur natif dans le bundle RDS. Depuis le
25 septembre, une voie native **à nœud modèle unique** construit les contrats R,
demande à DAG-ML FIT_CV/OOF/REFIT/PREDICT et fait tourner ces familles
de contrôleurs sur les lignes de folds prescrites. Elle accepte désormais une
liste nommée de pipelines complets comme variantes fixes et confie à DAG-ML
leur sélection OOF avant le refit du seul gagnant. Les prétraitements restent
des paramètres internes au nœud modèle. La CV accepte désormais des IDs de
groupe explicites : aucun groupe ne traverse un fold et le même groupe est
attesté dans les données, le FoldSet et les relations DAG-ML. Les graphes à branches, HPO adaptatif et
prédiction sur cohorte externe ne sont pas encore exposés. La demande d'une API
R de bout en bout, de contrôleurs `ranger`/`parsnip`/`mlr3`/`torch` et d'une
cadence CRAN indépendante atteint désormais le seuil où ces composants ne
doivent pas entrer dans Core. `nirs4all-core/bindings/r` a été retiré de
`main` par la PR [#14](https://github.com/GBeurier/nirs4all-core/pull/14),
avec les fixtures communes conservées ; le runner manuel ne doit pas être
généralisé.
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

La solution retenue est de **migrer**, et non dupliquer, le package public
`nirs4all` vers `nirs4all-r`. Le Core cesse de publier un paquet R. Aucun
`nirs4allcore` R séparé n'est prévu : il recréerait la redondance que le
transfert cherche à éliminer.

Publier simultanément deux dépôts avec un package nommé `nirs4all` est exclu.
Le dépôt `nirs4all-r` existe ; après la fusion de la PR #14,
`packages.json` de R-universe pointe maintenant sur sa branche de
développement. Le rebuild externe n'est pas encore une preuve de publication
effective ; les métadonnées CRAN attendent les gates scientifiques et de
dépendances. Le nom public reste exactement `nirs4all`.

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
- étendre les adaptateurs `ranger`/`glmnet`/`parsnip` existants, puis qualifier `xgboost` et les autres moteurs ;
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
| Sélection de variantes R par DAG-ML (`nirs4all-r` `b714fa7`) | Cinq pipelines SNV/SG/PLS de l'exemple Python sont évalués sur les mêmes cinq folds par le DSL natif ; les scores par fold et RMSE OOF sont recoupés avec des fits R séparés, le gagnant et son unique refit sont vérifiés. `R CMD check --as-cran --no-manual` strict, y compris formats et DAG natif : 0 erreur, 1 warning incoming sur version dev et dépendances externes. | La grille et les prétraitements sont contenus dans un seul nœud DAG ; ce n'est ni une sélection imbriquée non biaisée, ni un HPO adaptatif, ni une traduction automatique de pipelines Python. |
| Inférence externe depuis le refit R (`nirs4all-r` `96e6572`) | L'API produit vérifie le SHA-256 du sidecar RDS retenu dans le bundle avant prédiction locale ; tests sur les sept contrôleurs, sur le gagnant PLS et sur une entrée `nirs4all-formats`. Une seconde campagne compare PLS, ridge `n4m` et forêt `ranger` ; le gagnant natif coïncide avec le meilleur RMSE OOF recalculé dans R. Contrôle strict `R CMD check --as-cran --no-manual` : 0 erreur, 1 warning incoming sur version dev et dépendances externes. | Cette inférence ne repasse pas par une phase PREDICT DAG-ML et ne produit pas de score de test. Le sidecar RDS reste un objet R à ne charger que depuis une source de confiance. |
| Prétraitements `n4m` R étendus (branches `feat/r-preprocessing-parity` : Methods `8ccb28af`, produit R `c17e673`) | Bindings C ABI R pour SNV local/robuste, normalisation d'aire et detrend, plus les options complètes de SNV. Les quatre sorties matricielles concordent à `1e-12` avec un oracle Python `n4m` figé ; sauvegarde/rechargement et campagnes DAG natives avec folds et refit passent. `n4m` R 1.0.21.9000 : archive source autoportante avec 238 unités C/C++ vendorizées, installation autonome et `R CMD check --as-cran --no-manual` 0 erreur/0 warning/2 NOTEs (nouvelle soumission, `-march=nocona` injecté par R conda). Le produit R passe ses tests locaux et DAG contre cette installation ; son contrôle strict : 0 erreur, 1 warning incoming attendu. | Les opérateurs avec état entraîné exigent une sérialisation explicite. Le tarball autoportant est construit localement par `N4M_R_VENDOR=1` ; les branches ne sont pas fusionnées ni publiées par R-universe, et Windows/macOS ne sont pas validés. |
| MSC ajusté en R (`nirs4all-methods` `5cbd6744`, `nirs4all-r` `22961f5`) | L'ABI additive 2.7 expose l'extraction/restauration de la référence MSC native. Le binding R conserve seulement ce vecteur ; Python sait également l'extraire et le restaurer. Une fixture Python figée vérifie la transformation R sur un jeu indépendant. Le pipeline R ajuste MSC sur le train de chaque fold, la conserve dans le RDS du refit et la réutilise pour OOF, replay et inférence. Un cas croisé `nirs4all-formats` → MSC/PLS → DAG-ML passe. L'archive `n4m` vendorizée repasse `R CMD check --as-cran --no-manual` avec 0 erreur/0 warning/2 NOTEs ; le produit R repasse avec 0 erreur/1 warning incoming attendu, tests formats et DAG stricts activés. | Le vecteur de référence est portable entre bindings `n4m`, mais le bundle RDS du contrôleur ne l'est pas ; aucun pipeline Python complet n'est automatiquement traduit. Les autres opérateurs à état restent à couvrir. Linux/R 4.6.0 seulement ; versions développées sur des branches, non diffusées par R-universe/CRAN. |
| EMSC ajusté en R (`nirs4all-methods` `0d059c4d`, `nirs4all-r` `3a4a7c0` ; Methods ABI 2.8, `n4m` R 1.0.21.9001, `nirs4all` R 0.4.0.9001) | La référence moyenne native et le degré polynomial suffisent à restaurer l'état EMSC ; aucun spectre de validation n'est utilisé pour l'ajustement. Tests C ABI, aller-retour Python exact et fixture R contre les valeurs Python figées passent. Le pipeline R vérifie l'état dans le RDS, puis la CV/OOF, le refit, le rejeu et l'inférence avec contrôleurs PLS et autres ; le chemin `nirs4all-formats` → EMSC/PLS → DAG-ML est testé. Suite Python Methods : 583 réussis, 3 ignorés ; archives `n4m` et produit R sous `R CMD check --as-cran --no-manual` : respectivement 0 erreur/0 warning/2 NOTEs et 0 erreur/1 warning incoming. | Degré et ordre des features doivent rester identiques entre fit et prédiction. Le bundle contrôleur RDS est spécifique à R ; seul l'état numérique `n4m` est portable. Autres prétraitements avec état non couverts, pas de traduction de pipeline Python/R ni de validation multi-OS. Branches non fusionnées ; `nirs4all-core` occupe encore le nom public R, donc pas de publication R-universe du produit à ce stade. |
| N4MM format 2 embarqué (`n4m` R 1.0.21.9002, `nirs4all` R 0.4.0.9005) | Le binding R ajuste le profil natif SNV→SG→SIMPLS sur X brut, exporte et inspecte le modèle N4MM format 2 avec le décodeur n4m. Le produit R exporte les octets et importe le modèle depuis une recette `n4m.*` JSON/YAML ou un pipeline R, en vérifiant le profil, ses paramètres, la largeur, le solveur et les composantes. En processus Python neuf, R-fit→Python-predict et Python-fit via ABI n4m→R-predict sont identiques à `1e-10`. `R CMD check --as-cran --no-manual` Linux/R 4.6.0 : paquet n4m autonome 0 erreur/0 warning/2 NOTEs ; produit R 0 erreur/1 warning incoming, tous les tests et `Suggests` présents. | Le Python fit de conformance utilise l'ABI n4m directement, pas encore l'orchestrateur complet `nirs4all` Python. Seul ce profil ordonné est N4MM-embarqué ; pas d'Archive V2/V3 DAG-ML, d'identités/lineage dans le payload ni de reprise automatique d'entraînement. Windows/macOS et publication R-universe non qualifiés. |
| N4MM format 1 PLS seul (`nirs4all` R 0.4.0.9007) | Le produit R exporte/importe le PLS sans étape de prétraitement en N4MM format 1 ; le décodeur n4m valide les octets, dimensions, composantes, algorithme et solveur. Deux processus Python distincts prouvent R-fit→Python-predict et Python-fit via ABI n4m→R-predict à `1e-10`. Une recette SNV→SG est refusée pour ce payload, et réciproquement. Contrôle du tarball Linux/R 4.6.0 : 0 erreur, 1 warning incoming, tous les tests et `Suggests` présents. | Le format 1 n'expose pas les drapeaux de centrage/scaling au descripteur : leur exactitude dans une recette importée reste une assertion du fournisseur. Le modèle n'inclut ni recette, ni identités, ni lineage DAG-ML ; pas de preuve de réentraînement automatique interlangage. |
| CV groupée native (`nirs4all` R 0.4.0.9008) | L'API R accepte des IDs de groupe alignés, répartit des groupes entiers entre folds de façon déterministe et transmet les groupes aux relations, à l'enveloppe et au FoldSet validés par DAG-ML. Test avec quatre groupes de tailles inégales : aucun chevauchement train/validation, OOF égale aux fits R exclusifs du groupe à `1e-10`, et mêmes prédictions à `1e-10` avec un fit Python n4m indépendant sur chaque fold ; refit/rejeu et refus d'un sidecar d'IDs altéré. Tarball Linux/R 4.6.0 : 0 erreur, 1 warning incoming. | Un seul niveau de CV groupée à nœud modèle unique ; pas de répétitions, groupes multiples, nested CV ni parité garantie de l'assignation de folds avec `sklearn.GroupKFold`. |
| `parsnip` régression (`nirs4all` R 0.4.0.9006) | Spécification moteur R conservée dans un sidecar RDS avec empreinte SHA-256 validée par l'adaptateur processus ; fit/prédiction et RDS passent avec `parsnip` 1.6.0 sur les moteurs `lm` et `ranger` à graine fixe. CV/OOF/refit/rejeu DAG-ML et sélection inter-familles passent avec `lm`. `R CMD check --as-cran --no-manual` : 0 erreur, 1 warning incoming, tous les tests et `Suggests` présents. | Seule la régression numérique mono-cible est exposée ; ce sidecar reste R-spécifique et ne rend pas les modèles `parsnip` portables. La couverture DAG multi-moteurs et multi-OS est à faire. |
| `mlr3` régression (`nirs4all` R 0.4.0.9009) | Learner `LearnerRegr` vierge cloné profondément par fit, sans partager son état entre folds. `regr.rpart` a été comparé à un ajustement `mlr3` indépendant ; fit/prédiction, RDS, rejeu en processus neuf et campagne DAG-ML stricte CV/OOF/refit/replay passent. L'adaptateur valide l'empreinte SHA-256 du learner sérialisé et son identifiant avant exécution. `R CMD check --as-cran --no-manual` Linux/R 4.6.0 : 0 erreur, 1 warning incoming, tous les `Suggests` présents. | Régression numérique mono-cible seulement ; un seul moteur `mlr3` testé, sans promesse de couverture des extensions `mlr3`. Le RDS n'est pas interlangage. |
| Générateur `_or_`/`_cartesian_` natif borné (`nirs4all` R 0.4.0.9010) | La recette JSON/YAML de deux étages SNV/SG et d'une plage PLS est développée en huit pipelines nommés. L'ordre et les paramètres sont comparés directement au générateur `expand_spec()` de `nirs4all` Python ; chaque fit/prédiction R est recoupé avec un pipeline construit indépendamment. DAG-ML exécute la CV des huit variantes ; le gagnant OOF et son refit sont recoupés avec des fits indépendants par fold. Tarball Linux/R 4.6.0, tests Python et DAG stricts : 0 erreur, 1 warning CRAN-incoming. | Ne couvre que les étapes SNV/SG et le modèle PLS déjà reconnus, sans modificateurs de générateur, branches arbitraires ni réentraînement interlangage d'un bundle. |
| Lecture de six prétraitements n4m supplémentaires (`nirs4all` R 0.4.0.9011) | Les recettes JSON/YAML `n4m.LSNV`, `n4m.RNV`, `n4m.AreaNormalization`, `n4m.Detrend`, `n4m.MSC`, `n4m.EMSC` sont reconstruites en étapes Methods R. Pour chaque cas, les matrices train/validation coïncident à `1e-10` avec un processus Python n4m indépendant ; fit/prédiction PLS et parcours avec holdout Kennard–Stone concordent avec des pipelines R construits séparément. Tarball Linux/R 4.6.0 avec tests Python/DAG stricts : 0 erreur, 1 warning CRAN-incoming. | Les autres bindings Core/JS/WASM ne résolvent pas encore ces identifiants ; l'export R reste volontairement limité à SNV/SG/PLS. Pas d'état MSC/EMSC encapsulé en N4MM portable ni de preuve de réentraînement interlangage. |
| Module `torch` R personnalisable (`nirs4all` R 0.4.0.9012) | Un builder R crée une nouvelle architecture `nn_module` par fit ; le modèle est vérifié `N×p → N×1`, entraîné en CPU avec Adam/MSE, puis sérialisé par torch dans un bundle RDS. Test d'architecture à deux couches cachées, contrôle de déterminisme/états distincts, chargement dans un processus R neuf et CV/OOF/refit/replay DAG-ML stricts sur une architecture personnalisée. Le builder sérialisé dans le sidecar DAG a une empreinte SHA-256 validée. Tarball Linux/R 4.6.0 : 0 erreur, 1 warning CRAN-incoming. | Régression numérique mono-cible, CPU et entraînement full-batch Adam/MSE seulement ; code/poids R-spécifiques, pas de conversion de poids PyTorch ni ONNX, ni de garantie multi-OS/GPU. |
| Chaîne de vrais nœuds DAG de prétraitement (`nirs4all` R 0.4.0.9013) | L'option `split_steps = TRUE` abaisse chaque étape n4m en nœud `transform` séparé avant le modèle. Les adaptateurs R échangent les matrices train/validation avec identités de fold et de producteur vérifiées ; MSC et EMSC ajustent leur référence exclusivement sur le train du fold. Les états de refit sont des artefacts distincts, vérifiés par SHA-256 lors de l'inférence externe. Sur deux chaînes de deux étapes avec deux workers, OOF, replay et inférence concordent à `1e-10` avec les fits R indépendants ; l'altération d'un artefact est refusée. Le tarball Linux/R 4.6.0 passe `R CMD check --as-cran --no-manual` avec 0 erreur et 1 warning incoming, tests stricts DAG/Formats/Python activés. | Ce transport matriciel est un sidecar RDS local, non un format interlangage. Les variantes de graphes transform séparés, branches, concat, HPO adaptatif et prédiction externe par une nouvelle phase DAG-ML ne sont pas exposés. Windows/macOS et publication R-universe restent à qualifier. |
| Variantes sur graphe de transformations (`nirs4all` R 0.4.0.9014) | Une dimension DAG-ML applique désormais des `param_overrides` au modèle **et** à chaque transformation n4m. À topologie identique, les variantes peuvent choisir d'autres étapes (SNV ou MSC), paramètres et contrôleurs. Le score de chaque fold, le gagnant, le refit et l'inférence sont confrontés aux fits R indépendants. Les cinq candidats SNV→SG→PLS de l'exemple Python gardent le même gagnant et les mêmes prédictions en mode graphe ; le rejet des variantes de nombres d'étapes différents est explicite. Le tarball Linux/R 4.6.0 passe `R CMD check --as-cran --no-manual` avec 0 erreur et 1 warning incoming, tests stricts DAG/Formats/Python activés. | Topologie fixe seulement : pas de branchement ou de nombre d'étapes variant par candidat, ni d'état entraîné portable hors R. Windows/macOS et publication R-universe restent à qualifier. |
| Branches de features parallèles et concaténation (`nirs4all` R 0.4.0.9015) | `nirs4all_concat()` compose des branches n4m indépendantes en local ; l'option `split_steps = TRUE` abaisse un unique bloc concat en nœuds `transform` parallèles, un `feature_join`, puis le modèle. Deux branches à deux étapes (SNV→SG et MSC→detrend) exécutent CV/OOF/refit/replay dans DAG-ML ; les OOF et l'inférence externe coïncident à `1e-10` avec des fits R par fold, et les matrices train/validation avec un processus Python n4m indépendant. Un état de branche altéré est refusé par le contrôle SHA-256 des artefacts. Le tarball Linux/R 4.6.0 passe `R CMD check --as-cran --no-manual` avec 0 erreur et 1 warning incoming, tests stricts DAG/Formats/Python activés. | Ce n'est pas encore le graphe Python complet : une seule concat de features, pas de stacking de prédictions, de branches mixtes ou de variantes sur ce bloc. Les sidecars RDS et l'état de concat ne sont pas portables vers Python/WASM. Windows/macOS et publication R-universe restent à qualifier. |
| Recette Python `branch` → `merge: features` (`nirs4all` R 0.4.0.9016) | Le lecteur JSON/YAML R reconnaît les branches nommées, anonymes ou `name`/`steps`, reconstruit `nirs4all_concat()` et peut exécuter la recette avec PLS, y compris après Kennard–Stone. L'export R d'une concat SNV/SG par défaut émet cette syntaxe ; l'analyseur de topologie du produit Python y détecte un unique modèle et une feature-merge, et le réimport R conserve les prédictions. La recette MSC peut être importée mais son export est refusé. Le tarball Linux/R 4.6.0 passe `R CMD check --as-cran --no-manual` avec 0 erreur et 1 warning incoming, tests stricts DAG/Formats/Python activés. | Le test Python porte sur la topologie, non l'exécution complète du pipeline Python ; les matrices numériques sont déjà comparées séparément à Python n4m. Core/JS/WASM n'ont pas encore été qualifiés sur cette nouvelle forme, qui n'est donc pas annoncée comme portable tous langages. Windows/macOS et publication R-universe restent à qualifier. |
| Classification `ranger` (`nirs4all` R 0.4.0.9017) | Contrôleur forêt probabiliste à classes factorielles stables, sauvegarde/rechargement RDS, probabilités `N×K` validées et nœuds DAG-ML de prétraitement n4m séparés. La cible est codée en indices numériques attestés dans l'empreinte de schéma ; l'adaptateur émet probabilités et classes par fold pour la sélection OOF sur l'accuracy, puis restaure les étiquettes en inférence R. Le test Iris confronte les OOF aux fits R indépendants et rejette les plis d'entraînement incomplets. Le tarball exact passe `R CMD check --as-cran --no-manual` sous Linux/R 4.6.0, tests stricts Python/DAG/Formats activés : 0 erreur, 1 avertissement CRAN-incoming sur version de développement et dépendances hors CRAN. | Les poids de forêt sont R-spécifiques ; ni conversion sklearn↔ranger ni artefact entraîné interlangage. Pas de classification `parsnip`/`mlr3`/`torch`, pas de validation multi-OS. R-universe sert encore l'ancien paquet Core 0.3.31 au moment du test : le rebuild externe n'est pas qualifié. |
| Classification `parsnip`/`mlr3` (`nirs4all` R 0.4.0.9018) | Adaptateurs aux moteurs de classification explicites : `parsnip`/`rpart` et `mlr3`/`classif.rpart` testés. Probabilités et classes gardent l'ordre des niveaux ; le learner `mlr3` vierge est cloné par fold. Fit/prédiction, sauvegarde RDS, OOF contre fits R indépendants, refit, inférence externe et sélection entre familles par accuracy passent dans DAG-ML. Tarball exact sous Linux/R 4.6.0 avec tests stricts : 0 erreur, 1 avertissement CRAN-incoming. | Couverture limitée à ces moteurs et au CPU ; sidecars RDS spécifiques à R. Classification torch, conversion interlangage et multi-OS non qualifiés. |
| Cibles catégorielles `nirs4all-formats` (`nirs4all` R 0.4.0.9019) | Le lecteur Rust décode un CSV réel dont les labels textuels sont en métadonnées par enregistrement ; le convertisseur R accepte ce champ demandé explicitement, ainsi qu'une colonne de targets textuelles complète, et conserve les IDs/axes. Les labels absents, vides ou de types mixtes sont refusés. Fit classifieur, CV/OOF/refit et inférence sur une nouvelle vue formats passent ; le tarball exact sous Linux/R 4.6.0, tests Python/DAG/Formats stricts activés : 0 erreur, 1 avertissement CRAN-incoming. | Uniquement des signaux spectraux 1D homogènes ; les métadonnées ne sont pas inférées automatiquement comme cibles. La classification complète des autres formats et la publication R-universe 0.4.0.9019 demandent qualification séparée. |
| Classification `torch` CPU (`nirs4all` R 0.4.0.9020) | MLP et module `nn_module` personnalisé à logits `N×K`, perte cross-entropy avec indices R `1..K`, facteurs et softmax `N×K`. Les tests Iris comparent les OOF de chaque fold DAG-ML à des fits R indépendants, vérifient le refit/inférence, le rejet d'une sortie de mauvaise dimension, la répétabilité et la recharge du module dans un processus R neuf. Le test ciblé passe avec DAG strict ; le tarball exact 0.4.0.9020 (SHA-256 `3f24dedf21857dba5a6c9883e57f4fe785019af617e4e8597662eca4b8b30052`) passe `R CMD check --as-cran --no-manual` Linux/R 4.6.0 avec `torch` CPU et `dagml` publics installés : 0 erreur, 1 warning CRAN-incoming. | Les autres `Suggests` n'étaient pas tous installés dans ce contrôle ; pas de validation multi-OS du tarball exact. Builder/poids R-spécifiques, ni conversion PyTorch→R ni ONNX. |
| `dag-ml-cli` + opérateur R Ridge | Test `r_hpo_ridge` réussi : folds, HPO parallèle, reprise, sidecar et replay. | Pas un contrôleur `ranger`, `torch` ou Methods général. |
| `R CMD check` de `nirs4all` Core | 0 erreur, 0 warning, 0 NOTE après retrait de la notice `LICENSE` redondante, avec `Suggests` manquants non forcés. | Check CRAN complet avec toutes dépendances et plateformes non effectué. |

Les lignes ci-dessus sont des jalons historiques : leurs mentions de Core
comme propriétaire R ou de `dagml` absent de R-universe décrivent leur date de
test, pas l'état après le transfert. Au 25 septembre 2026, la PR Core
[#14](https://github.com/GBeurier/nirs4all-core/pull/14) est fusionnée après
neuf contrôles verts ; le paquet R `0.4.0.9012` a passé localement les tests
stricts Formats/DAG et `R CMD check --as-cran --no-manual` (0 erreur, 1 warning
incoming). Le contrôle a d'abord été lancé avec deux `Suggests` absents,
puis répété avec `dagmldata` et `nirs4alldatasets` installés depuis R-universe,
sans désactiver la vérification des `Suggests`. Les accès amont et le corpus
d'exécution JSON/YAML partagé sont repris dans `nirs4all-r`.
Les PR [Methods #26](https://github.com/GBeurier/nirs4all-methods/pull/26)
(44 contrôles applicables verts) et [produit R #1](https://github.com/GBeurier/nirs4all-r/pull/1)
sont fusionnées ; le registre R-universe suit maintenant leurs branches `main`.
L'export R de recettes porte désormais des alias natifs `n4m.*` plutôt que
des noms de classes sklearn ; la PR Core fusionnée
[#15](https://github.com/GBeurier/nirs4all-core/pull/15) les ajoute aux lecteurs
Python, Rust, JS/WASM et MATLAB. Ce test ne prouve pas la reprise de l'état
entraîné ni la compréhension de ces alias par le produit Python complet.

La publication du produit `nirs4all` R n'est donc **pas encore qualifiée de
fonctionnellement complète**. En outre,
le tarball `dagml` ne fournit ni le CLI ni la bibliothèque C ABI qu'exercent
ses tests R : le check vert ci-dessus les prend dans le build Rust local. Il
faut un mode d'installation autoportant ou une dépendance système explicitement
distribuée avant d'annoncer un package R-universe/CRAN installable hors de ce
workspace. Le dépôt R-universe pointe désormais sur `GBeurier/nirs4all-r`
`main` et inclut `dagml`. Le rebuild externe de `nirs4all` 0.4.0.9018 a
réussi sur Linux, Windows, macOS et WASM ; ses tarballs sources publics et
`n4m` 1.0.21.9002 se sont installés dans une bibliothèque R vierge, puis un
pipeline SNV→SG→PLS y a ajusté et prédit. Cela démontre l'installation
publique de **cette version**, pas la publication de 0.4.0.9020 ni la parité
complète des niveaux 1–2. Le tarball `dagml` 0.3.27 public s'installe et a
servi au test du classifieur `torch`, mais le CLI provenait encore du build
local ; le correctif de test R-universe de `dagml` fusionné doit encore être
repris par une synchronisation de ce dépôt.

Pour CRAN, un tarball source est soumis **par package**, dans l'ordre des
dépendances ; une archive composite de tous les tarballs n'est qu'un kit de
transmission, pas un package CRAN soumettable. Le kit de release doit inclure
les tarballs source autoportants, leurs SHA-256, les versions/commits, les
licences, le journal `R CMD check --as-cran` par plateforme, les résultats de
parité, les commentaires au mainteneur et la déclaration des dépendances non
présentes sur CRAN. Aucun texte ne doit prétendre à un check vert non exécuté.
`n4m` reste une dépendance forte hors CRAN/Bioconductor ; la
[politique CRAN](https://cran.r-project.org/web/packages/policies.html) impose
de résoudre ce point avant de soumettre `nirs4all`. Les textes définitifs des
formulaires devront être préparés sur les tarballs finaux, dans l'ordre de
soumission de leurs dépendances.

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
