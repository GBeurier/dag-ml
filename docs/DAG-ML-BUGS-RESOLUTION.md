# Validation et corrections de l’audit DAG-ML — 2026-10-05

**État après correction du second passage : les cinq défauts reproduits R1–R5 sont corrigés**, détaillés dans le [réaudit ciblé](DAG-ML-BUGS.md#réaudit-ciblé-du-2026-10-05--r1r5) et les [gates complémentaires](#corrections-complémentaires--r1r5). Quatre complètent les correctifs A2-06/A2-13/A2-14/A2-15 ; R4 concerne le helper multimodal ajouté ensuite. Le bilan chiffré et la « Validation finale » initiale restent historiques ; la qualification commune des interfaces est pilotée dans l'autre tâche.

Les 117 signalements de [DAG-ML-BUGS.md](DAG-ML-BUGS.md) ont été examinés contre le code, les contrats et les tests. Le tableau conserve les réserves des signalements composites : une correction porte sur le défaut validé, et un comportement intentionnel reste conservé. Les cinq faux positifs complets sont A1-06, A1-12, A5-03, A6-05 et A7-07.

Les corrections et la validation finale du passage initial sont terminées. Répartition initiale : 91 points corrigés, 18 corrigés avec réserve sur un comportement intentionnel, 2 corrigés avec compatibilité historique, 1 clarifié par contrat et 5 faux positifs. Les limites des anciens bundles sont explicites en A4-08/A4-11 et ci-dessous. Les manifests/checksums W1/D4 ont été rafraîchis pour les octets des sources corrigées, sans changer les cibles numériques des oracles. Le SHA du header ABI change pour sa documentation, sans changement de layout ni de signature C.

## Conclusions point par point

| ID | Conclusion | Justification et correction |
| --- | --- | --- |
| A1-01 | Corrigé | Les inner folds refusent l'effacement de groupes et valident aussi les relations/origines du scope imbriqué. `fold.rs`, `runtime/scheduler.rs`; reproduction groupée et suites de stacking. |
| A1-02 | Corrigé | Ranking et poids filtrent variante, folds déclarés, niveau et port ; la moyenne remplace le premier score rencontré. `runtime/oof.rs`. |
| A1-03 | Corrigé avec réserve | Une augmentation sans origine est refusée par la politique sûre. Une origine égale au sample reste légitime pour des observations augmentées ; l'interdiction proposée serait un faux positif. Reproduction des deux cas. |
| A1-04 | Corrigé | La recherche CapacityKFold est bornée par les univers d'entraînement utilisables. `runtime/stacking.rs`. |
| A1-05 | Corrigé | Les joints, merges et entrées de prédictions appellent la validation du contenu, incluant les valeurs non finies. |
| A1-06 | Faux positif | `select_candidate` classe des scores déjà attestés par l'appelant ; CandidateScore ne transporte pas leur provenance. Le training natif impose séparément l'évidence OOF. Inventer cette provenance dans le sélecteur ne fournirait aucune garantie. |
| A1-07 | Corrigé avec réserve | Fit sur FoldValidation/Predict est refusé. Une augmentation de features appliquée indépendamment à chaque ligne peut couvrir toutes les partitions sans fit sur le holdout ; ce cas reste autorisé. |
| A1-08 | Corrigé | Un couple absent d'un fold resamplé est ignoré ; les couples présents de part et d'autre restent interdits. Reproduction Resampled. |
| A1-09 | Corrigé | L'assertion de couverture FullTrain est réservée au mode Partition. Le mode Resampled garde son univers complet. |
| A1-10 | Corrigé | Le bloc et ses masques sont validés avant tout accès indexé. `runtime/oof.rs`. |
| A1-11 | Corrigé avec réserve | Les scores d'erreur négatifs sont refusés. Le choix du premier fold sans scores est le fallback déterministe du contrat existant ; il est conservé. |
| A1-12 | Faux positif | Ces sélecteurs prennent les meilleurs candidats (producteur, fold), puis projettent leurs producteurs. Ils ne promettent pas k producteurs distincts ; modifier cette sémantique changerait la sélection. |
| A1-13 | Corrigé | Les branches doivent fournir le même y_true pour un sample commun ; un conflit est refusé avant remplacement. `runtime/merge.rs`. |
| A1-14 | Corrigé | Les longueurs sample_ids/values sont vérifiées avant zip. `runtime/oof.rs`. |
| A2-01 | Corrigé avec réserve | Cardinalité vérifiée avant allocation ; plafond pré-pruning 10 000 et arithmétique checked. L'absence de max_variants ne devient pas arbitrairement une limite à un candidat. |
| A2-02 | Corrigé | Les tailles pick/arrange sont bornées avant expansion. `dsl/generation.rs`. |
| A2-03 | Corrigé | Log ranges, samples, grids, combinaisons et permutations ont des limites contrôlées, y compris avant pruning avec contraintes ; nombre de dimensions borné. |
| A2-04 | Corrigé | Le sérialiseur trie récursivement les maps et garde l'ordre déclaré des structs, préservant les empreintes natives historiques. Les membres JSON d'archive V2/V3 l'utilisent aussi : leur fermeture exacte est indépendante de preserve_order. Tests de clés inversées et reconstruction d'une archive réelle avec/sans cette feature. |
| A2-05 | Corrigé | Les clés des objets compat sont triées explicitement avant de produire indices, branches et grilles. |
| A2-06 | Corrigé, complété par R1 | Plan/folds/campaign, bindings et shapes sont recoupés. Les variantes ordinaires, Methods HPO et host HPO re-dérivent désormais identité, seed et empreinte ; un préfixe inconnu n'exempte aucun contrôle. L'objectif host HPO est transporté dans le choix signé. |
| A2-07 | Corrigé avec réserve | Le merge de features suivi d'un modèle garde le chemin normal ; la fusion de stacking est réservée aux modes qui consomment effectivement des prédictions. Les modes mixed/all restent légitimes. |
| A2-08 | Corrigé | Les valeurs intégrales des ranges sont des entiers JSON dans les paramètres transmis aux modèles. Reproduction qui vérifie is_i64. |
| A2-09 | Corrigé | Les bornes initiale/finale déclarées sont recopiées exactement. Reproduction log range. |
| A2-10 | Corrigé | Le look-ahead travaille sur une copie et ne commit ses effets que si la fusion réussit. Reproduction du split dupliqué. |
| A2-11 | Corrigé | Les métadonnées internes de branche/générateur priment sur le contexte extérieur. |
| A2-12 | Corrigé avec réserve | Edges identiques et fan-in multiple sur One/Optional sont refusés. Un input sans edge peut être alimenté par un provider externe et reste valide. |
| A2-13 | Corrigé avec réserve, complété par R5 | Noms/kinds/représentations et fan-in concret recoupés avec les manifests. Deux edges sont refusés pour One/Optional ; Many avec un edge reste compatible avec One. Les prototypes génériques, aliases prediction et lanes Methods/OOF restent légitimes. |
| A2-14 | Corrigé avec réserve, complété par R2 | Structures imbriquées et modes ajoutés au label ; opérateurs plats inchangés. Seuls les IDs des étapes DSL typées sont retirés : les IDs sémantiques des opérateurs, paramètres, metadata et selectors sont préservés. |
| A2-15 | Corrigé, complété par R3 | IDs tronqués ou modifiés par sanitization désambiguïsés, suffixes digest réservés également encodés. Deux générateurs g:a/g_a et un nom littéral imitant le digest compilent ensemble ; références de contraintes exactes conservées. |
| A2-16 | Corrigé | Les conversions u64 vers usize sont checked ; les tailles hors capacité sont refusées, y compris wasm32. |
| A2-17 | Corrigé avec réserve | La forme exacte {label,value} est la syntaxe Labeled documentée. deny_unknown_fields évite de perdre silencieusement les autres clés d'un objet. Test des deux formes. |
| A2-18 | Corrigé avec réserve | Les tailles dupliquées sont refusées. Deux valeurs identiques avec des labels ou métadonnées distincts peuvent représenter des essais distincts ; elles ne sont pas fusionnées arbitrairement. |
| A3-01 | Corrigé | Tous les filtres HPO/checkpoint utilisent le niveau et la grouping_key signés. Les rapports Sample et Group ne sont plus confondus. |
| A3-02 | Corrigé | Parallel refuse explicitement FIT_CV/REFIT imbriqués non pris en charge ; Sequential sans provider les refuse aussi avant invocation. |
| A3-03 | Corrigé | Les scores importés du bundle gardent la priorité pour le stacking en replay, même après l'émission d'un score local. |
| A3-04 | Corrigé | Les ensembles Test filtrent blocs, cibles et scores sur les folds de reporting extérieurs. |
| A3-05 | Corrigé | Les chemins canoniques sont comparés avant la copie pour éviter de tronquer le fichier source lui-même. |
| A3-06 | Corrigé avec réserve | Les contrôleurs FIT_CV sont vérifiés avant ask ; les erreurs OOF/plan/contrat et de rescoring se propagent. RuntimeValidation reste le protocole d'échec numérique de candidat ; aucune classification fiable I/O/modèle n'existe dans ce type générique. Si tous échouent, la cause initiale est conservée dans l'erreur finale. Tests Methods avec candidats invalides conservés. |
| A3-07 | Corrigé | Un sampler épuisé peut terminer sous le budget ; la cohérence du nombre de propositions/historique reste contrôlée. |
| A3-08 | Corrigé | Les entrées lineage sont comparées comme ensembles, puis rangées canoniquement. |
| A3-09 | Corrigé | Les doublons lineage n'écrasent plus l'original ; la capture d'artefacts est staged avant publication. Reproduction du doublon. |
| A3-10 | Corrigé | Le contexte doit être spécifique à une variante pour dépendances OOF/scoring groupé ; les collecteurs refusent aussi les ensembles multivariantes. L'API ordinaire sans dépendance reste compatible. |
| A3-11 | Corrigé | Le scheduler Parallel avec artifact store configure l'agrégation globale OOF en FIT_CV. |
| A3-12 | Corrigé | Les propositions ne changent que les paramètres déclarés de la cible ; métrique et direction sont recoupées avec l'étude. |
| A3-13 | Corrigé | L'override attendu est vérifié par slice pattern et node_id avant mutation, sans index non gardé. |
| A3-14 | Corrigé | Les wrappers délèguent les capacités facultatives des providers, y compris generated views, cohortes et refit IDs. |
| A3-15 | Corrigé | Les empreintes des contrats acceptent uniquement le SHA-256 hexadécimal minuscule. |
| A3-16 | Corrigé | Les cibles agrégées sont appariées par niveau, unités et noms de cibles compatibles ; les candidats contradictoires sont refusés. |
| A3-17 | Corrigé | Les noms drive-prefixed, contrôles et séparateurs sont refusés ; les chemins lus doivent rester dans la racine canonique du cache. |
| A3-18 | Corrigé | Le résultat matérialisé est capturé immédiatement ; un hook release_materialized couvre les sorties en erreur. |
| A3-19 | Corrigé avec réserve | Le parent doit appartenir au même run/node/input/phase/binding. Les handles de parent peuvent être réutilisés entre folds ; les relations d'enveloppe historiques ne prétendent pas toutes décrire un univers exhaustif. |
| A4-01 | Corrigé | V3 recoupe projection, topologie, folds, manifests, influence et closure des prédicteurs, puis valide sa projection vers le bundle runtime ; les artefacts stateful sont obligatoires. |
| A4-02 | Corrigé | PROV et RO-Crate sont reconstruits sémantiquement depuis les entrées validées avant comparaison. Des checksums auto-déclarés ne suffisent plus. |
| A4-03 | Corrigé | Rapports de rerun/rescoring comparés à 1e-12 absolu + 1e-12 relatif, identités exactes. Tests d'arrondi, altération et non-finitude. |
| A4-04 | Corrigé | Une erreur d'exécution et une erreur de libération hydratée sont toutes deux rapportées. |
| A4-05 | Corrigé | N4ME suit les limites de longueur, segments et séparateurs des autres membres Methods. |
| A4-06 | Corrigé | Facets avec _producer ; eventTime contrôlé comme RFC3339 avec calendrier ; runId dérivé aussi des runs exécutés. |
| A4-07 | Corrigé | EXPLAIN V3 valide les bindings demandés et refuse les outputs hors demande. |
| A4-08 | Corrigé avec compatibilité | Les namespaces générés sont re-dérivés avec le seed d'exécution enregistré dans le bundle. Les anciens bundles sans ce seed ne fournissent pas les données nécessaires à cette preuve ; leurs contrôles historiques d'identité, payload et empreinte restent appliqués. Cette limitation est explicite. |
| A4-09 | Corrigé | Toute évidence replay utilisée pour calibration doit attester target_content_fingerprint. |
| A4-10 | Corrigé avec réserve | Le replay REFIT recoupe chaque edge requires_oof avec ses requirements. Un package déjà refité destiné à PREDICT peut légitimement ne plus transporter ses caches d'entraînement. |
| A4-11 | Corrigé avec compatibilité | Les contextes replay utilisent le root_seed de campagne. Les anciens contrats qui ne stockent pas de limites de ressources ne permettent pas de reconstruire des limites absentes. |
| A4-12 | Corrigé | Les clés PROV composites ont une séparation non ambiguë ; seuls les producteurs REFIT attestent la génération des modèles consommés ensuite. |
| A4-13 | Corrigé | Le parseur des payloads limite les nœuds à 4 000 000 pendant parsing. L'export de bundle PROV évite l'indentation par octet ; les formats JSON historiques sont conservés. |
| A5-01 | Corrigé | Les cibles de classification explicites sans probabilités utilisent un vote ; Methods atteste aussi les probabilités Validation. Reproduction classes {0,2} et oracle natif des probabilités. |
| A5-02 | Corrigé | La direction HPO doit correspondre à l'objectif de la métrique avant toute proposition. |
| A5-03 | Faux positif | RobustBest est explicitement l'alias historique de Best dans le contrat et ses tests. Lui attribuer un nouvel estimateur changerait les résultats et les checkpoints existants. |
| A5-04 | Corrigé | SortedTuple.length est borné à 4096 avant construction des noms. Reproduction avec i32::MAX. |
| A5-05 | Corrigé | Les encoders multimodaux doivent être des objets avant IndexMut ; les recettes malformées retournent une erreur. |
| A5-06 | Corrigé | Recipe/model/params classifier sont contrôlés comme objets avant mutation. |
| A5-07 | Corrigé | Bornes log positives, doublons de catégories/ordinales et valeurs non finies contrôlés sans charger l'optimizer natif. |
| A5-08 | Corrigé | Sommes de budgets, budget positif et phase_index utilisent une arithmétique checked, y compris les workers. |
| A5-09 | Corrigé | Les intermédiaires du pruning parallèle sont traités dans un ordre déterministe de trial/step ; les workers sont joints. |
| A5-10 | Corrigé avec réserve | Le plancher positif évite une masse nulle en maximisation. La forte pondération d'un score d'erreur proche de zéro en minimisation est la convention inverse-score existante ; elle est conservée. |
| A5-11 | Corrigé | Les features Methods sont libérées quand le dernier lease du RunContext disparaît. Test de clones et isolation entre runs. |
| A5-12 | Corrigé | L'espace v1 est contrôlé avant ask ; une proposition native rejetée après ask est terminalisée Failed. |
| A5-13 | Corrigé | Les IDs HPO gardent leur forme compatible ; les égalités se départagent par numéro de trial. Test trials 2/10 et IDs ordinaires. |
| A5-14 | Corrigé | Calibration/apply partagent le contrat de noms facultatifs pour les blocs non nommés. |
| A5-15 | Corrigé | Accuracy, balanced accuracy et F1 comparent les labels selon la même convention d'arrondi. |
| A6-01 | Corrigé | Replay et initial-refit refusent les vtables owning avant toute adoption ; les APIs training gardent leur escrow transactionnel. |
| A6-02 | Corrigé | Le contrat borrowed-only des mêmes APIs est explicite ; un refus laisse intégralement la propriété à l'appelant. |
| A6-03 | Corrigé | Les frontières C à code de statut capturent les panics Rust et retournent PANIC. Test direct de la frontière et suite C/ownership. |
| A6-04 | Corrigé | dagml_version est dérivé de la version Cargo. |
| A6-05 | Faux positif | Le résultat owning conserve les handles pour ses replays attachés et les libère au detach/free. Une libération anticipée arbitraire casserait les aliases et le contrat de possession ; les features internes Methods ont, séparément, été corrigées en A5-11. |
| A6-06 | Corrigé | Un handle déjà suivi n'est pas relâché une seconde fois sur rejet de résultat. |
| A6-07 | Corrigé | len=0 signifie input facultatif omis ; NULL avec longueur positive est invalide. Documentation et parseurs harmonisés. |
| A6-08 | Corrigé par contrat | Les buffers sont toujours libérés ; les handles/out states ne deviennent valides que sur OK. Sur échec, le host garde la responsabilité de ses allocations non transférées. Ce protocole est documenté. |
| A6-09 | Corrigé | Les payloads du cache host passent le même UTF-8/TCV1/contrat strict que les résultats de contrôleurs. |
| A6-10 | Corrigé avec réserve | NULL n'est pas une adresse aliasée. destroy(NULL) reste possible pour les vtables stateless dont le nettoyage ne dépend pas d'un pointeur ; les tests existants le requièrent. |
| A6-11 | Corrigé | Header et ABI.md décrivent les frees F32, résultats owning, registre, last-error, APIs et slots réservés. Layouts et signatures C conservés ; snapshot SHA mis à jour pour les commentaires. |
| A6-12 | Corrigé | phase_index -> i32 est checked. Les bytes d'erreur des callbacks qui possèdent ce canal sont inclus avant libération. |
| A7-01 | Corrigé | TrainingResult utilise try_lock ; une réentrée échoue explicitement avec busy au lieu de bloquer GIL/mutex. Test de replay avec detach dans le callback. |
| A7-02 | Corrigé | Hash des floats conforme à json.dumps Python : exposants signés/paddés, seuils de notation, int/float et -0.0. Goldens et parcours de store Python. |
| A7-03 | Corrigé | Le lecteur et writer refusent explicitement les valeurs non finies, sans conversion silencieuse vers null. |
| A7-04 | Corrigé | Type et traceback des exceptions ordinaires conservés ; KeyboardInterrupt/SystemExit ressortent avec leur type Python, y compris en HPO. Tests successifs et parallèles. |
| A7-05 | Corrigé | Un pré-parcours des valeurs Python refuse les floats non finis avant depythonize, y compris dans les champs facultatifs. |
| A7-06 | Corrigé | Les entrées JSON Python utilisent le parseur strict commun ; doublons et collisions NFC sont refusés. |
| A7-07 | Faux positif | Le replay direct avec callback arbitraire documente un registre facultatif et laisse l'appelant posséder son contrôleur. Les façades archive imposent déjà un registre indépendant. La politique stricte WASM correspond à son autre frontière de confiance. |
| A7-08 | Corrigé | Nouveaux exports WASM u64 décimal compatibles avec les exports u32 existants ; chemins CV/terminal Python utilisent le seed de campagne. |
| A7-09 | Corrigé avec réserve | Absence de folds, différences de cardinalité et résultats malformés deviennent des erreurs contrôlées ou Failed par trial. L'évaluation finale complète reste nécessaire : le transcript scalaire par fold ne contient pas l'évidence par sample exigée par les rapports globaux OOF. |
| A7-10 | Corrigé | La création de view est rollback via discard_view si callback/receipt échoue ; wrappers délèguent aussi ce hook. |
| A7-11 | Corrigé | Hydratation transmet PyBytes ; export accepte bytes/bytearray directement. Contrat JSON des archives inchangé. Tests des deux types. |
| A7-12 | Corrigé avec réserve | IPC et manifeste écrits via temp/sync/rename, contenu adressé et manifeste en dernier ; metadata dupliquée retirée. Les anciennes générations peuvent encore servir des lecteurs ayant ouvert l'ancien manifeste et ne sont pas supprimées pendant publication. |
| A7-13 | Corrigé | Un n_jobs présent doit être entier ; 0 et les négatifs autres que -1 sont refusés. |
| A7-14 | Corrigé | Un ScoreSet est créé si les variantes supplémentaires ont des scores, même sans ScoreSet primaire. |
| A7-15 | Corrigé | Fallback create_new/copy sur filesystem sans hard links, avec publication du manifeste en dernier et rollback des seuls fichiers du writer. |
| A8-01 | Corrigé | Timeout operator distinct, défaut 0 illimité ; optimizer garde son timeout positif. CLI, R et MATLAB exposent les deux. |
| A8-02 | Corrigé | R conserve les grands seeds comme décimaux exacts et injecte un token numérique JSON ; MATLAB réutilise le token raw de NodeTask, sans conversion double. |
| A8-03 | Corrigé | Version R 0.3.34 ; la gate de release compare DESCRIPTION à Cargo. |
| A8-04 | Corrigé | Les sorties facultatives de bundle ne publient plus un gros JSON quand leur flag est absent. |
| A8-05 | Corrigé | score-output écrit explicitement null quand aucun score n'existe, remplaçant un éventuel ancien fichier. |
| A8-06 | Corrigé avec réserve | Comparaison lexicale Rust qui distingue dereferences/commentaires/littéraux ; Cargo.lock Python autonome suivi. Tests dédiés de tokens et historique Git. dag-ml-arrow n'est pas une dépendance de cette extension. |
| A8-07 | Corrigé | Les jobs CI concernés installent les dépendances et imposent les smokes via variables REQUIRE ; les skips locaux restent disponibles. |
| A8-08 | Corrigé | Les erreurs avant wait tuent et attendent le child et incluent stderr disponible. |
| A8-09 | Corrigé | Publish npm exige un tag stable qualifié ; cancellation des publications désactivée. Aucun publish exécuté dans cet audit. |
| A8-10 | Corrigé | Le job de parité construit et installe la wheel du checkout courant. |
| A8-11 | Corrigé | Un worker qui possède des artefacts ne peut plus être remplacé silencieusement ; erreur explicite sur perte d'artefact. |

## Validation finale

Gates exécutées sur les corrections, avec zéro échec :

| Gate | Résultat |
| --- | --- |
| `cargo test --workspace --no-fail-fast` | 1 097 tests passés, 3 ignorés explicitement ; les intégrations CLI Methods, R et sklearn étaient obligatoires via les variables `DAG_ML_REQUIRE_*`. |
| `scripts/test_methods_optimizer_local.sh` | 927 tests passés, 2 probes de performance ignorés ; 58 tests d'intégration avec la bibliothèque Methods native, incluant HPO, CV, refit et replay. |
| `cargo test -p dag-ml-core@0.3.34 --test bug_audit_regressions` | 14 régressions passées, dont la borne `u64::MAX` d'une plage pick ajoutée après la gate workspace. |
| Crate PyO3, feature Methods locale | 48 tests Rust passés ; clippy de cette crate avec `-D warnings` passé. |
| Surface Python avec l'extension reconstruite du checkout | 97 tests pytest et 72 subtests passés. Import vérifié depuis `crates/dag-ml-py/python/dag_ml/_dag_ml.abi3.so`. |
| WASM réel | Compilation wasm32 release passée ; `scripts/smoke_wasm_bindings.cjs`, refit/replay et métadonnées npm passés sous Node 24. Seeds décimaux jusqu'à `u64::MAX` et refus des overflows vérifiés. |
| R | `scripts/test_r_seed_json.R` passé : 42, 2^53+1, `u64::MAX`, résultats objet/JSON et refus des doubles déjà arrondis. |
| Contrats | Contrats locaux et croisés avec `dag-ml-data` passés ; contrat D4 validé ; 26 tests de replay passés. Manifests W1/D4 rafraîchis puis revalidés. |
| Qualité et livraison | `cargo fmt --all --check`, fmt de la crate Python, clippy workspace avec `-D warnings`, release metadata, snapshot ABI, fraîcheur de l'extension et autotests lexicaux de fraîcheur passés. |

La bibliothèque Methods utilisée est le build courant `nirs4all-methods/build/dev-release/cpp/src/libn4m.so.2.17.0`, indiqué explicitement aux runtimes Rust et Python. La wheel reconstruite a aussi servi à mettre à jour le binaire Python suivi. Les nombres des différentes gates se recouvrent et ne doivent pas être additionnés. Les trois tests ignorés du workspace sont deux probes de performance en release et un lecteur nécessitant une fixture externe du producteur Python ; aucun des smokes CLI obligatoires n'a été ignoré.

Limites et éléments utiles pour le réaudit :

- **A4-08** : les bundles historiques sans seed de namespace ne permettent pas de prouver sa préimage. La re-dérivation s'applique aux bundles nouvellement produits ; la compatibilité historique conserve les autres validations sans inventer un seed absent.
- **A4-11** : un ancien contrat sans limites de ressources ne permet pas de retrouver ces limites. Le seed de campagne est bien utilisé en replay.
- **A3-06** : les contrôles de contrat et les erreurs typées sont fatals ; `RuntimeValidation` reste le canal historique de refus numérique d'un candidat. La première cause est conservée si aucun candidat ne réussit. Le type générique ne fournit pas une distinction fiable entre toutes les causes I/O et modèle.
- Les wrappers MATLAB ont été corrigés et relus, mais MATLAB/Octave n'était pas disponible pour une exécution locale.
- L'archive publique `block3/wasm-workflow-model.n4a`, produite avant A5-15, contient d'anciens scores de classification et est correctement refusée par la validation actuelle. Ses métriques de régression concordent entre WASM et natif. Sa régénération appartient à la qualification des interfaces publiques, coordonnée dans `DAG_COORDINATION.md` ; cette archive antérieure ne sert pas de preuve de passage des nouveaux smokes WASM.

Les constats originaux restent dans `DAG-ML-BUGS.md` pour permettre une comparaison indépendante. Le détail de chaque faux positif et de chaque réserve figure dans le tableau, sans appliquer les changements de comportement proposés lorsqu'ils contredisaient le contrat existant.

### Complément d'intégration CPU/WASM — A2-04

La qualification des interfaces publiques a révélé une couverture manquante après les gates initiales : les empreintes utilisaient le sérialiseur stable, mais les membres d'archive étaient encore produits avec `serde_json::to_vec`. Une build CPU activant `preserve_order` et une build WASM sans cette feature pouvaient alors produire des membres différents pour des valeurs JSON identiques. La construction des membres V2/V3 utilise maintenant le même sérialiseur stable ; les comparaisons exactes de manifeste et d'octets restent inchangées.

La nouvelle régression sur l'inversion des clés imbriquées échoue avant le correctif et passe après ; les 10 tests d'archive passent. Une reconstruction depuis les membres bruts de l'archive réelle `root-integration-v2/sdk-cli-model.n4a`, dans un harness Cargo indépendant sans puis avec `preserve_order`, produit sept membres identiques octet par octet et le même manifeste. Les features des deux builds ont été vérifiées séparément. Clippy, fmt du fichier, recompilation wasm32 et smokes WASM passent. L'extension Python a été reconstruite une nouvelle fois et ses 97 tests/72 subtests repassent avec l'import public depuis le checkout ; les 26 tests D4 et les contrats croisés repassent également.

Les archives créées avec l'ancien ordre dépendant des features doivent être reconstruites avec le producteur corrigé pour cette fermeture exacte. Les nouveaux helpers des interfaces publiques et la proposition distincte sur l'identité de contenu du cohort PREDICT sont sous la responsabilité de l'autre root, selon `DAG_COORDINATION.md` ; leurs modifications ultérieures nécessitent leurs propres gates et reconstruction des bindings.

## Corrections complémentaires — R1–R5

Les cinq reproductions du second passage sont corrigées. Les sources ont d'abord été modifiées et testées dans une copie isolée pendant le gel des revues de l'autre tâche, puis intégrées après dégel, réservation explicite des fichiers et vérification des SHA de base. La liste des fichiers et empreintes finales est dans [applied-sources.json](../../_audits/2026-10-05-dagml-reaudit/fixes/applied-sources.json). Les modifications préexistantes, locks et binaires du checkout partagé ont été préservés.

| ID | Correction | Régression vérifiée |
| --- | --- | --- |
| R1 | Validation de tous les profils de variantes, sans exemption par préfixe ; re-dérivation du seed et du digest depuis la base canonique et les choices/trial/namespace/objectif. | IDs inventés ou HPO incomplets refusés ; essais Methods, scoped et host valides acceptés ; altérations de seed, choix, trial et objectif refusées. |
| R2 | Parcours des seuls enfants DSL typés ; IDs sémantiques opaques et absence des champs facultatifs préservés. | Labels distincts pour les IDs de catalogue dans operator/params/metadata/selector ; renommer seulement les IDs de nœuds ne change pas le label ; préimage explicite du générateur sans enfants optionnels conservée. |
| R3 | Digest pour les IDs modifiés, tronqués ou imitant un suffixe digest réservé. | Générateurs courts `g:a`/`g_a`, underscores initiaux et noms littéraux imitant l'encodage compilent ensemble ; compilation déterministe. |
| R4 | Validation objet/noms de `sources` et de chaque source avant insertion du descriptor. | Objet valide conservé sans mutation de l'entrée ; absent/null/array/number/noms manquants ou supplémentaires refusés sans panic. Le package U07 réel reste accepté. |
| R5 | Contrôle du nombre concret d'edges entrants contre One/Optional du manifeste. | Deux edges refusés avec One/Optional ; un edge Many→One accepté ; Many et manifest générique conservés. |

Neuf tests sont ajoutés dans `crates/dag-ml-core/tests/bug_reaudit_regressions.rs`, plus la matrice de formes invalides du helper multimodal. Les helpers des tests de contraintes décodent désormais le suffixe digest des IDs pour garder leur oracle de membres et d'ordre ; le mock HPO utilise une identité conforme au contrat. Aucun oracle numérique n'est modifié.

Le sélecteur natif a aussi révélé une assertion devenue obsolète après A2-04 : elle comparait un membre Archive V2 canonique à `serde_json::to_vec`, dépendant de `preserve_order`. Le JSON était strictement identique, seules les clés opaques changeaient d'ordre. Le test compare maintenant les **octets canoniques exacts**, comme les tests d'archive existants ; les contrôles d'égalité d'archive restent stricts.

| Gate complémentaire | Résultat |
| --- | --- |
| Workspace, copie des sources finales | `cargo test --offline --workspace` : **1 114 passés, 3 ignorés**, zéro échec. Pas de variables imposant les smokes host facultatifs ; cette gate ne remplace pas leur qualification finale. |
| Régressions du réaudit | **9 passées**, incluant les profils HPO valides et les tentatives de contournement. |
| Witnesses devenus régressions | **19 passés** dans le harness corrigé : cinq assertions inversées et les 14 tests du passage initial. Le harness historique reste conservé séparément. |
| Methods natif | Sélecteur `methods-optimizer-local` avec la feature `methods-optimizer` : **849 tests unitaires passés**, 2 probes ignorés ; **59 tests d'intégration passés**, dont les 58 scénarios métier et le test du sérialiseur inclus. Les neuf régressions, les 14 anciennes et les autres suites core passent aussi. |
| Qualité | Fmt de la copie complète, clippy workspace/all-targets et clippy core natif/all-targets avec `-D warnings` passés. |
| Contrats | Contrats locaux et croisés `dag-ml-data` passés sur la copie qualifiée, après synchronisation des digests de sources W1/D4. Aucun changement des fixtures ou goldens numériques. |
| CLI | Validation de `examples/minimal_graph.json` passée. |

Les [logs de correction](../../_audits/2026-10-05-dagml-reaudit/fixes/) distinguent les tentatives intermédiaires des résultats finaux. Le runtime Methods est explicitement `/home/delete/nirs4all/nirs4all-methods/build/dev-release/cpp/src/libn4m.so` ; la fixture estimator utilise `N4M_ESTIMATOR_ROLES_FIXTURE` absolu, car le lien relatif externe de la copie isolée ne résout pas le sibling original.

**Impact de qualification :** R3 change les IDs dérivés affectés ; R2 change les labels auparavant ambigus ; les nouveaux candidats host HPO transportent l'objectif nécessaire à R1 et les anciens sans cette liaison sont refusés. Les graphes/archives/checkpoints concernés doivent être régénérés. La reconstruction des bindings, le rafraîchissement commun des packs originaux après les autres changements et la qualification finale sont confiés au root d'intégration. Le guard de fraîcheur du checkout rapporte un état Rust/binaire tous deux modifiés : il exige un smoke d'import du binaire public reconstruit, et ne constitue pas une preuve que ce binaire contient R1–R5. Aucun binaire n'a été reconstruit ou remplacé par cette tâche.
