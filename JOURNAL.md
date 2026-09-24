# Journal de bord — ML-DMA

**Structure du projet** (depuis la réorganisation du 16/09/2026) :
- `dechets/` — exercice 1, complet : données SINOE, clustering + régression, rapport
- `menages/` — exercice 2, en cours : données INSEE Filosofi (revenu médian par département)
- `sante/` — exercice 3, en cours : annuaire RPPS (professionnels de santé), carte de densité faite, clustering à approfondir
- `JOURNAL.md` (ce fichier, à la racine) — journal partagé des trois exercices

Journal des étapes de réflexion du projet : prédiction/clustering des déchets ménagers et assimilés (DMA) à partir des données SINOE (ADEME).

---

## 2026-09-16 — Récupération et exploration des données

**Objectif initial** : télécharger les données SINOE "destination des DMA collectés par type de traitement" (data.gouv.fr) et préparer un premier tri/nettoyage en vue d'un projet de machine learning.

**Actions**
- Téléchargement de `sinoe_dma.csv` (ressource officielle ADEME, licence Ouverte 2.0) : 14 660 lignes × 10 colonnes, format "long" (une ligne = année × département × type de déchet × type de traitement × tonnage).
- Exploration (`scripts/01_explore.py`) : aucune valeur manquante, aucun doublon, 7 années (2009-2021, années impaires uniquement), 14 régions, 101 départements, 7 types de déchets, 7 types de traitement.
- Point de vigilance identifié : 54 lignes à `TONNAGE_DMA == 0` (pas de négatifs) — à surveiller mais pas bloquant.
- Redondance notée entre colonnes code/libellé (`C_REGION`/`L_REGION`, `C_DEPT`/`N_DEPT`, etc.) — à arbitrer au moment du feature engineering.

**Question soulevée** : pourquoi seulement des années impaires, et pourquoi rien après 2021 alors que la page data.gouv indique une mise à jour "août 2026" ?

**Réponse / conclusion** : ce n'est pas un problème de données manquantes au sens nettoyage — c'est la périodicité de l'enquête ADEME/SINOE elle-même, **biennale**. Vérifié sur les 4 jeux SINOE de l'ADEME (annuaire déchèteries, tonnage par type, chiffres-clés avec/hors gravats, destination par traitement) : tous mentionnent explicitement "années impaires à partir de 2009". La date de mise à jour sur data.gouv.fr reflète une republication de la ressource (métadonnées, re-upload), pas forcément l'ajout de nouvelles années. Qualité des métadonnées du dataset jugée faible (~56%) — pas de documentation méthodologique poussée sur data.gouv.fr.

---

## 2026-09-16 — Recherche de données complémentaires

**Décision utilisateur** : chercher d'autres jeux SINOE/ADEME + des données démographiques INSEE, plutôt que de rester sur le seul fichier initial.

**Actions**
- Repérage des autres jeux ADEME sur data.gouv.fr : *SINOE Annuaire des déchèteries DMA*, *SINOE Chiffres-clés DMA (avec gravats)*, *SINOE Chiffres-clés DMA (hors gravats)*.
- Téléchargement de `sinoe_chiffres_cles_avec_gravats.csv` (701 lignes) et `sinoe_chiffres_cles_hors_gravats.csv` (800 lignes) — format `;` / décimale `,`, à ne pas confondre avec le fichier principal (`,` / `.`).
- Téléchargement de `insee_population_dep_sexe_gca.xls` (INSEE, population par département/sexe/tranche d'âge, 1975-2023, une feuille Excel par année) — piste de secours, finalement non indispensable (voir plus bas).

**Découverte clé** : `sinoe_chiffres_cles_hors_gravats.csv` couvre en réalité **2009 à 2023** (8 enquêtes), contrairement au fichier principal limité à 2021. C'est le seul des fichiers téléchargés à inclure 2023. Petits trous ponctuels par département/année (99-101 lignes selon l'année sur 101 attendues) — normal pour une enquête déclarative, pas un problème de qualité.

- Vérifié que `avec_gravats` et `hors_gravats` partagent exactement la même colonne population (`VA_POPANNEE`) pour les couples (département, année) communs — 0 écart sur 701 lignes comparées. Ces fichiers contiennent déjà une population et un ratio (`RATIO_DMA`, kg/habitant) précalculés par l'ADEME.

---

## 2026-09-16 — Définition de l'objectif ML

**Réflexion** : avant d'aller plus loin dans le nettoyage/l'enrichissement, il fallait d'abord fixer *ce qu'on cherche à prédire* et *ce qu'on veut clusteriser* — c'est ce choix qui détermine quelles données sont réellement nécessaires (plutôt que d'enrichir à l'aveugle).

**Options envisagées** :
1. Régression du tonnage / ratio par habitant + clustering des départements par **profil de traitement** (répartition % incinération/stockage/valorisation/recyclage).
2. Classification du type de traitement dominant + clustering des régions par politique de traitement.
3. Séries temporelles (extrapolation 2023+) + clustering des trajectoires d'évolution — plus complexe, aurait nécessité des données socio-économiques supplémentaires (revenu, urbanisation).

**Choix retenu** : option 1 — régression de `RATIO_DMA`/`TONNAGE_DMA` par département/année, couplée à un clustering des départements selon leur profil de traitement. Réalisable intégralement avec les données déjà téléchargées, sans enrichissement externe supplémentaire.

**Conséquence** : le fichier population INSEE (âge/sexe) téléchargé n'est finalement pas indispensable dans l'immédiat — la population est déjà présente dans les fichiers ADEME "chiffres-clés". Gardé de côté pour un éventuel affinage futur (pyramide des âges comme feature).

---

## 2026-09-16 — Construction des datasets

**Clustering** (`scripts/02_clustering_profil_traitement.py`, source : `sinoe_dma.csv`) :
- Pivot du tonnage par département × type de traitement, normalisé en % (profil de traitement par département, agrégé 2009-2021).
- KMeans avec sélection de k par silhouette score : k=4 retenu (silhouette 0.276).
- Résultat : 2 clusters majeurs (42 départements orientés stockage/autres vs 52 orientés incinération ~39,5%), 2 clusters atypiques (Lozère/Mayotte — faible densité ; Charente/Landes/Savoie/Morbihan/Charente-Maritime — forte valorisation organique).
- Sortie : `data/processed_clusters_traitement.csv`.

**Régression** (`scripts/03_regression_prep.py`, source : `sinoe_chiffres_cles_hors_gravats.csv`) :
- Choisi comme base car seul fichier couvrant 2009-2023 (8 points temporels au lieu de 7).
- Jointure avec le cluster de profil de traitement (calculé sur 2009-2021) comme feature catégorielle de typologie territoriale.
- Ajout de features de lag (`RATIO_DMA_lag1`, `TONNAGE_DMA_lag1`) — valeur de l'enquête précédente par département, NaN attendu uniquement pour 2009 (101 lignes, première année de chaque département).
- Sortie : `data/processed_regression_dataset.csv` (800 lignes, aucune valeur manquante inattendue).

**Prochaine étape envisagée** : entraîner un modèle de régression (RandomForest/GradientBoosting) sur `RATIO_DMA`, avec split train/test **par année** (et non aléatoire) pour éviter la fuite temporelle — par exemple entraîner sur 2009-2019 et tester sur 2021/2023.

---

## 2026-09-16 — Cible = ratio par habitant, pas tonnage brut ; année vs population

**Question soulevée** : faut-il donner l'année en entrée du modèle, ou plutôt la population ?

**Vérification chiffrée** sur `processed_regression_dataset.csv` :
- `corr(TONNAGE_DMA, VA_POPANNEE) = 0.966` — quasi-parfaite. Logique : `RATIO_DMA = TONNAGE_DMA / population`, donc `TONNAGE_DMA ≈ RATIO_DMA × population`. Si la cible est le tonnage brut, la population résout le problème quasi toute seule — ce n'est plus un apprentissage, c'est une fuite (leakage) qui redonne une multiplication.
- `corr(RATIO_DMA, ANNEE) = 0.02` — quasi nulle. L'année brute ne porte presque aucun signal linéaire pour le ratio par habitant ; la variation observée vient surtout des différences entre départements, pas d'une tendance temporelle globale.

**Décision** :
1. Cible confirmée = `RATIO_DMA` (kg/habitant), pas `TONNAGE_DMA` brut, pour éviter la fuite via la population.
2. Population conservée comme feature, mais comme variable **structurelle** (effet taille/densité du département), pas comme proxy du temps.
3. `ANNEE` brute reléguée derrière `RATIO_DMA_lag1` (déjà dans le dataset) comme représentation du temps — le lag capture la trajectoire propre à chaque département, plus informatif qu'un simple numéro d'année.

**Note** : ça ne règle pas le problème des années manquantes (survey biennale, cf. entrée du 2026-09-16 plus haut) — c'est un choix de feature engineering orthogonal, pas un correctif du trou temporel. On n'aura toujours pas de vérité terrain pour les années paires.

Script `scripts/03_regression_prep.py` mis à jour pour afficher ces corrélations de vérification à chaque exécution.

---

## 2026-09-16 — Matrice de corrélation, split temporel et Min-Max scaling

**Contexte** : notions vues en cours par l'utilisateur — (1) une étape de corrélation pour savoir quelle donnée influe sur quelle statistique, (2) une réduction des intervalles de valeurs en mappant tout entre 0 et 1.

**Clarification pédagogique** : deux techniques de mise à l'échelle différentes, à ne pas confondre.
- `StandardScaler` (déjà utilisé pour le clustering, script 02) : centre-réduit (moyenne 0, écart-type 1), **pas borné**.
- `MinMaxScaler` (ce que décrit le cours) : `x' = (x - min) / (max - min)`, borne exactement entre 0 et 1.

**Corrélation** (`scripts/04_correlation.py`) : matrice de corrélation complète sur les variables numériques du dataset régression, heatmap sauvegardée dans `data/correlation_matrix.png`.
- `RATIO_DMA_lag1` domine largement (corr = 0.91 avec la cible `RATIO_DMA`) : le ratio d'un département varie peu d'une enquête à l'autre (2 ans d'écart) — cohérent avec des habitudes de collecte qui évoluent lentement.
- `VA_POPANNEE` : corrélation faible et négative (-0.17) — les départements plus peuplés ont tendance à un ratio par habitant légèrement plus faible (effet densité/économies d'échelle).
- `cluster` (typologie de traitement) : corrélation faible (0.13).
- `ANNEE` : confirme le quasi-non-signal déjà observé (0.02).
- `TONNAGE_DMA` / `TONNAGE_DMA_lag1` vs `VA_POPANNEE` : 0.97 — confirme à nouveau la quasi-tautologie évoquée précédemment (raison pour laquelle la cible reste `RATIO_DMA`, pas le tonnage brut).

**Split + scaling** (`scripts/05_split_scale.py`) :
- Suppression des 101 lignes 2009 (pas de `lag1` disponible) -> 699 lignes exploitables.
- Split **temporel**, pas aléatoire : train = années ≤ 2019 (501 lignes, 2011-2019), test = années > 2019 (198 lignes, 2021 et 2023). Objectif : simuler une vraie prédiction du futur, pas une interpolation.
- `MinMaxScaler` calibré (fit) **uniquement sur le train**, puis appliqué (transform seul) au test — règle essentielle pour éviter la fuite train/test.
- Vérification : sur le train, toutes les features scalées sont exactement dans `[0, 1]`. Sur le test, certaines dépassent légèrement (ex. `TONNAGE_DMA_lag1` max = 1.03) — **comportement normal et voulu** : ça signale que 2021/2023 contiennent des valeurs plus extrêmes que tout ce que le train (2011-2019) a vu. Si le scaler avait été calibré sur l'ensemble des données, ce signal aurait été invisible.
- Sorties : `data/train_scaled.csv`, `data/test_scaled.csv`.

**Prochaine étape** : entraîner le modèle de régression sur `train_scaled.csv` (features : `VA_POPANNEE`, `cluster`, `RATIO_DMA_lag1`, `TONNAGE_DMA_lag1` ; cible : `RATIO_DMA`), évaluer sur `test_scaled.csv`.

---

## 2026-09-16 — Rapport de projet (livrable)

**Demande** : formaliser dans un rapport ce qu'on a trouvé au total (exploration) et le nettoyage effectué, avec justification de chaque étape — destiné à être lu par un tiers (évaluateur), pas seulement une trace de réflexion interne comme ce journal.

**Action** : rapport rédigé en HTML — sources de données, constats d'exploration (tableau de corrélations inclus), et les 8 étapes de nettoyage/transformation avec justification pour chacune. Document vivant, à compléter avec la section modélisation une fois le modèle entraîné.

- Versionné dans le repo : [`rapport/rapport_dma.html`](rapport/rapport_dma.html) — à utiliser pour un envoi par mail ou un export PDF.

---

## 2026-09-16 — Réorganisation de `data/`

**Problème signalé** : tous les fichiers de données étaient en vrac dans `data/`, sans distinction entre bruts, générés, ou figures.

**Action** : restructuration en `data/raw/` (4 fichiers sources téléchargés, jamais modifiés), `data/processed/` (fichiers générés par les scripts), `data/figures/` (sorties graphiques). Chemins mis à jour dans les 5 scripts, pipeline complet rejoué de bout en bout pour vérifier qu'aucune référence n'était cassée.

**Effet de bord non demandé** : un `git init` a été exécuté par erreur au passage (aucun commit fait). À garder ou supprimer selon le choix de l'utilisateur.

---

## 2026-09-16 — Entraînement et évaluation du modèle de régression

**Script** : `scripts/06_train_model.py`. Cible `RATIO_DMA`, features `VA_POPANNEE`, `cluster`, `RATIO_DMA_lag1`, `TONNAGE_DMA_lag1`. Train = 2011-2019 (501 lignes), test = 2021 + 2023 (198 lignes, jamais vus à l'entraînement).

**Trois modèles comparés**, dont une baseline naïve (prédire = valeur de l'enquête précédente) servant de repère minimal :

| Modèle | MAE | RMSE | R² |
|---|---|---|---|
| Baseline naïve | 36.49 | 44.58 | 0.702 |
| Régression linéaire | 35.89 | 43.64 | 0.715 |
| Random Forest | 36.05 | 44.21 | 0.707 |

**Constat** : les modèles ML battent à peine la baseline. Importance des variables (Random Forest) : `RATIO_DMA_lag1` = 94.9 %, le reste (population, cluster, tonnage_lag) se partage moins de 5 %. Cohérent avec la corrélation observée précédemment (0.91 entre ratio et son lag).

**Interprétation** : ce n'est pas un échec de modélisation, c'est une conclusion sur les données — le ratio de déchets par habitant est très stable dans le temps par département (inertie forte), donc les caractéristiques stables du département (population, typologie de traitement) apportent peu par rapport à la simple persistance de la valeur précédente. Pour dépasser cette baseline, il faudrait des variables qui *varient* dans le temps (revenu, nouveaux dispositifs de tri, évolution de la densité urbaine) plutôt que des caractéristiques structurelles.

**Décision utilisateur** : ce résultat est documenté dans le rapport (section 04 ajoutée), la piste "améliorer le modèle" est mise de côté pour l'instant au profit de la finalisation du rapport ; le clustering pourra être approfondi dans une étape suivante.

Sorties : `data/processed/test_predictions.csv`, `data/processed/model_comparison.csv`.

---

## 2026-09-16 — Section clustering du rapport (graphique + interprétation)

**Contexte** : le clustering (script 02) était fait depuis le début mais jamais documenté visuellement — seule la régression avait une section complète. Complété pour boucler les deux volets ML choisis en début de projet.

**Action** : ajout d'une section 05 au rapport avec un graphique en barres empilées à 100 % (profil de traitement par cluster, palette catégorielle validée via le script du skill dataviz — contrôle CVD/contraste passé en clair et en sombre) et un tableau d'interprétation.

**Lecture des 4 clusters** (profils moyens en %, sur `sinoe_dma.csv` 2009-2021, gravats compris) :
- **Cluster 0 — Stockage** (42 dép.) : 44,3 % stockage, 27,3 % valorisation matière.
- **Cluster 1 — Incinération** (52 dép.) : 39,5 % incinération avec récupération d'énergie, ~6× plus que le cluster 0.
- **Cluster 2 — Atypique** (2 dép., Lozère et Mayotte) : 32,7 % « non précisé » + 45,7 % stockage — probablement un défaut de déclaration plutôt qu'une vraie politique, à traiter avec prudence (seulement 2 membres).
- **Cluster 3 — Valorisation** (5 dép. : Charente, Charente-Maritime, Landes, Morbihan, Savoie) : 53,6 % de valorisation cumulée (matière + organique), le plus haut des 4 ; seul cluster avec de l'incinération sans récupération notable (8,2 %).

Rapport republié (v6).

---

## 2026-09-16 — Nuage de points prédit vs réel, dans la section régression

**Question posée** : "il y a pas la prédiction ?" — clarification que prédiction = régression, et où/comment ça se passe dans le code (`fit` puis `.predict()` dans `scripts/06_train_model.py`).

**Bug trouvé en cherchant** : `pred_lr` (prédictions de la régression linéaire) était calculé et utilisé pour les métriques, mais jamais sauvegardé dans `test_predictions.csv` — seul `pred_rf` l'était. Corrigé dans `scripts/06_train_model.py`, script rejoué.

**Ajout au rapport** (section 04) : nuage de points (régression linéaire, 198 points = jeu de test 2021+2023) — ratio réel en abscisse, ratio prédit en ordonnée, ligne pointillée = prédiction parfaite. Un point = un département/année concret, survolable pour le détail. Couleur = accent du rapport (série unique, pas besoin de la palette catégorielle). Rendu conforme au skill dataviz (marqueurs, grille discrète, pas de double axe).

Rapport republié (v7).

---

## 2026-09-16 — Courbe d'apprentissage : plus de lignes aiderait-il ?

**Question posée** : "à partir de combien de lignes ça serait intéressant ?" — suite à la discussion sur le R² proche du baseline et l'hypothèse du manque de données.

**Méthode** : plutôt que de répondre en théorie, courbe d'apprentissage empirique — régression linéaire réentraînée sur des fractions croissantes du train (10 % à 100 %, tirage aléatoire fixe), R² mesuré à chaque fois sur le même jeu de test (198 lignes).

**Résultat** :

| % du train | Lignes | R² test |
|---|---|---|
| 10 % | 50 | 0,724 |
| 50 % | 250 | 0,699 |
| 100 % | 501 | 0,715 |

Le score **n'augmente pas** avec plus de lignes — il oscille entre 0,70 et 0,72 dès 50 lignes. C'est la signature empirique d'un problème limité par le contenu des variables (biais), pas par le volume d'exemples (variance) : si le manque de données était le facteur limitant, la courbe monterait progressivement puis plafonnerait plus haut. Ici elle est plate dès le départ.

**Conclusion communiquée** : pas de seuil de lignes à atteindre avec ces 4 variables — multiplier les lignes de ce même jeu ne débloquerait vraisemblablement rien. Le levier, c'est des variables réellement informatives (temps-variantes), pas plus d'exemples des mêmes variables.

**Ajout au rapport** (section 04) : graphique de la courbe d'apprentissage + encart de conclusion. Sortie : `data/processed/learning_curve.csv`. Rapport republié (v8).

---

## 2026-09-16 — Recherche Kaggle, puis pivot vers l'INSEE

**Piste envisagée** : chercher sur Kaggle un jeu de données déchets/municipalités avec plus de lignes ou de variables.

**Constat** : la quasi-totalité des jeux "waste" sur Kaggle sont des jeux d'images pour la classification (tri visuel de déchets) — hors sujet pour une régression tabulaire. La seule ressource tabulaire pertinente trouvée (base "What a Waste" de la Banque Mondiale) est au niveau ville/pays mondial, sans clé de jointure avec nos départements français, et surtout une photo à un instant donné plutôt qu'une série temporelle — ne résout ni le problème de granularité ni celui des variables manquantes.

**Décision** : rester sur l'échelle département française et chercher une vraie variable explicative complémentaire, via l'INSEE (même levier que la population : source officielle, jointure propre sur `C_DEPT`). Revenu fiscal des ménages choisi en priorité (hypothèse : le pouvoir d'achat influence le volume de déchets produits).

---

## 2026-09-16 — Assemblage du revenu médian INSEE (Filosofi), 2013-2021

**Difficulté rencontrée** : contrairement aux fichiers ADEME (un CSV = toutes les années), l'INSEE publie le revenu médian (Filosofi) dans une édition séparée par année, sans page listant proprement toutes les éditions — chaque année a dû être retrouvée individuellement par recherche, puis le fichier department-level extrait d'une archive zip contenant plusieurs niveaux géographiques.

**Formats rencontrés, différents d'une année à l'autre** :
- 2013, 2015 : fichier `.xls`, feuille "DEP", colonne `MED13`/`MED15`
- 2017, 2019 : fichier `.csv` dédié `cc_filosofi_YYYY_DEP.csv`, colonne `MEDyy`
- 2021 : format long (une ligne par indicateur), filtré sur `GEO_OBJECT='DEP'` et `FILOSOFI_MEASURE='MED_SL'`

**Script** : `menages/scripts/07_income_prep.py` — normalise les 5 formats en une seule table `C_DEPT, ANNEE, MEDIANE_REVENU`. Résultat : 491 lignes (96 à 101 départements selon l'année — écarts dus au secret statistique INSEE sur les petits départements, pas une erreur). Sortie : `menages/data/raw/insee_revenu_median_dep_2013_2021.csv`.

**Limite actée** : Filosofi ne couvre que 2013-2021, contre 2009-2023 pour les données déchets — toute jointure avec le revenu réduit mécaniquement la période exploitable.

---

## 2026-09-16 — Expérience : le revenu améliore-t-il la prédiction ?

**Script** : `dechets/scripts/08_income_experiment.py`. Jointure du dataset régression avec le revenu (restreint à 2015-2021 après retrait de la 1ère année sans lag), comparaison à features égales avec/sans revenu, même split, même baseline.

| | MAE | RMSE | R² |
|---|---|---|---|
| Baseline naïve | 21,72 | 29,83 | 0,871 |
| Sans revenu | 19,86 | 27,86 | 0,887 |
| Avec revenu | 20,17 | 28,10 | 0,885 |

**Résultat** : le revenu n'améliore pas la prédiction — très légèrement pire avec (0,885 vs 0,887). Importance Random Forest : `MEDIANE_REVENU` + `REVENU_lag1` réunis pèsent 2,7 %, contre 94,5 % pour le lag du ratio.

**Point de vigilance communiqué** : ces R² (~0,88) ne sont pas comparables à ceux du pipeline principal (0,715) — sous-ensemble différent (2015-2021 seulement, faute de revenu disponible avant 2013), pas une amélioration du modèle.

**Conclusion** : hypothèse testée et écartée — le revenu médian par département n'est pas la variable manquante qui expliquerait le changement de comportement. Une conclusion valable en soi, pas un échec de démarche.

---

## 2026-09-16 — Réorganisation en deux exercices séparés

**Demande** : séparer le projet en deux dossiers — un pour les déchets, un pour les ménages — pour refaire l'exercice complet (exploration → nettoyage → ML) une seconde fois, sur les données de revenu cette fois, indépendamment du premier.

**Action** : restructuration —
- `dechets/` reçoit tout l'exercice 1 : `data/raw`, `data/processed`, `data/figures`, `scripts/01` à `06` + `08` (l'expérience revenu, qui reste ici car sa cible est le tonnage déchets), `rapport/rapport_dma.html`.
- `menages/` reçoit `data/raw/insee_revenu_median_dep_2013_2021.csv` et `scripts/07_income_prep.py` (l'assemblage des 5 fichiers Filosofi) — c'est le point de départ du 2ᵉ exercice, exploration/nettoyage/ML restant à faire dessus en tant que sujet à part entière (pas juste une feature pour les déchets).

Chemins des scripts mis à jour (chaque script tourne désormais depuis la racine de son propre dossier), pipeline complet des deux dossiers rejoué pour vérifier — tout fonctionne. `JOURNAL.md` reste unique à la racine, partagé entre les deux exercices.

---

## 2026-09-16 — Exercice 3 : annuaire RPPS (professionnels de santé)

**Demande** : un 3ᵉ exercice indépendant, sur l'annuaire santé RPPS de data.gouv.fr (Agence du Numérique en Santé).

**Téléchargement** : fichier principal `PS_LibreAcces_Personne_activite.txt` (820 Mo, 2 287 724 lignes, 56 colonnes, séparateur `|`). Mémoire disponible limitée (5,5 Go) constatée avant chargement -> exploration d'abord sur un échantillon de 200 000 lignes plutôt que le fichier entier.

**Découverte majeure** : la colonne dédiée `Code Département (structure)` est **vide à 100 %** dans le fichier. Contournement : département reconstruit depuis les 2 premiers chiffres du code postal de la structure (3 chiffres pour l'outre-mer : 971, 972... — sinon tous les DOM se seraient mélangés dans un faux code "97"). Cas particulier : la Corse a un code postal "20" qui ne distingue pas 2A/2B -> repli sur le code commune INSEE pour ces lignes.

**Valeurs manquantes** : 24,9 % des lignes sans département reconstructible. Vérifié que ce n'est pas une erreur : ces lignes correspondent à des professionnels inscrits au RPPS **sans activité active déclarée** (leur "mode d'exercice" n'est renseigné que 34,5 % du temps contre 100 % pour le reste, "genre d'activité" seulement 6,5 %). Lignes retirées pour l'analyse territoriale — 1 718 220 lignes exploitables.

**Colonnes retenues** : 10 sur 56 (identité, profession, localisation) — le reste (téléphone, SIRET, adresse détaillée, savoir-faire...) n'apporte rien à une analyse par département et alourdit inutilement la mémoire.

**Dédoublonnage** : un praticien peut apparaître plusieurs fois (plusieurs structures) — dédoublonné par (praticien, département) avant tout comptage, sinon les effectifs auraient été gonflés artificiellement (1 718 220 → 1 477 031 lignes).

**Question posée sur l'objectif ML** : l'utilisateur voulait à terme une carte des praticiens + une prédiction d'évolution temporelle. **Vérifié et confirmé** : ni le fichier principal ni le fichier "Diplômes et autorisations d'exercice" (264 Mo, téléchargé pour vérifier) ne contiennent de date (ni inscription, ni diplôme, ni début d'exercice) — impossible de faire de la prédiction temporelle avec les extractions RPPS en open data telles quelles. Seule option : re-télécharger ce même jeu à intervalles réguliers pour constituer un historique à partir de maintenant, pas de reconstitution possible du passé.

**Décision** : carte de densité + clustering par profil de profession (miroir de l'exercice déchets), sans volet temporel.

**Carte réalisée** : densité de praticiens pour 10 000 habitants par département (population 2023 reprise du dataset déchets), choroplèthe en 6 classes (quantiles, palette séquentielle bleue validée par le script du skill dataviz), France métropolitaine (96 départements — contours simplifiés depuis `gregoiredavid/france-geojson`, outre-mer exclu de la carte pour limiter la complexité d'un encart séparé, mais chiffré séparément). 1,47M praticiens comptés, densité médiane 211 pour 10 000 hab.

**Lecture (hypothèses, pas de causalité démontrée)** : les plus fortes densités sont à Paris et dans les villes à CHU (Limoges/87, Lyon/69, Marseille/13) ; les plus faibles dans la grande couronne parisienne (Seine-Saint-Denis, Seine-et-Marne, Oise, Essonne) — cohérent avec le phénomène connu des "déserts médicaux" périurbains.

**Sorties** : `sante/data/processed/rpps_clean.csv` (1,72M lignes), `rpps_par_departement.csv`, `rpps_densite_departement.csv`, `dept_svg_paths.json`. Rapport carte : `sante/rapport/carte_rpps.html`.

**Prochaine étape envisagée** : clustering des départements par profil de profession (% médecins/infirmiers/kinés...), miroir du clustering profil de traitement des déchets.

---

## 2026-09-16 — Carte par points (commune) : les zones vides

**Remarque de l'utilisateur** : la moyenne par département masque les écarts internes — un département avec une grande ville bien dotée peut cacher des zones rurales vides. Demande de passer à un pointage plus fin pour voir les vraies zones sans praticien.

**Source ajoutée** : `sante/data/raw/communes_coords.csv` (data.gouv.fr, "Données sur les communes de France Métropolitaine") — code INSEE, latitude, longitude, ~35 000 communes.

**Choix d'échelle** : un point par praticien individuel (1,47M) aurait été trop lourd pour une carte statique. Agrégation au niveau **commune** à la place (rayon ∝ racine carrée de l'effectif, pour une aire proportionnelle) — assez fin pour révéler les vides, assez léger pour rester une carte statique (16 652 communes, ~1,3 Mo de SVG).

**Bug rencontré et corrigé** : la première jointure avec `communes_coords.csv` faisait gonfler les effectifs (une commune avec plusieurs codes postaux apparaît plusieurs fois dans ce fichier) — corrigé par un dédoublonnage sur le code INSEE avant la jointure.

**Résultat marquant** : seulement **16 652 communes sur ~34 900** ont au moins un professionnel de santé actif recensé — moins de la moitié. L'espace blanc entre les points sur la carte est directement lisible comme l'absence de praticien.

**Script** : `sante/scripts/04_points_commune.py`. Sortie : `sante/data/processed/rpps_points_commune.csv`. Carte ajoutée en section "02bis" du rapport (même repère de projection que la carte département, pour rester comparable). Rapport mis à jour (v2).

---

## 2026-09-16 — Bug de rayon + fusion des deux cartes en une seule

**Problème signalé** : les points n'apparaissaient pas sur la carte publiée.

**Diagnostic** : les points existaient bien et étaient bien positionnés, mais leur rayon (1 à 26 unités) était ridicule comparé à la taille du repère de la carte (~15 656 unités de large pour ~640px affichés à l'écran) — des fractions de pixel, invisibles. Corrigé : rayon recalculé pour être visible à l'écran (57 à 340 unités, soit environ 2 à 14px affichés).

**Deuxième remarque** : les points devaient être superposés sur la carte de France (départements), pas dans une section séparée sur fond blanc, pour que ce soit "cohérent" géographiquement. Les deux cartes (choroplèthe département + points commune) fusionnées en une seule SVG (points en corail par-dessus le fond bleu des départements) — la section "02bis" séparée a été supprimée.

Rapport mis à jour (v3).

---

## 2026-09-16 — Carte filtrable par profession

**Demande** : ajouter des filtres pour voir les points par type de profession (médecins seuls, pharmaciens seuls...) et repérer les trous propres à chaque profession, pas seulement au total.

**Méthode** : effectifs recalculés par commune **et** par groupe de profession (mêmes 8 catégories + "Autres" que pour le clustering département), au lieu d'un seul total par commune. Données embarquées en JSON dans la page (625 Ko, 16 652 communes × 10 catégories) plutôt que rechargées côté serveur — chaque cercle SVG porte un `data-i` reliant à sa ligne de données, un script JS recalcule le rayon (et masque les communes à 0) au clic sur un filtre, sans recharger la page.

**Filtres ajoutés** : Tous, Médecins, Infirmiers, Masseurs-kinés, Psychologues, Pharmaciens, Chirurgiens-dentistes, Sages-femmes, Orthophonistes, Autres professions — plus un compteur dynamique ("X communes avec au moins un·e [profession] actif·ve, sur ~34 900, soit Y %") qui se met à jour avec le filtre, pour chiffrer directement l'ampleur des trous par profession.

**Vérification avant publication** : JSON réembarqué validé (parseable, 16 652 lignes, 10 catégories), comptage des balises équilibré, avant republication — pour éviter de répéter l'erreur du rayon invisible sans avoir vérifié.

**Script** : `sante/scripts/05_carte_filtrable.py`. Sorties : `sante/data/processed/rpps_points_par_profession.csv`, `points_data.json`. Rapport mis à jour (v4).

**Note "démographie par type"** : interprété comme la répartition géographique par profession (ce que les filtres montrent), pas une pyramide des âges — le fichier RPPS ne contient aucune date de naissance ni d'âge, donc une vraie démographie (âge/sexe) par profession n'est pas possible avec cette source. Le champ "civilité" (M./Mme) donnerait un proxy de genre si besoin, pas encore exploité.

---

## 2026-09-16 — Points en pleine mer, toggle, et légende bleue réactive au filtre

**Signalement** : des points apparaissaient dans l'océan Atlantique, surtout autour de la Bretagne.

**Diagnostic** : coordonnées vérifiées une à une — toutes dans la zone plausible de la métropole, aucune erreur de donnée. La vraie cause : le fond de carte simplifié (pour rester léger) avait supprimé les petites îles lors de la simplification (Ouessant, Belle-Île, Groix, Île d'Yeu...), qui sont pourtant de vraies communes avec de vrais praticiens — leurs points, eux, gardaient les bonnes coordonnées et flottaient donc hors du contour terrestre simplifié.

**Correction** : contours complets (non simplifiés) réintégrés pour les 6 départements concernés par des îles significatives (17 Charente-Maritime, 29 Finistère, 50 Manche, 56 Morbihan, 83 Var, 85 Vendée), le reste de la métropole restant en version simplifiée pour la légèreté.

**Deux fonctionnalités ajoutées** :
1. **Bouton bascule des points** ("● Points communes") — permet de voir la carte bleue seule, sans les points, ou l'inverse.
2. **Légende bleue réactive au filtre** — jusque-là, filtrer sur "Médecins" changeait les points mais le fond bleu restait sur la densité totale. Densité par département calculée pour chaque profession séparément (pas seulement le total), avec ses propres seuils de quantiles (la distribution des sages-femmes par département n'a rien à voir avec celle des médecins) — le fond, la légende et les info-bulles des départements se recalculent maintenant avec le filtre actif.

**Bug intercepté avant publication** : le bouton bascule partageait la classe CSS `filter-btn` avec les boutons de profession, ce qui aurait déclenché la logique de filtre (et cassé l'état actif) au clic sur bascule. Repéré et corrigé avant republication en relisant le script, pas après coup.

**Sorties** : `sante/data/processed/dept_density_data.json` (densité par département × profession + seuils), `dept_paths_v2.json` (contours avec îles). Rapport mis à jour (v5).

---

## 2026-09-16 — 29 professions filtrables (pas 8 + "Autres") et grammaire française correcte

**Deux remarques de l'utilisateur** :
1. Pourquoi seulement 10 filtres alors que 29 professions existent dans les données ?
2. Bug d'accord grammatical : l'info-bulle affichait "2 touss" pour la catégorie "Tous" (pluriel naïf en ajoutant un "s", qui casse sur un mot déjà terminé par "s").

**Réponse au point 1** : le regroupement initial (8 professions les plus fréquentes + "Autres") avait été fait pour garder une ligne de boutons lisible, mais ça cache de l'info pour les 21 professions restantes. Recalculé pour les 29 professions individuellement, par commune et par département. Interface changée : une ligne de 9 boutons ne passait plus à l'échelle pour 30 options (29 + Tous) -> remplacée par un menu déroulant unique, trié alphabétiquement.

**Réponse au point 2** : construction d'une table de correspondance singulier/pluriel pour les 29 professions + "Tous" (`sante/data/processed/plurals.json`), avec les cas particuliers du français (accord des deux parties d'un nom composé : "chirurgien-dentiste" → "chirurgiens-dentistes", pas "chirurgien-dentistes" ; "Tous" traité à part avec "praticien"/"praticiens", pas de pluriel naïf). Appliqué aux 4 endroits du script qui construisaient du texte (info-bulle des points, info-bulle des départements, légende, compteur de communes couvertes).

**Scripts mis à jour** : `sante/scripts/05_carte_filtrable.py` (v2, 29 professions). Sorties : `points_data.json`, `dept_density_data.json`, `plurals.json`. Rapport mis à jour (v6), vérifié avant publication (30 catégories cohérentes partout, JSON valide, plus aucune trace de l'ancienne UI par boutons).

---

## 2026-09-16 — Pourquoi si peu de communes avec médecin, couverture RPPS, population vérifiée, zoom département

**Quatre questions posées d'un coup.**

**1. "Plus de la moitié des communes sans médecin, y a pas un problème ?"** Vérifié plus précisément : c'est même 72 % sans médecin (28 % en ont un, 9 784/34 900). Croisé avec la population des communes (via `communes_coords.csv`) : population médiane 629 hab. pour les communes sans médecin, contre 2170 pour celles qui en ont un. Sur ~34 900 communes françaises, 84,5 % ont moins de 2000 habitants — structure communale très morcelée, cohérent avec la réalité des déserts médicaux, pas un bug.

**2. "C'est juste leur lieu de travail non ?"** Confirmé — RPPS recense le lieu d'exercice, pas le lieu de résidence.

**3. "Quelles professions de santé n'ont pas de RPPS ?"** Recherché : depuis 2024 le RPPS a absorbé l'ancien registre ADELI et couvre quasi toutes les professions réglementées. Restent hors RPPS (pas d'obligation d'inscription) : aides-soignants, auxiliaires de puériculture, ambulanciers, assistants de régulation médicale. Les aides-soignants seuls représentent plusieurs centaines de milliers de personnes — leur absence fait sous-estimer largement l'effectif réel du système de santé.

**4. "Prends une population plus récente, pas 2023 !"** Téléchargé le fichier officiel INSEE dédié (`sante/data/raw/population_ref_2023.xlsx`, populations de référence, validées par décret du 26/12/2025, en vigueur au 1er janvier 2026 — la donnée la plus à jour qui existe, la prochaine ne sort qu'en décembre 2026). Résultat inattendu : **0 écart** sur les 99 départements comparés avec la population qu'on empruntait déjà au projet déchets — même source in fine (ADEME utilise la même donnée INSEE). Les chiffres ne changent donc pas, mais la source est maintenant téléchargée et vérifiée directement pour ce projet plutôt qu'empruntée. Ancien fichier `population_2023.csv` (emprunté) supprimé, remplacé par `population_insee_2023ref.csv` (source propre). Nouveau script `sante/scripts/06_departement_densite.py`.

**Fonctionnalité ajoutée : zoom par département.** Demande : "zoomer sur un département pour avoir les mêmes infos mais sur un truc précis". Bounding box calculée pour chacun des 96 départements (à partir des mêmes coordonnées géographiques que les contours), clic sur un département -> zoom animé (450ms, easing) sur son étendue avec un padding ; bouton "Vue France entière" pour revenir. Filtre par profession et couleurs restent actifs pendant le zoom (aucune donnée dupliquée, juste le `viewBox` du SVG qui change).

**Sorties** : `sante/data/processed/dept_bbox.json`. Rapport mis à jour (v7).

---

## 2026-09-16 — Taille des points recalculée localement au zoom

**Remarque** : en zoomant sur un département, les points gardaient la même échelle de taille que sur la carte nationale — un petit département rural, écrasé par le maximum national (Toulouse/Paris), affichait des points quasi tous identiques, sans info utile.

**Correctif** : ajout du département de chaque commune dans les données embarquées (`DATA.dept`, parallèle à `DATA.rows`). Au zoom sur un département, le maximum utilisé pour l'échelle des rayons (racine carrée) est recalculé **uniquement parmi les communes de ce département**, pour cette catégorie de profession filtrée — recalculé aussi si on change de filtre pendant qu'on est zoomé, et repli sur le maximum national si le zoom retourne à la vue France entière.

**Script** : `sante/scripts/05_carte_filtrable.py` mis à jour pour inclure le département par commune. Rapport mis à jour (v8).

---

## 2026-09-16 — Points énormes au zoom : rayon fixé en unités carte, pas en pixels écran

**Signalement** : après le correctif précédent, des points énormes envahissaient l'écran une fois zoomé.

**Diagnostic** : le vrai bug n'était pas l'échelle relative (corrigée avant) mais l'échelle absolue — le rayon (55 à 340) est exprimé en unités du repère SVG, pas en pixels. Ces unités restaient fixes alors que le cadre de vue (`viewBox`) rétrécit fortement au zoom (~15 656 unités de large pour la France entière, ~400 pour un petit département) — un rayon de 340 unités qui faisait ~14px à l'échelle France devient énorme rapporté à un cadre 40x plus petit, exactement comme les contours des départements grossissent au zoom (ce qui est voulu, pour eux), sauf que pour des points ce n'est pas le comportement souhaité.

**Correctif** : rayon rendu proportionnel au niveau de zoom (rapport entre la largeur du cadre de vue ciblé et la largeur de base) — les points gardent une taille stable à l'écran quel que soit le niveau de zoom, comme les marqueurs sur une carte interactive classique (Google Maps, Leaflet), au lieu de suivre le grossissement du fond de carte.

Rapport mis à jour (v9).

---

## 2026-09-16 — Nom de commune dans l'info-bulle + carte de détail au clic

**Demande 1** : afficher le nom de la commune sur les points, pas juste le nombre de praticiens.

**Action** : nom de commune ajouté aux données embarquées (`DATA.nom`, depuis `communes_coords.csv`), info-bulle changée de "12 médecins" à "Saint-Étienne : 12 médecins".

**Demande 2** : pour une grande ville avec plusieurs établissements, une carte de détail plutôt qu'un simple survol.

**Limite précisée à l'utilisateur** : les données actuelles sont agrégées par commune et par profession, pas par établissement individuel (cabinet/clinique précis) — cette information existe dans le fichier source RPPS (colonnes "Raison sociale site", adresse) mais a été écartée lors du nettoyage initial pour rester léger en mémoire. La carte de détail montre donc la répartition par profession dans la commune, pas la liste des établissements physiques.

**Réalisé** : carte qui s'ouvre au clic sur un point (pas juste au survol) — nom de la commune, effectif total, puis la répartition des 29 professions présentes dans cette commune, triée par effectif décroissant, avec une mini-barre proportionnelle par ligne. Bouton de fermeture. Utilise les données déjà embarquées (toutes les catégories par commune étaient déjà là pour le filtre), aucune nouvelle donnée à charger.

Rapport mis à jour (v10).
