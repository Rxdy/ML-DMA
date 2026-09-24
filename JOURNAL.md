# Journal de bord — ML-DMA

**Structure du projet** : le dépôt ne contient que le projet déchets (`dechets/`). Ce journal, à la racine, retrace le raisonnement au fil de l'eau.

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

> Note du 2026-09-24 : le dossier `menages/` a depuis été retiré du dépôt ; le script 07 et le fichier de revenu ont été rapatriés dans `dechets/`.

**Demande** : séparer le projet en deux dossiers — un pour les déchets, un pour les ménages — pour refaire l'exercice complet (exploration → nettoyage → ML) une seconde fois, sur les données de revenu cette fois, indépendamment du premier.

**Action** : restructuration —
- `dechets/` reçoit tout l'exercice 1 : `data/raw`, `data/processed`, `data/figures`, `scripts/01` à `06` + `08` (l'expérience revenu, qui reste ici car sa cible est le tonnage déchets), `rapport/rapport_dma.html`.
- `menages/` reçoit `data/raw/insee_revenu_median_dep_2013_2021.csv` et `scripts/07_income_prep.py` (l'assemblage des 5 fichiers Filosofi) — c'est le point de départ du 2ᵉ exercice, exploration/nettoyage/ML restant à faire dessus en tant que sujet à part entière (pas juste une feature pour les déchets).

Chemins des scripts mis à jour (chaque script tourne désormais depuis la racine de son propre dossier), pipeline complet des deux dossiers rejoué pour vérifier — tout fonctionne. `JOURNAL.md` reste unique à la racine, partagé entre les deux exercices.

---

## 2026-09-24 — Consignes du rendu (1ère note du module)

**Consignes du professeur** : reprendre les notebooks des ateliers 1 et 2 (versions mises à jour sur Teams) et finaliser un travail d'**apprentissage supervisé complet**, qui constitue la première note du module. Il faut savoir justifier chaque choix : préparation des données, modèle utilisé, métriques d'évaluation, interprétation des résultats.

Les notebooks mis à jour ajoutent : un prétraitement découpé plus finement, une comparaison de modèles (de la baseline aux modèles plus élaborés), une évaluation plus détaillée, et une première introduction à la classification.

L'après-midi, on passe à l'apprentissage non supervisé.

---

## 2026-09-24 — Doublons : contrôle à deux niveaux

**Demande** : réfléchir aux doublons potentiels et montrer dans le rendu qu'ils ont été pris en compte.

**Constat** : jusqu'ici, seul `sinoe_dma.csv` était contrôlé, et seulement sur les lignes strictement identiques (`df.duplicated()`). Ça ne détecte pas le cas le plus dangereux : une **clé répétée** avec des valeurs différentes (double saisie corrigée).

**Vérifié** (0 doublon partout) :
- `sinoe_dma.csv` : clé année × département × type de déchet × type de traitement (14 660 lignes).
- Chiffres-clés hors/avec gravats : clé département × année (800 / 701 lignes).
- Revenu INSEE : clé département × année (491 lignes).
- Clusters : 1 ligne par département (101).
- Train/test : 0 couple département × année commun aux deux jeux.

**Ajouté au code** (le contrôle est prouvé à chaque exécution, pas seulement affirmé) :
- `01_explore.py` : doublons de clé sur les 3 fichiers SINOE.
- `03_regression_prep.py` : `assert` sur l'unicité département × année (sinon le lag `shift(1)` lirait la ligne dupliquée comme « enquête précédente », soit une fuite de la cible), jointure `validate="many_to_one"`, nombre de lignes contrôlé avant et après.
- `05_split_scale.py` : `assert` qu'aucun couple département × année n'est à la fois en train et en test.
- `08_income_experiment.py` : jointure `validate="one_to_one"`.

Pipeline rejoué en entier : résultats identiques (R² 0,715 en régression linéaire).

**Précision pour le rendu** : un même département apparaît une fois par enquête. Ce n'est pas un doublon mais une donnée de panel, qui permet justement de calculer le lag.

**Rapport** : encart ajouté en section 02, nouvelle étape 3 « Contrôle des doublons aux jointures » en section 03 (les étapes passent de 8 à 9).

---

## 2026-09-24 — Preprocessor, pipeline et compte rendu

**Demande du professeur** : structurer la préparation en un **preprocessor** (l'étape de transformation) et une **pipeline** (la chaîne complète preprocessor → modèle).

**Script** : `dechets/scripts/09_pipeline.py`.
- Preprocessor = `ColumnTransformer` : `MinMaxScaler` sur les variables numériques, `OneHotEncoder` sur `cluster`.
- Pipeline = preprocessor + modèle. Trois modèles comparés à preprocessor identique : `DummyRegressor` (moyenne), régression linéaire, Random Forest, plus la baseline naïve hors pipeline.
- Validation croisée temporelle sur le train (fenêtre croissante, 4 plis : valide 2013 → 2019), en plus du test final.

**Défaut corrigé au passage** : dans les scripts 05/06, `cluster` (0-3) était mis à l'échelle comme un nombre, ce qui lui donnait un faux ordre. C'est une catégorie : il est désormais encodé en one-hot. Effet sur les scores quasi nul (R² test 0,715 → 0,716), ce qui est cohérent avec le poids très faible du cluster.

**Résultats** : validation croisée R² 0,865 (linéaire) / 0,845 (RF) ; test R² 0,716 (linéaire) / 0,705 (RF) / 0,702 (baseline naïve) / −0,010 (moyenne).

**Question : pourquoi le score chute-t-il entre la validation croisée et le test ?** Vérifié année par année : la baseline naïve chute aussi (R² 0,82-0,93 sur 2011-2019, 0,81 en 2021, 0,53 en 2023 ; ratio moyen 563 en 2021 puis 519 en 2023). C'est une rupture dans les données de la période de test, pas du surapprentissage.

**Livrable** : `dechets/COMPTE_RENDU.md` (objectif, sommaire, découpage en scripts, démarche complète, lien vers le dépôt GitHub) et un `README.md` à la racine du dépôt.

---

## 2026-09-24 — Correction : aucun modèle ne bat réellement la baseline naïve

**Question posée** : « avant, le modèle était au niveau de la baseline ; maintenant, il y a une différence notable ? Et le meilleur modèle, c'est le ML ? »

**Erreur repérée** : la validation croisée ajoutée au script 09 ne comparait pas les modèles à la baseline naïve. Le compte rendu concluait donc à tort que la régression linéaire était « la meilleure en validation croisée comme sur le test ».

**Vérifié** :
- Validation croisée : baseline naïve R² 0,872 / MAE 18,68, **meilleure** que la régression linéaire (0,865 / 19,55) et la Random Forest (0,845 / 20,88).
- Test : régression linéaire MAE 35,75 contre 36,49 pour la baseline naïve. Gain de 0,74 kg/hab, IC 95 % par bootstrap de 0,14 à 1,34 : réel statistiquement, négligeable en pratique (2 % de l'erreur). Meilleure sur seulement 54 % des départements-années.
- La régression linéaire a réappris la persistance : `ratio ≈ 0,97 × ratio précédent + 17`.

**Réponse** : rien n'a changé sur le fond. La « différence notable » n'existe que face à la nouvelle baseline moyenne (`DummyRegressor`, R² ≈ 0), qui est un plancher, pas le vrai repère. Face à la baseline naïve, le constat du 16/09 tient toujours.

**Corrigé** : script 09 (baseline naïve dans la validation croisée + bootstrap du gain), sections 9 et 10 du compte rendu. Ajout de `requirements.txt` (versions exactes) ; pipeline complet rejoué depuis une copie propre du dépôt.

---

## 2026-09-24 — Rapport chronologique en PDF

**Demande** : un rapport PDF structuré dans l'ordre chronologique du projet, avec une page de garde (membres du groupe, nom du projet, établissement, module, date, lien GitHub) et un sommaire.

**Réalisé** : `dechets/rapport/rapport_ML_dechets.pdf` (20 pages), généré par `dechets/rapport/generer_rapport_pdf.py` (reportlab). Les figures sont recalculées à partir des données et de la pipeline, pour rester cohérentes avec le code.

**Plan** : introduction et chronologie → 1. choix du thème → 2. source principale SINOE → 3. consolidation → 4. choix de la cible → 5. nettoyage → 6. suppression des données inutiles → 7. mise en forme → 8. baseline → 9. premiers modèles → 10. preprocessor et pipeline → 11. évaluation → 12. problèmes rencontrés et conclusion → annexes (clustering, dépôt).

---

## 2026-09-24 — Tri du dépôt, Makefile, analyse des erreurs et prédiction 2025

**Tri du dépôt** : seul le projet déchets reste publié. `sante/` et `menages/` sont retirés du dépôt (conservés en local, ignorés par git). Le script 07 et le fichier de revenu INSEE sont rapatriés dans `dechets/`. Les entrées du journal sur l'exercice santé sont déplacées dans un journal local séparé.

**Nouveaux scripts** :
- `10_detail_metriques.py` : MAE, RMSE et R² calculés pas à pas (sommes des erreurs, SS_res, SS_tot), pli par pli, année par année, coefficients de la régression.
- `11_analyse_erreurs.py` : l'erreur est surtout **commune à tous les départements** (sous-estimation de 22 kg/hab en 2021, surestimation de 43 kg/hab en 2023, quand 97 % des départements baissent). Si l'on connaissait l'évolution nationale, la MAE passerait de 36,5 à 19,9 et le R² de 0,70 à 0,88. Piste prioritaire : une information qui varie dans le temps, d'abord nationale. Allers-retours suspects à vérifier (Eure-et-Loir, Territoire-de-Belfort) ; biais fort en Corse (hypothèse : population touristique non comptée).
- `12_prediction_2025.py` : modèle réentraîné sur 2011–2023, prédiction de l'enquête 2025 (moyenne 518,0 kg/hab), à ± 50,4 kg/hab (80 % des erreurs observées sur le test).

**Question des données simulées (mocks)** : utiles pour tester le code, pas pour évaluer le modèle (score circulaire). La vérification « terrain » est le test sur 2021 et 2023, des années réelles jamais vues.

**Makefile** à la racine : une commande par étape (`make supervise`, `make tout`…), guide dans le README et en annexe C du rapport.

**Rapport PDF** : 31 pages. Ajouts : chapitre X et y, formules et calculs détaillés (annexe B), chapitre 13 (tester le modèle et prédire 2025), chapitre 14 (comment améliorer le modèle), guide d'utilisation (annexe C).
