# Compte rendu — Apprentissage supervisé : prédire les déchets ménagers par département

**Dépôt GitHub : [github.com/Rxdy/ML-DMA](https://github.com/Rxdy/ML-DMA)** (dossier [`dechets/`](https://github.com/Rxdy/ML-DMA/tree/main/dechets))

Ce document présente l'objectif du projet, son découpage et la logique suivie de bout en bout : récupération des données, préparation, choix de la cible, modèles, évaluation et interprétation. Le raisonnement détaillé, étape par étape et avec les hésitations, est tracé au fil de l'eau dans le [journal de bord](../JOURNAL.md). Le [rapport HTML](rapport/rapport_dma.html) reprend les mêmes résultats avec des graphiques.

---

## Sommaire

1. [Objectif du projet](#1-objectif-du-projet)
2. [Découpage du projet](#2-découpage-du-projet)
3. [Récupération des données](#3-récupération-des-données)
4. [Exploration](#4-exploration)
5. [Choix de la cible](#5-choix-de-la-cible)
6. [Préparation des données](#6-préparation-des-données)
7. [Preprocessor et pipeline](#7-preprocessor-et-pipeline)
8. [Modèles comparés](#8-modèles-comparés)
9. [Métriques et résultats](#9-métriques-et-résultats)
10. [Interprétation](#10-interprétation)
11. [Limites et pistes](#11-limites-et-pistes)
12. [Reproduire les résultats](#12-reproduire-les-résultats)

---

## 1. Objectif du projet

Prédire la quantité de **déchets ménagers et assimilés (DMA) produite par habitant** (kg/habitant) dans chaque département français, pour la prochaine enquête, à partir de ce que l'on sait des enquêtes précédentes.

Il s'agit d'un problème de **régression supervisée** : la cible est une valeur numérique continue, connue pour les années passées, qui sert d'exemple au modèle.

Un volet non supervisé (clustering des départements selon leur façon de traiter les déchets) a été mené en parallèle. Son résultat est réutilisé ici comme variable d'entrée.

## 2. Découpage du projet

Chaque étape est un script numéroté, exécuté dans l'ordre depuis le dossier `dechets/`. Chaque script lit les sorties du précédent : on peut rejouer n'importe quelle étape de façon isolée.

| Étape | Script | Rôle | Sortie |
|---|---|---|---|
| Exploration | [`01_explore.py`](scripts/01_explore.py) | Dimensions, types, valeurs manquantes, doublons, valeurs aberrantes | affichage console |
| Clustering | [`02_clustering_profil_traitement.py`](scripts/02_clustering_profil_traitement.py) | Profil de traitement de chaque département → 4 clusters (KMeans) | `processed_clusters_traitement.csv` |
| Préparation | [`03_regression_prep.py`](scripts/03_regression_prep.py) | Jointure avec le cluster, variables d'historique (lag) | `processed_regression_dataset.csv` |
| Corrélations | [`04_correlation.py`](scripts/04_correlation.py) | Matrice de corrélation, choix des variables | `correlation_matrix.png` |
| Split + scaling | [`05_split_scale.py`](scripts/05_split_scale.py) | Découpage train/test temporel, Min-Max (version décomposée à la main) | `train_scaled.csv`, `test_scaled.csv` |
| Modèles | [`06_train_model.py`](scripts/06_train_model.py) | Baseline, régression linéaire, Random Forest (version décomposée à la main) | `model_comparison.csv` |
| Expérience | [`08_income_experiment.py`](scripts/08_income_experiment.py) | Test d'une variable supplémentaire : le revenu médian | affichage console |
| **Pipeline** | [`09_pipeline.py`](scripts/09_pipeline.py) | **Chaîne complète preprocessor → modèle, validation croisée, évaluation** | `model_comparison_pipeline.csv` |
| Détail des calculs | [`10_detail_metriques.py`](scripts/10_detail_metriques.py) | MAE, RMSE et R² pas à pas, pli par pli, année par année | `detail_*.csv` |
| Analyse des erreurs | [`11_analyse_erreurs.py`](scripts/11_analyse_erreurs.py) | Où le modèle se trompe, et quelle information manque | `erreurs_test.csv` |

Le script [`07_income_prep.py`](scripts/07_income_prep.py) assemble les données de revenu INSEE utilisées par le script 08. Il lit des fichiers téléchargés à la main et non conservés ; son résultat est inclus dans `data/raw/`.

Chaque script se lance aussi par une commande `make` depuis la racine du dépôt (voir le [README](../README.md) : `make supervise`, `make all`…).

Les scripts 05 et 06 montrent chaque étape de façon explicite (scaler calibré, puis modèle entraîné). Le script 09 refait la même chose avec les outils scikit-learn prévus pour ça (`ColumnTransformer` et `Pipeline`). C'est la version de référence.

Arborescence :

```
dechets/
├── data/
│   ├── raw/          fichiers téléchargés, jamais modifiés
│   ├── processed/    fichiers générés par les scripts
│   └── figures/      graphiques
├── scripts/          01 → 11
├── rapport/          rapport HTML (graphiques)
└── COMPTE_RENDU.md   ce document
```

## 3. Récupération des données

Toutes les sources sont publiques (data.gouv.fr, licence Ouverte 2.0).

| Fichier | Producteur | Lignes | Période | Utilisation |
|---|---|---|---|---|
| `sinoe_dma.csv` | ADEME (SINOE) | 14 660 | 2009–2021 | Tonnage par département × type de déchet × type de traitement → **clustering** |
| `sinoe_chiffres_cles_hors_gravats.csv` | ADEME (SINOE) | 800 | 2009–2023 | Tonnage, population, ratio kg/hab → **base de la régression** |
| `sinoe_chiffres_cles_avec_gravats.csv` | ADEME (SINOE) | 701 | 2009–2021 | Contrôle croisé des populations (0 écart), puis écarté |
| `insee_revenu_median_dep_2013_2021.csv` | INSEE (Filosofi) | 491 | 2013–2021 | Test d'une variable explicative supplémentaire (script 08) |

**Pourquoi le fichier « hors gravats » comme base :**
- c'est le seul qui va jusqu'en 2023, soit une enquête de plus ;
- les gravats viennent surtout des chantiers (BTP), pas du comportement des ménages. Ils ajoutent de la variabilité sans lien avec la population : l'écart-type du ratio est de 100,6 kg avec gravats contre 80,3 kg sans.

## 4. Exploration

Ce qu'on a vérifié et ce qu'on a trouvé :

- **Valeurs manquantes : aucune** dans les fichiers sources.
- **Doublons : aucun, recherchés à deux niveaux.**
  - Lignes strictement identiques.
  - **Clés répétées**, cas plus dangereux : une même combinaison déclarée deux fois avec des valeurs différentes, que `duplicated()` seul ne voit pas. On a vérifié l'unicité de la clé année × département × type de déchet × type de traitement pour `sinoe_dma.csv`, et département × année pour les autres fichiers.
  - Un même département apparaît bien une fois par enquête : ce n'est pas un doublon, ce sont des observations successives du même territoire (données de panel).
- **Années impaires uniquement** (2009, 2011, … 2023). Ce n'est pas un trou à combler : l'enquête SINOE est biennale par construction.
- **54 lignes à tonnage = 0**, aucune négative : absence réelle de tonnage pour une combinaison donnée, conservée.

**Corrélations** avec la cible (`RATIO_DMA`) :

| Variable | Corrélation | Lecture |
|---|---|---|
| `RATIO_DMA_lag1` (ratio à l'enquête précédente) | 0,91 | signal très fort : un département change peu en 2 ans |
| `VA_POPANNEE` (population) | −0,17 | signal faible : les départements peuplés produisent un peu moins par habitant |
| `cluster` (profil de traitement) | 0,13 | signal faible |
| `ANNEE` | 0,02 | quasi nul : l'année seule n'explique rien |

## 5. Choix de la cible

**Cible retenue : `RATIO_DMA` (kg/habitant), et non le tonnage brut `TONNAGE_DMA`.**

La corrélation entre le tonnage et la population vaut 0,97. Prédire le tonnage reviendrait à réapprendre une multiplication (ratio × population). Le modèle aurait l'air excellent sans avoir rien appris d'utile. Ramener la cible par habitant retire cet effet de taille, et le modèle doit alors expliquer les vraies différences de comportement entre départements.

## 6. Préparation des données

Chaque étape répond à un constat de l'exploration.

1. **Harmonisation des formats.** Les fichiers « chiffres-clés » utilisent `;` et la décimale `,` (convention Excel française), contrairement au fichier principal. Sans harmonisation, les nombres sont lus comme du texte.
2. **Codes département sur 2 caractères** (`1` → `01`). Sans ça, la jointure entre fichiers perd des départements sans aucun message d'erreur.
3. **Contrôle des doublons aux jointures.**
   - Unicité de la clé vérifiée par `assert`.
   - Jointures déclarées avec `validate="many_to_one"` ou `validate="one_to_one"`.
   - Nombre de lignes comparé avant et après chaque jointure.
   - Une clé en double multiplierait les lignes en silence. Elle casserait aussi le lag : l'« enquête précédente » serait la ligne elle-même, soit une fuite de la cible.
4. **Profil de traitement en %** (pour le clustering). On calcule la part de chaque mode de traitement (incinération, stockage, recyclage…) dans le total du département, pour comparer des départements de tailles différentes.
5. **Variables d'historique (lag).** `RATIO_DMA_lag1` et `TONNAGE_DMA_lag1` prennent la valeur de l'enquête précédente du même département. Elles portent la vraie information temporelle (corrélation 0,91), contrairement à l'année brute (0,02).
6. **Suppression de l'année 2009** (101 lignes). Première enquête, donc pas d'historique disponible. Il reste 699 lignes exploitables.
7. **Découpage train/test temporel, pas aléatoire.**
   - Train : 2011–2019 (501 lignes).
   - Test : 2021 et 2023 (198 lignes), jamais vus pendant l'entraînement.
   - Un découpage aléatoire mélangerait passé et futur. Le découpage par année simule une vraie prédiction. Contrôle : aucun couple département × année n'est à la fois dans le train et dans le test.
8. **Mise à l'échelle et encodage.** Cette étape est faite par le preprocessor (section suivante).

## 7. Preprocessor et pipeline

### Le preprocessor : l'étape de transformation

Le preprocessor est un `ColumnTransformer` qui applique à chaque type de colonne la transformation adaptée :

```python
preprocessor = ColumnTransformer([
    ("num", MinMaxScaler(), ["VA_POPANNEE", "RATIO_DMA_lag1", "TONNAGE_DMA_lag1"]),
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["cluster"]),
])
```

- **Variables numériques → `MinMaxScaler`.** Chaque valeur est ramenée entre 0 et 1 : `x' = (x − min) / (max − min)`. Sans ça, la population (des centaines de milliers) écraserait le ratio (quelques centaines) dans les modèles sensibles aux échelles.
- **Variable catégorielle → `OneHotEncoder`.** Le cluster (0, 1, 2 ou 3) est une typologie, pas une quantité : « cluster 3 » n'est pas « trois fois cluster 1 ». On le transforme en 4 colonnes 0/1 (`cluster_0` … `cluster_3`).

Dans la version décomposée (script 05), le cluster était mis à l'échelle comme un nombre, ce qui introduisait un faux ordre. Le preprocessor corrige ce point.

### La pipeline : la chaîne complète

La pipeline enchaîne le preprocessor et le modèle en un seul objet :

```python
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("modele", LinearRegression()),
])
pipeline.fit(X_train, y_train)   # calibre le preprocessor PUIS entraîne le modèle, sur le train seul
pipeline.predict(X_test)         # applique les mêmes transformations PUIS prédit
```

Ce que la pipeline apporte :
- **Pas de fuite de données par construction.** Le min/max du scaler est calculé sur le train uniquement. Dans la version décomposée, c'était une règle à respecter à la main. Avec la pipeline, c'est garanti, y compris en validation croisée où le preprocessor est recalibré dans chaque pli.
- **Même traitement partout.** Les données de test (ou de nouvelles données) passent exactement par les mêmes transformations que le train.
- **Un seul objet** prend les données brutes et renvoie une prédiction. C'est la seule chose à sauvegarder pour réutiliser le modèle.
- **Comparaison équitable.** Tous les modèles reçoivent le même preprocessor : on ne change que le modèle.

## 8. Modèles comparés

Des plus simples aux plus élaborés :

| Modèle | Rôle |
|---|---|
| **Baseline moyenne** (`DummyRegressor`) | Prédit toujours la moyenne du train. Plancher absolu : un modèle qui fait moins bien n'a rien appris. |
| **Baseline naïve** | Prédit « comme à l'enquête précédente » (= `RATIO_DMA_lag1`), sans aucun modèle. C'est le vrai repère à battre, parce que la corrélation avec l'enquête précédente est de 0,91. |
| **Régression linéaire** | Modèle simple et interprétable : une combinaison pondérée des variables. |
| **Random Forest** | Ensemble de 300 arbres de décision (profondeur max 6). Capte les relations non linéaires et fournit l'importance de chaque variable. |

## 9. Métriques et résultats

**Métriques utilisées :**
- **MAE** (erreur absolue moyenne) : en moyenne, de combien de kg/habitant la prédiction se trompe. C'est la plus lisible.
- **RMSE** (racine de l'erreur quadratique moyenne) : même unité que la MAE, mais pénalise davantage les grosses erreurs.
- **R²** : part de la variation entre départements expliquée par le modèle. 1 = parfait, 0 = pas mieux que la moyenne.

**Validation croisée temporelle** sur le train (4 plis : on entraîne sur les enquêtes passées et on valide sur la suivante, 2013 → 2019) :

| Modèle | MAE | R² (± écart-type) |
|---|---|---|
| Baseline moyenne | 62,23 | −0,013 (± 0,009) |
| **Baseline naïve (enquête précédente)** | **18,68** | **0,872 (± 0,043)** |
| Régression linéaire | 19,55 | 0,865 (± 0,039) |
| Random Forest | 20,88 | 0,845 (± 0,045) |

**Évaluation finale sur le test** (2021 + 2023, jamais vus) :

| Modèle | MAE (kg/hab) | RMSE (kg/hab) | R² |
|---|---|---|---|
| Baseline moyenne | 61,99 | 82,14 | −0,010 |
| Baseline naïve (enquête précédente) | 36,49 | 44,58 | 0,702 |
| **Régression linéaire** | **35,75** | **43,53** | **0,716** |
| Random Forest | 36,08 | 44,36 | 0,705 |

**Importance des variables** (Random Forest) : `RATIO_DMA_lag1` = 94,9 %. Tout le reste (population, tonnage précédent, clusters) se partage environ 5 %.

**Le gain sur la baseline naïve est-il réel ?** Sur le test, la régression linéaire se trompe en moyenne de 0,74 kg/hab de moins que la baseline naïve. Un bootstrap (5 000 rééchantillonnages des 198 départements-années) donne un intervalle de confiance à 95 % de 0,14 à 1,34 kg/hab. Le gain est donc réel au sens statistique, mais minuscule : 2 % de l'erreur, sur un ratio moyen d'environ 520 kg/hab. La régression linéaire ne fait mieux que la baseline que sur 54 % des départements-années, à peine plus qu'un tirage à pile ou face.

**Modèle retenu : aucun modèle ne bat réellement la baseline naïve.**
- En validation croisée, la baseline naïve est la meilleure (R² de 0,872 contre 0,865 pour la régression linéaire).
- Sur le test, la régression linéaire la devance de très peu (0,716 contre 0,702).
- Aucun des deux écarts n'est significatif en pratique.

Si l'on doit retenir un modèle entraîné, c'est la régression linéaire : meilleure des deux modèles sur les deux évaluations, plus simple et plus interprétable que la Random Forest, qui fait moins bien partout avec ses 300 arbres. Mais la conclusion honnête est que la règle « comme à l'enquête précédente » fait aussi bien. La régression linéaire l'a d'ailleurs réapprise : réentraînée sur la seule variable `RATIO_DMA_lag1`, elle donne `ratio ≈ 0,97 × ratio précédent + 17`, c'est-à-dire presque la valeur précédente telle quelle.

## 10. Interprétation

**Les modèles battent largement la moyenne, mais pas la baseline naïve.** Ce n'est pas un échec de modélisation, c'est une conclusion sur les données : la production de déchets par habitant d'un département est très stable d'une enquête à l'autre. Presque toute l'information utile est déjà dans la valeur précédente. Les autres variables (population, type de traitement) sont des caractéristiques qui ne bougent pas dans le temps : elles ne peuvent pas expliquer un changement.

**Plus de lignes n'aiderait pas.** On a tracé une courbe d'apprentissage : le modèle réentraîné sur 10 %, 50 % puis 100 % du train obtient toujours un R² entre 0,70 et 0,72. La courbe est plate dès 50 lignes. La limite vient du contenu des variables (biais), pas du volume de données (variance).

**Pourquoi le score baisse entre la validation croisée (0,865) et le test (0,716) :** ce n'est pas du surapprentissage. La baseline naïve, qui n'apprend rien, chute de la même façon :

| Année | R² baseline naïve | MAE baseline naïve | Ratio moyen (kg/hab) |
|---|---|---|---|
| 2011–2019 | 0,82 à 0,93 | 15 à 21 | 525 à 540 |
| 2021 | 0,81 | 28 | 563 |
| 2023 | **0,53** | **45** | 519 |

Les années de test marquent une rupture : un pic en 2021 puis une baisse nette en 2023. « Comme la dernière fois » devient une moins bonne hypothèse, et le modèle, qui repose surtout sur cette hypothèse, suit la même baisse. L'origine de cette rupture n'est pas identifiable avec nos données. Deux hypothèses possibles, non vérifiées : l'effet de la période Covid sur la consommation, et l'extension des consignes de tri.

**Une hypothèse testée et écartée : le revenu.** L'ajout du revenu médian INSEE (script 08) n'améliore pas la prédiction : R² de 0,885 avec contre 0,887 sans, sur la période 2015–2021 commune aux deux sources. Le niveau de revenu n'explique pas les écarts d'évolution entre départements.

## 11. Limites et pistes

- **Peu de points dans le temps** : 8 enquêtes seulement, une tous les 2 ans. On n'a aucune donnée pour les années paires.
- **Variables trop stables** : pour dépasser la baseline, il faudrait des variables qui changent dans le temps et au niveau du département. Par exemple la mise en place de la tarification incitative, l'extension des consignes de tri, ou l'évolution de la part de population urbaine.
- **Périmètres légèrement différents** : le clustering inclut les gravats (`sinoe_dma.csv`), la cible de régression les exclut. Cet écart est documenté et assumé.

## 12. Reproduire les résultats

```bash
git clone https://github.com/Rxdy/ML-DMA.git
cd ML-DMA
make installer
make all
```

Les fichiers de `data/raw/` sont inclus dans le dépôt : aucun téléchargement n'est nécessaire. Le rapport PDF complet ([`rapport/rapport_ML_dechets.pdf`](rapport/rapport_ML_dechets.pdf)) détaille en plus X et y, les calculs des métriques, l'analyse des erreurs et les pistes d'amélioration.
