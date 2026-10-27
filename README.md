# Les déchets des départements français : prédire et segmenter

Projet de machine learning sur les déchets ménagers et assimilés (DMA) des départements français, à partir de données publiques de l'ADEME (base SINOE) et de l'INSEE.

Deux questions, deux approches :

| | Question | Approche | Résultat principal |
|---|---|---|---|
| **Supervisé** | Peut-on prédire les kg de déchets par habitant d'un département à la prochaine enquête ? | Régression : baseline, régression linéaire, arbre de décision, Random Forest | R² de 0,716 sur le test, contre 0,702 pour la règle « comme à l'enquête précédente » : la production change très peu d'une enquête à l'autre |
| **Non supervisé** | Les départements se ressemblent-ils par ce qu'ils jettent ? | K-means (k = 4), PCA, clustering hiérarchique, DBSCAN | 4 profils géographiques : Grand Ouest, grandes agglomérations, Méditerranée et Corse, France intermédiaire |

Projet iRUP Alternance : ALVES Rudy, ANTHONY Josselin, BENASSIE Noé, TAVERNIER Florian.

## Par où commencer

| Je veux… | Je lis |
|---|---|
| Présenter le projet à l'oral (30 min) | [`dechets/presentation/presentation.pptx`](dechets/presentation/presentation.pptx), avec une version PDF ; les notes de l'orateur sont dans le fichier source `presentation.md` |
| Comprendre le projet en 16 pages | [`dechets/rapports/synthese_projet.pdf`](dechets/rapports/synthese_projet.pdf) |
| Rejouer le projet sans rien installer | [le notebook dans Google Colab](https://colab.research.google.com/github/Rxdy/ML-DMA/blob/main/dechets/notebooks/projet_dechets.ipynb) |
| Le détail du volet supervisé | [`dechets/rapports/rapport_supervise.pdf`](dechets/rapports/rapport_supervise.pdf) et sa [synthèse](dechets/docs/supervise.md) |
| Le détail du clustering | [`dechets/rapports/rapport_clustering.pdf`](dechets/rapports/rapport_clustering.pdf) et sa [synthèse](dechets/docs/clustering.md) |
| Suivre le raisonnement pas à pas, avec les hésitations | [`docs/JOURNAL.md`](docs/JOURNAL.md) |

## Organisation du dépôt

```
.
├── README.md
├── Makefile                  commandes pour tout rejouer (make aide)
├── requirements.txt          dépendances Python
├── docs/
│   └── JOURNAL.md            journal de bord : décisions et résultats, jour par jour
└── dechets/
    ├── data/
    │   ├── raw/              données sources téléchargées, jamais modifiées
    │   ├── processed/        tableaux produits par les scripts (CSV)
    │   └── figures/          graphiques produits par les scripts (PNG)
    ├── scripts/              01 à 14, à exécuter dans l'ordre
    ├── notebooks/            notebook Colab / Jupyter qui rejoue tout
    ├── docs/                 synthèses écrites (supervisé, clustering)
    ├── presentation/         diaporama de 30 minutes (source Marp, PowerPoint, PDF)
    └── rapports/             rapports PDF (synthèse, supervisé, clustering)
```

Les scripts sont numérotés et chacun lit la sortie du précédent : on peut en rejouer un seul. Les données brutes sont dans le dépôt, il n'y a rien à télécharger.

| Étapes | Scripts | Contenu |
|---|---|---|
| Explorer | 01 | Dimensions, valeurs manquantes, doublons |
| Préparer | 02 à 05, 07 | Profil de traitement (k-means), jointures, variables d'historique, corrélations, découpage train/test |
| Prédire et évaluer | 06, 08 à 11 | Modèles, test du revenu médian, pipeline avec validation croisée, détail des métriques, analyse des erreurs (carte comprise) |
| Segmenter | 12 à 14 | Variables, coude et silhouette, 4 groupes, PCA, clustering hiérarchique, DBSCAN |

## Lancer le projet

### Option 1 : Google Colab (rien à installer)

[![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rxdy/ML-DMA/blob/main/dechets/notebooks/projet_dechets.ipynb)

Ouvrir le lien, puis *Exécution > Tout exécuter*. Le notebook télécharge le dépôt et rejoue chaque étape avec ses explications : partie 1 supervisée, partie 2 clustering (à partir de la section 11). Il fonctionne aussi dans Jupyter en local.

### Option 2 : Linux (testé sur Ubuntu 22.04 et 24.04)

```bash
sudo apt update && sudo apt install -y git make python3 python3-venv
git clone https://github.com/Rxdy/ML-DMA.git
cd ML-DMA
make installer      # crée .venv et installe les dépendances
make all            # rejoue tout le projet
```

Python 3.10 ou plus récent est requis. `make` seul affiche la liste des commandes ; si `python3-venv` manque, `make installer` l'indique. Les résultats s'affichent dans la console et s'enregistrent dans `dechets/data/processed/` (tableaux) et `dechets/data/figures/` (graphiques).

| Commande | Script | Ce qu'elle fait |
|---|---|---|
| `make explorer` | 01 | Exploration : dimensions, valeurs manquantes, doublons |
| `make clustering` | 02 | Profil de traitement et k-means |
| `make preparation` | 03 | Jointure avec le cluster, variables d'historique |
| `make correlation` | 04 | Matrice de corrélation |
| `make decoupage` | 05 | Découpage train/test et mise à l'échelle (version manuelle) |
| `make modeles` | 06 | Baseline et premiers modèles (version manuelle) |
| `make revenu` | 08 | Test du revenu médian |
| `make pipeline` | 09 | Preprocessor, pipeline, validation croisée, test |
| `make details` | 10 | Détail des calculs de MAE, RMSE et R² |
| `make erreurs` | 11 | Analyse des erreurs et carte par département |
| `make supervise` | 09 à 11 | Volet supervisé complet |
| `make profils` | 12 | Clustering : variables, coude, silhouette, 4 groupes, carte |
| `make analyses-profils` | 13 | Corrélations, PCA (PC1, PC2, PC3), silhouette par département, robustesse |
| `make autres-methodes` | 14 | Clustering hiérarchique (dendrogramme) et DBSCAN |
| `make non-supervise` | 12 à 14 | Volet clustering complet |
| `make all` | 01 à 14 | Rejoue tout le projet |

## Modifier la présentation

Le diaporama est écrit en [Marp](https://marp.app) (Markdown) dans [`dechets/presentation/presentation.md`](dechets/presentation/presentation.md) ; les figures viennent de `dechets/data/figures/`. Le PowerPoint généré contient une image par slide (rendu identique au PDF) : pour modifier un texte, on change `presentation.md` puis on régénère. Commandes :

```bash
cd dechets/presentation
npx @marp-team/marp-cli presentation.md --allow-local-files --pptx -o presentation.pptx            # PowerPoint (une image par slide)
npx @marp-team/marp-cli presentation.md --allow-local-files -o presentation.pdf                    # PDF
```

## Données

Toutes les sources sont publiques (data.gouv.fr, licence Ouverte 2.0).

| Fichier (`dechets/data/raw/`) | Producteur | Période | Utilisation |
|---|---|---|---|
| `sinoe_dma.csv` | ADEME (SINOE) | 2009–2021 | Tonnage par département, type de déchet et traitement |
| `sinoe_chiffres_cles_hors_gravats.csv` | ADEME (SINOE) | 2009–2023 | Base de la régression |
| `sinoe_chiffres_cles_avec_gravats.csv` | ADEME (SINOE) | 2009–2021 | Contrôle des populations |
| `insee_revenu_median_dep_2013_2021.csv` | INSEE (Filosofi) | 2013–2021 | Test du revenu médian |
| `insee_population_dep_sexe_gca.xls` | INSEE | 1975–2023 | Piste de secours, non utilisée par les scripts |
| `contours-departements.geojson` | | | Fond de carte des départements |
