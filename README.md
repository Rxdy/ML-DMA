# Machine learning : Déchets

Prédire la production de déchets ménagers par habitant et par département, à partir des données publiques de l'ADEME (SINOE) et de l'INSEE.

Projet d'apprentissage supervisé et non supervisé, iRUP Alternance — ALVES Rudy, ANTHONY Josselin, BENASSIE Noé, TAVERNIER Florian.

- **Rapport apprentissage supervisé (PDF)** : [`dechets/rapport/rapport_ML_dechets.pdf`](dechets/rapport/rapport_ML_dechets.pdf) — synthèse : [`dechets/COMPTE_RENDU.md`](dechets/COMPTE_RENDU.md)
- **Rapport clustering (PDF)** : [`dechets/rapport/rapport_ML_dechets_clustering.pdf`](dechets/rapport/rapport_ML_dechets_clustering.pdf) — synthèse : [`dechets/CLUSTERING.md`](dechets/CLUSTERING.md)
- **Journal de bord** : [`JOURNAL.md`](JOURNAL.md)

## Lancer le projet

### Option 1 : Google Colab (rien à installer)

[![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rxdy/ML-DMA/blob/main/dechets/notebooks/projet_dechets.ipynb)

Ouvrir le lien, puis *Exécution > Tout exécuter*. Le notebook télécharge le dépôt et rejoue chaque étape avec ses explications : partie 1 supervisée, partie 2 clustering (à partir de la section 11). Il fonctionne aussi dans Jupyter en local : `dechets/notebooks/projet_dechets.ipynb`.

### Option 2 : Linux (testé sur Ubuntu 22.04 et 24.04)

```bash
sudo apt update && sudo apt install -y git make python3 python3-venv
git clone https://github.com/Rxdy/ML-DMA.git
cd ML-DMA
make installer      # crée .venv et installe les dépendances
make all            # rejoue tout le projet (supervisé et clustering)
```

`make` seul affiche la liste des commandes. Python 3.10 ou plus récent est requis ; si le paquet `python3-venv` manque, `make installer` l'indique.

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
| `make erreurs` | 11 | Analyse des erreurs |
| `make supervise` | 09 à 11 | Volet supervisé complet |
| `make profils` | 12 | Clustering : variables, coude, silhouette, 4 groupes, carte |
| `make analyses-profils` | 13 | Corrélations, ACP, silhouette par département, robustesse |
| `make non-supervise` | 12 et 13 | Volet clustering complet |
| `make all` | 01 à 13 | Rejoue tout le projet |

Les résultats sont affichés dans la console et enregistrés en CSV dans `dechets/data/processed/`.
