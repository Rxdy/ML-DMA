# Machine learning : Déchets

Prédire la production de déchets ménagers par habitant et par département, à partir des données publiques de l'ADEME (SINOE) et de l'INSEE.

Projet d'apprentissage supervisé, iRUP Alternance — ALVES Rudy, ANTHONY Josselin, BENASSIE Noé, TAVERNIER Florian.

- **Rapport complet (PDF)** : [`dechets/rapport/rapport_ML_dechets.pdf`](dechets/rapport/rapport_ML_dechets.pdf)
- **Synthèse** : [`dechets/COMPTE_RENDU.md`](dechets/COMPTE_RENDU.md)
- **Journal de bord** : [`JOURNAL.md`](JOURNAL.md)

## Lancer le projet

Prérequis : Python 3, `make`, `git`. Les données sont incluses, rien à télécharger.

```bash
git clone https://github.com/Rxdy/ML-DMA.git
cd ML-DMA
make installer      # crée .venv et installe les dépendances
make                # liste des commandes
make supervise      # volet supervisé complet : pipeline, détail des calculs, analyse des erreurs
```

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
| `make all` | 01 à 11 | Rejoue tout le projet |

Les résultats sont affichés dans la console et enregistrés en CSV dans `dechets/data/processed/`.
