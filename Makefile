# Lancer le projet déchets pas à pas.  `make` ou `make aide` affiche la liste des commandes.
# Toutes les commandes s'exécutent dans le dossier dechets/ avec l'environnement .venv.

PY := $(abspath .venv/bin/python)
RUN := cd dechets && $(PY)

.DEFAULT_GOAL := aide
.PHONY: aide installer explorer clustering preparation correlation decoupage modeles revenu \
        pipeline details erreurs supervise all

aide: ## Affiche cette aide
	@echo "Commandes disponibles :"
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*## "}{printf "  make %-13s %s\n", $$1, $$2}'

installer: ## Crée l'environnement .venv et installe les dépendances
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt

explorer: ## 01 - Exploration : dimensions, valeurs manquantes, doublons
	$(RUN) scripts/01_explore.py

clustering: ## 02 - Profil de traitement et k-means (4 groupes)
	$(RUN) scripts/02_clustering_profil_traitement.py

preparation: clustering ## 03 - Jointure avec le cluster, variables d'historique
	$(RUN) scripts/03_regression_prep.py

correlation: preparation ## 04 - Matrice de corrélation
	$(RUN) scripts/04_correlation.py

decoupage: preparation ## 05 - Découpage train/test par année et mise à l'échelle
	$(RUN) scripts/05_split_scale.py

modeles: decoupage ## 06 - Baseline et premiers modèles (version manuelle)
	$(RUN) scripts/06_train_model.py

revenu: preparation ## 08 - Le revenu médian améliore-t-il la prédiction ?
	$(RUN) scripts/08_income_experiment.py

pipeline: preparation ## 09 - Preprocessor + pipeline, validation croisée, test
	$(RUN) scripts/09_pipeline.py

details: preparation ## 10 - Détail des calculs de MAE, RMSE et R²
	$(RUN) scripts/10_detail_metriques.py

erreurs: preparation ## 11 - Analyse des erreurs du modèle
	$(RUN) scripts/11_analyse_erreurs.py

supervise: pipeline details erreurs ## Le volet supervisé complet (09 à 11)

all: explorer correlation modeles revenu supervise ## Rejoue tout le projet, de 01 à 11
