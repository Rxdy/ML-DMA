# Lancer le projet déchets pas à pas.  `make` ou `make aide` affiche la liste des commandes.
# Toutes les commandes s'exécutent dans le dossier dechets/ avec l'environnement .venv.
# Testé sur Ubuntu 22.04 et 24.04. Prérequis :  sudo apt install git make python3 python3-venv

PY := $(abspath .venv/bin/python)
RUN := cd dechets && $(PY)

.DEFAULT_GOAL := aide
.PHONY: aide installer verifier-env explorer clustering preparation correlation decoupage modeles revenu \
        pipeline details erreurs supervise profils analyses-profils autres-methodes non-supervise all

aide: ## Affiche cette aide
	@echo "Commandes disponibles :"
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*## "}{printf "  make %-17s %s\n", $$1, $$2}'

installer: ## Crée l'environnement .venv et installe les dépendances
	@command -v python3 >/dev/null || { echo "Python 3 est introuvable. Sur Ubuntu : sudo apt install python3"; exit 1; }
	@python3 -c 'import sys; sys.exit(sys.version_info < (3, 10))' || { echo "Python 3.10 ou plus récent est requis (version actuelle : $$(python3 --version))."; exit 1; }
	@python3 -c 'import ensurepip, venv' 2>/dev/null || { echo "Le module venv de Python est absent. Sur Ubuntu : sudo apt install python3-venv"; exit 1; }
	rm -rf .venv
	python3 -m venv .venv
	.venv/bin/pip install --quiet --upgrade pip
	.venv/bin/pip install --quiet -r requirements.txt
	@echo "Installation terminée. Lancez maintenant : make all"

verifier-env:
	@test -x .venv/bin/python || { echo "Environnement absent : lancez d'abord  make installer"; exit 1; }

explorer: verifier-env ## 01 - Exploration : dimensions, valeurs manquantes, doublons
	$(RUN) scripts/01_explore.py

clustering: verifier-env ## 02 - Profil de traitement et k-means (4 groupes)
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

profils: verifier-env ## 12 - Clustering : choix des variables, coude, silhouette, 4 groupes
	$(RUN) scripts/12_clustering_production.py

analyses-profils: profils ## 13 - Corrélations, PCA (PC1, PC2, PC3), silhouette, robustesse
	$(RUN) scripts/13_clustering_analyses.py

autres-methodes: profils ## 14 - Clustering hiérarchique (dendrogramme) et DBSCAN
	$(RUN) scripts/14_clustering_hierarchique_dbscan.py

non-supervise: analyses-profils autres-methodes ## Le volet non supervisé complet (12 à 14)

all: explorer correlation modeles revenu supervise non-supervise ## Rejoue tout le projet, de 01 à 14
