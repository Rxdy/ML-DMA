"""
Où et pourquoi le modèle se trompe : analyse des erreurs sur le test (2021, 2023),
pour savoir quelles informations supplémentaires pourraient l'améliorer.

Question clé : l'erreur est-elle propre à chaque département, ou commune à tous ?
  - si elle est commune, il manque une information NATIONALE qui varie dans le temps ;
  - si elle est propre au département, il manque une information LOCALE.

Sortie : data/processed/erreurs_test.csv
"""
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, r2_score

ns = {}
exec(open("scripts/09_pipeline.py").read().split("\n# --- Validation croisée")[0]
     .replace("print(", "(lambda *a, **k: None)("), ns)
test = ns["test"].copy()
lin = clone(ns["pipelines"]["Régression linéaire"]).fit(ns["X_train"], ns["y_train"])
test["prediction"] = lin.predict(ns["X_test"])
test["erreur"] = test["prediction"] - test["RATIO_DMA"]  # > 0 : le modèle surestime

print("=== Erreur moyenne par année (> 0 = le modèle surestime) ===")
print(test.groupby("ANNEE")["erreur"].agg(moyenne="mean", ecart_type="std").round(1))

# Évolution réelle entre deux enquêtes, département par département
test["evolution"] = test["RATIO_DMA"] - test["RATIO_DMA_lag1"]
for a in [2021, 2023]:
    ev = test.loc[test["ANNEE"] == a, "evolution"]
    print(f"{a} : évolution moyenne {ev.mean():+.1f} kg/hab, {(ev < 0).mean():.0%} des départements en baisse")

# Expérience de pensée : si l'on connaissait l'évolution nationale moyenne de l'année
# à prédire, et qu'on l'ajoutait à la valeur précédente de chaque département ?
nat = test.groupby("ANNEE")["evolution"].transform("mean")
y = test["RATIO_DMA"]
print("\n=== Valeur d'une information nationale ===")
for nom, pred in [("Baseline naïve", test["RATIO_DMA_lag1"]),
                  ("Régression linéaire", test["prediction"]),
                  ("Naïve + évolution nationale connue", test["RATIO_DMA_lag1"] + nat)]:
    print(f"{nom:36s} MAE={mean_absolute_error(y, pred):6.2f}  R2={r2_score(y, pred):.3f}")

print("\n=== Les 10 plus grosses erreurs ===")
top = test.assign(abs_err=test["erreur"].abs()).nlargest(10, "abs_err")
print(top[["N_DEPT", "ANNEE", "RATIO_DMA_lag1", "RATIO_DMA", "prediction", "erreur"]].round(0).to_string(index=False))

print("\n=== Erreur moyenne par région ===")
print(test.groupby("L_REGION")["erreur"].mean().round(1).sort_values().to_string())

test[["C_DEPT", "N_DEPT", "L_REGION", "ANNEE", "RATIO_DMA_lag1", "RATIO_DMA", "prediction", "erreur"]] \
    .to_csv("data/processed/erreurs_test.csv", index=False)
print("\nSauvegardé -> data/processed/erreurs_test.csv")
