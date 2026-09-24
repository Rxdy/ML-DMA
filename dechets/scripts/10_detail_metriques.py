"""
Détail de tous les calculs d'évaluation : comment on obtient MAE, RMSE et R²,
pli par pli et année par année.

  MAE  = (1/n) * Σ |y - ŷ|                   erreur moyenne, en kg/habitant
  RMSE = sqrt( (1/n) * Σ (y - ŷ)² )          pénalise davantage les grosses erreurs
  R²   = 1 - SS_res / SS_tot                  part des écarts expliquée
         SS_res = Σ (y - ŷ)²                  ce que le modèle n'explique pas
         SS_tot = Σ (y - ȳ)²                  la variabilité totale autour de la moyenne

Sorties : data/processed/detail_calcul_test.csv, detail_cv_plis.csv,
          detail_test_par_annee.csv, detail_coefficients.csv
"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

ns = {}
exec(open("scripts/09_pipeline.py").read().split("\n# --- Validation croisée")[0]
     .replace("print(", "(lambda *a, **k: None)("), ns)
train, test = ns["train"], ns["test"]
X_train, y_train, X_test, y_test = ns["X_train"], ns["y_train"], ns["X_test"], ns["y_test"]
pipelines = ns["pipelines"]


def detail(y, y_pred):
    e = y - y_pred
    ss_res = float((e ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {"n": len(y), "somme_erreurs_abs": float(e.abs().sum()), "MAE": float(e.abs().mean()),
            "somme_erreurs_carrees": ss_res, "RMSE": float(np.sqrt(ss_res / len(y))),
            "moyenne_y": float(y.mean()), "SS_tot": ss_tot, "R2": 1 - ss_res / ss_tot}


# --- 1. Calcul pas à pas sur le test, pour chaque approche ---
predictions = {"Baseline moyenne": pd.Series(y_train.mean(), index=y_test.index),
               "Baseline naïve": test["RATIO_DMA_lag1"]}
for nom in ["Régression linéaire", "Arbre de décision", "Random Forest"]:
    predictions[nom] = pd.Series(clone(pipelines[nom]).fit(X_train, y_train).predict(X_test), index=y_test.index)

lignes = [{"modele": nom, **detail(y_test, pr)} for nom, pr in predictions.items()]
calc = pd.DataFrame(lignes)
print("=== Test 2021 + 2023 : calcul pas à pas ===")
for _, r in calc.iterrows():
    print(f"\n{r['modele']}")
    print(f"  MAE  = {r['somme_erreurs_abs']:.1f} / {r['n']} = {r['MAE']:.2f} kg/hab")
    print(f"  RMSE = sqrt({r['somme_erreurs_carrees']:.0f} / {r['n']}) = {r['RMSE']:.2f} kg/hab")
    print(f"  R²   = 1 - {r['somme_erreurs_carrees']:.0f} / {r['SS_tot']:.0f} = {r['R2']:.3f}")

# Exemple sur 3 départements pour montrer d'où viennent les erreurs
ex = test.assign(pred=predictions["Régression linéaire"]).set_index("N_DEPT")
ex = ex.loc[["Ain", "Landes", "Paris"]]
ex = ex[ex["ANNEE"] == 2023][["RATIO_DMA_lag1", "RATIO_DMA", "pred"]]
ex["erreur"] = ex["RATIO_DMA"] - ex["pred"]
print("\n=== Exemple : 3 départements en 2023 (régression linéaire) ===")
print(ex.round(1))

# --- 2. Validation croisée, pli par pli ---
annees = sorted(train["ANNEE"].unique())
plis = []
for a in annees[1:]:
    tr, va = train["ANNEE"] < a, train["ANNEE"] == a
    yv = y_train[va]
    plis.append({"annee_validee": a, "n_train": int(tr.sum()), "n_valid": int(va.sum()),
                 "modele": "Baseline naïve", **detail(yv, X_train.loc[va, "RATIO_DMA_lag1"])})
    for nom in ["Baseline moyenne (DummyRegressor)", "Régression linéaire", "Arbre de décision", "Random Forest"]:
        pr = pd.Series(clone(pipelines[nom]).fit(X_train[tr], y_train[tr]).predict(X_train[va]), index=yv.index)
        plis.append({"annee_validee": a, "n_train": int(tr.sum()), "n_valid": int(va.sum()),
                     "modele": nom.replace(" (DummyRegressor)", ""), **detail(yv, pr)})
cv = pd.DataFrame(plis)
print("\n=== Validation croisée, pli par pli ===")
print(cv.pivot(index="annee_validee", columns="modele", values="R2").round(3))
print(cv.groupby("modele")[["MAE", "RMSE", "R2"]].mean().round(3))

# --- 3. Test, année par année ---
par_annee = []
for a in [2021, 2023]:
    m = test["ANNEE"] == a
    for nom, pr in predictions.items():
        par_annee.append({"annee": a, "modele": nom, **detail(y_test[m], pr[m])})
pa = pd.DataFrame(par_annee)
print("\n=== Test, année par année ===")
print(pa.pivot(index="modele", columns="annee", values=["MAE", "R2"]).round(3))

# --- 4. Coefficients de la régression linéaire ---
lin = clone(pipelines["Régression linéaire"]).fit(X_train, y_train)
noms = lin.named_steps["preprocessor"].get_feature_names_out()
coefs = pd.DataFrame({"variable": noms, "coefficient": lin.named_steps["modele"].coef_})
print("\n=== Coefficients (variables après MinMaxScaler, donc entre 0 et 1) ===")
print(coefs.round(2).to_string(index=False), f"\nordonnée à l'origine : {lin.named_steps['modele'].intercept_:.2f}")
simple = LinearRegression().fit(train[["RATIO_DMA_lag1"]], y_train)
print(f"Sur la seule variable d'historique, en kg/hab : ratio = {simple.coef_[0]:.3f} × ratio précédent "
      f"+ {simple.intercept_:.1f}")

# --- 5. Arbre de décision : choisir la profondeur ---
# Plus l'arbre est profond, mieux il colle à l'entraînement ; la validation croisée
# dit à partir de quand il se met à apprendre par cœur (surapprentissage).
profondeurs = []
for d in [2, 3, 4, 5, 6, 8, None]:
    arbre = Pipeline([("preprocessor", clone(ns["preprocessor"])),
                      ("modele", DecisionTreeRegressor(max_depth=d, random_state=42))])
    r2_cv = np.mean([detail(y_train[train["ANNEE"] == a],
                            pd.Series(clone(arbre).fit(X_train[train["ANNEE"] < a], y_train[train["ANNEE"] < a])
                                      .predict(X_train[train["ANNEE"] == a]),
                                      index=y_train[train["ANNEE"] == a].index))["R2"] for a in annees[1:]])
    arbre.fit(X_train, y_train)
    profondeurs.append({"profondeur": "illimitée" if d is None else d,
                        "R2_entrainement": detail(y_train, pd.Series(arbre.predict(X_train), index=y_train.index))["R2"],
                        "R2_validation_croisee": r2_cv,
                        "R2_test": detail(y_test, pd.Series(arbre.predict(X_test), index=y_test.index))["R2"]})
prof = pd.DataFrame(profondeurs)
print("\n=== Arbre de décision : R² selon la profondeur ===")
print(prof.round(3).to_string(index=False))

calc.to_csv("data/processed/detail_calcul_test.csv", index=False)
prof.to_csv("data/processed/detail_profondeur_arbre.csv", index=False)
cv.to_csv("data/processed/detail_cv_plis.csv", index=False)
pa.to_csv("data/processed/detail_test_par_annee.csv", index=False)
coefs.to_csv("data/processed/detail_coefficients.csv", index=False)
print("\nSauvegardé -> data/processed/detail_*.csv")
