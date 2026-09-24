"""
Entraînement et évaluation du modèle de régression : prédire RATIO_DMA
(kg/habitant) par département/année.

Comparaison à 3 niveaux :
  1. Baseline naïve  : on prédit simplement la valeur de l'enquête précédente
                       (RATIO_DMA_lag1) sans aucun modèle.
  2. Régression linéaire
  3. Random Forest

Le but de la baseline est de vérifier que le modèle apporte réellement
quelque chose par rapport à "ne rien faire d'intelligent".
"""
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

train = pd.read_csv("data/processed/train_scaled.csv")
test = pd.read_csv("data/processed/test_scaled.csv")

feature_cols = ["VA_POPANNEE", "cluster", "RATIO_DMA_lag1", "TONNAGE_DMA_lag1"]
target_col = "RATIO_DMA"

X_train, y_train = train[feature_cols], train[target_col]
X_test, y_test = test[feature_cols], test[target_col]


def evaluate(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    print(f"{name:35s} MAE={mae:7.2f} kg/hab   RMSE={rmse:7.2f} kg/hab   R2={r2:.3f}")
    return {"modele": name, "MAE": mae, "RMSE": rmse, "R2": r2}


print("=== Évaluation sur le test (2021 + 2023) ===\n")
results = []

# --- 1. Baseline naïve : lag1 non scalé (repris du dataset brut) ---
raw = pd.read_csv("data/processed/processed_regression_dataset.csv").dropna(subset=["RATIO_DMA_lag1"])
raw_test = raw[raw["ANNEE"] > 2019]
results.append(evaluate("Baseline naïve (= enquête précédente)", raw_test["RATIO_DMA"], raw_test["RATIO_DMA_lag1"]))

# --- 2. Régression linéaire ---
lr = LinearRegression().fit(X_train, y_train)
pred_lr = lr.predict(X_test)
results.append(evaluate("Régression linéaire", y_test, pred_lr))

# --- 3. Random Forest ---
rf = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42).fit(X_train, y_train)
pred_rf = rf.predict(X_test)
results.append(evaluate("Random Forest", y_test, pred_rf))

print("\n=== Importance des variables (Random Forest) ===")
importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print(importances.round(3))

# --- Erreur par cluster (typologie de traitement) ---
test_out = test.copy()
test_out["pred_rf"] = pred_rf
test_out["pred_lr"] = pred_lr
test_out["erreur_abs_rf"] = (test_out["pred_rf"] - test_out[target_col]).abs()

raw_cluster_map = raw[["C_DEPT", "ANNEE", "cluster"]].rename(columns={"cluster": "cluster_raw"})
test_out = test_out.merge(raw_cluster_map, on=["C_DEPT", "ANNEE"], how="left")

print("\n=== Erreur absolue moyenne (Random Forest) par cluster de traitement ===")
print(test_out.groupby("cluster_raw")["erreur_abs_rf"].agg(["mean", "count"]).round(2))

test_out.to_csv("data/processed/test_predictions.csv", index=False)
pd.DataFrame(results).to_csv("data/processed/model_comparison.csv", index=False)
print("\nSauvegardé -> data/processed/test_predictions.csv, data/processed/model_comparison.csv")
