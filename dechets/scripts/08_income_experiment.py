"""
Expérience : est-ce que le revenu médian (INSEE) améliore la prédiction de
RATIO_DMA par rapport au pipeline précédent (sans revenu) ?

Contrainte du revenu : disponible seulement pour 2013-2021 (5 enquêtes),
contre 2009-2023 (8 enquêtes) sans lui. On restreint donc cette expérience
à ces 5 années, et on recalcule les lags dans ce sous-ensemble (le lag de
2013 n'a pas de valeur précédente ici, il est donc retiré comme 2009 dans
le pipeline original).

Comparaison à deux features sets, mêmes lignes, même split, pour isoler
l'effet du revenu :
  A) sans revenu (population, cluster, lag) — comme avant
  B) avec revenu (+ REVENU, REVENU_lag1)
"""
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

base = pd.read_csv("data/processed/processed_regression_dataset.csv")
revenu = pd.read_csv("data/raw/insee_revenu_median_dep_2013_2021.csv")

df = base.merge(revenu, on=["C_DEPT", "ANNEE"], how="inner", validate="one_to_one")
print(f"Lignes après jointure avec le revenu (années 2013-2021 seulement) : {df.shape[0]}")

df = df.sort_values(["C_DEPT", "ANNEE"])
df["REVENU_lag1"] = df.groupby("C_DEPT")["MEDIANE_REVENU"].shift(1)
df["RATIO_DMA_lag1_v2"] = df.groupby("C_DEPT")["RATIO_DMA"].shift(1)
df = df.dropna(subset=["REVENU_lag1", "RATIO_DMA_lag1_v2"])
print(f"Lignes après retrait de la 1ère année sans lag (2013) : {df.shape[0]}")
print("Années disponibles :", sorted(df["ANNEE"].unique()))

train = df[df["ANNEE"] <= 2017]
test = df[df["ANNEE"] > 2017]
print(f"Train (<=2017) : {len(train)} lignes | Test (>2017) : {len(test)} lignes")

target_col = "RATIO_DMA"
features_A = ["VA_POPANNEE", "cluster", "RATIO_DMA_lag1_v2", "TONNAGE_DMA_lag1"]
features_B = features_A + ["MEDIANE_REVENU", "REVENU_lag1"]


def run(name, features):
    X_train, y_train = train[features], train[target_col]
    X_test, y_test = test[features], test[target_col]
    results = {}
    for model_name, model in [("Régression linéaire", LinearRegression()),
                               ("Random Forest", RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42))]:
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, pred)
        rmse = mean_squared_error(y_test, pred) ** 0.5
        r2 = r2_score(y_test, pred)
        print(f"  [{name:18s}] {model_name:20s} MAE={mae:7.2f}  RMSE={rmse:7.2f}  R2={r2:.3f}")
        results[model_name] = (mae, rmse, r2)
    return results


print("\n=== A) SANS revenu (features d'origine) ===")
res_a = run("sans revenu", features_A)

print("\n=== B) AVEC revenu ===")
res_b = run("avec revenu", features_B)

print("\n=== Baseline naïve (recopier l'enquête précédente), sur ces mêmes lignes ===")
mae_base = mean_absolute_error(test[target_col], test["RATIO_DMA_lag1_v2"])
rmse_base = mean_squared_error(test[target_col], test["RATIO_DMA_lag1_v2"]) ** 0.5
r2_base = r2_score(test[target_col], test["RATIO_DMA_lag1_v2"])
print(f"  Baseline naïve      MAE={mae_base:7.2f}  RMSE={rmse_base:7.2f}  R2={r2_base:.3f}")

# Importance des variables avec revenu
rf_b = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42).fit(train[features_B], train[target_col])
print("\n=== Importance des variables (Random Forest, avec revenu) ===")
print(pd.Series(rf_b.feature_importances_, index=features_B).sort_values(ascending=False).round(3))
