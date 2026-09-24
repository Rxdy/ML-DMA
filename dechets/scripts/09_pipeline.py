"""
Même chaîne que les scripts 05 + 06, mais encapsulée dans les objets scikit-learn
prévus pour ça :

  - preprocessor (ColumnTransformer) : l'étape de TRANSFORMATION des données.
      * variables numériques  -> MinMaxScaler (ramenées entre 0 et 1)
      * variable catégorielle -> OneHotEncoder (cluster = catégorie, pas un nombre)

  - pipeline (Pipeline) : la CHAÎNE COMPLÈTE preprocessor -> modèle.
      pipeline.fit(X_train)   calibre le preprocessor PUIS entraîne le modèle, sur le train seul
      pipeline.predict(X)     applique les mêmes transformations PUIS prédit

Ce que ça change par rapport aux scripts 05/06 :
  1. `cluster` n'est plus mis à l'échelle comme un nombre (0 < 1 < 2 < 3 n'a pas
     de sens, c'est une typologie) : il est encodé en one-hot.
  2. La règle "fit sur le train uniquement" n'est plus une discipline manuelle :
     elle est garantie par construction, y compris dans la validation croisée
     (le preprocessor est recalibré dans chaque pli).
  3. Le modèle prend directement les données brutes en entrée : une seule
     chose à sauvegarder et à réutiliser, pas un scaler + un modèle séparés.
"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

df = pd.read_csv("data/processed/processed_regression_dataset.csv", dtype={"C_DEPT": str})
df = df.dropna(subset=["RATIO_DMA_lag1"]).reset_index(drop=True)  # 2009 : pas de lag

num_cols = ["VA_POPANNEE", "RATIO_DMA_lag1", "TONNAGE_DMA_lag1"]
cat_cols = ["cluster"]
target_col = "RATIO_DMA"

# --- Split temporel (identique au script 05) ---
train = df[df["ANNEE"] <= 2019].reset_index(drop=True)
test = df[df["ANNEE"] > 2019].reset_index(drop=True)
X_train, y_train = train[num_cols + cat_cols], train[target_col]
X_test, y_test = test[num_cols + cat_cols], test[target_col]
print(f"Train : {len(train)} lignes (2011-2019) | Test : {len(test)} lignes (2021, 2023)")

# --- Preprocessor : étape de transformation ---
preprocessor = ColumnTransformer([
    ("num", MinMaxScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

# --- Pipelines : chaîne complète preprocessor -> modèle ---
modeles = {
    "Baseline moyenne (DummyRegressor)": DummyRegressor(strategy="mean"),
    "Régression linéaire": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42),
}
pipelines = {nom: Pipeline([("preprocessor", clone(preprocessor)), ("modele", m)]) for nom, m in modeles.items()}


def scores(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "R2": r2_score(y_true, y_pred),
    }


# --- Validation croisée temporelle sur le train ---
# Fenêtre croissante : on entraîne sur les enquêtes passées, on valide sur la suivante.
# Pas de KFold aléatoire : il mélangerait futur et passé, comme un split aléatoire.
annees = sorted(train["ANNEE"].unique())
plis = [
    (train.index[train["ANNEE"] < a].to_numpy(), train.index[train["ANNEE"] == a].to_numpy())
    for a in annees[1:]
]

print("\n=== Validation croisée temporelle (train seul, 4 plis : valide 2013, 2015, 2017, 2019) ===")
# La baseline naïve n'apprend rien : elle se calcule directement sur chaque pli de validation.
naive_r2 = [r2_score(y_train[va], X_train.loc[va, "RATIO_DMA_lag1"]) for _, va in plis]
naive_mae = [mean_absolute_error(y_train[va], X_train.loc[va, "RATIO_DMA_lag1"]) for _, va in plis]
print(f"{'Baseline naïve (= enquête précédente)':35s} MAE={pd.Series(naive_mae).mean():6.2f}  "
      f"R2={pd.Series(naive_r2).mean():.3f} (± {pd.Series(naive_r2).std(ddof=0):.3f})")
for nom, pipe in pipelines.items():
    cv = cross_validate(pipe, X_train, y_train, cv=plis, scoring=("neg_mean_absolute_error", "r2"))
    print(f"{nom:35s} MAE={-cv['test_neg_mean_absolute_error'].mean():6.2f}  "
          f"R2={cv['test_r2'].mean():.3f} (± {cv['test_r2'].std():.3f})")

# --- Évaluation finale sur le test (jamais vu) ---
print("\n=== Évaluation sur le test (2021 + 2023) ===")
resultats = [{"modele": "Baseline naïve (= enquête précédente)", **scores(y_test, test["RATIO_DMA_lag1"])}]
for nom, pipe in pipelines.items():
    pipe.fit(X_train, y_train)  # calibre le preprocessor puis entraîne, sur le train seul
    resultats.append({"modele": nom, **scores(y_test, pipe.predict(X_test))})

res = pd.DataFrame(resultats)

# --- Le gain sur la baseline naïve est-il réel ou dû au hasard ? ---
# Bootstrap : on retire 5 000 fois 198 départements-années avec remise et on
# recalcule le gain de MAE. Si l'intervalle contient 0, le gain n'est pas établi.
lin = pipelines["Régression linéaire"]
err_lin = (lin.predict(X_test) - y_test).abs().to_numpy()
err_naive = (test["RATIO_DMA_lag1"] - y_test).abs().to_numpy()
rng = np.random.default_rng(42)
tirages = rng.integers(0, len(y_test), size=(5000, len(y_test)))
gains = (err_naive[tirages] - err_lin[tirages]).mean(axis=1)
bas, haut = np.percentile(gains, [2.5, 97.5])
print(f"\nGain de MAE régression linéaire vs baseline naïve : {err_naive.mean() - err_lin.mean():.2f} kg/hab "
      f"(IC 95 % : {bas:.2f} à {haut:.2f}), meilleure sur {(err_lin < err_naive).mean():.0%} des départements-années")
for _, r in res.iterrows():
    print(f"{r['modele']:38s} MAE={r['MAE']:6.2f}  RMSE={r['RMSE']:6.2f}  R2={r['R2']:.3f}")

# --- Ce que voit le modèle après le preprocessor ---
rf = pipelines["Random Forest"]
noms = rf.named_steps["preprocessor"].get_feature_names_out()
print(f"\nColonnes en sortie du preprocessor : {list(noms)}")
imp = pd.Series(rf.named_steps["modele"].feature_importances_, index=noms).sort_values(ascending=False)
print("\n=== Importance des variables (Random Forest) ===")
print(imp.round(3))

res.to_csv("data/processed/model_comparison_pipeline.csv", index=False)
print("\nSauvegardé -> data/processed/model_comparison_pipeline.csv")
