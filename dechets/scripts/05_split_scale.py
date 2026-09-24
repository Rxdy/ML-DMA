"""
Split train/test (par année, pas aléatoire) + Min-Max scaling [0, 1].

Point pédagogique important : le scaler est calibré (fit) UNIQUEMENT sur le
train. Si on le calibrait sur tout le dataset, les min/max du test
"fuiteraient" dans le train (data leakage) -> évaluation trop optimiste.
"""
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("data/processed/processed_regression_dataset.csv")

# La 1ère année (2009) n'a pas de lag -> on ne peut pas l'utiliser comme
# exemple d'entraînement pour un modèle qui s'appuie sur RATIO_DMA_lag1.
df = df.dropna(subset=["RATIO_DMA_lag1"])
print(f"Shape après suppression des lignes sans lag (2009) : {df.shape}")

# --- Split temporel : pas de mélange aléatoire, pour ne pas apprendre sur le "futur" ---
train = df[df["ANNEE"] <= 2019].copy()
test = df[df["ANNEE"] > 2019].copy()
print(f"Train (<=2019) : {train.shape[0]} lignes, années {sorted(train['ANNEE'].unique())}")
print(f"Test  (>2019)  : {test.shape[0]} lignes, années {sorted(test['ANNEE'].unique())}")

# Aucun couple (département, année) ne doit se retrouver à la fois en train et en
# test : sinon le modèle serait évalué sur des exemples déjà vus.
commun = train.merge(test, on=["C_DEPT", "ANNEE"])
print(f"Couples département × année présents dans les deux jeux : {len(commun)}")
assert commun.empty

feature_cols = ["VA_POPANNEE", "cluster", "RATIO_DMA_lag1", "TONNAGE_DMA_lag1"]
target_col = "RATIO_DMA"

# --- Min-Max scaling : fit sur train uniquement ---
scaler = MinMaxScaler()
train_scaled = train.copy()
test_scaled = test.copy()

train_scaled[feature_cols] = scaler.fit_transform(train[feature_cols])
test_scaled[feature_cols] = scaler.transform(test[feature_cols])  # transform seul, pas fit !

print("\n=== Avant scaling (train) ===")
print(train[feature_cols].describe().loc[["min", "max"]])

print("\n=== Après scaling (train) : doit être exactement [0, 1] ===")
print(train_scaled[feature_cols].describe().loc[["min", "max"]])

print("\n=== Après scaling (test) : peut dépasser [0, 1] si min/max du test diffèrent du train ===")
print(test_scaled[feature_cols].describe().loc[["min", "max"]])

train_scaled.to_csv("data/processed/train_scaled.csv", index=False)
test_scaled.to_csv("data/processed/test_scaled.csv", index=False)
print("\nSauvegardé -> data/processed/train_scaled.csv, data/processed/test_scaled.csv")
