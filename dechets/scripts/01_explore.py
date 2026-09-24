"""Exploration initiale du dataset SINOE - destination des DMA collectés par type de traitement."""
import pandas as pd

df = pd.read_csv(
    "data/raw/sinoe_dma.csv",
    decimal=",",
)

print("=== Dimensions ===")
print(df.shape)

print("\n=== Colonnes et types ===")
print(df.dtypes)

print("\n=== Aperçu ===")
print(df.head())

print("\n=== Valeurs manquantes par colonne ===")
print(df.isnull().sum())

print("\n=== Doublons ===")
print(f"Lignes dupliquées : {df.duplicated().sum()}")

print("\n=== Valeurs uniques par colonne catégorielle ===")
for col in ["ANNEE", "C_REGION", "L_REGION", "C_DEPT", "N_DEPT",
            "C_TYP_REG_DECHET", "L_TYP_REG_DECHET",
            "C_TYP_REG_SERVICE", "L_TYP_REG_SERVICE"]:
    print(f"\n--- {col} ({df[col].nunique()} valeurs uniques) ---")
    print(sorted(df[col].dropna().unique().tolist())[:30])

print("\n=== Statistiques TONNAGE_DMA ===")
print(df["TONNAGE_DMA"].describe())

print("\n=== Tonnages négatifs ou nuls ===")
print((df["TONNAGE_DMA"] <= 0).sum())
