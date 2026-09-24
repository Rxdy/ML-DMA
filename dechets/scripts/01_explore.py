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
print(f"Lignes strictement identiques : {df.duplicated().sum()}")
# Plus fin qu'une ligne identique : la même combinaison année × département ×
# type de déchet × type de traitement déclarée deux fois avec un tonnage différent
# (double saisie corrigée). Le pivot du clustering additionnerait les deux.
cle = ["ANNEE", "C_DEPT", "C_TYP_REG_DECHET", "C_TYP_REG_SERVICE"]
print(f"Doublons sur la clé {cle} : {df.duplicated(subset=cle).sum()}")

print("\n=== Doublons dans les autres sources (clé département × année) ===")
for f in ["sinoe_chiffres_cles_hors_gravats.csv", "sinoe_chiffres_cles_avec_gravats.csv"]:
    cc = pd.read_csv(f"data/raw/{f}", encoding="utf-8-sig", sep=";", decimal=",")
    print(f"{f} : {cc.duplicated().sum()} lignes identiques, "
          f"{cc.duplicated(subset=['C_DEPT', 'Annee']).sum()} doublons département × année")

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
