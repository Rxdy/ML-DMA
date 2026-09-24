"""
Préparation du jeu de données pour la régression :
prédire le tonnage / ratio de DMA par habitant, par département et par année.

Source principale : data/sinoe_chiffres_cles_hors_gravats.csv
  -> seul fichier couvrant 2009 à 2023 (8 enquêtes), avec population et ratio déjà calculés.

Enrichissement : cluster de profil de traitement (data/processed_clusters_traitement.csv)
  calculé sur 2009-2021, utilisé comme feature catégorielle (typologie du département).
"""
import pandas as pd

cc = pd.read_csv(
    "data/raw/sinoe_chiffres_cles_hors_gravats.csv",
    encoding="utf-8-sig", sep=";", decimal=",",
)
cc = cc.rename(columns={"Annee": "ANNEE", "TONNAGE_ DMA": "TONNAGE_DMA"})

clusters = pd.read_csv("data/processed/processed_clusters_traitement.csv")[["C_DEPT", "cluster"]]
clusters["C_DEPT"] = clusters["C_DEPT"].astype(str).str.zfill(2)
cc["C_DEPT"] = cc["C_DEPT"].astype(str).str.zfill(2)

# Une ligne par (département, année) attendue : un doublon ici fausserait le lag
# (shift(1) prendrait la ligne dupliquée comme "enquête précédente").
assert not cc.duplicated(subset=["C_DEPT", "ANNEE"]).any(), "doublon département × année"
# validate : chaque département n'a qu'un seul cluster, sinon la jointure
# multiplierait les lignes sans le signaler.
df = cc.merge(clusters, on="C_DEPT", how="left", validate="many_to_one")
assert len(df) == len(cc), "la jointure a changé le nombre de lignes"

print("=== Shape avant traitement ===", df.shape)
print("\n=== Valeurs manquantes ===")
print(df.isnull().sum())

# Départements sans cluster (ex: Mayotte non présent dans sinoe_dma.csv 2009-2021 ?)
missing_cluster = df[df["cluster"].isnull()]["N_DEPT"].unique()
print(f"\nDépartements sans cluster de traitement : {list(missing_cluster)}")

# --- Feature d'historique : tonnage/ratio de l'enquête précédente (lag) ---
df = df.sort_values(["C_DEPT", "ANNEE"])
df["RATIO_DMA_lag1"] = df.groupby("C_DEPT")["RATIO_DMA"].shift(1)
df["TONNAGE_DMA_lag1"] = df.groupby("C_DEPT")["TONNAGE_DMA"].shift(1)

print("\n=== Aperçu final ===")
print(df.head(10))

print("\n=== Valeurs manquantes après feature engineering ===")
print(df.isnull().sum())
print("(lag1 = NaN attendu pour la 1ère année de chaque département, 2009)")

# --- Vérification du choix de cible ---
# TONNAGE_DMA est quasi-déterminé par la population (fuite : RATIO = TONNAGE/POP),
# donc la cible retenue est RATIO_DMA (kg/habitant), pas le tonnage brut.
# ANNEE seule porte très peu de signal (corr ~0 avec RATIO_DMA) : le lag1
# capture mieux la trajectoire propre à chaque département que l'année brute.
print("\n=== Vérification cible/features ===")
print("corr(TONNAGE_DMA, VA_POPANNEE) =", df["TONNAGE_DMA"].corr(df["VA_POPANNEE"]).round(3), "-> fuite si cible = tonnage brut")
print("corr(RATIO_DMA, ANNEE)         =", df["RATIO_DMA"].corr(df["ANNEE"]).round(3), "-> peu de signal dans l'année brute")
print("Cible retenue : RATIO_DMA (kg/habitant)")

df.to_csv("data/processed/processed_regression_dataset.csv", index=False)
print("\nSauvegardé -> data/processed/processed_regression_dataset.csv")
print(f"Shape finale : {df.shape}")
