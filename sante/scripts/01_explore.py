"""
Exploration initiale de l'annuaire RPPS (professionnels de santé).

Fichier volumineux (820 Mo, ~2,29M lignes) : on explore d'abord sur un
échantillon pour comprendre la structure avant de décider comment le
traiter en entier (colonnes utiles seulement, chunking, agrégation...).
"""
import pandas as pd

PATH = "data/raw/PS_LibreAcces_Personne_activite.txt"

sample = pd.read_csv(PATH, sep="|", nrows=200_000, dtype=str)

print("=== Dimensions (échantillon) ===")
print(sample.shape)

print("\n=== Colonnes ===")
for c in sample.columns:
    print(f"  - {c}")

print("\n=== Taux de valeurs manquantes par colonne (sur l'échantillon, %) ===")
print((sample.isnull().mean() * 100).round(1).sort_values(ascending=False))

print("\n=== Professions les plus fréquentes ===")
print(sample["Libellé profession"].value_counts().head(15))

print("\n=== Départements les plus fréquents (structure d'exercice) ===")
print(sample["Libellé Département (structure)"].value_counts().head(10))

print("\n=== Doublons sur Identifiant PP (un pro peut avoir plusieurs lignes = plusieurs activités/structures) ===")
print("Identifiants uniques :", sample["Identifiant PP"].nunique(), "/ lignes :", len(sample))
