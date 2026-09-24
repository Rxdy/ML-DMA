"""
Agrégation RPPS par département : densité de praticiens + profil de professions
(miroir du script clustering des déchets : profil de traitement -> profil de professions).
"""
import pandas as pd

df = pd.read_csv("data/processed/rpps_clean.csv", dtype=str)

# Un praticien peut apparaître plusieurs fois dans le même département (plusieurs
# structures) -> on déduplique par (praticien, département) avant de compter,
# pour ne pas gonfler artificiellement les effectifs.
dedup = df.drop_duplicates(subset=["Identifiant PP", "C_DEPT"])
print(f"Lignes avant dédoublonnage (praticien x département) : {len(df)}")
print(f"Lignes après dédoublonnage : {len(dedup)}")

# --- Regroupement des 29 professions en 8 grandes catégories + Autres ---
TOP_PROFESSIONS = [
    "Médecin", "Infirmier", "Masseur-Kinésithérapeute", "Psychologue",
    "Pharmacien", "Chirurgien-Dentiste", "Sage-Femme", "Orthophoniste",
]
dedup = dedup.copy()
dedup["PROFESSION_GROUPE"] = dedup["Libellé profession"].where(
    dedup["Libellé profession"].isin(TOP_PROFESSIONS), "Autres professions"
)

# --- Effectif total par département ---
effectif_dept = dedup.groupby("C_DEPT")["Identifiant PP"].nunique().rename("NB_PRATICIENS")

# --- Profil de professions (%) par département ---
pivot = dedup.pivot_table(
    index="C_DEPT", columns="PROFESSION_GROUPE", values="Identifiant PP",
    aggfunc="nunique", fill_value=0,
)
profil_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

print("\n=== Effectif total par département (top 10) ===")
print(effectif_dept.sort_values(ascending=False).head(10))

print("\n=== Profil de professions (%), aperçu ===")
print(profil_pct.head())

resultat = effectif_dept.to_frame().join(profil_pct)
resultat.to_csv("data/processed/rpps_par_departement.csv")
print(f"\nSauvegardé -> data/processed/rpps_par_departement.csv ({resultat.shape[0]} départements)")
