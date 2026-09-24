"""
Nettoyage de l'annuaire RPPS : sélection des colonnes utiles + reconstruction
du département (la colonne dédiée est vide à 100 %).

Le fichier complet (56 colonnes, 2,29M lignes) est trop lourd à charger tel
quel avec une mémoire limitée -> on ne garde que les colonnes utiles dès la
lecture (usecols).
"""
import pandas as pd

PATH = "data/raw/PS_LibreAcces_Personne_activite.txt"

USECOLS = [
    "Identifiant PP",
    "Nom d'exercice",
    "Prénom d'exercice",
    "Libellé profession",
    "Libellé catégorie professionnelle",
    "Libellé mode exercice",
    "Code postal (coord. structure)",
    "Code commune (coord. structure)",
    "Libellé commune (coord. structure)",
    "Libellé genre activité",
]

df = pd.read_csv(PATH, sep="|", usecols=USECOLS, dtype=str)
print("=== Shape brut ===", df.shape)

# --- Département : la colonne dédiée est vide -> reconstruction depuis le code postal ---
cp = df["Code postal (coord. structure)"]
commune = df["Code commune (coord. structure)"]

# Outre-mer : le département se code sur 3 chiffres (971, 972, 974...),
# pas 2 -> sinon tous les DOM-TOM sont mélangés dans un faux "97".
def extraire_dept(code, n_metropole=2):
    is_om = code.str.startswith(("97", "98"), na=False)
    dept = code.str[:n_metropole].copy()
    dept[is_om] = code[is_om].str[:3]
    return dept

dept_from_cp = extraire_dept(cp)
dept_from_commune = extraire_dept(commune)

# Corse : code postal 20xxx ne distingue pas 2A/2B -> on utilise le code commune
# INSEE (qui lui distingue 2A/2B) en priorité pour ces lignes.
is_corse_ambigu = dept_from_cp == "20"
dept = dept_from_cp.copy()
dept[is_corse_ambigu] = dept_from_commune[is_corse_ambigu]

# Repli sur le code commune quand le code postal est manquant
dept = dept.fillna(dept_from_commune)

df["C_DEPT"] = dept

print("\n=== Département reconstruit : taux de succès ===")
print(f"Lignes avec département identifié : {df['C_DEPT'].notna().sum()} / {len(df)} "
      f"({df['C_DEPT'].notna().mean()*100:.1f} %)")

print("\n=== Valeurs manquantes par colonne (sur le fichier complet) ===")
print((df.isnull().mean() * 100).round(1).sort_values(ascending=False))

print("\n=== Lignes sans département identifiable : quelques exemples ===")
sans_dept = df[df["C_DEPT"].isna()]
print(sans_dept[["Libellé profession", "Libellé mode exercice", "Libellé genre activité"]].head(10))

# On retire les lignes sans département (pas exploitables pour une analyse territoriale)
df_clean = df[df["C_DEPT"].notna()].copy()

# Codes département non-métropolitains/invalides à vérifier
print("\n=== Codes département trouvés (aperçu) ===")
print(sorted(df_clean["C_DEPT"].unique())[:15], "...")
print("Nombre de codes département distincts :", df_clean["C_DEPT"].nunique())

df_clean.to_csv("data/processed/rpps_clean.csv", index=False)
print(f"\nSauvegardé -> data/processed/rpps_clean.csv ({df_clean.shape[0]} lignes)")
