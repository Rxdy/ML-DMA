"""
Assemblage du revenu médian disponible par département, 2013-2021 (Filosofi/INSEE).

Contrairement aux fichiers ADEME (un seul CSV multi-années), l'INSEE publie
une édition séparée par année, avec un nom de colonne qui change (MED13,
MED15, MED17...) ou un format "long" pour l'édition la plus récente (2021).
Ce script normalise tout ça en un seul format : C_DEPT, ANNEE, MEDIANE_REVENU.

Sources téléchargées manuellement dans /tmp/filosofi/ (voir JOURNAL.md pour
les URLs) :
  - 2013, 2015 : fichiers .xls, feuille "DEP", colonne MEDxx
  - 2017, 2019 : fichiers cc_filosofi_YYYY_DEP.csv, colonne MEDxx
  - 2021       : format long DS_FILOSOFI_CC_data.csv (GEO_OBJECT='DEP', FILOSOFI_MEASURE='MED_SL')
"""
import pandas as pd

rows = []

# --- 2013, 2015 : Excel, feuille DEP ---
for annee, path in [
    (2013, "/tmp/filosofi/filo2013/filo-revenu-pauvrete-menage-2013/filo-revenu-pauvrete-menage-2013.xls"),
    (2015, "/tmp/filosofi/filo2015/filo-revenu-pauvrete-menage-2015/base-cc-filosofi-2015.xls"),
]:
    df = pd.read_excel(path, sheet_name="DEP", header=5)
    col = f"MED{str(annee)[2:]}"
    sub = df[["CODGEO", col]].rename(columns={"CODGEO": "C_DEPT", col: "MEDIANE_REVENU"})
    sub["ANNEE"] = annee
    rows.append(sub)

# --- 2017, 2019 : CSV, colonne MEDxx ---
for annee, path in [
    (2017, "/tmp/filosofi/filo2017/cc_filosofi_2017_DEP.CSV"),
    (2019, "/tmp/filosofi/filo2019/cc_filosofi_2019_DEP.csv"),
]:
    df = pd.read_csv(path, sep=";")
    col = f"MED{str(annee)[2:]}"
    sub = df[["CODGEO", col]].rename(columns={"CODGEO": "C_DEPT", col: "MEDIANE_REVENU"})
    sub["ANNEE"] = annee
    rows.append(sub)

# --- 2021 : format long ---
df21 = pd.read_csv("/tmp/filosofi/filo2021/DS_FILOSOFI_CC_data.csv", sep=";", low_memory=False)
dep21 = df21[(df21["GEO_OBJECT"] == "DEP") & (df21["FILOSOFI_MEASURE"] == "MED_SL")]
sub21 = dep21[["GEO", "OBS_VALUE"]].rename(columns={"GEO": "C_DEPT", "OBS_VALUE": "MEDIANE_REVENU"})
sub21["ANNEE"] = 2021
rows.append(sub21)

revenu = pd.concat(rows, ignore_index=True)
revenu["C_DEPT"] = revenu["C_DEPT"].astype(str).str.zfill(2)

print("=== Shape par année ===")
print(revenu.groupby("ANNEE").size())

print("\n=== Valeurs manquantes ===")
print(revenu.isnull().sum())

print("\n=== Aperçu ===")
print(revenu.sort_values(["C_DEPT", "ANNEE"]).head(10))

revenu.to_csv("data/raw/insee_revenu_median_dep_2013_2021.csv", index=False)
print(f"\nSauvegardé -> data/raw/insee_revenu_median_dep_2013_2021.csv ({revenu.shape[0]} lignes)")
