"""
Densité par département ET par profession, pour la carte (fond bleu réactif
au filtre). Population : fichier INSEE officiel téléchargé directement pour
ce projet (populations de référence, millésime 2023, validées décembre 2025,
en vigueur depuis le 1er janvier 2026 — le plus récent disponible à ce jour).
"""
import pandas as pd
import json

rpps = pd.read_csv("data/processed/rpps_clean.csv", dtype=str,
                    usecols=["Identifiant PP", "C_DEPT", "Libellé profession"])
DOM = {"971", "972", "973", "974", "976", "975", "977", "978"}
rpps_metro = rpps[~rpps["C_DEPT"].isin(DOM)]

ALL_PROFS = sorted(rpps_metro["Libellé profession"].unique().tolist())
CATS_ALL = ALL_PROFS + ["Tous"]

dedup = rpps_metro.drop_duplicates(subset=["Identifiant PP", "C_DEPT"])
pivot = dedup.pivot_table(index="C_DEPT", columns="Libellé profession", values="Identifiant PP",
                           aggfunc="nunique", fill_value=0)
pivot = pivot.reindex(columns=ALL_PROFS, fill_value=0)
pivot["Tous"] = pivot.sum(axis=1)
pivot = pivot.reset_index()

pop = pd.read_csv("data/processed/population_insee_2023ref.csv").rename(columns={"POPULATION": "VA_POPANNEE"})
merged = pivot.merge(pop[["C_DEPT", "VA_POPANNEE"]], on="C_DEPT", how="inner")
print("Départements avec population :", len(merged), "/", len(pivot))

for c in CATS_ALL:
    merged[c + "_dens"] = merged[c] / merged["VA_POPANNEE"] * 10000

breaks_by_cat = {c: merged[c + "_dens"].quantile([0, 1/6, 2/6, 3/6, 4/6, 5/6, 1]).round(2).tolist() for c in CATS_ALL}
dept_density = {row["C_DEPT"]: {c: round(row[c + "_dens"], 2) for c in CATS_ALL} for _, row in merged.iterrows()}

with open("data/processed/dept_density_data.json", "w", encoding="utf-8") as f:
    json.dump({"cats": CATS_ALL, "breaks": breaks_by_cat, "dept_density": dept_density},
               f, separators=(",", ":"), ensure_ascii=False)
print("Sauvegardé -> data/processed/dept_density_data.json")
