"""
Prépare les données pour la carte filtrable par profession : compte par
commune ET par profession, coordonnées projetées, plus le SVG initial
(catégorie "Tous") avec un index reliant chaque cercle à sa ligne de données.

v2 : les 29 professions individuelles (plus "Tous"), pas un regroupement en
8 catégories + "Autres" — cf. JOURNAL.md, l'utilisateur voulait filtrer sur
n'importe quelle profession, pas juste les plus fréquentes.
"""
import pandas as pd
import math
import json

rpps = pd.read_csv("data/processed/rpps_clean.csv", dtype=str,
                    usecols=["Identifiant PP", "Code commune (coord. structure)", "C_DEPT", "Libellé profession"])
rpps = rpps.rename(columns={"Code commune (coord. structure)": "code_insee"})
rpps = rpps.dropna(subset=["code_insee"])

DOM = {"971", "972", "973", "974", "976", "975", "977", "978"}
rpps = rpps[~rpps["C_DEPT"].isin(DOM)]

TOP_PROFESSIONS = [
    "Médecin", "Infirmier", "Masseur-Kinésithérapeute", "Psychologue",
    "Pharmacien", "Chirurgien-Dentiste", "Sage-Femme", "Orthophoniste",
]
rpps["GROUPE"] = rpps["Libellé profession"].where(rpps["Libellé profession"].isin(TOP_PROFESSIONS), "Autres professions")

dedup = rpps.drop_duplicates(subset=["Identifiant PP", "code_insee"])
pivot = dedup.pivot_table(index="code_insee", columns="GROUPE", values="Identifiant PP", aggfunc="nunique", fill_value=0)

CATS = TOP_PROFESSIONS + ["Autres professions"]
pivot = pivot.reindex(columns=CATS, fill_value=0)
pivot["Tous"] = pivot.sum(axis=1)
pivot = pivot.reset_index()
CATS_ALL = CATS + ["Tous"]

coords = pd.read_csv("data/raw/communes_coords.csv").rename(columns={"Code Insee": "code_insee"})
coords_dedup = coords.drop_duplicates(subset=["code_insee"], keep="first")

merged = pivot.merge(coords_dedup[["code_insee", "latitude", "longitude"]], on="code_insee", how="left")
merged = merged.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
print("Communes avec coordonnées :", len(merged))

LAT0, SCALE = 46.6, 1550
def project(lon, lat):
    return round((lon - 2.5) * math.cos(math.radians(LAT0)) * SCALE, 1), round(-(lat - LAT0) * SCALE, 1)

xs, ys = zip(*[project(lon, lat) for lon, lat in zip(merged["longitude"], merged["latitude"])])
merged["x"], merged["y"] = xs, ys

rows = merged[["x", "y"] + CATS_ALL].values.tolist()
rows = [[r[0], r[1]] + [int(v) for v in r[2:]] for r in rows]
maxvals = [max(row[2 + i] for row in rows) for i in range(len(CATS_ALL))]
print("Max par catégorie :", dict(zip(CATS_ALL, maxvals)))

with open("data/processed/points_data.json", "w", encoding="utf-8") as f:
    json.dump({"cats": CATS_ALL, "max": maxvals, "rows": rows}, f, separators=(",", ":"), ensure_ascii=False)

# --- SVG initial (catégorie "Tous", index i_data pour retrouver la ligne en JS) ---
i_tous = CATS_ALL.index("Tous")
rmin, rmax = 55, 340
nmax = maxvals[i_tous]
def radius(n):
    return round(rmin + (rmax - rmin) * (n / nmax) ** 0.5, 1) if n > 0 else 0

circles = []
for i, row in enumerate(rows):
    x, y, n = row[0], row[1], row[2 + i_tous]
    if n == 0:
        continue
    circles.append(f'<circle data-i="{i}" cx="{x}" cy="{y}" r="{radius(n)}"><title>{n} praticien(s)</title></circle>')

with open("data/processed/_points_svg_filtrable.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(circles))
print(f"{len(circles)} cercles générés")
