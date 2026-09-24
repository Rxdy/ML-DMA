"""
Carte par points (niveau commune) : montre les zones sans praticien, que la
moyenne par département masque.

Un point = une commune avec au moins 1 praticien actif, rayon ~ sqrt(effectif).
"""
import pandas as pd
import math

rpps = pd.read_csv("data/processed/rpps_clean.csv", dtype=str,
                    usecols=["Identifiant PP", "Code commune (coord. structure)", "C_DEPT"])
rpps = rpps.rename(columns={"Code commune (coord. structure)": "code_insee"})
rpps = rpps.dropna(subset=["code_insee"])

DOM = {"971", "972", "973", "974", "976", "975", "977", "978"}
rpps_metro = rpps[~rpps["C_DEPT"].isin(DOM)]

dedup = rpps_metro.drop_duplicates(subset=["Identifiant PP", "code_insee"])
counts = dedup.groupby("code_insee").size().rename("n").reset_index()
print("Communes avec au moins 1 praticien :", len(counts))

coords = pd.read_csv("data/raw/communes_coords.csv").rename(columns={"Code Insee": "code_insee"})
# Une commune peut avoir plusieurs codes postaux -> plusieurs lignes dans ce
# fichier pour le même code_insee. On déduplique, sinon les effectifs sont
# comptés en double lors de la jointure.
coords_dedup = coords.drop_duplicates(subset=["code_insee"], keep="first")

merged = counts.merge(coords_dedup[["code_insee", "latitude", "longitude"]], on="code_insee", how="left")
print("Communes sans coordonnées trouvées :", merged["latitude"].isna().sum())
merged = merged.dropna(subset=["latitude", "longitude"])
print("Communes finales avec coordonnées :", len(merged))
print("Total praticiens représentés :", merged["n"].sum())

merged.to_csv("data/processed/rpps_points_commune.csv", index=False)

# --- Projection (même repère que la carte département, pour comparabilité) ---
LAT0, SCALE = 46.6, 1550
def project(lon, lat):
    x = (lon - 2.5) * math.cos(math.radians(LAT0)) * SCALE
    y = -(lat - LAT0) * SCALE
    return x, y

rmin, rmax = 1.1, 26
nmax = merged["n"].max()
def radius(n):
    return round(rmin + (rmax - rmin) * (n / nmax) ** 0.5, 2)

circles = [
    f'<circle cx="{project(row.longitude, row.latitude)[0]:.1f}" '
    f'cy="{project(row.longitude, row.latitude)[1]:.1f}" r="{radius(row.n)}">'
    f'<title>{int(row.n)} praticien(s)</title></circle>'
    for row in merged.itertuples()
]
with open("data/processed/_points_svg.txt", "w") as f:
    f.write("\n".join(circles))
print(f"\n{len(circles)} points générés -> data/processed/_points_svg.txt (à coller dans le rapport)")
