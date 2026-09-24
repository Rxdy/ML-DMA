"""
Clustering des départements selon ce que leurs habitants jettent.

Étape 1 — variables : 5 types de déchets, en kg par habitant, moyenne des enquêtes
          2019 et 2021 (une ligne par département) :
          ordures ménagères résiduelles (OMR), recyclables, déchets verts et biodéchets,
          encombrants, déchets dangereux.
          Écartés : gravats (chantiers, pas les ménages), « autres » (catégorie quasi vide),
          total (somme des autres : doublon).
          Standardisation (StandardScaler) : sinon les OMR (~240 kg) écraseraient les
          déchets dangereux (~10 kg) dans le calcul des distances.

Étape 2 — nombre de groupes k : méthode du coude (inertie) et score de silhouette,
          pour k de 2 à 10.

Étape 3 — k-means avec le k retenu, profils des groupes et interprétation à l'aide de
          variables qui n'ont PAS servi au clustering (population, revenu, traitement).

Sorties : data/processed/clustering_choix_k.csv, clustering_production.csv,
          data/figures/clustering_coude_silhouette.png, clustering_profils.png,
          clustering_carte.png
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Polygon
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

K_RETENU = 4
ENQUETES = [2019, 2021]
VARIABLES = {
    "Ordures ménagères résiduelles": "OMR",
    "Matériaux recyclables": "Recyclables",
    "Déchets verts et biodéchets": "Verts",
    "Encombrants": "Encombrants",
    "Déchets dangereux (y.c. DEEE)": "Dangereux",
}
COLS = list(VARIABLES.values())
COULEURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]

# ------------------------------------------------------------------ étape 1 : variables
dma = pd.read_csv("data/raw/sinoe_dma.csv", decimal=",", dtype={"C_DEPT": str})
dma["C_DEPT"] = dma["C_DEPT"].str.zfill(2)
cc = pd.read_csv("data/raw/sinoe_chiffres_cles_hors_gravats.csv", sep=";", decimal=",",
                 encoding="utf-8-sig", dtype={"C_DEPT": str}).rename(columns={"Annee": "ANNEE"})
cc["C_DEPT"] = cc["C_DEPT"].str.zfill(2)

sel = dma[dma["ANNEE"].isin(ENQUETES) & dma["L_TYP_REG_DECHET"].isin(VARIABLES)]
tonnes = (sel.pivot_table(index=["C_DEPT", "ANNEE"], columns="L_TYP_REG_DECHET", values="TONNAGE_DMA",
                          aggfunc="sum").rename(columns=VARIABLES).reset_index())
tonnes = tonnes.merge(cc[["C_DEPT", "N_DEPT", "L_REGION", "ANNEE", "VA_POPANNEE", "RATIO_DMA"]],
                      on=["C_DEPT", "ANNEE"], how="inner", validate="one_to_one")
for c in COLS:
    tonnes[c] = tonnes[c] * 1000 / tonnes["VA_POPANNEE"]  # tonnes -> kg par habitant
X = tonnes.groupby(["C_DEPT", "N_DEPT", "L_REGION"])[COLS + ["VA_POPANNEE", "RATIO_DMA"]].mean().reset_index()
assert not X[COLS].isna().any().any()
print(f"=== Étape 1 : {len(X)} départements × {len(COLS)} variables (kg/hab, moyenne {ENQUETES}) ===")
print("Absent (pas de détail par type de déchet) :",
      ", ".join(cc.loc[~cc["C_DEPT"].isin(X["C_DEPT"]), "N_DEPT"].unique()))
print(X[COLS].describe().loc[["mean", "min", "max"]].round(1))

Z = StandardScaler().fit_transform(X[COLS])

# ------------------------------------------------------------------ étape 2 : choix de k
lignes = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, n_init=20, random_state=42).fit(Z)
    lignes.append({"k": k, "inertie": km.inertia_, "silhouette": silhouette_score(Z, km.labels_),
                   "plus_petit_groupe": int(np.bincount(km.labels_).min())})
choix = pd.DataFrame(lignes)
choix["baisse_inertie_pct"] = -choix["inertie"].pct_change() * 100
print("\n=== Étape 2 : méthode du coude et silhouette ===")
print(choix.round(3).to_string(index=False))

fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.4))
a1.plot(choix.k, choix.inertie, marker="o", color="#2F5236")
a1.axvline(K_RETENU, ls="--", color="#9C6B2A", lw=1)
a1.set(title="Méthode du coude", xlabel="nombre de groupes k", ylabel="inertie (intra-groupe)")
a2.plot(choix.k, choix.silhouette, marker="o", color="#2F5236")
a2.axvline(K_RETENU, ls="--", color="#9C6B2A", lw=1)
a2.set(title="Score de silhouette", xlabel="nombre de groupes k", ylabel="silhouette moyenne")
for ax in (a1, a2):
    ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("data/figures/clustering_coude_silhouette.png", dpi=200)
plt.close(fig)

# ------------------------------------------------------------------ étape 3 : k-means retenu
km = KMeans(n_clusters=K_RETENU, n_init=20, random_state=42).fit(Z)
X["cluster"] = km.labels_
X["silhouette"] = silhouette_samples(Z, km.labels_)
profils = X.groupby("cluster")[COLS].mean()

# Nom de chaque groupe d'après son profil (indépendant du numéro attribué par k-means)
noms = {}
restants = list(profils.index)
for regle, nom in [(lambda p: p["OMR"].idxmax(), "Ordures ménagères élevées"),
                   (lambda p: p["Recyclables"].idxmin(), "Grandes agglomérations"),
                   (lambda p: p["Verts"].idxmax(), "Déchets verts et déchèteries")]:
    c = regle(profils.loc[restants])
    noms[c] = nom
    restants.remove(c)
for c in restants:
    noms[c] = "Profil intermédiaire"
X["groupe"] = X["cluster"].map(noms)

# Variables illustratives : elles n'ont pas servi à former les groupes
rev = pd.read_csv("data/raw/insee_revenu_median_dep_2013_2021.csv", dtype={"C_DEPT": str})
X = X.merge(rev[rev["ANNEE"].isin(ENQUETES)].groupby("C_DEPT")["MEDIANE_REVENU"].mean().rename("revenu"),
            left_on="C_DEPT", right_index=True, how="left")
trait = dma[dma["ANNEE"].isin(ENQUETES)].pivot_table(index="C_DEPT", columns="L_TYP_REG_SERVICE",
                                                     values="TONNAGE_DMA", aggfunc="sum").fillna(0)
trait = trait.div(trait.sum(axis=1), axis=0) * 100
X = X.merge(trait[["Incinération avec récupération d'énergie", "Stockage", "Valorisation matière",
                   "Valorisation organique"]], left_on="C_DEPT", right_index=True, how="left")

ordre = X.groupby("groupe")["OMR"].size().sort_values(ascending=False).index
print(f"\n=== Étape 3 : k-means avec k = {K_RETENU} ===")
resume = X.groupby("groupe").agg(departements=("N_DEPT", "size"), silhouette=("silhouette", "mean"),
                                 **{c: (c, "mean") for c in COLS}, total_kg_hab=("RATIO_DMA", "mean"))
print(resume.loc[ordre].round(1).to_string())
print("\nMoyenne nationale (kg/hab) :", X[COLS].mean().round(0).to_dict())
illus = X.groupby("groupe").agg(population=("VA_POPANNEE", "mean"), revenu=("revenu", "mean"),
                                incineration=("Incinération avec récupération d'énergie", "mean"),
                                stockage=("Stockage", "mean"), valo_matiere=("Valorisation matière", "mean"),
                                valo_organique=("Valorisation organique", "mean"))
print("\nVariables illustratives (moyennes par groupe ; traitement en %) :")
print(illus.loc[ordre].round(0).to_string())
print("\nMembres (du plus typique au moins typique) :")
for g in ordre:
    m = X[X["groupe"] == g].sort_values("silhouette", ascending=False)
    print(f"- {g} ({len(m)}) : {', '.join(m['N_DEPT'])}")
print("\nDépartements à silhouette négative (plus proches d'un autre groupe) :",
      ", ".join(X.loc[X["silhouette"] < 0, "N_DEPT"]) or "aucun")

# Figure : profils en indice (100 = moyenne nationale)
indice = (resume.loc[ordre, COLS] / X[COLS].mean() * 100)
fig, ax = plt.subplots(figsize=(8, 3.2))
largeur = 0.8 / len(ordre)
couleur = {g: COULEURS[i] for i, g in enumerate(ordre)}
for i, g in enumerate(ordre):
    ax.bar(np.arange(len(COLS)) + (i - (len(ordre) - 1) / 2) * largeur, indice.loc[g], largeur,
           color=couleur[g], label=f"{g} ({int(resume.loc[g, 'departements'])})")
ax.axhline(100, color="#5B6559", lw=1)
ax.set_xticks(np.arange(len(COLS)), ["Ordures\nménagères", "Recyclables", "Déchets\nverts", "Encombrants", "Déchets\ndangereux"])
ax.set_ylabel("indice (100 = moyenne)")
ax.set_ylim(0, 215)
ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper left")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig("data/figures/clustering_profils.png", dpi=200)
plt.close(fig)

# Figure : carte (métropole ; outre-mer cité en légende)
geo = json.load(open("data/raw/contours-departements.geojson", encoding="utf-8"))
groupe_de = dict(zip(X["C_DEPT"], X["groupe"]))
fig, ax = plt.subplots(figsize=(6.2, 6.6))
for f in geo["features"]:
    code = f["properties"]["code"]
    geom = f["geometry"]
    polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
    teinte = couleur.get(groupe_de.get(code), "#DDDDDD")
    for poly in polys:
        pts = np.array(poly[0])
        if pts[:, 0].min() < -6 or pts[:, 1].min() < 41:  # outre-mer : hors de la carte métropolitaine
            continue
        ax.add_patch(Polygon(pts, closed=True, facecolor=teinte, edgecolor="white", linewidth=0.4))
ax.set_xlim(-5.3, 9.8); ax.set_ylim(41.2, 51.2); ax.set_aspect(1.45); ax.axis("off")
outre_mer = X[X["C_DEPT"].str.len() == 3]
ax.legend(handles=[Patch(color=couleur[g], label=g) for g in ordre], frameon=False, fontsize=8,
          loc="upper left", bbox_to_anchor=(-0.02, 1.0))
fig.text(0.5, 0.02, "Outre-mer : " + " ; ".join(f"{r.N_DEPT} ({r.groupe})" for r in outre_mer.itertuples())
         + ". Mayotte : pas de détail par type de déchet.", fontsize=7, ha="center", color="#5B6559", wrap=True)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig("data/figures/clustering_carte.png", dpi=200)
plt.close(fig)

choix.to_csv("data/processed/clustering_choix_k.csv", index=False)
X.to_csv("data/processed/clustering_production.csv", index=False)
print("\nSauvegardé -> data/processed/clustering_*.csv, data/figures/clustering_*.png")
