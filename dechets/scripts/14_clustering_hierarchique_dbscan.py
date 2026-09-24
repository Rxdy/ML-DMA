"""
Les deux autres méthodes de clustering du cours, appliquées aux mêmes données que le
k-means (script 12) : 5 types de déchets en kg par habitant, standardisés.

  1. Clustering hiérarchique agglomératif (bottom-up), liaison de Ward :
     chaque département part seul, les deux groupes les plus proches fusionnent à chaque
     étape. Le dendrogramme montre ces fusions ; le couper à une hauteur donne k groupes.
  2. DBSCAN : groupes fondés sur la densité. Deux réglages : ε (rayon du voisinage) et
     MinPts (nombre minimal de voisins pour être un point de cœur). Les points isolés
     sont classés comme bruit. ε est choisi avec la courbe des k-distances.

À lancer après le script 12.
Sorties : data/processed/clustering_dbscan.csv, clustering_hierarchique.csv,
          data/figures/clustering_dendrogramme.png, clustering_dbscan_kdistance.png,
          clustering_dbscan.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

pd.set_option("display.width", 200)

COLS = ["OMR", "Recyclables", "Verts", "Encombrants", "Dangereux"]
ORDRE = ["Profil intermédiaire", "Déchets verts et déchèteries", "Grandes agglomérations", "Ordures ménagères élevées"]
COULEUR = dict(zip(ORDRE, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))

X = pd.read_csv("data/processed/clustering_production.csv", dtype={"C_DEPT": str})
Z = StandardScaler().fit_transform(X[COLS])

# ------------------------------------------------------------------ 1. hiérarchique
fusions = linkage(Z, method="ward")  # distance euclidienne, liaison de Ward
hauteurs = fusions[::-1, 2]           # hauteur de la dernière fusion, de l'avant-dernière...
print("=== 1. Clustering hiérarchique (Ward) ===")
print("Hauteur de fusion quand on passe de k à k-1 groupes :")
for k in range(2, 9):
    print(f"  {k} -> {k - 1} groupes : {hauteurs[k - 2]:.2f}")
ecarts = -np.diff(hauteurs[:8])
print("Saut de hauteur entre deux fusions successives (grand saut = bon endroit pour couper) :")
for k in range(2, 9):
    print(f"  couper en {k} groupes : saut de {ecarts[k - 2]:.2f}")

K = 4
coupe = (hauteurs[K - 2] + hauteurs[K - 1]) / 2  # entre la fusion 5 -> 4 et la fusion 4 -> 3
hier = fcluster(fusions, t=K, criterion="maxclust")
X["groupe_hierarchique"] = hier
ari = adjusted_rand_score(X["cluster"], hier)
print(f"\nCoupe en {K} groupes (hauteur {coupe:.2f}) : silhouette {silhouette_score(Z, hier):.3f}, "
      f"ARI avec le k-means {ari:.3f}")
print(pd.crosstab(X["groupe"], hier).loc[ORDRE])

fig, ax = plt.subplots(figsize=(11, 4.8))
dendrogram(fusions, labels=X["N_DEPT"].tolist(), color_threshold=coupe, above_threshold_color="#8B958A",
           leaf_font_size=5.5, leaf_rotation=90, ax=ax)
ax.axhline(coupe, ls="--", color="#9C6B2A", lw=1)
ax.text(ax.get_xlim()[1], coupe + 0.3, f"coupe : {K} groupes", ha="right", fontsize=8, color="#9C6B2A")
ax.set_ylabel("distance de fusion (Ward)")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig("data/figures/clustering_dendrogramme.png", dpi=200)
plt.close(fig)

# ------------------------------------------------------------------ 2. DBSCAN
MINPTS = 2 * len(COLS)  # règle usuelle : deux fois le nombre de dimensions
distances, _ = NearestNeighbors(n_neighbors=MINPTS).fit(Z).kneighbors(Z)
kdist = np.sort(distances[:, -1])
# coude de la courbe des k-distances : point le plus éloigné de la droite premier -> dernier point
x = np.arange(len(kdist))
droite = kdist[0] + (kdist[-1] - kdist[0]) * x / x[-1]
EPS = float(kdist[np.argmax(droite - kdist)])
print(f"\n=== 2. DBSCAN (MinPts = {MINPTS}, ε = {EPS:.2f} choisi au coude des k-distances) ===")

db = DBSCAN(eps=EPS, min_samples=MINPTS).fit(Z)
lab = db.labels_
coeur = np.zeros(len(Z), dtype=bool)
coeur[db.core_sample_indices_] = True
X["dbscan_groupe"] = lab
X["dbscan_type"] = np.where(lab == -1, "bruit", np.where(coeur, "cœur", "bordure"))
print(f"Groupes denses trouvés : {len(set(lab)) - (1 if -1 in lab else 0)}")
print(X["dbscan_type"].value_counts().to_string())
print("Points de bruit :", ", ".join(X.loc[lab == -1, "N_DEPT"]))
print("Groupe k-means des points de bruit :")
print(X.loc[lab == -1, "groupe"].value_counts().to_string())

print("\nSensibilité aux réglages (nombre de groupes denses / points de bruit) :")
sens = []
for mp in [5, 10]:
    for eps in [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]:
        l = DBSCAN(eps=eps, min_samples=mp).fit(Z).labels_
        sens.append({"MinPts": mp, "eps": eps, "groupes": len(set(l)) - (1 if -1 in l else 0),
                     "bruit": int((l == -1).sum())})
sens = pd.DataFrame(sens)
print(sens.pivot(index="eps", columns="MinPts", values=["groupes", "bruit"]).to_string())

fig, ax = plt.subplots(figsize=(6.4, 3.2))
ax.plot(kdist, color="#2F5236")
ax.axhline(EPS, ls="--", color="#9C6B2A", lw=1)
ax.text(2, EPS + 0.06, f"ε = {EPS:.2f}".replace(".", ","), fontsize=8, color="#9C6B2A")
ax.set(xlabel="départements, triés", ylabel=f"distance au {MINPTS}ᵉ voisin")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("data/figures/clustering_dbscan_kdistance.png", dpi=200)
plt.close(fig)

pcs = PCA(n_components=2).fit(Z).transform(Z)
fig, ax = plt.subplots(figsize=(7.4, 5.2))
style = {"cœur": dict(color="#2F5236", s=26), "bordure": dict(color="#8FC397", s=26),
         "bruit": dict(color="#C0392B", s=40, marker="x")}
for t, st in style.items():
    m = (X["dbscan_type"] == t).to_numpy()
    ax.scatter(pcs[m, 0], pcs[m, 1], label=f"point de {t} ({m.sum()})" if t != "bruit" else f"bruit ({m.sum()})", **st)
for i in np.where(lab == -1)[0]:
    ax.annotate(X.loc[i, "N_DEPT"], pcs[i], xytext=(4, 3), textcoords="offset points", fontsize=7)
ax.axhline(0, color="#C9D2C3", lw=0.8); ax.axvline(0, color="#C9D2C3", lw=0.8)
ax.set(xlabel="PC1 : tri et apports en déchèterie  →", ylabel="PC2 : ordures ménagères et encombrants  →")
ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig("data/figures/clustering_dbscan.png", dpi=200)
plt.close(fig)

X[["C_DEPT", "N_DEPT", "groupe", "groupe_hierarchique"]].to_csv("data/processed/clustering_hierarchique.csv", index=False)
X[["C_DEPT", "N_DEPT", "groupe", "dbscan_groupe", "dbscan_type"]].to_csv("data/processed/clustering_dbscan.csv", index=False)
sens.to_csv("data/processed/clustering_dbscan_sensibilite.csv", index=False)
pd.DataFrame({"k": range(2, 9), "hauteur_fusion": hauteurs[:7], "saut": ecarts[:7]}) \
    .to_csv("data/processed/clustering_hierarchique_hauteurs.csv", index=False)
print("\nSauvegardé -> data/processed/clustering_hierarchique*.csv, clustering_dbscan*.csv, data/figures/clustering_*.png")
