"""
Analyses complémentaires du clustering (script 12) : voir les données en plusieurs
dimensions et vérifier la solidité des 4 groupes.

  1. Corrélations entre les 5 variables.
  2. ACP (analyse en composantes principales) : résumer les 5 dimensions en 2 axes
     pour pouvoir les dessiner ; cercle des corrélations ; projection des départements.
  3. Nuages de points variable par variable (matrice de nuages), colorés par groupe.
  4. Silhouette de chaque département.
  5. Robustesse : autres graines du k-means, autre méthode (classification hiérarchique
     de Ward), sans l'outre-mer, et comparaison avec le clustering du traitement.

À lancer après le script 12.
Sorties : data/processed/clustering_acp.csv, clustering_robustesse.csv,
          data/figures/clustering_correlations.png, clustering_acp_variance.png,
          clustering_acp_cercle.png, clustering_acp_projection.png,
          clustering_nuages.png, clustering_silhouette_departements.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

COLS = ["OMR", "Recyclables", "Verts", "Encombrants", "Dangereux"]
LIBELLES = {"OMR": "Ordures ménagères", "Recyclables": "Recyclables", "Verts": "Déchets verts",
            "Encombrants": "Encombrants", "Dangereux": "Déchets dangereux"}
ORDRE = ["Profil intermédiaire", "Déchets verts et déchèteries", "Grandes agglomérations", "Ordures ménagères élevées"]
COULEUR = dict(zip(ORDRE, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))

X = pd.read_csv("data/processed/clustering_production.csv", dtype={"C_DEPT": str})
Z = StandardScaler().fit_transform(X[COLS])
couleurs = X["groupe"].map(COULEUR)

# ------------------------------------------------------------------ 1. corrélations
corr = X[COLS].corr()
print("=== 1. Corrélations entre les 5 variables ===")
print(corr.round(2))
fig, ax = plt.subplots(figsize=(5.4, 4.4))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(5), [LIBELLES[c] for c in COLS], rotation=35, ha="right")
ax.set_yticks(range(5), [LIBELLES[c] for c in COLS])
for i in range(5):
    for j in range(5):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}".replace(".", ","), ha="center", va="center", fontsize=9,
                color="white" if abs(corr.iloc[i, j]) > 0.6 else "black")
fig.colorbar(im, ax=ax, shrink=0.8, label="corrélation")
fig.tight_layout()
fig.savefig("data/figures/clustering_correlations.png", dpi=200)
plt.close(fig)

# ------------------------------------------------------------------ 2. ACP
acp = PCA().fit(Z)
variance = acp.explained_variance_ratio_
coords = acp.transform(Z)
X["axe1"], X["axe2"] = coords[:, 0], coords[:, 1]
# corrélation de chaque variable avec chaque axe (= coordonnées sur le cercle des corrélations)
charges = pd.DataFrame(acp.components_.T * np.sqrt(acp.explained_variance_), index=COLS,
                       columns=[f"axe{i + 1}" for i in range(len(COLS))])
print("\n=== 2. ACP ===")
print("Part de l'information (variance) portée par chaque axe :",
      ", ".join(f"axe {i + 1} : {v:.1%}" for i, v in enumerate(variance)))
print(f"Les 2 premiers axes résument {variance[:2].sum():.1%} de l'information.")
print("Corrélation des variables avec les axes :")
print(charges.iloc[:, :3].round(2))
print("Position moyenne des groupes sur les axes :")
print(X.groupby("groupe")[["axe1", "axe2"]].mean().loc[ORDRE].round(2))

fig, ax = plt.subplots(figsize=(5.6, 2.8))
ax.bar(range(1, 6), variance * 100, color="#2F5236")
ax.plot(range(1, 6), variance.cumsum() * 100, marker="o", color="#9C6B2A", label="cumul")
for i, v in enumerate(variance):
    ax.text(i + 1, v * 100 + 2, f"{v:.0%}", ha="center", fontsize=8)
ax.set(xlabel="axe", ylabel="% de l'information", ylim=(0, 105))
ax.legend(frameon=False, fontsize=8)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig("data/figures/clustering_acp_variance.png", dpi=200)
plt.close(fig)

fig, ax = plt.subplots(figsize=(5.2, 5.2))
ax.add_patch(plt.Circle((0, 0), 1, fill=False, color="#8B958A"))
ax.axhline(0, color="#C9D2C3", lw=0.8); ax.axvline(0, color="#C9D2C3", lw=0.8)
for c in COLS:
    x, y = charges.loc[c, "axe1"], charges.loc[c, "axe2"]
    ax.annotate("", xy=(x, y), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="#2F5236", lw=1.4))
    ax.text(x * 1.12, y * 1.12, LIBELLES[c], ha="center", va="center", fontsize=8.5)
ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2); ax.set_aspect("equal")
ax.set_xlabel(f"axe 1 ({variance[0]:.0%}) : tri et apports en déchèterie")
ax.set_ylabel(f"axe 2 ({variance[1]:.0%}) : ordures ménagères et encombrants")
fig.tight_layout()
fig.savefig("data/figures/clustering_acp_cercle.png", dpi=200)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7.4, 5.4))
for g in ORDRE:
    m = X["groupe"] == g
    ax.scatter(X.loc[m, "axe1"], X.loc[m, "axe2"], s=22, color=COULEUR[g], alpha=0.85, label=g, edgecolors="none")
extremes = pd.concat([X.nlargest(3, "axe1"), X.nsmallest(3, "axe1"), X.nlargest(3, "axe2"), X.nsmallest(2, "axe2")])
for r in extremes.drop_duplicates("C_DEPT").itertuples():
    ax.annotate(r.N_DEPT, (r.axe1, r.axe2), xytext=(4, 3), textcoords="offset points", fontsize=7, color="#1B221C")
ax.axhline(0, color="#C9D2C3", lw=0.8); ax.axvline(0, color="#C9D2C3", lw=0.8)
ax.set_xlabel(f"axe 1 ({variance[0]:.0%}) : tri et apports en déchèterie  →")
ax.set_ylabel(f"axe 2 ({variance[1]:.0%}) : ordures ménagères et encombrants  →")
ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig("data/figures/clustering_acp_projection.png", dpi=200)
plt.close(fig)

# ------------------------------------------------------------------ 3. matrice de nuages
fig, axes = plt.subplots(5, 5, figsize=(9, 9))
for i, ci in enumerate(COLS):
    for j, cj in enumerate(COLS):
        ax = axes[i, j]
        if i == j:
            for g in ORDRE:
                ax.hist(X.loc[X["groupe"] == g, ci], bins=12, color=COULEUR[g], alpha=0.6)
        else:
            ax.scatter(X[cj], X[ci], s=6, c=couleurs, edgecolors="none")
        ax.tick_params(labelsize=6)
        if i == 4:
            ax.set_xlabel(LIBELLES[cj], fontsize=7.5)
        if j == 0:
            ax.set_ylabel(LIBELLES[ci], fontsize=7.5)
fig.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=COULEUR[g], label=g) for g in ORDRE],
           loc="upper center", ncol=4, frameon=False, fontsize=8)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig("data/figures/clustering_nuages.png", dpi=170)
plt.close(fig)

# ------------------------------------------------------------------ 4. silhouette par département
fig, ax = plt.subplots(figsize=(7, 6))
y = 0
for g in ORDRE:
    s = X.loc[X["groupe"] == g].sort_values("silhouette")
    ax.barh(range(y, y + len(s)), s["silhouette"], color=COULEUR[g], height=1.0)
    ax.text(-0.26, y + len(s) / 2, f"{g}\n({len(s)})", fontsize=7.5, va="center", ha="right")
    for r, pos in zip(s.itertuples(), range(y, y + len(s))):
        if r.silhouette < 0:
            ax.text(r.silhouette - 0.01, pos, r.N_DEPT, fontsize=6.5, va="center", ha="right")
    y += len(s) + 3
moy = X["silhouette"].mean()
ax.axvline(moy, ls="--", color="#9C6B2A", lw=1)
ax.text(moy + 0.01, y - 2, f"moyenne {moy:.3f}".replace(".", ","), fontsize=7.5, color="#9C6B2A")
ax.axvline(0, color="#5B6559", lw=0.8)
ax.set_yticks([]); ax.set_xlabel("silhouette du département"); ax.set_xlim(-0.55, 0.7)
fig.tight_layout()
fig.savefig("data/figures/clustering_silhouette_departements.png", dpi=200)
plt.close(fig)
print("\n=== 4. Silhouette par groupe ===")
print(X.groupby("groupe")["silhouette"].agg(["mean", "min", "max"]).loc[ORDRE].round(3))

# ------------------------------------------------------------------ 5. robustesse
ref = X["cluster"].to_numpy()
rob = []
for graine in range(10):
    lab = KMeans(n_clusters=4, n_init=20, random_state=graine).fit(Z).labels_
    rob.append({"essai": f"k-means, graine {graine}", "ARI": adjusted_rand_score(ref, lab),
                "silhouette": silhouette_score(Z, lab)})
ward = AgglomerativeClustering(n_clusters=4, linkage="ward").fit(Z).labels_
rob.append({"essai": "Classification hiérarchique (Ward)", "ARI": adjusted_rand_score(ref, ward),
            "silhouette": silhouette_score(Z, ward)})
metro = X["C_DEPT"].str.len() == 2
Zm = StandardScaler().fit_transform(X.loc[metro, COLS])
lab_m = KMeans(n_clusters=4, n_init=20, random_state=42).fit(Zm).labels_
rob.append({"essai": "k-means sans l'outre-mer (96 départements)", "ARI": adjusted_rand_score(ref[metro.to_numpy()], lab_m),
            "silhouette": silhouette_score(Zm, lab_m)})
trait = pd.read_csv("data/processed/processed_clusters_traitement.csv", dtype={"C_DEPT": str})
trait["C_DEPT"] = trait["C_DEPT"].str.zfill(2)
mt = X.merge(trait[["C_DEPT", "cluster"]].rename(columns={"cluster": "cluster_traitement"}), on="C_DEPT")
rob.append({"essai": "Clustering du traitement (partie supervisée)", "ARI": adjusted_rand_score(mt["cluster"], mt["cluster_traitement"]),
            "silhouette": np.nan})
rob = pd.DataFrame(rob)
print("\n=== 5. Robustesse (ARI : 1 = mêmes groupes, 0 = aucun lien) ===")
print(rob.round(3).to_string(index=False))
print("\nk-means contre Ward (lignes : groupes k-means ; colonnes : groupes Ward) :")
print(pd.crosstab(X["groupe"], ward).loc[ORDRE])
print("\nGroupes de production contre groupes de traitement :")
print(pd.crosstab(mt["groupe"], mt["cluster_traitement"]).loc[ORDRE])

X[["C_DEPT", "N_DEPT", "groupe", "axe1", "axe2"]].to_csv("data/processed/clustering_acp.csv", index=False)
charges.to_csv("data/processed/clustering_acp_charges.csv")
rob.to_csv("data/processed/clustering_robustesse.csv", index=False)
print("\nSauvegardé -> data/processed/clustering_acp*.csv, clustering_robustesse.csv, data/figures/clustering_*.png")
