"""
Analyses complémentaires du clustering (script 12) : voir les données en plusieurs
dimensions et vérifier la solidité des 4 groupes.

  1. Corrélations entre les 5 variables.
  2. PCA / ACP (analyse en composantes principales), étape par étape comme dans le cours :
     standardisation, matrice de covariance, valeurs et vecteurs propres, choix du nombre
     de composantes (éboulis, règle des 80-90 %), projection sur PC1, PC2, PC3 ;
     cercle des corrélations (chargements) ; centroïdes des groupes sur la projection.
  3. Nuages de points variable par variable (matrice de nuages), colorés par groupe.
  4. Silhouette de chaque département.
  5. Robustesse : autres graines du k-means, autre méthode (classification hiérarchique
     de Ward), sans l'outre-mer, et comparaison avec le clustering du traitement.

À lancer après le script 12.
Sorties : data/processed/clustering_acp.csv, clustering_robustesse.csv,
          clustering_acp_valeurs_propres.csv,
          data/figures/clustering_correlations.png, clustering_acp_variance.png,
          clustering_acp_cercle.png, clustering_acp_projection.png, clustering_acp_projection_pc3.png,
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

# ------------------------------------------------------------------ 2. PCA
# Étape 1 : standardisation (déjà faite : Z). Étape 2 : matrice de covariance des données
# standardisées -- identique à la matrice de corrélation. Étapes 3 et 4 : valeurs propres
# (variance portée par chaque composante) et vecteurs propres (direction de chaque
# composante), triés par valeur propre décroissante. Étape 5 : projection.
covariance = np.cov(Z, rowvar=False)
assert np.allclose(covariance * (len(Z) - 1) / len(Z), corr.to_numpy())
acp = PCA().fit(Z)
valeurs_propres = acp.explained_variance_
variance = acp.explained_variance_ratio_
PCS = [f"PC{i + 1}" for i in range(len(COLS))]
vecteurs = pd.DataFrame(acp.components_.T, index=COLS, columns=PCS)
coords = acp.transform(Z)
for i in range(3):
    X[PCS[i]] = coords[:, i]
# chargements : corrélation de chaque variable avec chaque composante (= vecteur propre × √valeur propre)
charges = vecteurs * np.sqrt(valeurs_propres)
n_80 = int(np.argmax(variance.cumsum() >= 0.8)) + 1
print("\n=== 2. PCA ===")
print("Matrice de covariance des données standardisées = matrice de corrélation : vérifié.")
vp = pd.DataFrame({"composante": PCS, "valeur_propre": valeurs_propres, "part_variance": variance,
                   "part_cumulee": variance.cumsum()})
print(vp.round(3).to_string(index=False))
print(f"Règle des 80-90 % : {n_80} composantes ({variance[:n_80].sum():.1%} de la variance).")
print("Vecteurs propres (coefficients de chaque variable dans chaque composante) :")
print(vecteurs.iloc[:, :3].round(2))
print("Chargements (corrélation variable / composante) :")
print(charges.iloc[:, :3].round(2))
print("Position moyenne des groupes (centroïdes) sur les composantes :")
print(X.groupby("groupe")[PCS[:3]].mean().loc[ORDRE].round(2))

NOM_PC = {"PC1": "tri et apports en déchèterie", "PC2": "ordures ménagères et encombrants",
          "PC3": "déchets verts plutôt que tri"}

# Éboulis des valeurs propres
fig, ax = plt.subplots(figsize=(6.4, 3.0))
ax.bar(range(1, 6), valeurs_propres, color="#2F5236")
for i, v in enumerate(valeurs_propres):
    ax.text(i + 1, v + 0.05, f"{v:.2f}".replace(".", ","), ha="center", fontsize=8)
ax.set(xlabel="composante principale", ylabel="valeur propre", xticks=range(1, 6), xticklabels=PCS, ylim=(0, 2.7))
ax2 = ax.twinx()
ax2.plot(range(1, 6), variance.cumsum() * 100, marker="o", color="#9C6B2A")
ax2.axhline(80, ls="--", lw=0.8, color="#9C6B2A")
ax2.text(5.3, 80, "80 %", fontsize=7.5, color="#9C6B2A", va="center")
ax2.set(ylabel="% de variance cumulée", ylim=(0, 105))
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig("data/figures/clustering_acp_variance.png", dpi=200)
plt.close(fig)

# Cercle des corrélations (chargements) sur PC1-PC2
fig, ax = plt.subplots(figsize=(5.2, 5.2))
ax.add_patch(plt.Circle((0, 0), 1, fill=False, color="#8B958A"))
ax.axhline(0, color="#C9D2C3", lw=0.8); ax.axvline(0, color="#C9D2C3", lw=0.8)
for c in COLS:
    x, y = charges.loc[c, "PC1"], charges.loc[c, "PC2"]
    ax.annotate("", xy=(x, y), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="#2F5236", lw=1.4))
    ax.text(x * 1.12, y * 1.12, LIBELLES[c], ha="center", va="center", fontsize=8.5)
ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2); ax.set_aspect("equal")
ax.set_xlabel(f"PC1 ({variance[0]:.0%}) : {NOM_PC['PC1']}")
ax.set_ylabel(f"PC2 ({variance[1]:.0%}) : {NOM_PC['PC2']}")
fig.tight_layout()
fig.savefig("data/figures/clustering_acp_cercle.png", dpi=200)
plt.close(fig)


def projection(axe_x, axe_y, fichier, n_extremes=3, legende="upper left"):
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    for g in ORDRE:
        m = X["groupe"] == g
        ax.scatter(X.loc[m, axe_x], X.loc[m, axe_y], s=22, color=COULEUR[g], alpha=0.8, label=g, edgecolors="none")
    centres = X.groupby("groupe")[[axe_x, axe_y]].mean()
    for g in ORDRE:  # centroïdes : la moyenne du groupe, projetée comme les départements
        ax.scatter(*centres.loc[g], marker="X", s=180, color=COULEUR[g], edgecolors="#1B221C", linewidths=1.2, zorder=5)
    ax.scatter([], [], marker="X", s=90, color="white", edgecolors="#1B221C", label="centroïde du groupe")
    ext = pd.concat([X.nlargest(n_extremes, axe_x), X.nsmallest(n_extremes, axe_x),
                     X.nlargest(n_extremes, axe_y), X.nsmallest(2, axe_y)]).drop_duplicates("C_DEPT")
    for r in ext.itertuples():
        ax.annotate(r.N_DEPT, (getattr(r, axe_x), getattr(r, axe_y)), xytext=(4, 3), textcoords="offset points",
                    fontsize=7, color="#1B221C")
    ax.axhline(0, color="#C9D2C3", lw=0.8); ax.axvline(0, color="#C9D2C3", lw=0.8)
    i, j = PCS.index(axe_x), PCS.index(axe_y)
    ax.set_xlabel(f"{axe_x} ({variance[i]:.0%}) : {NOM_PC[axe_x]}  →")
    ax.set_ylabel(f"{axe_y} ({variance[j]:.0%}) : {NOM_PC[axe_y]}  →")
    ax.legend(frameon=False, fontsize=8, loc=legende)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(fichier, dpi=200)
    plt.close(fig)


projection("PC1", "PC2", "data/figures/clustering_acp_projection.png")
projection("PC1", "PC3", "data/figures/clustering_acp_projection_pc3.png", legende="lower right")

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

X[["C_DEPT", "N_DEPT", "groupe", "PC1", "PC2", "PC3"]].to_csv("data/processed/clustering_acp.csv", index=False)
charges.to_csv("data/processed/clustering_acp_charges.csv")
vecteurs.to_csv("data/processed/clustering_acp_vecteurs_propres.csv")
vp.to_csv("data/processed/clustering_acp_valeurs_propres.csv", index=False)
rob.to_csv("data/processed/clustering_robustesse.csv", index=False)
print("\nSauvegardé -> data/processed/clustering_acp*.csv, clustering_robustesse.csv, data/figures/clustering_*.png")
