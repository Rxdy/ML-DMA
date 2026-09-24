"""
Clustering des départements selon leur profil de traitement des DMA.

Objectif : regrouper les départements qui ont une stratégie de traitement
similaire (part de tonnage en incinération / stockage / valorisation / etc.)
pour faire émerger des typologies territoriales.

Source : data/sinoe_dma.csv (2009-2021, seul fichier avec la ventilation
par type de traitement).
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = pd.read_csv("data/raw/sinoe_dma.csv", decimal=",")

# --- Profil de traitement par département (agrégé sur toutes les années) ---
pivot = df.pivot_table(
    index=["C_DEPT", "N_DEPT", "C_REGION", "L_REGION"],
    columns="L_TYP_REG_SERVICE",
    values="TONNAGE_DMA",
    aggfunc="sum",
    fill_value=0,
)

# Conversion en pourcentages (chaque ligne somme à 100%)
profil_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
profil_pct = profil_pct.reset_index()

print("=== Profil de traitement (%) par département ===")
print(profil_pct.head())
print(f"\nShape: {profil_pct.shape}")

feature_cols = [c for c in profil_pct.columns
                 if c not in ("C_DEPT", "N_DEPT", "C_REGION", "L_REGION")]

X = profil_pct[feature_cols].values
X_scaled = StandardScaler().fit_transform(X)

# --- Choix du nombre de clusters (silhouette score) ---
print("\n=== Recherche du k optimal ===")
scores = {}
for k in range(2, 8):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    scores[k] = score
    print(f"k={k}: silhouette={score:.3f}")

best_k = max(scores, key=scores.get)
print(f"\nMeilleur k = {best_k}")

km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
profil_pct["cluster"] = km.fit_predict(X_scaled)

print("\n=== Taille des clusters ===")
print(profil_pct["cluster"].value_counts().sort_index())

print("\n=== Profil moyen par cluster ===")
print(profil_pct.groupby("cluster")[feature_cols].mean().round(1))

print("\n=== Départements par cluster (exemples) ===")
for c in sorted(profil_pct["cluster"].unique()):
    depts = profil_pct[profil_pct["cluster"] == c]["N_DEPT"].tolist()
    print(f"\nCluster {c} ({len(depts)} départements) : {depts[:10]}{'...' if len(depts) > 10 else ''}")

# Sauvegarde
profil_pct.to_csv("data/processed/processed_clusters_traitement.csv", index=False)
print("\n Sauvegardé -> data/processed/processed_clusters_traitement.csv")
