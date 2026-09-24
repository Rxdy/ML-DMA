"""
Étape de corrélation : quelles variables influencent quelles autres ?

Objectif pédagogique : identifier les variables fortement corrélées
(risque de redondance / fuite) avant de choisir les features du modèle.
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("data/processed/processed_regression_dataset.csv")

num_cols = ["ANNEE", "VA_POPANNEE", "cluster",
            "RATIO_DMA_lag1", "TONNAGE_DMA_lag1",
            "TONNAGE_DMA", "RATIO_DMA"]

corr = df[num_cols].corr()

print("=== Matrice de corrélation ===")
print(corr.round(2))

print("\n=== Corrélations avec la cible RATIO_DMA (triées) ===")
print(corr["RATIO_DMA"].drop("RATIO_DMA").sort_values(key=abs, ascending=False).round(3))

# Heatmap
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(num_cols)))
ax.set_yticks(range(len(num_cols)))
ax.set_xticklabels(num_cols, rotation=45, ha="right")
ax.set_yticklabels(num_cols)
for i in range(len(num_cols)):
    for j in range(len(num_cols)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
plt.colorbar(im, label="corrélation")
plt.title("Matrice de corrélation - dataset régression")
plt.tight_layout()
plt.savefig("data/figures/correlation_matrix.png", dpi=120)
print("\nHeatmap sauvegardée -> data/figures/correlation_matrix.png")
