"""
Prédiction de l'enquête 2025, qui n'est pas encore publiée par l'ADEME.

Le modèle retenu (pipeline régression linéaire) est réentraîné sur TOUTES les
enquêtes disponibles (2011 à 2023), puis appliqué aux valeurs de 2023 :
  - RATIO_DMA_lag1 / TONNAGE_DMA_lag1 = valeurs réelles 2023 ;
  - VA_POPANNEE = population 2023 (celle de 2025 n'est pas connue : approximation) ;
  - cluster     = profil de traitement du département (inchangé).

Ces prédictions ne peuvent pas encore être vérifiées : elles le seront quand
l'ADEME publiera l'enquête 2025. L'intervalle donné vient des erreurs réellement
observées sur le test (2021, 2023) : 80 % des erreurs étaient inférieures à ce seuil.

Sortie : data/processed/predictions_2025.csv
"""
import pandas as pd
from sklearn.base import clone

ns = {}
exec(open("scripts/09_pipeline.py").read().split("\n# --- Validation croisée")[0]
     .replace("print(", "(lambda *a, **k: None)("), ns)
df, num_cols, cat_cols = ns["df"], ns["num_cols"], ns["cat_cols"]
pipeline = ns["pipelines"]["Régression linéaire"]

# Largeur d'intervalle : erreurs du modèle entraîné sur 2011-2019, mesurées sur 2021-2023
evaluation = clone(pipeline).fit(ns["X_train"], ns["y_train"])
erreurs = (evaluation.predict(ns["X_test"]) - ns["y_test"]).abs()
marge = erreurs.quantile(0.8)

# Modèle final : toutes les enquêtes
final = clone(pipeline).fit(df[num_cols + cat_cols], df["RATIO_DMA"])

# Entrées pour 2025 : les valeurs de 2023 deviennent « l'enquête précédente »
base = df[df["ANNEE"] == 2023]
X_2025 = pd.DataFrame({
    "VA_POPANNEE": base["VA_POPANNEE"],
    "RATIO_DMA_lag1": base["RATIO_DMA"],
    "TONNAGE_DMA_lag1": base["TONNAGE_DMA"],
    "cluster": base["cluster"],
})
pred = pd.DataFrame({
    "C_DEPT": base["C_DEPT"], "N_DEPT": base["N_DEPT"],
    "ratio_2023_reel": base["RATIO_DMA"].round(1),
    "ratio_2025_predit": final.predict(X_2025).round(1),
})
pred["borne_basse"] = (pred["ratio_2025_predit"] - marge).round(1)
pred["borne_haute"] = (pred["ratio_2025_predit"] + marge).round(1)

print(f"Modèle final entraîné sur {len(df)} lignes (2011-2023)")
print(f"Marge d'erreur (80 % des erreurs observées sur le test) : ± {marge:.1f} kg/hab")
print(f"\nMoyenne 2023 réelle : {pred['ratio_2023_reel'].mean():.1f} kg/hab")
print(f"Moyenne 2025 prédite : {pred['ratio_2025_predit'].mean():.1f} kg/hab")
print("\nExtrait :")
print(pred.head(10).to_string(index=False))
print("\nLimite : la plus grosse source d'erreur est l'évolution nationale d'une enquête à l'autre\n"
      "(+22 kg/hab en 2021, -44 en 2023), que le modèle ne peut pas anticiper (script 11).")

pred.to_csv("data/processed/predictions_2025.csv", index=False)
print("\nSauvegardé -> data/processed/predictions_2025.csv")
