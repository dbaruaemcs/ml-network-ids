import pandas as pd
import joblib

from pathlib import Path

import matplotlib.pyplot as plt


# ==========================================
# CONFIGURATION
# ==========================================

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

INPUT_FILE = DATA_DIR / "ddos_clean.csv"
MODEL_FILE = MODEL_DIR / "random_forest_ids.pkl"


# ==========================================
# LOAD DATA AND MODEL
# ==========================================

df = pd.read_csv(INPUT_FILE)

X = df.drop(columns=["Label"])

model = joblib.load(MODEL_FILE)


# ==========================================
# FEATURE IMPORTANCE
# ==========================================

importance = model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})


# Sort
feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)


# ==========================================
# DISPLAY TOP FEATURES
# ==========================================

print("\n==========================================")
print("TOP 20 IMPORTANT FEATURES")
print("==========================================")

print(
    feature_importance.head(20).to_string(index=False)
)


# ==========================================
# PLOT
# ==========================================

top_features = feature_importance.head(15)

plt.figure(figsize=(10, 7))

plt.barh(
    top_features["Feature"][::-1],
    top_features["Importance"][::-1]
)

plt.xlabel("Importance")

plt.ylabel("Network Feature")

plt.title("Top Network Features Used by Random Forest")

plt.tight_layout()


output_file = RESULTS_DIR / "feature_importance.png"

plt.savefig(output_file)

plt.close()

print("\nFeature importance chart saved to:")
print(output_file)