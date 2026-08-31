import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# ==========================================
# CONFIGURATION
# ==========================================

DATA_DIR = Path("data")
MODEL_DIR = Path("models")

INPUT_FILE = DATA_DIR / "ddos_clean.csv"
MODEL_FILE = MODEL_DIR / "random_forest_ids.pkl"


# ==========================================
# 1. LOAD CLEAN DATASET
# ==========================================

print("Loading cleaned dataset...")

df = pd.read_csv(INPUT_FILE)

print("Dataset shape:", df.shape)


# ==========================================
# 2. SEPARATE FEATURES AND LABEL
# ==========================================

X = df.drop(columns=["Label"])

y = df["Label"]

print("\nNumber of features:", X.shape[1])

print("\nLabel distribution:")
print(y.value_counts())


# ==========================================
# 3. TRAIN / TEST SPLIT
# ==========================================

print("\nCreating train/test split...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ==========================================
# 4. CREATE RANDOM FOREST MODEL
# ==========================================

print("\nCreating Random Forest model...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)


# ==========================================
# 5. TRAIN MODEL
# ==========================================

print("\nTraining model...")

model.fit(X_train, y_train)

print("Training completed.")


# ==========================================
# 6. TEST MODEL
# ==========================================

print("\nGenerating predictions...")

y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)[:, 1]


# ==========================================
# 7. BASIC RESULTS
# ==========================================

accuracy = (y_pred == y_test).mean()

print("\n==========================================")
print("INITIAL MODEL RESULT")
print("==========================================")

print("Accuracy:", round(accuracy, 4))


# ==========================================
# 8. SAVE MODEL
# ==========================================

MODEL_DIR.mkdir(exist_ok=True)

joblib.dump(model, MODEL_FILE)

print("\nModel saved to:")
print(MODEL_FILE)

print("\nTraining process completed.")