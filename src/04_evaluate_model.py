import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt


# ==========================================
# CONFIGURATION
# ==========================================

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

INPUT_FILE = DATA_DIR / "ddos_clean_no_port.csv"
MODEL_FILE = MODEL_DIR / "random_forest_ids_no_port.pkl"


# ==========================================
# CREATE RESULTS DIRECTORY
# ==========================================

RESULTS_DIR.mkdir(exist_ok=True)


# ==========================================
# 1. LOAD DATA
# ==========================================

print("Loading dataset...")

df = pd.read_csv(INPUT_FILE)

X = df.drop(columns=["Label"])
y = df["Label"]


# ==========================================
# 2. SAME TRAIN/TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# 3. LOAD TRAINED MODEL
# ==========================================

print("Loading trained model...")

model = joblib.load(MODEL_FILE)


# ==========================================
# 4. PREDICTIONS
# ==========================================

print("Generating predictions...")

y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)[:, 1]


# ==========================================
# 5. CALCULATE METRICS
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


# ==========================================
# 6. DISPLAY RESULTS
# ==========================================

print("\n==========================================")
print("MODEL EVALUATION")
print("==========================================")

print(f"Accuracy :  {accuracy:.4f}")
print(f"Precision:  {precision:.4f}")
print(f"Recall   :  {recall:.4f}")
print(f"F1 Score :  {f1:.4f}")
print(f"ROC-AUC  :  {roc_auc:.4f}")


# ==========================================
# 7. CLASSIFICATION REPORT
# ==========================================

print("\n==========================================")
print("CLASSIFICATION REPORT")
print("==========================================")

report = classification_report(
    y_test,
    y_pred,
    target_names=["BENIGN", "ATTACK"],
    zero_division=0
)

print(report)


# ==========================================
# 8. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\n==========================================")
print("CONFUSION MATRIX")
print("==========================================")

print(cm)


# ==========================================
# 9. SAVE CONFUSION MATRIX
# ==========================================

plt.figure(figsize=(7, 6))

plt.imshow(cm)

plt.title("Random Forest IDS Confusion Matrix")

plt.xlabel("Predicted Label")

plt.ylabel("Actual Label")

plt.xticks(
    [0, 1],
    ["BENIGN", "ATTACK"]
)

plt.yticks(
    [0, 1],
    ["BENIGN", "ATTACK"]
)

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()

output_file = RESULTS_DIR / "confusion_matrix_no_port.png"

plt.savefig(output_file)

plt.close()

print("\nConfusion matrix saved to:")
print(output_file)


# ==========================================
# 10. SAVE TEXT RESULTS
# ==========================================

results_file = RESULTS_DIR / "model_results_no_port.txt"

with open(results_file, "w") as file:

    file.write("Random Forest IDS Evaluation\n")
    file.write("============================\n\n")

    file.write(f"Accuracy : {accuracy:.4f}\n")
    file.write(f"Precision: {precision:.4f}\n")
    file.write(f"Recall   : {recall:.4f}\n")
    file.write(f"F1 Score : {f1:.4f}\n")
    file.write(f"ROC-AUC  : {roc_auc:.4f}\n\n")

    file.write("Classification Report\n")
    file.write("=====================\n")
    file.write(report)

    file.write("\n\nConfusion Matrix\n")
    file.write("================\n")
    file.write(str(cm))

print("\nResults saved to:")
print(results_file)