from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import joblib


# ============================================================
# EXPERIMENT 3A
# MULTICLASS RANDOM FOREST IDS
# ============================================================

DATA_FILE = Path("data/multiclass_ids.csv")
RESULTS_DIR = Path("results")
MODELS_DIR = Path("models")

RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


RANDOM_STATE = 42

# Maximum number of samples per class.
# This prevents BENIGN and very large attack classes
# from dominating the experiment.
MAX_SAMPLES_PER_CLASS = 100_000


print("=" * 70)
print("EXPERIMENT 3A - MULTICLASS RANDOM FOREST IDS")
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(
    DATA_FILE,
    low_memory=False
)

print(f"Dataset shape: {df.shape}")


# ============================================================
# CHECK LABELS
# ============================================================

print("\nOriginal class distribution:")

print(
    df["Label"]
    .value_counts()
)


# ============================================================
# REMOVE SOURCE FILE FROM MODEL FEATURES
# ============================================================

if "Source_File" in df.columns:

    df = df.drop(
        columns=["Source_File"]
    )


# ============================================================
# SAMPLE EACH CLASS
# ============================================================

print("\nApplying class sampling...")

sampled_parts = []


for label, group in df.groupby("Label"):

    original_count = len(group)

    sample_size = min(
        original_count,
        MAX_SAMPLES_PER_CLASS
    )

    if original_count > sample_size:

        group = group.sample(
            n=sample_size,
            random_state=RANDOM_STATE
        )

    sampled_parts.append(group)

    print(
        f"{label:<25}"
        f"{original_count:>12,}"
        f" -> "
        f"{len(group):>12,}"
    )


df = pd.concat(
    sampled_parts,
    ignore_index=True
)


# ============================================================
# SHUFFLE
# ============================================================

df = df.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)


print(
    f"\nBalanced dataset shape: "
    f"{df.shape}"
)


# ============================================================
# SEPARATE FEATURES AND LABEL
# ============================================================

X = df.drop(
    columns=["Label"]
)

y = df["Label"]


# ============================================================
# ENSURE NUMERIC FEATURES
# ============================================================

print("\nPreparing numeric features...")

X = X.apply(
    pd.to_numeric,
    errors="coerce"
)


# Replace infinite values

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)


# Remove rows with missing feature values

valid_rows = X.notna().all(axis=1)

X = X.loc[valid_rows].reset_index(drop=True)

y = y.loc[valid_rows].reset_index(drop=True)


print(
    f"Dataset after numeric cleanup: "
    f"{X.shape}"
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

print("\nCreating stratified 80/20 train-test split...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)


print(
    f"Training samples: {len(X_train):,}"
)

print(
    f"Testing samples:  {len(X_test):,}"
)


# ============================================================
# TRAIN RANDOM FOREST
# ============================================================

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=RANDOM_STATE,
    class_weight="balanced",
    n_jobs=-1
)


model.fit(
    X_train,
    y_train
)


print("Training completed.")


# ============================================================
# PREDICTION
# ============================================================

print("\nGenerating predictions...")

y_pred = model.predict(
    X_test
)


# ============================================================
# EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)


print("\n" + "=" * 70)
print("MODEL EVALUATION")
print("=" * 70)

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1 Score : {f1:.4f}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_test,
    y_pred,
    zero_division=0
)


print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

labels = sorted(
    y.unique()
)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)


print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print("Labels:")

print(labels)

print("\nMatrix:")

print(cm)


# ============================================================
# SAVE RESULTS
# ============================================================

results_file = (
    RESULTS_DIR /
    "experiment3a_results.txt"
)


with open(
    results_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "EXPERIMENT 3A - MULTICLASS RANDOM FOREST IDS\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    f.write(
        "Dataset: CIC-IDS2017\n"
    )

    f.write(
        "Classes: 10\n"
    )

    f.write(
        "Sampling limit per class: "
        f"{MAX_SAMPLES_PER_CLASS:,}\n"
    )

    f.write(
        "Train/test split: 80/20 stratified\n"
    )

    f.write(
        f"Random state: {RANDOM_STATE}\n\n"
    )

    f.write(
        "MODEL EVALUATION\n"
    )

    f.write(
        "=" * 70 + "\n"
    )

    f.write(
        f"Accuracy : {accuracy:.4f}\n"
    )

    f.write(
        f"Precision: {precision:.4f}\n"
    )

    f.write(
        f"Recall   : {recall:.4f}\n"
    )

    f.write(
        f"F1 Score : {f1:.4f}\n\n"
    )

    f.write(
        "CLASSIFICATION REPORT\n"
    )

    f.write(
        "=" * 70 + "\n"
    )

    f.write(report)

    f.write(
        "\n\nCONFUSION MATRIX\n"
    )

    f.write(
        "=" * 70 + "\n"
    )

    f.write(
        "Labels:\n"
    )

    f.write(
        str(labels)
    )

    f.write(
        "\n\n"
    )

    f.write(
        str(cm)
    )


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

cm_file = (
    RESULTS_DIR /
    "experiment3a_confusion_matrix.csv"
)

pd.DataFrame(
    cm,
    index=labels,
    columns=labels
).to_csv(
    cm_file
)


# ============================================================
# SAVE MODEL
# ============================================================

model_file = (
    MODELS_DIR /
    "random_forest_multiclass_experiment3a.joblib"
)

joblib.dump(
    model,
    model_file
)


# ============================================================
# SAVE FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)


importance_file = (
    RESULTS_DIR /
    "experiment3a_feature_importance.csv"
)

importance.to_csv(
    importance_file,
    index=False
)


print("\n" + "=" * 70)
print("EXPERIMENT 3A COMPLETED")
print("=" * 70)

print("\nResults:")
print(results_file)

print("\nConfusion matrix:")
print(cm_file)

print("\nModel:")
print(model_file)

print("\nFeature importance:")
print(importance_file)