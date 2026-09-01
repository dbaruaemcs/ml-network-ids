from pathlib import Path

import pandas as pd
import numpy as np

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
# EXPERIMENT 3B
# CROSS-DAY BINARY GENERALIZATION
# ============================================================

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
MODELS_DIR = Path("models")

RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42

MAX_BENIGN_TRAIN = 100_000
MAX_ATTACK_TRAIN = 100_000

MAX_BENIGN_TEST = 100_000
MAX_ATTACK_TEST = 100_000


# ============================================================
# TRAINING FILES
# ============================================================

TRAIN_FILES = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
]


# ============================================================
# TESTING FILES
# ============================================================

TEST_FILES = [
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
]


# ============================================================
# FUNCTION: LOAD AND PREPARE FILE
# ============================================================

def load_file(filename):

    file_path = DATA_DIR / filename

    print(f"\nLoading: {filename}")

    df = pd.read_csv(
        file_path,
        low_memory=False
    )

    df.columns = df.columns.str.strip()

    if "Label" not in df.columns:

        raise ValueError(
            f"Label column not found in {filename}"
        )

    df["Label"] = (
        df["Label"]
        .astype(str)
        .str.strip()
    )

    # Convert to binary classification
    df["Binary_Label"] = np.where(
        df["Label"] == "BENIGN",
        "BENIGN",
        "ATTACK"
    )

    return df


# ============================================================
# FUNCTION: CLEAN FEATURES
# ============================================================

def prepare_features(df):

    # Keep label separately
    y = df["Binary_Label"].copy()

    # Remove columns that must not become model features
    columns_to_remove = [
        "Label",
        "Binary_Label",
        "Flow ID",
        "Source IP",
        "Source Port",
        "Destination IP",
        "Destination Port",
        "Timestamp",
    ]

    columns_to_remove = [
        c for c in columns_to_remove
        if c in df.columns
    ]

    X = df.drop(
        columns=columns_to_remove
    )

    # Convert everything possible to numeric
    X = X.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # Replace infinity
    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Remove rows containing invalid values
    valid_rows = X.notna().all(axis=1)

    X = X.loc[
        valid_rows
    ].reset_index(drop=True)

    y = y.loc[
        valid_rows
    ].reset_index(drop=True)

    return X, y


# ============================================================
# FUNCTION: BALANCED SAMPLE
# ============================================================

def balanced_sample(
    X,
    y,
    max_benign,
    max_attack
):

    benign_mask = y == "BENIGN"
    attack_mask = y == "ATTACK"

    benign_X = X.loc[benign_mask]
    benign_y = y.loc[benign_mask]

    attack_X = X.loc[attack_mask]
    attack_y = y.loc[attack_mask]

    benign_n = min(
        len(benign_X),
        max_benign
    )

    attack_n = min(
        len(attack_X),
        max_attack
    )

    benign_indices = benign_X.sample(
        n=benign_n,
        random_state=RANDOM_STATE
    ).index

    attack_indices = attack_X.sample(
        n=attack_n,
        random_state=RANDOM_STATE
    ).index

    selected_indices = (
        list(benign_indices)
        + list(attack_indices)
    )

    X_selected = X.loc[
        selected_indices
    ]

    y_selected = y.loc[
        selected_indices
    ]

    combined = X_selected.copy()

    combined["Binary_Label"] = (
        y_selected.values
    )

    combined = combined.sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    y_selected = combined[
        "Binary_Label"
    ]

    X_selected = combined.drop(
        columns=["Binary_Label"]
    )

    return X_selected, y_selected


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("EXPERIMENT 3B - CROSS-DAY BINARY GENERALIZATION")
print("=" * 70)

print("\nTRAINING DAYS:")
print("Monday + Tuesday + Wednesday")

print("\nTESTING DAYS:")
print("Thursday + Friday")


# ============================================================
# LOAD TRAINING DATA
# ============================================================

train_parts = []

for filename in TRAIN_FILES:

    df = load_file(filename)

    train_parts.append(df)


train_df = pd.concat(
    train_parts,
    ignore_index=True
)


print(
    f"\nRaw training records: "
    f"{len(train_df):,}"
)


# ============================================================
# LOAD TEST DATA
# ============================================================

test_parts = []

for filename in TEST_FILES:

    df = load_file(filename)

    test_parts.append(df)


test_df = pd.concat(
    test_parts,
    ignore_index=True
)


print(
    f"Raw testing records: "
    f"{len(test_df):,}"
)


# ============================================================
# PREPARE FEATURES
# ============================================================

print("\nPreparing training features...")

X_train, y_train = prepare_features(
    train_df
)


print(
    f"Training features: "
    f"{X_train.shape}"
)


print("\nPreparing testing features...")

X_test, y_test = prepare_features(
    test_df
)


print(
    f"Testing features: "
    f"{X_test.shape}"
)


# ============================================================
# MATCH FEATURE COLUMNS
# ============================================================

common_features = [
    column
    for column in X_train.columns
    if column in X_test.columns
]


X_train = X_train[
    common_features
]

X_test = X_test[
    common_features
]


print(
    f"\nCommon model features: "
    f"{len(common_features)}"
)


# ============================================================
# BALANCE TRAINING DATA
# ============================================================

print("\nSampling training data...")

X_train, y_train = balanced_sample(
    X_train,
    y_train,
    MAX_BENIGN_TRAIN,
    MAX_ATTACK_TRAIN
)


# ============================================================
# BALANCE TEST DATA
# ============================================================

print("Sampling testing data...")

X_test, y_test = balanced_sample(
    X_test,
    y_test,
    MAX_BENIGN_TEST,
    MAX_ATTACK_TEST
)


print(
    f"\nFinal training samples: "
    f"{len(X_train):,}"
)

print(
    f"Final testing samples: "
    f"{len(X_test):,}"
)


print("\nTraining class distribution:")

print(
    y_train.value_counts()
)


print("\nTesting class distribution:")

print(
    y_test.value_counts()
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
    pos_label="ATTACK",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    pos_label="ATTACK",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    pos_label="ATTACK",
    zero_division=0
)


print("\n" + "=" * 70)
print("EXPERIMENT 3B RESULTS")
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

labels = [
    "BENIGN",
    "ATTACK"
]

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)


print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(cm)


# ============================================================
# SAVE RESULTS
# ============================================================

results_file = (
    RESULTS_DIR /
    "experiment3b_results.txt"
)

with open(
    results_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "EXPERIMENT 3B - CROSS-DAY BINARY GENERALIZATION\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        "Training days: Monday, Tuesday, Wednesday\n"
    )

    f.write(
        "Testing days: Thursday, Friday\n"
    )

    f.write(
        "Model: Random Forest\n"
    )

    f.write(
        "Train/test design: temporal cross-day validation\n"
    )

    f.write(
        "Destination Port: REMOVED\n"
    )

    f.write(
        f"Random state: {RANDOM_STATE}\n\n"
    )

    f.write(
        "MODEL EVALUATION\n"
    )

    f.write("=" * 70 + "\n")

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

    f.write("=" * 70 + "\n")

    f.write(report)

    f.write(
        "\nCONFUSION MATRIX\n"
    )

    f.write("=" * 70 + "\n")

    f.write(str(cm))


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

cm_file = (
    RESULTS_DIR /
    "experiment3b_confusion_matrix.csv"
)

pd.DataFrame(
    cm,
    index=labels,
    columns=labels
).to_csv(
    cm_file
)


# ============================================================
# SAVE FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

importance_file = (
    RESULTS_DIR /
    "experiment3b_feature_importance.csv"
)

importance.to_csv(
    importance_file,
    index=False
)


# ============================================================
# SAVE MODEL
# ============================================================

model_file = (
    MODELS_DIR /
    "random_forest_experiment3b_cross_day.joblib"
)

joblib.dump(
    model,
    model_file
)


print("\n" + "=" * 70)
print("EXPERIMENT 3B COMPLETED")
print("=" * 70)

print("\nResults saved to:")
print(results_file)

print("\nConfusion matrix saved to:")
print(cm_file)

print("\nFeature importance saved to:")
print(importance_file)

print("\nModel saved to:")
print(model_file)