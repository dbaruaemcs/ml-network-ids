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
# EXPERIMENT 3C
# DO-SFAMILY CROSS-DAY GENERALIZATION
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


print("=" * 70)
print("EXPERIMENT 3C - DO-SFAMILY CROSS-DAY GENERALIZATION")
print("=" * 70)


# ============================================================
# TRAINING DATA
# ============================================================

TRAIN_FILE = (
    "Wednesday-workingHours.pcap_ISCX.csv"
)


# ============================================================
# TESTING DATA
# ============================================================

TEST_FILE = (
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
)


# ============================================================
# DO-SFAMILY ATTACK LABELS
# ============================================================

DOS_LABELS = [
    "DoS Hulk",
    "DoS GoldenEye",
    "DoS slowloris",
    "DoS Slowhttptest"
]


# ============================================================
# LOAD FILE
# ============================================================

def load_dataset(filename):

    file_path = DATA_DIR / filename

    print(f"\nLoading: {filename}")

    df = pd.read_csv(
        file_path,
        low_memory=False
    )

    df.columns = df.columns.str.strip()

    df["Label"] = (
        df["Label"]
        .astype(str)
        .str.strip()
    )

    return df


# ============================================================
# PREPARE TRAINING DATA
# ============================================================

print("\nPreparing training data...")

train_df = load_dataset(
    TRAIN_FILE
)


# Keep BENIGN and DoS-family attacks only

train_df = train_df[
    (train_df["Label"] == "BENIGN")
    |
    (train_df["Label"].isin(DOS_LABELS))
].copy()


train_df["Binary_Label"] = np.where(
    train_df["Label"] == "BENIGN",
    "BENIGN",
    "ATTACK"
)


print("\nTraining labels before sampling:")

print(
    train_df["Label"].value_counts()
)


# ============================================================
# PREPARE TEST DATA
# ============================================================

print("\nPreparing testing data...")

test_df = load_dataset(
    TEST_FILE
)


# Friday DDoS dataset contains:
# BENIGN
# DDoS

test_df = test_df[
    (test_df["Label"] == "BENIGN")
    |
    (test_df["Label"] == "DDoS")
].copy()


test_df["Binary_Label"] = np.where(
    test_df["Label"] == "BENIGN",
    "BENIGN",
    "ATTACK"
)


print("\nTesting labels before sampling:")

print(
    test_df["Label"].value_counts()
)


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(df):

    y = df["Binary_Label"].copy()

    columns_to_remove = [
        "Label",
        "Binary_Label",
        "Flow ID",
        "Source IP",
        "Source Port",
        "Destination IP",
        "Destination Port",
        "Timestamp"
    ]

    columns_to_remove = [
        column
        for column in columns_to_remove
        if column in df.columns
    ]

    X = df.drop(
        columns=columns_to_remove
    )

    X = X.apply(
        pd.to_numeric,
        errors="coerce"
    )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    valid_rows = X.notna().all(
        axis=1
    )

    X = X.loc[
        valid_rows
    ].reset_index(drop=True)

    y = y.loc[
        valid_rows
    ].reset_index(drop=True)

    return X, y


X_train, y_train = prepare_features(
    train_df
)

X_test, y_test = prepare_features(
    test_df
)


print(
    f"\nTraining feature shape: "
    f"{X_train.shape}"
)

print(
    f"Testing feature shape: "
    f"{X_test.shape}"
)


# ============================================================
# MATCH FEATURES
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
    f"Common features: "
    f"{len(common_features)}"
)


# ============================================================
# BALANCED SAMPLING FUNCTION
# ============================================================

def balanced_sample(
    X,
    y,
    max_benign,
    max_attack
):

    benign_indices = y[
        y == "BENIGN"
    ].index

    attack_indices = y[
        y == "ATTACK"
    ].index


    benign_n = min(
        len(benign_indices),
        max_benign
    )

    attack_n = min(
        len(attack_indices),
        max_attack
    )


    selected_benign = np.random.RandomState(
        RANDOM_STATE
    ).choice(
        benign_indices,
        size=benign_n,
        replace=False
    )


    selected_attack = np.random.RandomState(
        RANDOM_STATE + 1
    ).choice(
        attack_indices,
        size=attack_n,
        replace=False
    )


    selected_indices = np.concatenate([
        selected_benign,
        selected_attack
    ])


    X_selected = X.loc[
        selected_indices
    ].copy()

    y_selected = y.loc[
        selected_indices
    ].copy()


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
# SAMPLE TRAINING DATA
# ============================================================

print("\nSampling training data...")

X_train, y_train = balanced_sample(
    X_train,
    y_train,
    MAX_BENIGN_TRAIN,
    MAX_ATTACK_TRAIN
)


# ============================================================
# SAMPLE TESTING DATA
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


print("\nTraining distribution:")

print(
    y_train.value_counts()
)


print("\nTesting distribution:")

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
# PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

y_pred = model.predict(
    X_test
)


# ============================================================
# METRICS
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
print("EXPERIMENT 3C RESULTS")
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
    "ATTACK",
    "BENIGN"
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
# FEATURE IMPORTANCE
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
    "experiment3c_feature_importance.csv"
)


importance.to_csv(
    importance_file,
    index=False
)


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

cm_file = (
    RESULTS_DIR /
    "experiment3c_confusion_matrix.csv"
)


pd.DataFrame(
    cm,
    index=labels,
    columns=labels
).to_csv(
    cm_file
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_file = (
    RESULTS_DIR /
    "experiment3c_results.txt"
)


with open(
    results_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "EXPERIMENT 3C - DO-SFAMILY CROSS-DAY GENERALIZATION\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    f.write(
        "Training file: Wednesday-workingHours.pcap_ISCX.csv\n"
    )

    f.write(
        "Training attack family: DoS\n"
    )

    f.write(
        "Testing file: Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv\n"
    )

    f.write(
        "Testing attack type: DDoS\n"
    )

    f.write(
        "Destination Port: REMOVED\n"
    )

    f.write(
        "Random Forest trees: 100\n"
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
        str(cm)
    )


# ============================================================
# SAVE MODEL
# ============================================================

model_file = (
    MODELS_DIR /
    "random_forest_experiment3c_dos_generalization.joblib"
)


joblib.dump(
    model,
    model_file
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("EXPERIMENT 3C COMPLETED")
print("=" * 70)

print("\nResults saved to:")
print(results_file)

print("\nConfusion matrix saved to:")
print(cm_file)

print("\nFeature importance saved to:")
print(importance_file)

print("\nModel saved to:")
print(model_file)