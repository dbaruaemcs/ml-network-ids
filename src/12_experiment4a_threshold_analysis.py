from pathlib import Path

import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# EXPERIMENT 4A
# RANDOM FOREST DECISION THRESHOLD ANALYSIS
# ============================================================

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
MODELS_DIR = Path("models")

RESULTS_DIR.mkdir(exist_ok=True)


RANDOM_STATE = 42

MAX_BENIGN_TEST = 100_000
MAX_ATTACK_TEST = 100_000


TEST_FILE = (
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
)


MODEL_FILE = (
    MODELS_DIR /
    "random_forest_experiment3c_dos_generalization.joblib"
)


THRESHOLDS = [
    0.90,
    0.80,
    0.70,
    0.60,
    0.50,
    0.40,
    0.30,
    0.20,
    0.10
]


print("=" * 70)
print("EXPERIMENT 4A - DECISION THRESHOLD ANALYSIS")
print("=" * 70)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading Experiment 3C model...")

model = joblib.load(
    MODEL_FILE
)

print("Model loaded successfully.")


# ============================================================
# LOAD TEST DATA
# ============================================================

print(
    f"\nLoading test dataset:\n{TEST_FILE}"
)

test_path = DATA_DIR / TEST_FILE

df = pd.read_csv(
    test_path,
    low_memory=False
)

df.columns = df.columns.str.strip()

df["Label"] = (
    df["Label"]
    .astype(str)
    .str.strip()
)


# ============================================================
# KEEP BENIGN + DDOS
# ============================================================

df = df[
    (df["Label"] == "BENIGN")
    |
    (df["Label"] == "DDoS")
].copy()


df["Binary_Label"] = np.where(
    df["Label"] == "BENIGN",
    "BENIGN",
    "ATTACK"
)


print("\nOriginal test distribution:")

print(
    df["Label"].value_counts()
)


# ============================================================
# PREPARE FEATURES
# ============================================================

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


y = df.loc[
    valid_rows,
    "Binary_Label"
].reset_index(drop=True)


# ============================================================
# MATCH MODEL FEATURES
# ============================================================

model_features = list(
    model.feature_names_in_
)


missing_features = [
    feature
    for feature in model_features
    if feature not in X.columns
]


if missing_features:

    print("\nERROR: Missing model features:")

    for feature in missing_features:
        print(feature)

    raise ValueError(
        "Test dataset does not contain all model features."
    )


X = X[
    model_features
]


# ============================================================
# REPRODUCE EXPERIMENT 3C TEST SAMPLING
# ============================================================

print("\nReproducing Experiment 3C test sampling...")


benign_indices = y[
    y == "BENIGN"
].index


attack_indices = y[
    y == "ATTACK"
].index


benign_n = min(
    len(benign_indices),
    MAX_BENIGN_TEST
)


attack_n = min(
    len(attack_indices),
    MAX_ATTACK_TEST
)


rng_benign = np.random.RandomState(
    RANDOM_STATE
)

rng_attack = np.random.RandomState(
    RANDOM_STATE + 1
)


selected_benign = rng_benign.choice(
    benign_indices,
    size=benign_n,
    replace=False
)


selected_attack = rng_attack.choice(
    attack_indices,
    size=attack_n,
    replace=False
)


selected_indices = np.concatenate([
    selected_benign,
    selected_attack
])


X = X.loc[
    selected_indices
].copy()


y = y.loc[
    selected_indices
].copy()


combined = X.copy()

combined["Binary_Label"] = (
    y.values
)


combined = combined.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)


y = combined[
    "Binary_Label"
]

X = combined.drop(
    columns=["Binary_Label"]
)


print(
    f"Final test samples: "
    f"{len(X):,}"
)


print("\nTest distribution:")

print(
    y.value_counts()
)


# ============================================================
# GENERATE ATTACK PROBABILITIES
# ============================================================

print("\nGenerating attack probabilities...")

probabilities = model.predict_proba(
    X
)


class_names = list(
    model.classes_
)


attack_class_index = class_names.index(
    "ATTACK"
)


attack_probability = probabilities[
    :,
    attack_class_index
]


print("Probabilities generated.")


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

results = []


print("\n" + "=" * 70)
print("THRESHOLD RESULTS")
print("=" * 70)


print(
    f"{'Threshold':<12}"
    f"{'Accuracy':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1':<12}"
    f"{'FP':<10}"
    f"{'FN':<10}"
)


for threshold in THRESHOLDS:

    y_pred = np.where(
        attack_probability >= threshold,
        "ATTACK",
        "BENIGN"
    )


    accuracy = accuracy_score(
        y,
        y_pred
    )


    precision = precision_score(
        y,
        y_pred,
        pos_label="ATTACK",
        zero_division=0
    )


    recall = recall_score(
        y,
        y_pred,
        pos_label="ATTACK",
        zero_division=0
    )


    f1 = f1_score(
        y,
        y_pred,
        pos_label="ATTACK",
        zero_division=0
    )


    cm = confusion_matrix(
        y,
        y_pred,
        labels=[
            "BENIGN",
            "ATTACK"
        ]
    )


    tn, fp, fn, tp = cm.ravel()


    results.append({
        "Threshold": threshold,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "True_Negatives": tn,
        "False_Positives": fp,
        "False_Negatives": fn,
        "True_Positives": tp
    })


    print(
        f"{threshold:<12.2f}"
        f"{accuracy:<12.4f}"
        f"{precision:<12.4f}"
        f"{recall:<12.4f}"
        f"{f1:<12.4f}"
        f"{fp:<10}"
        f"{fn:<10}"
    )


# ============================================================
# CREATE RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# SAVE RESULTS CSV
# ============================================================

csv_file = (
    RESULTS_DIR /
    "experiment4a_threshold_results.csv"
)


results_df.to_csv(
    csv_file,
    index=False
)


# ============================================================
# FIND BEST THRESHOLD BY F1
# ============================================================

best_f1_row = results_df.loc[
    results_df["F1"].idxmax()
]


# ============================================================
# FIND BEST THRESHOLD WITH PRECISION >= 0.95
# ============================================================

high_precision_results = (
    results_df[
        results_df["Precision"] >= 0.95
    ]
)


if not high_precision_results.empty:

    best_high_precision = (
        high_precision_results.loc[
            high_precision_results["Recall"].idxmax()
        ]
    )

else:

    best_high_precision = None


# ============================================================
# PRINT BEST RESULTS
# ============================================================

print("\n" + "=" * 70)
print("BEST THRESHOLD BY F1 SCORE")
print("=" * 70)


print(
    f"Threshold : "
    f"{best_f1_row['Threshold']:.2f}"
)


print(
    f"Accuracy  : "
    f"{best_f1_row['Accuracy']:.4f}"
)


print(
    f"Precision : "
    f"{best_f1_row['Precision']:.4f}"
)


print(
    f"Recall    : "
    f"{best_f1_row['Recall']:.4f}"
)


print(
    f"F1 Score  : "
    f"{best_f1_row['F1']:.4f}"
)


if best_high_precision is not None:

    print("\n" + "=" * 70)
    print(
        "BEST THRESHOLD WITH PRECISION >= 95%"
    )
    print("=" * 70)


    print(
        f"Threshold : "
        f"{best_high_precision['Threshold']:.2f}"
    )


    print(
        f"Accuracy  : "
        f"{best_high_precision['Accuracy']:.4f}"
    )


    print(
        f"Precision : "
        f"{best_high_precision['Precision']:.4f}"
    )


    print(
        f"Recall    : "
        f"{best_high_precision['Recall']:.4f}"
    )


    print(
        f"F1 Score  : "
        f"{best_high_precision['F1']:.4f}"
    )


# ============================================================
# SAVE TEXT REPORT
# ============================================================

report_file = (
    RESULTS_DIR /
    "experiment4a_results.txt"
)


with open(
    report_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "EXPERIMENT 4A - DECISION THRESHOLD ANALYSIS\n"
    )

    f.write("=" * 70 + "\n\n")


    f.write(
        "Model: Random Forest from Experiment 3C\n"
    )

    f.write(
        "Training: Wednesday DoS-family traffic\n"
    )

    f.write(
        "Testing: Friday DDoS traffic\n"
    )

    f.write(
        "Destination Port: REMOVED\n\n"
    )


    f.write(
        "THRESHOLD RESULTS\n"
    )

    f.write("=" * 70 + "\n\n")


    f.write(
        results_df.to_string(
            index=False
        )
    )


    f.write(
        "\n\nBEST THRESHOLD BY F1 SCORE\n"
    )

    f.write("=" * 70 + "\n")


    f.write(
        f"Threshold: "
        f"{best_f1_row['Threshold']:.2f}\n"
    )

    f.write(
        f"Accuracy: "
        f"{best_f1_row['Accuracy']:.4f}\n"
    )

    f.write(
        f"Precision: "
        f"{best_f1_row['Precision']:.4f}\n"
    )

    f.write(
        f"Recall: "
        f"{best_f1_row['Recall']:.4f}\n"
    )

    f.write(
        f"F1: "
        f"{best_f1_row['F1']:.4f}\n"
    )


    if best_high_precision is not None:

        f.write(
            "\n\nBEST THRESHOLD WITH PRECISION >= 95%\n"
        )

        f.write("=" * 70 + "\n")


        f.write(
            f"Threshold: "
            f"{best_high_precision['Threshold']:.2f}\n"
        )

        f.write(
            f"Accuracy: "
            f"{best_high_precision['Accuracy']:.4f}\n"
        )

        f.write(
            f"Precision: "
            f"{best_high_precision['Precision']:.4f}\n"
        )

        f.write(
            f"Recall: "
            f"{best_high_precision['Recall']:.4f}\n"
        )

        f.write(
            f"F1: "
            f"{best_high_precision['F1']:.4f}\n"
        )


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 70)
print("EXPERIMENT 4A COMPLETED")
print("=" * 70)


print("\nThreshold results saved to:")
print(csv_file)


print("\nFull report saved to:")
print(report_file)