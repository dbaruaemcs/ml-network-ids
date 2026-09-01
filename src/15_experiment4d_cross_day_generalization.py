from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

warnings.filterwarnings("ignore")


# ============================================================
# EXPERIMENT 4D
# CROSS-DAY GENERALIZATION AFTER DOMAIN ADAPTATION
# ============================================================

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
MODELS_DIR = Path("models")

RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42

# Training limits
MAX_WEDNESDAY_BENIGN = 100_000
MAX_WEDNESDAY_ATTACK = 100_000

# Friday DDoS adaptation
FRIDAY_DDOS_EXPOSURE = 0.05

# Tuesday test sampling
MAX_TUESDAY_BENIGN = 100_000
MAX_TUESDAY_ATTACK = 100_000


WEDNESDAY_FILE = (
    "Wednesday-workingHours.pcap_ISCX.csv"
)

FRIDAY_FILE = (
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
)

TUESDAY_FILE = (
    "Tuesday-WorkingHours.pcap_ISCX.csv"
)


print("=" * 70)
print("EXPERIMENT 4D - CROSS-DAY GENERALIZATION")
print("=" * 70)

print("\nResearch question:")
print(
    "Does adding representative Friday DDoS traffic "
    "improve generalization to a different unseen attack day?"
)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading Wednesday dataset...")

wednesday = pd.read_csv(
    DATA_DIR / WEDNESDAY_FILE,
    low_memory=False
)

wednesday.columns = wednesday.columns.str.strip()

wednesday["Label"] = (
    wednesday["Label"]
    .astype(str)
    .str.strip()
)


print("Loading Friday DDoS dataset...")

friday = pd.read_csv(
    DATA_DIR / FRIDAY_FILE,
    low_memory=False
)

friday.columns = friday.columns.str.strip()

friday["Label"] = (
    friday["Label"]
    .astype(str)
    .str.strip()
)


print("Loading Tuesday dataset...")

tuesday = pd.read_csv(
    DATA_DIR / TUESDAY_FILE,
    low_memory=False
)

tuesday.columns = tuesday.columns.str.strip()

tuesday["Label"] = (
    tuesday["Label"]
    .astype(str)
    .str.strip()
)


# ============================================================
# CONVERT TO BINARY LABELS
# ============================================================

wednesday["Binary_Label"] = np.where(
    wednesday["Label"] == "BENIGN",
    "BENIGN",
    "ATTACK"
)

friday["Binary_Label"] = np.where(
    friday["Label"] == "BENIGN",
    "BENIGN",
    "ATTACK"
)

tuesday["Binary_Label"] = np.where(
    tuesday["Label"] == "BENIGN",
    "BENIGN",
    "ATTACK"
)


# ============================================================
# WEDNESDAY TRAINING DATA
# ============================================================

print("\nPreparing Wednesday training data...")

wednesday_benign_all = wednesday[
    wednesday["Binary_Label"] == "BENIGN"
]

wednesday_attack_all = wednesday[
    wednesday["Binary_Label"] == "ATTACK"
]


wednesday_benign = wednesday_benign_all.sample(
    n=min(
        MAX_WEDNESDAY_BENIGN,
        len(wednesday_benign_all)
    ),
    random_state=RANDOM_STATE
)


wednesday_attack = wednesday_attack_all.sample(
    n=min(
        MAX_WEDNESDAY_ATTACK,
        len(wednesday_attack_all)
    ),
    random_state=RANDOM_STATE
)


wednesday_train = pd.concat(
    [
        wednesday_benign,
        wednesday_attack
    ],
    ignore_index=True
)


# ============================================================
# FRIDAY DDOS ADAPTATION DATA
# ============================================================

print("\nPreparing Friday DDoS adaptation data...")

friday_attack = friday[
    friday["Binary_Label"] == "ATTACK"
].sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)


# Use 5% of Friday DDoS
n_friday_train = int(
    len(friday_attack) *
    FRIDAY_DDOS_EXPOSURE
)


friday_ddos_train = friday_attack[
    :n_friday_train
].copy()


print(
    f"Friday DDoS exposure: "
    f"{FRIDAY_DDOS_EXPOSURE * 100:.0f}%"
)

print(
    f"Friday DDoS training samples: "
    f"{len(friday_ddos_train):,}"
)


# ============================================================
# FINAL TRAINING DATA
# ============================================================

train_df = pd.concat(
    [
        wednesday_train,
        friday_ddos_train
    ],
    ignore_index=True
)


train_df = train_df.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)


print(
    f"\nTotal training samples: "
    f"{len(train_df):,}"
)


print("\nTraining class distribution:")

print(
    train_df["Binary_Label"].value_counts()
)


# ============================================================
# TUESDAY TEST DATA
# ============================================================

print("\nPreparing Tuesday unseen test data...")

tuesday_benign_all = tuesday[
    tuesday["Binary_Label"] == "BENIGN"
]

tuesday_attack_all = tuesday[
    tuesday["Binary_Label"] == "ATTACK"
]


tuesday_benign = tuesday_benign_all.sample(
    n=min(
        MAX_TUESDAY_BENIGN,
        len(tuesday_benign_all)
    ),
    random_state=RANDOM_STATE
)


tuesday_attack = tuesday_attack_all.sample(
    n=min(
        MAX_TUESDAY_ATTACK,
        len(tuesday_attack_all)
    ),
    random_state=RANDOM_STATE
)


test_df = pd.concat(
    [
        tuesday_benign,
        tuesday_attack
    ],
    ignore_index=True
)


test_df = test_df.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)


print(
    f"Tuesday test samples: "
    f"{len(test_df):,}"
)

print("\nTuesday test class distribution:")

print(
    test_df["Binary_Label"].value_counts()
)


# ============================================================
# FEATURE PREPARATION
# ============================================================

def prepare_features(train_df, test_df):

    train_df = train_df.copy()
    test_df = test_df.copy()

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

    train_remove = [
        c for c in columns_to_remove
        if c in train_df.columns
    ]

    test_remove = [
        c for c in columns_to_remove
        if c in test_df.columns
    ]

    y_train = train_df[
        "Binary_Label"
    ].copy()

    y_test = test_df[
        "Binary_Label"
    ].copy()

    X_train = train_df.drop(
        columns=train_remove
    )

    X_test = test_df.drop(
        columns=test_remove
    )

    common_features = [
        c for c in X_train.columns
        if c in X_test.columns
    ]

    X_train = X_train[
        common_features
    ]

    X_test = X_test[
        common_features
    ]

    X_train = X_train.apply(
        pd.to_numeric,
        errors="coerce"
    )

    X_test = X_test.apply(
        pd.to_numeric,
        errors="coerce"
    )

    X_train = X_train.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X_test = X_test.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Training medians only
    medians = X_train.median()

    X_train = X_train.fillna(
        medians
    )

    X_test = X_test.fillna(
        medians
    )

    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    return (
        X_train,
        y_train,
        X_test,
        y_test
    )


# ============================================================
# PREPARE DATA
# ============================================================

print("\nPreparing model features...")

X_train, y_train, X_test, y_test = (
    prepare_features(
        train_df,
        test_df
    )
)


print(
    f"Number of features: "
    f"{X_train.shape[1]}"
)


# ============================================================
# TRAIN RANDOM FOREST
# ============================================================

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    class_weight="balanced_subsample",
    max_features="sqrt"
)


model.fit(
    X_train,
    y_train
)


print("Training completed.")


# ============================================================
# PREDICTION
# ============================================================

print("\nEvaluating on unseen Tuesday traffic...")

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


cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[
        "BENIGN",
        "ATTACK"
    ]
)


tn, fp, fn, tp = cm.ravel()


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("EXPERIMENT 4D RESULTS")
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

print(
    f"False Positives: {fp:,}"
)

print(
    f"False Negatives: {fn:,}"
)

print(
    f"True Positives : {tp:,}"
)

print(
    f"True Negatives : {tn:,}"
)


print("\n")
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


print("\n")
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(
    "Labels: ['BENIGN', 'ATTACK']"
)

print(cm)


# ============================================================
# SAVE RESULTS
# ============================================================

results_file = (
    RESULTS_DIR /
    "experiment4d_results.txt"
)


with open(
    results_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "EXPERIMENT 4D - CROSS-DAY GENERALIZATION\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        "Dataset: CIC-IDS2017\n"
    )

    f.write(
        "Training domain: Wednesday WorkingHours\n"
    )

    f.write(
        "Adaptation domain: Friday Afternoon DDoS\n"
    )

    f.write(
        "Testing domain: Tuesday WorkingHours\n"
    )

    f.write(
        "Friday DDoS exposure: 5%\n"
    )

    f.write(
        f"Friday DDoS training samples: "
        f"{len(friday_ddos_train):,}\n"
    )

    f.write(
        f"Total training samples: "
        f"{len(train_df):,}\n"
    )

    f.write(
        f"Tuesday test samples: "
        f"{len(test_df):,}\n"
    )

    f.write(
        f"Number of features: "
        f"{X_train.shape[1]}\n\n"
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
        f"F1 Score : {f1:.4f}\n"
    )

    f.write(
        f"False Positives: {fp:,}\n"
    )

    f.write(
        f"False Negatives: {fn:,}\n"
    )

    f.write(
        f"True Positives : {tp:,}\n"
    )

    f.write(
        f"True Negatives : {tn:,}\n\n"
    )

    f.write(
        "CONFUSION MATRIX\n"
    )

    f.write("=" * 70 + "\n")

    f.write(
        str(cm)
    )

    f.write("\n\n")

    f.write(
        "CLASSIFICATION REPORT\n"
    )

    f.write("=" * 70 + "\n")

    f.write(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

cm_df = pd.DataFrame(
    cm,
    index=[
        "BENIGN",
        "ATTACK"
    ],
    columns=[
        "BENIGN",
        "ATTACK"
    ]
)


cm_file = (
    RESULTS_DIR /
    "experiment4d_confusion_matrix.csv"
)


cm_df.to_csv(
    cm_file
)


# ============================================================
# SAVE MODEL
# ============================================================

model_file = (
    MODELS_DIR /
    "random_forest_experiment4d_cross_day.joblib"
)


import joblib

joblib.dump(
    model,
    model_file
)


# ============================================================
# COMPLETED
# ============================================================

print("\n")
print("=" * 70)
print("EXPERIMENT 4D COMPLETED")
print("=" * 70)

print("\nResults saved to:")
print(results_file)

print("\nConfusion matrix saved to:")
print(cm_file)

print("\nModel saved to:")
print(model_file)