from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import joblib

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
# EXPERIMENT 4E
# REVERSE ATTACK-TYPE GENERALIZATION
# ============================================================

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
MODELS_DIR = Path("models")

RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42

MAX_BENIGN_PER_SOURCE = 100_000
MAX_ATTACK_PER_SOURCE = 100_000


TUESDAY_FILE = "Tuesday-WorkingHours.pcap_ISCX.csv"

WEDNESDAY_FILE = "Wednesday-workingHours.pcap_ISCX.csv"

FRIDAY_FILE = (
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
)


print("=" * 70)
print("EXPERIMENT 4E - REVERSE ATTACK-TYPE GENERALIZATION")
print("=" * 70)

print("""
Training:
  Tuesday FTP-Patator + SSH-Patator
  Wednesday DoS attacks
  Tuesday + Wednesday BENIGN

Testing:
  Friday DDoS + BENIGN

Friday DDoS is completely excluded from training.
""")


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading Tuesday dataset...")

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


print("Loading Wednesday dataset...")

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


# ============================================================
# CREATE BINARY LABELS
# ============================================================

tuesday["Binary_Label"] = np.where(
    tuesday["Label"] == "BENIGN",
    "BENIGN",
    "ATTACK"
)

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


# ============================================================
# TUESDAY TRAINING DATA
# ============================================================

print("\nPreparing Tuesday attack traffic...")

tuesday_attacks = tuesday[
    tuesday["Label"].isin([
        "FTP-Patator",
        "SSH-Patator"
    ])
].copy()

tuesday_attacks = tuesday_attacks.sample(
    n=min(
        MAX_ATTACK_PER_SOURCE,
        len(tuesday_attacks)
    ),
    random_state=RANDOM_STATE
)


tuesday_benign_all = tuesday[
    tuesday["Binary_Label"] == "BENIGN"
]

tuesday_benign = tuesday_benign_all.sample(
    n=min(
        MAX_BENIGN_PER_SOURCE,
        len(tuesday_benign_all)
    ),
    random_state=RANDOM_STATE
)


print(
    f"Tuesday attack samples: "
    f"{len(tuesday_attacks):,}"
)

print(
    f"Tuesday BENIGN samples: "
    f"{len(tuesday_benign):,}"
)


# ============================================================
# WEDNESDAY TRAINING DATA
# ============================================================

print("\nPreparing Wednesday DoS traffic...")

wednesday_attacks = wednesday[
    wednesday["Binary_Label"] == "ATTACK"
].copy()

wednesday_attacks = wednesday_attacks.sample(
    n=min(
        MAX_ATTACK_PER_SOURCE,
        len(wednesday_attacks)
    ),
    random_state=RANDOM_STATE
)


wednesday_benign_all = wednesday[
    wednesday["Binary_Label"] == "BENIGN"
]

wednesday_benign = wednesday_benign_all.sample(
    n=min(
        MAX_BENIGN_PER_SOURCE,
        len(wednesday_benign_all)
    ),
    random_state=RANDOM_STATE
)


print(
    f"Wednesday attack samples: "
    f"{len(wednesday_attacks):,}"
)

print(
    f"Wednesday BENIGN samples: "
    f"{len(wednesday_benign):,}"
)


# ============================================================
# COMBINE TRAINING DATA
# ============================================================

train_df = pd.concat(
    [
        tuesday_attacks,
        tuesday_benign,
        wednesday_attacks,
        wednesday_benign
    ],
    ignore_index=True
)


train_df = train_df.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)


print("\nTotal training samples:")
print(f"{len(train_df):,}")

print("\nTraining labels:")
print(train_df["Binary_Label"].value_counts())

print("\nTraining original attack types:")
print(
    train_df[
        train_df["Binary_Label"] == "ATTACK"
    ]["Label"].value_counts()
)


# ============================================================
# FRIDAY TEST DATA
# ============================================================

print("\nPreparing Friday DDoS test data...")

friday_ddos = friday[
    friday["Label"] == "DDoS"
].copy()

friday_benign_all = friday[
    friday["Binary_Label"] == "BENIGN"
]


friday_ddos = friday_ddos.sample(
    n=min(
        MAX_ATTACK_PER_SOURCE,
        len(friday_ddos)
    ),
    random_state=RANDOM_STATE
)


friday_benign = friday_benign_all.sample(
    n=min(
        MAX_BENIGN_PER_SOURCE,
        len(friday_benign_all)
    ),
    random_state=RANDOM_STATE
)


test_df = pd.concat(
    [
        friday_ddos,
        friday_benign
    ],
    ignore_index=True
)


test_df = test_df.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)


print(
    f"Friday DDoS test samples: "
    f"{len(friday_ddos):,}"
)

print(
    f"Friday BENIGN test samples: "
    f"{len(friday_benign):,}"
)

print(
    f"Total Friday test samples: "
    f"{len(test_df):,}"
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

    # IMPORTANT:
    # Calculate imputation values only from training data.
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


print("\nPreparing numerical features...")

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

print("\nTesting on unseen Friday DDoS...")

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
# RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("EXPERIMENT 4E RESULTS")
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

report = classification_report(
    y_test,
    y_pred,
    zero_division=0
)

print(report)


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
    "experiment4e_results.txt"
)


with open(
    results_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "EXPERIMENT 4E - REVERSE ATTACK-TYPE GENERALIZATION\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        "Dataset: CIC-IDS2017\n"
    )

    f.write(
        "Training domains: Tuesday + Wednesday\n"
    )

    f.write(
        "Testing domain: Friday DDoS\n"
    )

    f.write(
        "Friday DDoS used for training: NO\n\n"
    )

    f.write(
        f"Total training samples: "
        f"{len(train_df):,}\n"
    )

    f.write(
        f"Friday test samples: "
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
        "CLASSIFICATION REPORT\n"
    )

    f.write("=" * 70 + "\n")

    f.write(report)

    f.write("\nCONFUSION MATRIX\n")

    f.write("=" * 70 + "\n")

    f.write(str(cm))


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
    "experiment4e_confusion_matrix.csv"
)


cm_df.to_csv(
    cm_file
)


# ============================================================
# SAVE MODEL
# ============================================================

model_file = (
    MODELS_DIR /
    "random_forest_experiment4e_reverse_attack.joblib"
)


joblib.dump(
    model,
    model_file
)


# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 70)
print("EXPERIMENT 4E COMPLETED")
print("=" * 70)

print("\nResults saved to:")
print(results_file)

print("\nConfusion matrix saved to:")
print(cm_file)

print("\nModel saved to:")
print(model_file)