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
    confusion_matrix
)

import joblib

warnings.filterwarnings("ignore")


# ============================================================
# EXPERIMENT 4C
# CONTROLLED DOMAIN ADAPTATION
# ============================================================

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
MODELS_DIR = Path("models")

RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42

MAX_WEDNESDAY_BENIGN = 100_000
MAX_WEDNESDAY_ATTACK = 100_000

MAX_FRIDAY_BENIGN = 100_000
MAX_FRIDAY_DDOS = 100_000

# Friday DDoS exposure levels
EXPOSURE_LEVELS = [0.00, 0.05, 0.10, 0.25, 0.50]


TRAIN_FILE = "Wednesday-workingHours.pcap_ISCX.csv"
FRIDAY_FILE = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"


print("=" * 70)
print("EXPERIMENT 4C - CONTROLLED DOMAIN ADAPTATION")
print("=" * 70)

print("\nObjective:")
print("Measure how representative Friday DDoS training data")
print("affects cross-day DDoS detection performance.")


# ============================================================
# LOAD WEDNESDAY
# ============================================================

print("\nLoading Wednesday dataset...")

train_path = DATA_DIR / TRAIN_FILE

wednesday = pd.read_csv(
    train_path,
    low_memory=False
)

wednesday.columns = wednesday.columns.str.strip()

wednesday["Label"] = (
    wednesday["Label"]
    .astype(str)
    .str.strip()
)


wednesday["Binary_Label"] = np.where(
    wednesday["Label"] == "BENIGN",
    "BENIGN",
    "ATTACK"
)


# ============================================================
# LOAD FRIDAY DDOS
# ============================================================

print("Loading Friday DDoS dataset...")

friday_path = DATA_DIR / FRIDAY_FILE

friday = pd.read_csv(
    friday_path,
    low_memory=False
)

friday.columns = friday.columns.str.strip()

friday["Label"] = (
    friday["Label"]
    .astype(str)
    .str.strip()
)


friday["Binary_Label"] = np.where(
    friday["Label"] == "BENIGN",
    "BENIGN",
    "ATTACK"
)


# ============================================================
# RANDOM SPLIT OF FRIDAY DATA
# ============================================================

print("\nCreating fixed Friday train/test split...")

friday_attack = friday[
    friday["Binary_Label"] == "ATTACK"
].sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)


friday_benign = friday[
    friday["Binary_Label"] == "BENIGN"
].sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True
)


# Reserve 50% of Friday traffic permanently for testing.
# The test portion is NEVER used during training.

friday_attack_test = friday_attack.iloc[
    :min(MAX_FRIDAY_DDOS, len(friday_attack) // 2)
].copy()


friday_attack_pool = friday_attack.iloc[
    len(friday_attack_test):
].copy()


friday_benign_test = friday_benign.iloc[
    :min(MAX_FRIDAY_BENIGN, len(friday_benign) // 2)
].copy()


print(
    f"\nHeld-out Friday DDoS test samples: "
    f"{len(friday_attack_test):,}"
)

print(
    f"Held-out Friday BENIGN test samples: "
    f"{len(friday_benign_test):,}"
)


# ============================================================
# SAMPLE WEDNESDAY
# ============================================================

wednesday_benign = wednesday[
    wednesday["Binary_Label"] == "BENIGN"
].sample(
    n=min(
        MAX_WEDNESDAY_BENIGN,
        len(
            wednesday[
                wednesday["Binary_Label"] == "BENIGN"
            ]
        )
    ),
    random_state=RANDOM_STATE
)


wednesday_attack = wednesday[
    wednesday["Binary_Label"] == "ATTACK"
].sample(
    n=min(
        MAX_WEDNESDAY_ATTACK,
        len(
            wednesday[
                wednesday["Binary_Label"] == "ATTACK"
            ]
        )
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
# FEATURE PREPARATION
# ============================================================

def prepare_features(
    train_df,
    test_df
):

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
# FIXED TEST DATA
# ============================================================

fixed_test = pd.concat(
    [
        friday_attack_test,
        friday_benign_test
    ],
    ignore_index=True
)

fixed_test = fixed_test.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)


# ============================================================
# EXPERIMENT LOOP
# ============================================================

all_results = []


for exposure in EXPOSURE_LEVELS:

    print("\n")
    print("=" * 70)

    print(
        f"FRIDAY DDOS EXPOSURE: "
        f"{exposure * 100:.0f}%"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # Determine number of Friday DDoS samples
    # --------------------------------------------------------

    n_friday_attack = int(
        len(friday_attack_pool) *
        exposure
    )


    if n_friday_attack > 0:

        friday_attack_train = (
            friday_attack_pool
            .iloc[:n_friday_attack]
            .copy()
        )

    else:

        friday_attack_train = (
            friday_attack_pool
            .iloc[:0]
            .copy()
        )


    # --------------------------------------------------------
    # Build training dataset
    # --------------------------------------------------------

    experiment_train = pd.concat(
        [
            wednesday_train,
            friday_attack_train
        ],
        ignore_index=True
    )


    experiment_train = experiment_train.sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)


    print(
        f"Wednesday training samples: "
        f"{len(wednesday_train):,}"
    )

    print(
        f"Friday DDoS samples added: "
        f"{len(friday_attack_train):,}"
    )

    print(
        f"Total training samples: "
        f"{len(experiment_train):,}"
    )


    # --------------------------------------------------------
    # Prepare features
    # --------------------------------------------------------

    X_train, y_train, X_test, y_test = (
        prepare_features(
            experiment_train,
            fixed_test
        )
    )


    # --------------------------------------------------------
    # Train model
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    y_pred = model.predict(
        X_test
    )


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

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


    print("\nRESULTS")

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


    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    all_results.append({
        "Friday_DDoS_Exposure": exposure,
        "Friday_DDoS_Training_Samples":
            len(friday_attack_train),
        "Total_Training_Samples":
            len(experiment_train),
        "Accuracy":
            accuracy,
        "Precision":
            precision,
        "Recall":
            recall,
        "F1":
            f1,
        "False_Positive":
            fp,
        "False_Negative":
            fn,
        "True_Positive":
            tp,
        "True_Negative":
            tn
    })


# ============================================================
# RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(
    all_results
)


print("\n")
print("=" * 70)
print("EXPERIMENT 4C SUMMARY")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# SAVE CSV
# ============================================================

csv_file = (
    RESULTS_DIR /
    "experiment4c_domain_adaptation.csv"
)


results_df.to_csv(
    csv_file,
    index=False
)


# ============================================================
# SAVE TEXT REPORT
# ============================================================

report_file = (
    RESULTS_DIR /
    "experiment4c_results.txt"
)


with open(
    report_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "EXPERIMENT 4C - CONTROLLED DOMAIN ADAPTATION\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        "Dataset: CIC-IDS2017\n"
    )

    f.write(
        "Base training domain: Wednesday WorkingHours\n"
    )

    f.write(
        "Adaptation domain: Friday Afternoon DDoS\n"
    )

    f.write(
        "Evaluation domain: Held-out Friday traffic\n"
    )

    f.write(
        "Random state: 42\n"
    )

    f.write(
        "Destination Port: REMOVED\n\n"
    )

    f.write(
        "EXPERIMENTAL QUESTION\n"
    )

    f.write(
        "How does exposure to representative Friday DDoS "
        "training data affect cross-day detection?\n\n"
    )

    f.write(
        "RESULTS\n"
    )

    f.write("=" * 70 + "\n")

    f.write(
        results_df.to_string(
            index=False
        )
    )

    f.write("\n\n")

    f.write(
        "INTERPRETATION\n"
    )

    f.write("=" * 70 + "\n")

    f.write(
        "The experiment compares increasing levels of "
        "representative Friday DDoS traffic in training "
        "while evaluating against a fixed held-out "
        "Friday test set.\n"
    )


# ============================================================
# COMPLETION
# ============================================================

print("\n")
print("=" * 70)
print("EXPERIMENT 4C COMPLETED")
print("=" * 70)

print("\nResults saved to:")
print(csv_file)

print("\nFull report saved to:")
print(report_file)