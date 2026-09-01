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

import joblib

warnings.filterwarnings("ignore")


# ============================================================
# EXPERIMENT 4B
# FEATURE ENGINEERING FOR CROSS-DAY GENERALIZATION
# ============================================================

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
MODELS_DIR = Path("models")

RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42

MAX_BENIGN = 100_000
MAX_ATTACK = 100_000


TRAIN_FILE = "Wednesday-workingHours.pcap_ISCX.csv"
TEST_FILE = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"


print("=" * 70)
print("EXPERIMENT 4B - FEATURE ENGINEERING")
print("=" * 70)

print("\nObjective:")
print("Improve unseen DDoS detection using engineered")
print("traffic-behavior features.")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_columns(df):
    df.columns = df.columns.str.strip()
    return df


def safe_divide(a, b):
    """
    Element-wise division with protection against zero.
    """
    return a / b.replace(0, np.nan)


def add_engineered_features(df):
    """
    Create general traffic-behavior features.
    """

    df = df.copy()

    # --------------------------------------------------------
    # Packet count ratios
    # --------------------------------------------------------

    if (
        "Total Fwd Packets" in df.columns
        and "Total Backward Packets" in df.columns
    ):
        df["Fwd_Bwd_Packet_Ratio"] = safe_divide(
            df["Total Fwd Packets"],
            df["Total Backward Packets"]
        )

        df["Bwd_Fwd_Packet_Ratio"] = safe_divide(
            df["Total Backward Packets"],
            df["Total Fwd Packets"]
        )


    # --------------------------------------------------------
    # Byte ratios
    # --------------------------------------------------------

    if (
        "Total Length of Fwd Packets" in df.columns
        and "Total Length of Bwd Packets" in df.columns
    ):
        df["Fwd_Bwd_Byte_Ratio"] = safe_divide(
            df["Total Length of Fwd Packets"],
            df["Total Length of Bwd Packets"]
        )

        df["Bwd_Fwd_Byte_Ratio"] = safe_divide(
            df["Total Length of Bwd Packets"],
            df["Total Length of Fwd Packets"]
        )


    # --------------------------------------------------------
    # Average bytes per forward/backward packet
    # --------------------------------------------------------

    if (
        "Total Length of Fwd Packets" in df.columns
        and "Total Fwd Packets" in df.columns
    ):
        df["Fwd_Bytes_Per_Packet"] = safe_divide(
            df["Total Length of Fwd Packets"],
            df["Total Fwd Packets"]
        )


    if (
        "Total Length of Bwd Packets" in df.columns
        and "Total Backward Packets" in df.columns
    ):
        df["Bwd_Bytes_Per_Packet"] = safe_divide(
            df["Total Length of Bwd Packets"],
            df["Total Backward Packets"]
        )


    # --------------------------------------------------------
    # Forward packet proportion
    # --------------------------------------------------------

    if (
        "Total Fwd Packets" in df.columns
        and "Total Backward Packets" in df.columns
    ):
        total_packets = (
            df["Total Fwd Packets"]
            + df["Total Backward Packets"]
        )

        df["Fwd_Packet_Proportion"] = safe_divide(
            df["Total Fwd Packets"],
            total_packets
        )

        df["Bwd_Packet_Proportion"] = safe_divide(
            df["Total Backward Packets"],
            total_packets
        )


    # --------------------------------------------------------
    # Forward byte proportion
    # --------------------------------------------------------

    if (
        "Total Length of Fwd Packets" in df.columns
        and "Total Length of Bwd Packets" in df.columns
    ):
        total_bytes = (
            df["Total Length of Fwd Packets"]
            + df["Total Length of Bwd Packets"]
        )

        df["Fwd_Byte_Proportion"] = safe_divide(
            df["Total Length of Fwd Packets"],
            total_bytes
        )

        df["Bwd_Byte_Proportion"] = safe_divide(
            df["Total Length of Bwd Packets"],
            total_bytes
        )


    # --------------------------------------------------------
    # Packet length relationships
    # --------------------------------------------------------

    if (
        "Fwd Packet Length Mean" in df.columns
        and "Bwd Packet Length Mean" in df.columns
    ):
        df["Fwd_Bwd_Packet_Length_Ratio"] = safe_divide(
            df["Fwd Packet Length Mean"],
            df["Bwd Packet Length Mean"]
        )


    # --------------------------------------------------------
    # Packet-rate ratios
    # --------------------------------------------------------

    if (
        "Fwd Packets/s" in df.columns
        and "Bwd Packets/s" in df.columns
    ):
        df["Fwd_Bwd_Packet_Rate_Ratio"] = safe_divide(
            df["Fwd Packets/s"],
            df["Bwd Packets/s"]
        )


    # --------------------------------------------------------
    # Header-length relationships
    # --------------------------------------------------------

    if (
        "Fwd Header Length" in df.columns
        and "Bwd Header Length" in df.columns
    ):
        df["Fwd_Bwd_Header_Ratio"] = safe_divide(
            df["Fwd Header Length"],
            df["Bwd Header Length"]
        )


    # --------------------------------------------------------
    # IAT relationships
    # --------------------------------------------------------

    if (
        "Fwd IAT Mean" in df.columns
        and "Bwd IAT Mean" in df.columns
    ):
        df["Fwd_Bwd_IAT_Ratio"] = safe_divide(
            df["Fwd IAT Mean"],
            df["Bwd IAT Mean"]
        )


    # --------------------------------------------------------
    # Flow duration normalized features
    # --------------------------------------------------------

    if "Flow Duration" in df.columns:

        duration_seconds = (
            df["Flow Duration"] / 1_000_000
        )

        if "Total Fwd Packets" in df.columns:
            df["Fwd_Packets_Per_Duration"] = safe_divide(
                df["Total Fwd Packets"],
                duration_seconds
            )

        if "Total Backward Packets" in df.columns:
            df["Bwd_Packets_Per_Duration"] = safe_divide(
                df["Total Backward Packets"],
                duration_seconds
            )

        if "Total Length of Fwd Packets" in df.columns:
            df["Fwd_Bytes_Per_Duration"] = safe_divide(
                df["Total Length of Fwd Packets"],
                duration_seconds
            )

        if "Total Length of Bwd Packets" in df.columns:
            df["Bwd_Bytes_Per_Duration"] = safe_divide(
                df["Total Length of Bwd Packets"],
                duration_seconds
            )


    # --------------------------------------------------------
    # Log transforms for heavily skewed features
    # --------------------------------------------------------

    log_candidates = [
        "Flow Duration",
        "Total Fwd Packets",
        "Total Backward Packets",
        "Total Length of Fwd Packets",
        "Total Length of Bwd Packets",
        "Flow Packets/s",
        "Fwd Packets/s",
        "Bwd Packets/s",
        "Packet Length Mean"
    ]

    for feature in log_candidates:

        if feature in df.columns:

            numeric_values = pd.to_numeric(
                df[feature],
                errors="coerce"
            )

            numeric_values = numeric_values.clip(
                lower=0
            )

            df[
                "Log_" + feature.replace(
                    " ",
                    "_"
                ).replace(
                    "/",
                    "_per_"
                )
            ] = np.log1p(
                numeric_values
            )


    return df


# ============================================================
# LOAD TRAINING DATA
# ============================================================

print("\nLoading training dataset:")
print(TRAIN_FILE)

train_path = DATA_DIR / TRAIN_FILE

train_df = pd.read_csv(
    train_path,
    low_memory=False
)

train_df = clean_columns(train_df)

train_df["Label"] = (
    train_df["Label"]
    .astype(str)
    .str.strip()
)


# Wednesday contains multiple DoS attacks.
# We treat all non-BENIGN traffic as ATTACK.

train_df["Binary_Label"] = np.where(
    train_df["Label"] == "BENIGN",
    "BENIGN",
    "ATTACK"
)


print("\nTraining label distribution:")
print(
    train_df["Binary_Label"].value_counts()
)


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\nLoading test dataset:")
print(TEST_FILE)

test_path = DATA_DIR / TEST_FILE

test_df = pd.read_csv(
    test_path,
    low_memory=False
)

test_df = clean_columns(test_df)

test_df["Label"] = (
    test_df["Label"]
    .astype(str)
    .str.strip()
)


# Friday DDoS dataset:
# DDoS = ATTACK
# BENIGN = BENIGN

test_df["Binary_Label"] = np.where(
    test_df["Label"] == "BENIGN",
    "BENIGN",
    "ATTACK"
)


print("\nTesting label distribution:")
print(
    test_df["Binary_Label"].value_counts()
)


# ============================================================
# SAMPLE DATA
# ============================================================

print("\nSampling data...")


train_benign = train_df[
    train_df["Binary_Label"] == "BENIGN"
]


train_attack = train_df[
    train_df["Binary_Label"] == "ATTACK"
]


test_benign = test_df[
    test_df["Binary_Label"] == "BENIGN"
]


test_attack = test_df[
    test_df["Binary_Label"] == "ATTACK"
]


rng = np.random.RandomState(
    RANDOM_STATE
)


train_benign = train_benign.sample(
    n=min(
        MAX_BENIGN,
        len(train_benign)
    ),
    random_state=RANDOM_STATE
)


train_attack = train_attack.sample(
    n=min(
        MAX_ATTACK,
        len(train_attack)
    ),
    random_state=RANDOM_STATE
)


test_benign = test_benign.sample(
    n=min(
        MAX_BENIGN,
        len(test_benign)
    ),
    random_state=RANDOM_STATE
)


test_attack = test_attack.sample(
    n=min(
        MAX_ATTACK,
        len(test_attack)
    ),
    random_state=RANDOM_STATE
)


train_df = pd.concat(
    [
        train_benign,
        train_attack
    ],
    ignore_index=True
)


test_df = pd.concat(
    [
        test_benign,
        test_attack
    ],
    ignore_index=True
)


train_df = train_df.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)


test_df = test_df.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)


print(
    f"\nTraining samples: {len(train_df):,}"
)

print(
    f"Testing samples:  {len(test_df):,}"
)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

print("\nCreating engineered features...")

train_df = add_engineered_features(
    train_df
)

test_df = add_engineered_features(
    test_df
)


# ============================================================
# REMOVE NON-FEATURE COLUMNS
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


# ============================================================
# ALIGN FEATURES
# ============================================================

common_features = [
    feature
    for feature in X_train.columns
    if feature in X_test.columns
]


X_train = X_train[
    common_features
]


X_test = X_test[
    common_features
]


# Convert to numeric

X_train = X_train.apply(
    pd.to_numeric,
    errors="coerce"
)

X_test = X_test.apply(
    pd.to_numeric,
    errors="coerce"
)


# Replace infinities

X_train = X_train.replace(
    [np.inf, -np.inf],
    np.nan
)

X_test = X_test.replace(
    [np.inf, -np.inf],
    np.nan
)


# Replace missing values using TRAINING medians only

train_medians = X_train.median()

X_train = X_train.fillna(
    train_medians
)

X_test = X_test.fillna(
    train_medians
)


# Any columns that remain completely unusable

X_train = X_train.fillna(0)
X_test = X_test.fillna(0)


print(
    f"\nOriginal + engineered features: "
    f"{len(common_features)}"
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

print("\nEvaluating on unseen Friday DDoS traffic...")

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
# DISPLAY RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("EXPERIMENT 4B RESULTS")
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


print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(
    "Labels: ['BENIGN', 'ATTACK']"
)

print(cm)


print("\n")
print("=" * 70)
print("FALSE-NEGATIVE ANALYSIS")
print("=" * 70)

print(
    f"Total DDoS attacks : {tp + fn:,}"
)

print(
    f"Detected           : {tp:,}"
)

print(
    f"Missed             : {fn:,}"
)

print(
    f"Attack recall      : {recall:.4f}"
)

print(
    f"False-negative rate: {fn / (tp + fn):.4f}"
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
# FEATURE IMPORTANCE
# ============================================================

importance_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
})


importance_df = importance_df.sort_values(
    "Importance",
    ascending=False
)


print("\n" + "=" * 70)
print("TOP 20 ENGINEERED MODEL FEATURES")
print("=" * 70)

print(
    importance_df.head(20).to_string(
        index=False
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
    "experiment4b_confusion_matrix.csv"
)


cm_df.to_csv(
    cm_file
)


# ============================================================
# SAVE FEATURE IMPORTANCE
# ============================================================

feature_file = (
    RESULTS_DIR /
    "experiment4b_feature_importance.csv"
)


importance_df.to_csv(
    feature_file,
    index=False
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_file = (
    RESULTS_DIR /
    "experiment4b_results.txt"
)


with open(
    results_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "EXPERIMENT 4B - FEATURE ENGINEERING\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        "Dataset: CIC-IDS2017\n"
    )

    f.write(
        "Training: Wednesday WorkingHours\n"
    )

    f.write(
        "Testing: Friday Afternoon DDoS\n"
    )

    f.write(
        "Maximum samples per class: 100,000\n"
    )

    f.write(
        "Random state: 42\n"
    )

    f.write(
        "Model: Random Forest\n"
    )

    f.write(
        "Destination Port: REMOVED\n\n"
    )

    f.write(
        "METRICS\n"
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
        f"\nTrue Negatives : {tn:,}\n"
    )

    f.write(
        f"False Positives: {fp:,}\n"
    )

    f.write(
        f"False Negatives: {fn:,}\n"
    )

    f.write(
        f"True Positives  : {tp:,}\n"
    )

    f.write(
        "\nCLASSIFICATION REPORT\n"
    )

    f.write("=" * 70 + "\n")

    f.write(report)

    f.write(
        "\nTOP 20 FEATURES\n"
    )

    f.write("=" * 70 + "\n")

    f.write(
        importance_df.head(20).to_string(
            index=False
        )
    )


# ============================================================
# SAVE MODEL
# ============================================================

model_file = (
    MODELS_DIR /
    "random_forest_experiment4b_feature_engineered.joblib"
)


joblib.dump(
    model,
    model_file
)


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 70)
print("EXPERIMENT 4B COMPLETED")
print("=" * 70)

print("\nResults saved to:")
print(results_file)

print("\nConfusion matrix saved to:")
print(cm_file)

print("\nFeature importance saved to:")
print(feature_file)

print("\nModel saved to:")
print(model_file)