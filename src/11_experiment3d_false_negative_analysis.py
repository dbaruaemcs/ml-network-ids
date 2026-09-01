from pathlib import Path

import pandas as pd
import numpy as np
import joblib


# ============================================================
# EXPERIMENT 3D
# FALSE-NEGATIVE ANALYSIS
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


print("=" * 70)
print("EXPERIMENT 3D - FALSE-NEGATIVE ANALYSIS")
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


original_labels = df.loc[
    valid_rows,
    "Label"
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
# REPRODUCE EXPERIMENT 3C SAMPLING
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


original_labels = original_labels.loc[
    selected_indices
].copy()


combined = X.copy()

combined["Binary_Label"] = (
    y.values
)

combined["Original_Label"] = (
    original_labels.values
)


combined = combined.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)


y = combined[
    "Binary_Label"
]

original_labels = combined[
    "Original_Label"
]

X = combined.drop(
    columns=[
        "Binary_Label",
        "Original_Label"
    ]
)


print(
    f"Final test samples: "
    f"{len(X):,}"
)


# ============================================================
# PREDICT
# ============================================================

print("\nGenerating predictions...")

y_pred = model.predict(
    X
)


# ============================================================
# CREATE ANALYSIS DATAFRAME
# ============================================================

analysis_df = X.copy()

analysis_df["True_Label"] = y.values

analysis_df["Predicted_Label"] = y_pred


analysis_df["Original_Label"] = (
    original_labels.values
)


# ============================================================
# IDENTIFY FALSE NEGATIVES
# ============================================================

false_negative_mask = (
    (analysis_df["True_Label"] == "ATTACK")
    &
    (analysis_df["Predicted_Label"] == "BENIGN")
)


true_positive_mask = (
    (analysis_df["True_Label"] == "ATTACK")
    &
    (analysis_df["Predicted_Label"] == "ATTACK")
)


false_negatives = analysis_df[
    false_negative_mask
].copy()


true_positives = analysis_df[
    true_positive_mask
].copy()


print("\n" + "=" * 70)
print("FALSE-NEGATIVE SUMMARY")
print("=" * 70)


total_attacks = (
    analysis_df["True_Label"] == "ATTACK"
).sum()


detected_attacks = (
    analysis_df[
        true_positive_mask
    ].shape[0]
)


missed_attacks = (
    analysis_df[
        false_negative_mask
    ].shape[0]
)


recall = (
    detected_attacks / total_attacks
    if total_attacks > 0
    else 0
)


false_negative_rate = (
    missed_attacks / total_attacks
    if total_attacks > 0
    else 0
)


print(
    f"Total DDoS attacks: "
    f"{total_attacks:,}"
)


print(
    f"Correctly detected: "
    f"{detected_attacks:,}"
)


print(
    f"False negatives: "
    f"{missed_attacks:,}"
)


print(
    f"Attack recall: "
    f"{recall:.4f}"
)


print(
    f"False-negative rate: "
    f"{false_negative_rate:.4f}"
)


# ============================================================
# FEATURE COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("COMPARING DETECTED VS MISSED DDoS FLOWS")
print("=" * 70)


feature_rows = []


for feature in model_features:

    detected_values = (
        true_positives[feature]
    )

    missed_values = (
        false_negatives[feature]
    )


    detected_mean = (
        detected_values.mean()
    )

    missed_mean = (
        missed_values.mean()
    )


    detected_median = (
        detected_values.median()
    )

    missed_median = (
        missed_values.median()
    )


    detected_std = (
        detected_values.std()
    )

    missed_std = (
        missed_values.std()
    )


    # Difference relative to detected mean
    denominator = (
        abs(detected_mean)
        + 1e-9
    )


    relative_difference = (
        abs(
            missed_mean
            - detected_mean
        )
        / denominator
    )


    feature_rows.append({
        "Feature": feature,
        "Detected_Mean": detected_mean,
        "Missed_Mean": missed_mean,
        "Detected_Median": detected_median,
        "Missed_Median": missed_median,
        "Detected_Std": detected_std,
        "Missed_Std": missed_std,
        "Relative_Difference": relative_difference
    })


feature_comparison = pd.DataFrame(
    feature_rows
)


feature_comparison = (
    feature_comparison
    .sort_values(
        by="Relative_Difference",
        ascending=False
    )
)


# ============================================================
# SAVE FEATURE COMPARISON
# ============================================================

comparison_file = (
    RESULTS_DIR /
    "experiment3d_feature_comparison.csv"
)


feature_comparison.to_csv(
    comparison_file,
    index=False
)


# ============================================================
# TOP 20 DIFFERENCES
# ============================================================

print("\n" + "=" * 70)
print("TOP 20 FEATURES DIFFERING BETWEEN DETECTED AND MISSED DDoS")
print("=" * 70)


print(
    feature_comparison[
        [
            "Feature",
            "Detected_Mean",
            "Missed_Mean",
            "Relative_Difference"
        ]
    ].head(20).to_string(
        index=False
    )
)


# ============================================================
# MODEL FEATURE IMPORTANCE
# ============================================================

importance_df = pd.DataFrame({
    "Feature": model_features,
    "Model_Importance":
        model.feature_importances_
})


feature_analysis = feature_comparison.merge(
    importance_df,
    on="Feature",
    how="left"
)


feature_analysis[
    [
        "Feature",
        "Model_Importance",
        "Detected_Mean",
        "Missed_Mean",
        "Relative_Difference"
    ]
].to_csv(
    RESULTS_DIR /
    "experiment3d_feature_analysis.csv",
    index=False
)


# ============================================================
# SAVE FALSE NEGATIVES
# ============================================================

false_negative_file = (
    RESULTS_DIR /
    "experiment3d_false_negatives.csv"
)


false_negatives.to_csv(
    false_negative_file,
    index=False
)


# ============================================================
# SAVE DETECTED ATTACKS
# ============================================================

true_positive_file = (
    RESULTS_DIR /
    "experiment3d_detected_attacks.csv"
)


true_positives.to_csv(
    true_positive_file,
    index=False
)


# ============================================================
# FALSE NEGATIVE FEATURE SUMMARY
# ============================================================

fn_summary = []


for feature in model_features:

    values = false_negatives[
        feature
    ]

    fn_summary.append({
        "Feature": feature,
        "Mean": values.mean(),
        "Median": values.median(),
        "Std": values.std(),
        "Min": values.min(),
        "Max": values.max()
    })


fn_summary_df = pd.DataFrame(
    fn_summary
)


fn_summary_df = (
    fn_summary_df
    .sort_values(
        by="Mean",
        ascending=False
    )
)


fn_summary_file = (
    RESULTS_DIR /
    "experiment3d_false_negative_feature_summary.csv"
)


fn_summary_df.to_csv(
    fn_summary_file,
    index=False
)


# ============================================================
# WRITE TEXT REPORT
# ============================================================

report_file = (
    RESULTS_DIR /
    "experiment3d_results.txt"
)


with open(
    report_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "EXPERIMENT 3D - FALSE-NEGATIVE ANALYSIS\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        "Model: Experiment 3C Random Forest\n"
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
        "FALSE-NEGATIVE SUMMARY\n"
    )

    f.write("=" * 70 + "\n")

    f.write(
        f"Total DDoS attacks: "
        f"{total_attacks:,}\n"
    )

    f.write(
        f"Correctly detected: "
        f"{detected_attacks:,}\n"
    )

    f.write(
        f"False negatives: "
        f"{missed_attacks:,}\n"
    )

    f.write(
        f"Attack recall: "
        f"{recall:.4f}\n"
    )

    f.write(
        f"False-negative rate: "
        f"{false_negative_rate:.4f}\n\n"
    )


    f.write(
        "TOP 20 FEATURE DIFFERENCES\n"
    )

    f.write("=" * 70 + "\n\n")


    f.write(
        feature_comparison[
            [
                "Feature",
                "Detected_Mean",
                "Missed_Mean",
                "Relative_Difference"
            ]
        ].head(20).to_string(
            index=False
        )
    )


print("\n" + "=" * 70)
print("EXPERIMENT 3D COMPLETED")
print("=" * 70)


print("\nResults saved to:")
print(report_file)


print("\nFeature comparison saved to:")
print(comparison_file)


print("\nFeature analysis saved to:")
print(
    RESULTS_DIR /
    "experiment3d_feature_analysis.csv"
)


print("\nFalse negatives saved to:")
print(false_negative_file)


print("\nDetected attacks saved to:")
print(true_positive_file)


print("\nFalse-negative feature summary saved to:")
print(fn_summary_file)