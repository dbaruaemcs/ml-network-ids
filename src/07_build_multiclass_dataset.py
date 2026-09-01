from pathlib import Path
import pandas as pd


# ============================================================
# EXPERIMENT 3A
# BUILD MULTICLASS DATASET
# ============================================================

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")

RESULTS_DIR.mkdir(exist_ok=True)


# ------------------------------------------------------------
# Source files
# ------------------------------------------------------------

RAW_FILES = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
]


# ------------------------------------------------------------
# Classes selected for Experiment 3A
# ------------------------------------------------------------

SELECTED_CLASSES = [
    "BENIGN",
    "DDoS",
    "DoS Hulk",
    "DoS GoldenEye",
    "DoS slowloris",
    "DoS Slowhttptest",
    "PortScan",
    "FTP-Patator",
    "SSH-Patator",
    "Bot",
]


print("=" * 70)
print("EXPERIMENT 3A - BUILD MULTICLASS DATASET")
print("=" * 70)


all_data = []


# ============================================================
# READ DATASETS
# ============================================================

for filename in RAW_FILES:

    file_path = DATA_DIR / filename

    print("\nLoading:")
    print(filename)

    if not file_path.exists():
        print("WARNING: File not found. Skipping.")
        continue

    try:

        df = pd.read_csv(
            file_path,
            low_memory=False
        )

    except Exception as e:

        print(f"ERROR loading file: {e}")
        continue


    # --------------------------------------------------------
    # Clean column names
    # --------------------------------------------------------

    df.columns = df.columns.str.strip()


    # --------------------------------------------------------
    # Clean labels
    # --------------------------------------------------------

    if "Label" not in df.columns:

        print("ERROR: Label column not found.")
        continue

    df["Label"] = (
        df["Label"]
        .astype(str)
        .str.strip()
    )


    # --------------------------------------------------------
    # Keep selected classes only
    # --------------------------------------------------------

    df = df[
        df["Label"].isin(SELECTED_CLASSES)
    ].copy()


    # --------------------------------------------------------
    # Record source day
    # --------------------------------------------------------

    df["Source_File"] = filename


    print(
        f"Selected records: {len(df):,}"
    )


    all_data.append(df)


# ============================================================
# COMBINE
# ============================================================

print("\nCombining datasets...")

combined_df = pd.concat(
    all_data,
    ignore_index=True
)


print(
    f"Combined records: {len(combined_df):,}"
)


# ============================================================
# REMOVE UNNECESSARY IDENTIFIER COLUMNS
# ============================================================

columns_to_remove = [
    "Flow ID",
    "Source IP",
    "Source Port",
    "Destination IP",
    "Timestamp",
]


for column in columns_to_remove:

    if column in combined_df.columns:

        combined_df.drop(
            columns=column,
            inplace=True
        )


# Source_File is only needed for later analysis.
# Keep it for now.


# ============================================================
# HANDLE INFINITE VALUES
# ============================================================

import numpy as np

numeric_columns = combined_df.select_dtypes(
    include=[np.number]
).columns

combined_df[numeric_columns] = (
    combined_df[numeric_columns]
    .replace([np.inf, -np.inf], np.nan)
)

# ============================================================
# REMOVE MISSING VALUES
# ============================================================

before = len(combined_df)

combined_df.dropna(
    inplace=True
)

after = len(combined_df)


print(
    f"Removed rows containing missing values: "
    f"{before - after:,}"
)


# ============================================================
# REMOVE DUPLICATES
# ============================================================

before = len(combined_df)

combined_df.drop_duplicates(
    inplace=True
)

after = len(combined_df)


print(
    f"Removed duplicate rows: "
    f"{before - after:,}"
)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\nFinal class distribution:")

class_counts = (
    combined_df["Label"]
    .value_counts()
)


for label, count in class_counts.items():

    percentage = (
        count / len(combined_df)
    ) * 100

    print(
        f"{label:<25}"
        f"{count:>12,}"
        f"  ({percentage:6.2f}%)"
    )


# ============================================================
# SAVE DATASET
# ============================================================

output_file = (
    DATA_DIR /
    "multiclass_ids.csv"
)


combined_df.to_csv(
    output_file,
    index=False
)


print("\n" + "=" * 70)
print("EXPERIMENT 3A DATASET CREATED")
print("=" * 70)

print("\nSaved to:")
print(output_file)

print(
    f"\nFinal dataset shape: "
    f"{combined_df.shape}"
)

print(
    f"Number of classes: "
    f"{combined_df['Label'].nunique()}"
)