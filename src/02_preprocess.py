import pandas as pd
import numpy as np
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================

DATA_DIR = Path("data")

INPUT_FILE = DATA_DIR / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
OUTPUT_FILE = DATA_DIR / "ddos_clean.csv"


# ==========================================
# 1. LOAD DATASET
# ==========================================

print("Loading CIC-IDS2017 dataset...")

df = pd.read_csv(INPUT_FILE)

print("\nOriginal dataset shape:")
print(df.shape)


# ==========================================
# 2. CLEAN COLUMN NAMES
# ==========================================

df.columns = df.columns.str.strip()

print("\nColumn names cleaned.")


# ==========================================
# 3. CHECK LABELS
# ==========================================

print("\nOriginal labels:")

print(df["Label"].value_counts())


# ==========================================
# 4. REMOVE IDENTIFIER COLUMNS
# ==========================================

columns_to_remove = [
    "Flow ID",
    "Source IP",
    "Source Port",
    "Destination IP",
    "Timestamp"
]

existing_columns = [
    column for column in columns_to_remove
    if column in df.columns
]

df = df.drop(columns=existing_columns)

print("\nRemoved identifier columns:")

for column in existing_columns:
    print(" -", column)


# ==========================================
# 5. HANDLE INFINITE VALUES
# ==========================================

print("\nChecking infinite values...")

df = df.replace([np.inf, -np.inf], np.nan)


# ==========================================
# 6. REMOVE MISSING VALUES
# ==========================================

before = len(df)

df = df.dropna()

after = len(df)

print("Rows removed because of missing values:",
      before - after)


# ==========================================
# 7. CONVERT LABEL TO BINARY
# ==========================================

df["Label"] = df["Label"].apply(
    lambda x: 0 if x == "BENIGN" else 1
)

print("\nBinary labels:")

print(df["Label"].value_counts())

print("\n0 = BENIGN")
print("1 = ATTACK")


# ==========================================
# 8. REMOVE NON-NUMERIC FEATURES
# ==========================================

X = df.drop(columns=["Label"])

non_numeric_columns = X.select_dtypes(
    exclude=np.number
).columns

if len(non_numeric_columns) > 0:

    print("\nRemoving non-numeric columns:")

    for column in non_numeric_columns:
        print(" -", column)

    X = X.drop(columns=non_numeric_columns)


# ==========================================
# 9. REBUILD CLEAN DATASET
# ==========================================

y = df["Label"]

df_clean = X.copy()

df_clean["Label"] = y


# ==========================================
# 10. REMOVE DUPLICATES
# ==========================================

before = len(df_clean)

df_clean = df_clean.drop_duplicates()

after = len(df_clean)

print("\nDuplicate rows removed:",
      before - after)


# ==========================================
# 11. SAVE CLEAN DATASET
# ==========================================

df_clean.to_csv(
    OUTPUT_FILE,
    index=False
)


# ==========================================
# 12. FINAL REPORT
# ==========================================

print("\n==========================================")
print("PREPROCESSING COMPLETED")
print("==========================================")

print("\nFinal dataset shape:")

print(df_clean.shape)

print("\nNumber of features:")

print(df_clean.shape[1] - 1)

print("\nFinal label distribution:")

print(df_clean["Label"].value_counts())

print("\nClean dataset saved as:")

print(OUTPUT_FILE)

print("\n==========================================")