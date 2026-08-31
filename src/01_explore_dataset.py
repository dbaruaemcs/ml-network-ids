import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")

file_path = DATA_DIR / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

print("Loading dataset...")

df = pd.read_csv(file_path)

print("\n========== DATASET INFORMATION ==========")

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\n========== LABEL DISTRIBUTION ==========")

print(df[" Label"].value_counts())

print("\n========== MISSING VALUES ==========")

missing = df.isnull().sum()
print(missing[missing > 0])

print("\n========== DATA TYPES ==========")

print(df.dtypes.value_counts())

print("\n========== FIRST 5 ROWS ==========")

print(df.head())