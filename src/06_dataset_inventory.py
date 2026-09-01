from pathlib import Path
import pandas as pd


# ============================================================
# CIC-IDS2017 DATASET INVENTORY
# ============================================================

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")

RESULTS_DIR.mkdir(exist_ok=True)


# Files that belong to the original CIC-IDS2017 dataset
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


print("=" * 70)
print("CIC-IDS2017 DATASET INVENTORY")
print("=" * 70)


all_results = []


for filename in RAW_FILES:

    file_path = DATA_DIR / filename

    print("\n" + "-" * 70)
    print(f"FILE: {filename}")
    print("-" * 70)

    if not file_path.exists():
        print("WARNING: File not found.")
        continue

    try:
        # Read only the Label column.
        # This saves memory and is much faster than loading
        # the complete dataset.
        df = pd.read_csv(
            file_path,
            usecols=[" Label"],
            low_memory=False
        )

    except ValueError:

        # Some versions of CIC-IDS2017 may not contain
        # the leading space in " Label".
        df = pd.read_csv(
            file_path,
            usecols=["Label"],
            low_memory=False
        )

    except Exception as e:
        print(f"ERROR: {e}")
        continue


    # Normalize label column
    df.columns = df.columns.str.strip()

    df["Label"] = df["Label"].astype(str).str.strip()


    # Count labels
    label_counts = df["Label"].value_counts()


    total_records = len(df)

    print(f"Total records: {total_records:,}")

    print("\nLabels:")

    for label, count in label_counts.items():

        percentage = (count / total_records) * 100

        print(
            f"  {label:<30} "
            f"{count:>12,} "
            f"({percentage:6.2f}%)"
        )

        all_results.append({
            "File": filename,
            "Label": label,
            "Records": count,
            "Percentage": percentage
        })


# ============================================================
# SAVE INVENTORY
# ============================================================

inventory_df = pd.DataFrame(all_results)

output_file = RESULTS_DIR / "dataset_inventory.csv"

inventory_df.to_csv(output_file, index=False)


print("\n" + "=" * 70)
print("INVENTORY COMPLETED")
print("=" * 70)

print(f"\nInventory saved to:")
print(output_file)

print("\nUnique labels found:")

for label in sorted(inventory_df["Label"].unique()):
    print(f"  - {label}")

print("\nTotal unique labels:", inventory_df["Label"].nunique())