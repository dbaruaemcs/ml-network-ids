from pathlib import Path
import re
import pandas as pd


# ============================================================
# MASTER EXPERIMENT RESULTS
# ============================================================

RESULTS_DIR = Path("results")


print("=" * 75)
print("BUILDING MASTER EXPERIMENT RESULTS TABLE")
print("=" * 75)


# ------------------------------------------------------------
# Helper: extract metric from text
# ------------------------------------------------------------

def extract_metric(text, metric):
    pattern = rf"{metric}\s*:\s*([0-9.]+)"
    match = re.search(pattern, text, re.IGNORECASE)

    if match:
        return float(match.group(1))

    return None


# ------------------------------------------------------------
# Experiment 3A
# ------------------------------------------------------------

rows = []


def read_results_file(filename, experiment, purpose):

    path = RESULTS_DIR / filename

    if not path.exists():
        print(f"WARNING: {filename} not found")
        return

    text = path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    rows.append({

        "Experiment": experiment,

        "Purpose": purpose,

        "Accuracy": extract_metric(
            text,
            "Accuracy"
        ),

        "Precision": extract_metric(
            text,
            "Precision"
        ),

        "Recall": extract_metric(
            text,
            "Recall"
        ),

        "F1": extract_metric(
            text,
            "F1 Score"
        ),

        "Source": filename

    })


# ------------------------------------------------------------
# Add experiments with standard metrics
# ------------------------------------------------------------

read_results_file(
    "experiment3a_results.txt",
    "3A",
    "Multiclass classification"
)


read_results_file(
    "experiment3b_results.txt",
    "3B",
    "Cross-day generalization"
)


read_results_file(
    "experiment3c_results.txt",
    "3C",
    "DDoS generalization"
)


read_results_file(
    "experiment4a_results.txt",
    "4A",
    "Classification threshold analysis"
)


read_results_file(
    "experiment4b_results.txt",
    "4B",
    "Feature engineering"
)


read_results_file(
    "experiment4d_results.txt",
    "4D",
    "Unseen attack-family testing"
)


read_results_file(
    "experiment4e_results.txt",
    "4E",
    "Cross-family DDoS testing"
)


# ------------------------------------------------------------
# Experiment 3D
# ------------------------------------------------------------

path = RESULTS_DIR / "experiment3d_results.txt"

if path.exists():

    text = path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    recall = extract_metric(
        text,
        "Attack recall"
    )

    rows.append({

        "Experiment": "3D",

        "Purpose":
            "DDoS false-negative analysis",

        "Accuracy": None,

        "Precision": None,

        "Recall": recall,

        "F1": None,

        "Source":
            "experiment3d_results.txt"

    })


# ------------------------------------------------------------
# Experiment 4F
# ------------------------------------------------------------

path = RESULTS_DIR / "experiment4f_attack_generalization_matrix.csv"

if path.exists():

    df4f = pd.read_csv(path)

    print("\n")
    print("=" * 75)
    print("EXPERIMENT 4F ATTACK GENERALIZATION")
    print("=" * 75)

    print(df4f.to_string(index=False))

    print("\n4F is a multi-condition experiment.")
    print("It will be preserved separately rather than reduced")
    print("to a single misleading metric.")


# ------------------------------------------------------------
# Experiment 4C
# ------------------------------------------------------------

path = RESULTS_DIR / "experiment4c_domain_adaptation.csv"

if path.exists():

    df4c = pd.read_csv(path)

    print("\n")
    print("=" * 75)
    print("EXPERIMENT 4C DOMAIN ADAPTATION")
    print("=" * 75)

    print(df4c.to_string(index=False))

    # Best F1 result

    if "F1" in df4c.columns:

        best = df4c.loc[
            df4c["F1"].idxmax()
        ]

        rows.append({

            "Experiment": "4C",

            "Purpose":
                "Target-domain exposure",

            "Accuracy":
                best.get("Accuracy"),

            "Precision":
                best.get("Precision"),

            "Recall":
                best.get("Recall"),

            "F1":
                best.get("F1"),

            "Source":
                "experiment4c_domain_adaptation.csv"

        })


# ------------------------------------------------------------
# Experiment 4G
# ------------------------------------------------------------

path = RESULTS_DIR / "experiment4g_pair_summary.csv"

if path.exists():

    df4g = pd.read_csv(path)

    print("\n")
    print("=" * 75)
    print("EXPERIMENT 4G DISTRIBUTION SHIFT")
    print("=" * 75)

    print(df4g.to_string(index=False))

    # Highest mean KS

    best_shift = df4g.loc[
        df4g["Mean_KS"].idxmax()
    ]

    print("\nLargest mean distribution shift:")

    print(
        f"{best_shift['Family_A']} -> "
        f"{best_shift['Family_B']}: "
        f"KS = {best_shift['Mean_KS']:.4f}"
    )


# ------------------------------------------------------------
# Create master dataframe
# ------------------------------------------------------------

master_df = pd.DataFrame(rows)


# ------------------------------------------------------------
# Format metrics
# ------------------------------------------------------------

for column in [
    "Accuracy",
    "Precision",
    "Recall",
    "F1"
]:

    master_df[column] = pd.to_numeric(
        master_df[column],
        errors="coerce"
    )


# ------------------------------------------------------------
# Save CSV
# ------------------------------------------------------------

csv_path = (
    RESULTS_DIR /
    "master_experiment_results.csv"
)


master_df.to_csv(
    csv_path,
    index=False
)


# ------------------------------------------------------------
# Save readable TXT report
# ------------------------------------------------------------

txt_path = (
    RESULTS_DIR /
    "master_experiment_results.txt"
)


with open(
    txt_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "MASTER EXPERIMENT RESULTS\n"
    )

    f.write(
        "=" * 75 + "\n\n"
    )

    f.write(
        "Dataset: CIC-IDS2017\n"
    )

    f.write(
        "Project: Machine Learning Network Intrusion Detection System\n\n"
    )

    f.write(
        master_df.to_string(
            index=False
        )
    )

    f.write("\n\n")

    f.write(
        "=" * 75 + "\n"
    )

    f.write(
        "EXPERIMENT 4F\n"
    )

    f.write(
        "=" * 75 + "\n\n"
    )

    if (
        RESULTS_DIR /
        "experiment4f_attack_generalization_matrix.csv"
    ).exists():

        f.write(
            df4f.to_string(
                index=False
            )
        )

    f.write("\n\n")

    f.write(
        "=" * 75 + "\n"
    )

    f.write(
        "EXPERIMENT 4G\n"
    )

    f.write(
        "=" * 75 + "\n\n"
    )

    if (
        RESULTS_DIR /
        "experiment4g_pair_summary.csv"
    ).exists():

        f.write(
            df4g.to_string(
                index=False
            )
        )


# ------------------------------------------------------------
# Display final table
# ------------------------------------------------------------

print("\n")
print("=" * 75)
print("MASTER RESULTS TABLE")
print("=" * 75)

print(
    master_df.to_string(
        index=False
    )
)


print("\n")
print("=" * 75)
print("MASTER RESULTS CREATED")
print("=" * 75)

print("\nCSV:")
print(csv_path)

print("\nText report:")
print(txt_path)