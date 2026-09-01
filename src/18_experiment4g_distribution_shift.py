from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from scipy.stats import ks_2samp


warnings.filterwarnings("ignore")


# ============================================================
# EXPERIMENT 4G
# FEATURE DISTRIBUTION / DOMAIN SHIFT ANALYSIS
# ============================================================

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")

RESULTS_DIR.mkdir(exist_ok=True)


RANDOM_STATE = 42

SAMPLE_SIZE = 20_000


print("=" * 70)
print("EXPERIMENT 4G - FEATURE DISTRIBUTION SHIFT ANALYSIS")
print("=" * 70)


# ============================================================
# DATA FILES
# ============================================================

FILES = {

    "patator":
        "Tuesday-WorkingHours.pcap_ISCX.csv",

    "dos":
        "Wednesday-workingHours.pcap_ISCX.csv",

    "ddos":
        "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
}


# ============================================================
# FEATURES
# ============================================================

FEATURES = [

    "Flow Duration",

    "Flow IAT Mean",

    "Flow IAT Max",

    "Flow IAT Min",

    "Packet Length Mean",

    "Packet Length Std",

    "Packet Length Variance",

    "Average Packet Size",

    "Fwd Packets/s",

    "Bwd Packets/s",

    "Flow Packets/s",

    "Fwd Packet Length Max",

    "Bwd Packet Length Max",

    "Fwd Packet Length Min",

    "Bwd Packet Length Min",

    "Total Length of Fwd Packets",

    "Total Length of Bwd Packets",

    "Init_Win_bytes_forward",

    "Init_Win_bytes_backward",

    "Bwd Header Length"

]


# ============================================================
# LOAD ATTACK DATA
# ============================================================

family_samples = {}


for family, filename in FILES.items():

    print("\n")
    print("-" * 70)

    print(
        f"Loading {family.upper()}"
    )

    print(filename)

    df = pd.read_csv(
        DATA_DIR / filename,
        low_memory=False
    )

    df.columns = df.columns.str.strip()

    df["Label"] = (
        df["Label"]
        .astype(str)
        .str.strip()
    )


    # --------------------------------------------------------
    # SELECT ATTACK RECORDS
    # --------------------------------------------------------

    if family == "patator":

        attack_labels = [
            "FTP-Patator",
            "SSH-Patator"
        ]

        attack_df = df[
            df["Label"].isin(
                attack_labels
            )
        ].copy()


    elif family == "dos":

        attack_labels = [
            "DoS Hulk",
            "DoS GoldenEye",
            "DoS slowloris",
            "DoS Slowhttptest",
            "Heartbleed"
        ]

        attack_df = df[
            df["Label"].isin(
                attack_labels
            )
        ].copy()


    elif family == "ddos":

        attack_df = df[
            df["Label"] == "DDoS"
        ].copy()


    # --------------------------------------------------------
    # SAMPLE
    # --------------------------------------------------------

    sample_n = min(
        SAMPLE_SIZE,
        len(attack_df)
    )


    attack_df = attack_df.sample(
        n=sample_n,
        random_state=RANDOM_STATE
    )


    print(
        f"Attack records selected: "
        f"{len(attack_df):,}"
    )


    family_samples[
        family
    ] = attack_df


    # Free memory

    del df
    del attack_df


# ============================================================
# CHECK FEATURES
# ============================================================

print("\n")
print("=" * 70)
print("CHECKING FEATURES")
print("=" * 70)


available_features = []


for feature in FEATURES:

    exists_everywhere = all(
        feature in family_samples[family].columns
        for family in family_samples
    )

    if exists_everywhere:

        available_features.append(
            feature
        )


print(
    f"Requested features: "
    f"{len(FEATURES)}"
)

print(
    f"Available features: "
    f"{len(available_features)}"
)


# ============================================================
# PREPARE NUMERIC DATA
# ============================================================

numeric_data = {}


for family in family_samples:

    numeric_data[family] = {}

    for feature in available_features:

        values = pd.to_numeric(
            family_samples[
                family
            ][feature],
            errors="coerce"
        )


        values = values.replace(
            [np.inf, -np.inf],
            np.nan
        )


        values = values.dropna()


        numeric_data[
            family
        ][feature] = values.values


# ============================================================
# BASIC DISTRIBUTION STATISTICS
# ============================================================

print("\n")
print("=" * 70)
print("CALCULATING DISTRIBUTION STATISTICS")
print("=" * 70)


summary_rows = []


for family in [
    "patator",
    "dos",
    "ddos"
]:

    for feature in available_features:

        values = numeric_data[
            family
        ][feature]


        if len(values) == 0:

            continue


        summary_rows.append({

            "Attack_Family":
                family,

            "Feature":
                feature,

            "Count":
                len(values),

            "Mean":
                np.mean(values),

            "Median":
                np.median(values),

            "Std":
                np.std(values),

            "Min":
                np.min(values),

            "Max":
                np.max(values)

        })


summary_df = pd.DataFrame(
    summary_rows
)


summary_file = (
    RESULTS_DIR /
    "experiment4g_feature_statistics.csv"
)


summary_df.to_csv(
    summary_file,
    index=False
)


# ============================================================
# KS DISTRIBUTION SHIFT
# ============================================================

print("\n")
print("=" * 70)
print("CALCULATING KS DISTRIBUTION SHIFT")
print("=" * 70)


pairs = [

    ("dos", "ddos"),

    ("ddos", "patator"),

    ("dos", "patator")

]


shift_rows = []


for family_a, family_b in pairs:

    print(
        f"\nComparing "
        f"{family_a.upper()} vs "
        f"{family_b.upper()}"
    )


    for feature in available_features:

        a = numeric_data[
            family_a
        ][feature]

        b = numeric_data[
            family_b
        ][feature]


        if len(a) < 10 or len(b) < 10:

            continue


        statistic, p_value = ks_2samp(
            a,
            b
        )


        shift_rows.append({

            "Family_A":
                family_a,

            "Family_B":
                family_b,

            "Feature":
                feature,

            "KS_Statistic":
                statistic,

            "P_Value":
                p_value

        })


shift_df = pd.DataFrame(
    shift_rows
)


# ============================================================
# SAVE KS RESULTS
# ============================================================

shift_file = (
    RESULTS_DIR /
    "experiment4g_distribution_shift.csv"
)


shift_df.to_csv(
    shift_file,
    index=False
)


# ============================================================
# TOP DISTRIBUTION SHIFTS
# ============================================================

print("\n")
print("=" * 70)
print("TOP 20 DISTRIBUTION SHIFTS")
print("=" * 70)


top_shift = (
    shift_df
    .sort_values(
        "KS_Statistic",
        ascending=False
    )
    .head(20)
)


print(
    top_shift.to_string(
        index=False
    )
)


# ============================================================
# AVERAGE KS BY ATTACK PAIR
# ============================================================

pair_summary = (

    shift_df
    .groupby(
        [
            "Family_A",
            "Family_B"
        ]
    )[
        "KS_Statistic"
    ]
    .agg(
        [
            "mean",
            "median",
            "max"
        ]
    )
    .reset_index()

)


pair_summary.columns = [

    "Family_A",

    "Family_B",

    "Mean_KS",

    "Median_KS",

    "Max_KS"

]


print("\n")
print("=" * 70)
print("PAIRWISE DISTRIBUTION SHIFT")
print("=" * 70)


print(
    pair_summary.to_string(
        index=False
    )
)


pair_file = (
    RESULTS_DIR /
    "experiment4g_pair_summary.csv"
)


pair_summary.to_csv(
    pair_file,
    index=False
)


# ============================================================
# TEXT REPORT
# ============================================================

report_file = (
    RESULTS_DIR /
    "experiment4g_results.txt"
)


with open(
    report_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "EXPERIMENT 4G - FEATURE DISTRIBUTION SHIFT ANALYSIS\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )


    f.write(
        "Dataset: CIC-IDS2017\n"
    )

    f.write(
        "Attack families: DDoS, DoS, Patator\n"
    )

    f.write(
        f"Sample size per attack family: "
        f"{SAMPLE_SIZE:,}\n"
    )

    f.write(
        "Random state: 42\n\n"
    )


    f.write(
        "PAIRWISE DISTRIBUTION SHIFT\n"
    )

    f.write(
        "=" * 70 + "\n"
    )


    f.write(
        pair_summary.to_string(
            index=False
        )
    )


    f.write(
        "\n\nTOP 20 DISTRIBUTION SHIFTS\n"
    )

    f.write(
        "=" * 70 + "\n"
    )


    f.write(
        top_shift.to_string(
            index=False
        )
    )


    f.write(
        "\n\nINTERPRETATION\n"
    )

    f.write(
        "=" * 70 + "\n"
    )


    f.write(
        "\nThe Kolmogorov-Smirnov statistic measures "
        "the difference between the empirical "
        "distributions of two attack families. "
        "Higher values indicate greater distribution "
        "difference.\n"
    )


    f.write(
        "\nLarge distribution differences can indicate "
        "feature-level domain shift and may help "
        "explain poor cross-attack generalization.\n"
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 70)
print("EXPERIMENT 4G COMPLETED")
print("=" * 70)


print("\nFeature statistics saved to:")
print(summary_file)


print("\nDistribution shift results saved to:")
print(shift_file)


print("\nPair summary saved to:")
print(pair_file)


print("\nFull report saved to:")
print(report_file)