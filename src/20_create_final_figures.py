from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# FINAL RESEARCH FIGURES
# CIC-IDS2017 ML NETWORK IDS
# ============================================================

RESULTS_DIR = Path("results")


print("=" * 75)
print("CREATING FINAL RESEARCH FIGURES")
print("=" * 75)


# ============================================================
# FIGURE 1
# Generalization Performance
# ============================================================

print("\nCreating Figure 1: Generalization Performance...")


experiments = [
    "3A\nRandom Split",
    "3B\nCross-Day",
    "3C\nDDoS Generalization"
]

accuracy = [
    0.9990,
    0.6398,
    0.8149
]

recall = [
    0.9990,
    0.2810,
    0.6351
]

f1 = [
    0.9990,
    0.4382,
    0.7764
]


x = range(len(experiments))

plt.figure(figsize=(10, 6))

plt.plot(
    x,
    accuracy,
    marker="o",
    label="Accuracy"
)

plt.plot(
    x,
    recall,
    marker="o",
    label="Recall"
)

plt.plot(
    x,
    f1,
    marker="o",
    label="F1 Score"
)

plt.xticks(
    list(x),
    experiments
)

plt.ylim(0, 1.05)

plt.ylabel("Score")

plt.title(
    "Model Performance Under Different Evaluation Settings"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

figure1 = (
    RESULTS_DIR /
    "figure1_generalization.png"
)

plt.savefig(
    figure1,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Saved: {figure1}")


# ============================================================
# FIGURE 2
# Domain Adaptation / DDoS Exposure
# ============================================================

print("\nCreating Figure 2: Domain Adaptation...")


domain_file = (
    RESULTS_DIR /
    "experiment4c_domain_adaptation.csv"
)

domain_df = pd.read_csv(
    domain_file
)


# Identify exposure column

exposure_column = None

for column in domain_df.columns:

    if "Exposure" in column:

        exposure_column = column

        break


if exposure_column is None:

    raise ValueError(
        "Could not find DDoS exposure column "
        "in experiment4c_domain_adaptation.csv"
    )


# Convert exposure to percentage

exposure = (
    domain_df[exposure_column] * 100
)


plt.figure(figsize=(10, 6))

plt.plot(
    exposure,
    domain_df["Accuracy"],
    marker="o",
    label="Accuracy"
)

plt.plot(
    exposure,
    domain_df["Recall"],
    marker="o",
    label="Recall"
)

plt.plot(
    exposure,
    domain_df["F1"],
    marker="o",
    label="F1 Score"
)

plt.xlabel(
    "DDoS Training Exposure (%)"
)

plt.ylabel(
    "Score"
)

plt.title(
    "Effect of Target-Domain DDoS Exposure on Detection Performance"
)

plt.ylim(
    0.5,
    1.02
)

plt.xticks(
    exposure
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

figure2 = (
    RESULTS_DIR /
    "figure2_domain_adaptation.png"
)

plt.savefig(
    figure2,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Saved: {figure2}")


# ============================================================
# FIGURE 3
# Attack Family Generalization
# ============================================================

print("\nCreating Figure 3: Attack Family Generalization...")


family_file = (
    RESULTS_DIR /
    "experiment4f_attack_generalization_matrix.csv"
)

family_df = pd.read_csv(
    family_file
)


# ------------------------------------------------------------
# Determine metric column
# ------------------------------------------------------------

metric_candidates = [
    "Recall",
    "F1",
    "Accuracy"
]

metric_column = None

for candidate in metric_candidates:

    if candidate in family_df.columns:

        metric_column = candidate

        break


if metric_column is None:

    raise ValueError(
        "Could not identify metric column "
        "in experiment4f_attack_generalization_matrix.csv"
    )


# ------------------------------------------------------------
# Create matrix
# ------------------------------------------------------------

matrix = family_df.pivot(
    index="Train_Attack_Family",
    columns="Test_Attack_Family",
    values=metric_column
)


plt.figure(figsize=(9, 7))

image = plt.imshow(
    matrix.values,
    aspect="auto",
    vmin=0,
    vmax=1
)

plt.colorbar(
    image,
    label=metric_column
)

plt.xticks(
    range(len(matrix.columns)),
    matrix.columns,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(matrix.index)),
    matrix.index
)

plt.xlabel(
    "Test Attack Family"
)

plt.ylabel(
    "Training Attack Family"
)

plt.title(
    f"Cross-Attack-Family Generalization ({metric_column})"
)


# Add values to cells

for i in range(
    len(matrix.index)
):

    for j in range(
        len(matrix.columns)
    ):

        value = matrix.iloc[i, j]

        if pd.notna(value):

            plt.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center"
            )


plt.tight_layout()

figure3 = (
    RESULTS_DIR /
    "figure3_attack_family_generalization.png"
)

plt.savefig(
    figure3,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Saved: {figure3}")


# ============================================================
# FIGURE 4
# Distribution Shift
# ============================================================

print("\nCreating Figure 4: Distribution Shift...")


shift_file = (
    RESULTS_DIR /
    "experiment4g_pair_summary.csv"
)

shift_df = pd.read_csv(
    shift_file
)


shift_df["Pair"] = (
    shift_df["Family_A"]
    + " → "
    + shift_df["Family_B"]
)


plt.figure(figsize=(10, 6))

bars = plt.bar(
    shift_df["Pair"],
    shift_df["Mean_KS"]
)

plt.ylabel(
    "Mean KS Statistic"
)

plt.xlabel(
    "Attack-Family Pair"
)

plt.title(
    "Feature Distribution Shift Between Attack Families"
)

plt.ylim(
    0,
    max(
        shift_df["Mean_KS"]
    ) * 1.15
)

plt.xticks(
    rotation=20
)


# Add values above bars

for bar, value in zip(
    bars,
    shift_df["Mean_KS"]
):

    plt.text(
        bar.get_x()
        + bar.get_width() / 2,
        bar.get_height()
        + 0.01,
        f"{value:.3f}",
        ha="center"
    )


plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

figure4 = (
    RESULTS_DIR /
    "figure4_distribution_shift.png"
)

plt.savefig(
    figure4,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Saved: {figure4}")


# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 75)
print("FINAL FIGURES COMPLETED")
print("=" * 75)

print("\nCreated files:")

print(
    "1.",
    RESULTS_DIR /
    "figure1_generalization.png"
)

print(
    "2.",
    RESULTS_DIR /
    "figure2_domain_adaptation.png"
)

print(
    "3.",
    RESULTS_DIR /
    "figure3_attack_family_generalization.png"
)

print(
    "4.",
    RESULTS_DIR /
    "figure4_distribution_shift.png"
)

print("\nNo model was retrained.")