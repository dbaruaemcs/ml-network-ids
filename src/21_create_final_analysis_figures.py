from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# FINAL ANALYSIS FIGURES
# ============================================================

RESULTS_DIR = Path("results")

print("=" * 75)
print("CREATING FINAL ANALYSIS FIGURES")
print("=" * 75)


# ============================================================
# FIGURE 5
# DDoS Confusion Matrix
# ============================================================

print("\nCreating Figure 5: DDoS Confusion Matrix...")

# Use the actual values from Experiment 4B.
# This avoids depending on the CSV's formatting.

cm = [
    [97631, 87],
    [36544, 63456]
]

labels = ["BENIGN", "ATTACK"]

plt.figure(figsize=(7, 6))

image = plt.imshow(
    cm,
    interpolation="nearest"
)

plt.colorbar(image)

plt.xticks(
    [0, 1],
    labels
)

plt.yticks(
    [0, 1],
    labels
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.title(
    "DDoS Detection Confusion Matrix"
)

for i in range(2):
    for j in range(2):

        plt.text(
            j,
            i,
            f"{cm[i][j]:,}",
            ha="center",
            va="center"
        )

plt.tight_layout()

figure5 = (
    RESULTS_DIR /
    "figure5_ddos_confusion_matrix.png"
)

plt.savefig(
    figure5,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Saved: {figure5}")


# ============================================================
# FIGURE 6
# FEATURE IMPORTANCE
# ============================================================

print("\nCreating Figure 6: Feature Importance...")

feature_file = (
    RESULTS_DIR /
    "experiment4b_feature_importance.csv"
)

feature_df = pd.read_csv(
    feature_file
)

feature_df = feature_df.sort_values(
    "Importance",
    ascending=True
).tail(20)

plt.figure(figsize=(10, 8))

plt.barh(
    feature_df["Feature"],
    feature_df["Importance"]
)

plt.xlabel(
    "Feature Importance"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Top 20 Features in Feature-Engineered DDoS Detector"
)

plt.tight_layout()

figure6 = (
    RESULTS_DIR /
    "figure6_feature_importance.png"
)

plt.savefig(
    figure6,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Saved: {figure6}")


# ============================================================
# FIGURE 7
# FALSE-NEGATIVE FEATURE ANALYSIS
# ============================================================

print("\nCreating Figure 7: False-Negative Analysis...")

fn_file = (
    RESULTS_DIR /
    "experiment3d_feature_comparison.csv"
)

fn_df = pd.read_csv(
    fn_file
)

# Remove invalid values

fn_df = fn_df.replace(
    [float("inf"), float("-inf")],
    pd.NA
)

fn_df = fn_df.dropna(
    subset=["Relative_Difference"]
)

# Select the 15 largest differences

fn_df = fn_df.sort_values(
    "Relative_Difference",
    ascending=False
).head(15)

# Reverse order for horizontal bar chart

fn_df = fn_df.iloc[::-1]

plt.figure(figsize=(10, 8))

plt.barh(
    fn_df["Feature"],
    fn_df["Relative_Difference"]
)

plt.xlabel(
    "Relative Difference"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Features Differentiating Detected and Missed DDoS Attacks"
)

plt.tight_layout()

figure7 = (
    RESULTS_DIR /
    "figure7_false_negative_features.png"
)

plt.savefig(
    figure7,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Saved: {figure7}")

# ============================================================
# FIGURE 8
# CLASSIFICATION THRESHOLD ANALYSIS
# ============================================================

print("\nCreating Figure 8: Threshold Analysis...")

threshold_file = (
    RESULTS_DIR /
    "experiment4a_threshold_results.csv"
)

threshold_df = pd.read_csv(
    threshold_file
)

plt.figure(figsize=(10, 6))

plt.plot(
    threshold_df["Threshold"],
    threshold_df["Precision"],
    marker="o",
    label="Precision"
)

plt.plot(
    threshold_df["Threshold"],
    threshold_df["Recall"],
    marker="o",
    label="Recall"
)

plt.plot(
    threshold_df["Threshold"],
    threshold_df["F1"],
    marker="o",
    label="F1 Score"
)

plt.xlabel(
    "Classification Threshold"
)

plt.ylabel(
    "Score"
)

plt.title(
    "Effect of Classification Threshold on DDoS Detection"
)

plt.ylim(
    0,
    1.05
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

figure8 = (
    RESULTS_DIR /
    "figure8_threshold_analysis.png"
)

plt.savefig(
    figure8,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Saved: {figure8}")


# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 75)
print("FINAL ANALYSIS FIGURES COMPLETED")
print("=" * 75)

print("\nCreated:")

print(
    "5.",
    figure5
)

print(
    "6.",
    figure6
)

print(
    "7.",
    figure7
)

print(
    "8.",
    figure8
)

print("\nNo model was retrained.")