from pathlib import Path
import pandas as pd


# ============================================================
# MASTER RESULTS CONSOLIDATION
# ============================================================

RESULTS_DIR = Path("results")

print("=" * 75)
print("CREATING MASTER EXPERIMENT RESULTS")
print("=" * 75)


# ============================================================
# EXPERIMENT RESULTS
# ============================================================

results = [
    {
        "Experiment": "3A",
        "Purpose": "Multiclass classification",
        "Accuracy": 0.9990,
        "Precision": 0.9990,
        "Recall": 0.9990,
        "F1": 0.9990,
        "Source": "experiment3a_results.txt"
    },
    {
        "Experiment": "3B",
        "Purpose": "Cross-day generalization",
        "Accuracy": 0.6398,
        "Precision": 0.9952,
        "Recall": 0.2810,
        "F1": 0.4382,
        "Source": "experiment3b_results.txt"
    },
    {
        "Experiment": "3C",
        "Purpose": "DDoS generalization",
        "Accuracy": 0.8149,
        "Precision": 0.9986,
        "Recall": 0.6351,
        "F1": 0.7764,
        "Source": "experiment3c_results.txt"
    },
    {
        "Experiment": "3D",
        "Purpose": "DDoS false-negative analysis",
        "Accuracy": None,
        "Precision": None,
        "Recall": 0.6351,
        "F1": None,
        "Source": "experiment3d_results.txt"
    },
    {
        "Experiment": "4A",
        "Purpose": "Classification threshold analysis",
        "Accuracy": 0.8150,
        "Precision": 0.9991,
        "Recall": 0.6349,
        "F1": 0.7764,
        "Source": "experiment4a_results.txt"
    },
    {
        "Experiment": "4B",
        "Purpose": "Feature engineering",
        "Accuracy": 0.8147,
        "Precision": 0.9986,
        "Recall": 0.6346,
        "F1": 0.7760,
        "Source": "experiment4b_results.txt"
    },
    {
        "Experiment": "4C",
        "Purpose": "Target-domain exposure",
        "Accuracy": 0.999238,
        "Precision": 0.999157,
        "Recall": 0.999500,
        "F1": 0.999328,
        "Source": "experiment4c_domain_adaptation.csv"
    },
    {
        "Experiment": "4D",
        "Purpose": "Unseen attack-family testing",
        "Accuracy": 0.8710,
        "Precision": 0.0035,
        "Recall": 0.0002,
        "F1": 0.0004,
        "Source": "experiment4d_results.txt"
    },
    {
        "Experiment": "4E",
        "Purpose": "Cross-family DDoS testing",
        "Accuracy": 0.8143,
        "Precision": 0.9989,
        "Recall": 0.6335,
        "F1": 0.7753,
        "Source": "experiment4e_results.txt"
    },
]


# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(results)

print("\nMASTER RESULTS")
print("=" * 75)

print(
    df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# SAVE CSV
# ============================================================

csv_file = (
    RESULTS_DIR /
    "master_experiment_results.csv"
)

df.to_csv(
    csv_file,
    index=False
)

print(
    f"\nMaster CSV saved to:\n{csv_file}"
)


# ============================================================
# CREATE TEXT REPORT
# ============================================================

txt_file = (
    RESULTS_DIR /
    "master_experiment_results.txt"
)

with open(
    txt_file,
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
        "Project: Machine Learning Based Network Intrusion Detection\n\n"
    )

    f.write(
        "The table below consolidates the principal results "
        "from Experiments 3A through 4E.\n\n"
    )

    f.write(
        df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    f.write("\n\n")

    f.write(
        "=" * 75 + "\n"
    )

    f.write(
        "KEY OBSERVATIONS\n"
    )

    f.write(
        "=" * 75 + "\n\n"
    )

    f.write(
        "1. Random-split multiclass classification achieved "
        "very high performance (Experiment 3A, F1=0.9990).\n\n"
    )

    f.write(
        "2. Cross-day evaluation produced substantially lower "
        "performance (Experiment 3B, F1=0.4382), demonstrating "
        "limited temporal generalization.\n\n"
    )

    f.write(
        "3. DDoS-focused evaluation achieved F1=0.7764 with "
        "recall=0.6351, showing that false negatives remain "
        "a major challenge.\n\n"
    )

    f.write(
        "4. Feature engineering in Experiment 4B produced "
        "performance very similar to the baseline DDoS model, "
        "indicating that engineered features alone did not "
        "solve the generalization problem.\n\n"
    )

    f.write(
        "5. Target-domain exposure in Experiment 4C produced "
        "a dramatic improvement. With 25% exposure, F1 reached "
        "approximately 0.9992, demonstrating the importance "
        "of representative target-domain training data.\n\n"
    )

    f.write(
        "6. Experiment 4D showed extremely poor transfer to "
        "an unseen attack family, with recall=0.0002 and "
        "F1=0.0004. High accuracy in this setting is therefore "
        "misleading because the model failed to detect attacks.\n\n"
    )

    f.write(
        "7. Experiment 4G demonstrated substantial statistical "
        "distribution shifts between attack families, supporting "
        "the observed transfer-learning limitations.\n\n"
    )

    f.write(
        "8. Overall, the experiments indicate that conventional "
        "random train/test evaluation can substantially "
        "overestimate the real-world generalization capability "
        "of ML-based intrusion detection models.\n"
    )


print(
    f"Master report saved to:\n{txt_file}"
)


# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 75)
print("MASTER RESULTS CONSOLIDATION COMPLETED")
print("=" * 75)