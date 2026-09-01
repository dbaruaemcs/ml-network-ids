from pathlib import Path
import warnings
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

warnings.filterwarnings("ignore")


# ============================================================
# EXPERIMENT 4F
# CROSS ATTACK-TYPE GENERALIZATION MATRIX
# ============================================================

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
MODELS_DIR = Path("models")

RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42

MAX_ATTACK = 50_000
MAX_BENIGN = 50_000


print("=" * 70)
print("EXPERIMENT 4F - CROSS ATTACK-TYPE GENERALIZATION")
print("=" * 70)


# ============================================================
# DATA SOURCES
# ============================================================

FILES = {
    "patator": "Tuesday-WorkingHours.pcap_ISCX.csv",
    "dos": "Wednesday-workingHours.pcap_ISCX.csv",
    "ddos": "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
}


# ============================================================
# LOAD DATA
# ============================================================

datasets = {}

for name, filename in FILES.items():

    print(f"\nLoading {name.upper()}:")
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

    datasets[name] = df

    print(
        f"Total records: {len(df):,}"
    )


# ============================================================
# CREATE ATTACK FAMILY DATASETS
# ============================================================

def build_family_dataset(
    df,
    family
):

    if family == "patator":

        attack_mask = df["Label"].isin([
            "FTP-Patator",
            "SSH-Patator"
        ])

    elif family == "dos":

        attack_mask = (
            df["Label"].isin([
                "DoS Hulk",
                "DoS GoldenEye",
                "DoS slowloris",
                "DoS Slowhttptest",
                "Heartbleed"
            ])
        )

    elif family == "ddos":

        attack_mask = (
            df["Label"] == "DDoS"
        )

    else:

        raise ValueError(
            f"Unknown family: {family}"
        )


    attacks = df[
        attack_mask
    ].copy()

    benign = df[
        df["Label"] == "BENIGN"
    ].copy()


    attacks = attacks.sample(
        n=min(
            MAX_ATTACK,
            len(attacks)
        ),
        random_state=RANDOM_STATE
    )


    benign = benign.sample(
        n=min(
            MAX_BENIGN,
            len(benign)
        ),
        random_state=RANDOM_STATE
    )


    result = pd.concat(
        [
            attacks,
            benign
        ],
        ignore_index=True
    )


    result["Binary_Label"] = np.where(
        result["Label"] == "BENIGN",
        "BENIGN",
        "ATTACK"
    )


    result = result.sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)


    return result


family_data = {}

for family in [
    "patator",
    "dos",
    "ddos"
]:

    family_data[family] = (
        build_family_dataset(
            datasets[
                family
            ],
            family
        )
    )

    print(
        f"\n{family.upper()} DATASET"
    )

    print(
        family_data[
            family
        ]["Binary_Label"].value_counts()
    )


# ============================================================
# FEATURE PREPARATION
# ============================================================

def prepare_features(
    train_df,
    test_df
):

    train_df = train_df.copy()
    test_df = test_df.copy()


    remove_columns = [
        "Label",
        "Binary_Label",
        "Flow ID",
        "Source IP",
        "Source Port",
        "Destination IP",
        "Destination Port",
        "Timestamp"
    ]


    train_remove = [
        c for c in remove_columns
        if c in train_df.columns
    ]

    test_remove = [
        c for c in remove_columns
        if c in test_df.columns
    ]


    y_train = train_df[
        "Binary_Label"
    ]

    y_test = test_df[
        "Binary_Label"
    ]


    X_train = train_df.drop(
        columns=train_remove
    )

    X_test = test_df.drop(
        columns=test_remove
    )


    common_features = [
        c for c in X_train.columns
        if c in X_test.columns
    ]


    X_train = X_train[
        common_features
    ]

    X_test = X_test[
        common_features
    ]


    X_train = X_train.apply(
        pd.to_numeric,
        errors="coerce"
    )

    X_test = X_test.apply(
        pd.to_numeric,
        errors="coerce"
    )


    X_train = X_train.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X_test = X_test.replace(
        [np.inf, -np.inf],
        np.nan
    )


    # Imputation values are learned ONLY
    # from training data.

    medians = X_train.median()

    X_train = X_train.fillna(
        medians
    )

    X_test = X_test.fillna(
        medians
    )


    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)


    return (
        X_train,
        y_train,
        X_test,
        y_test
    )


# ============================================================
# CROSS ATTACK MATRIX
# ============================================================

families = [
    "ddos",
    "dos",
    "patator"
]


results = []


for train_family in families:

    for test_family in families:

        if train_family == test_family:

            continue


        print("\n")
        print("=" * 70)

        print(
            f"TRAIN: {train_family.upper()}"
        )

        print(
            f"TEST : {test_family.upper()}"
        )

        print("=" * 70)


        train_df = family_data[
            train_family
        ]

        test_df = family_data[
            test_family
        ]


        X_train, y_train, X_test, y_test = (
            prepare_features(
                train_df,
                test_df
            )
        )


        print(
            f"Training samples: "
            f"{len(X_train):,}"
        )

        print(
            f"Testing samples: "
            f"{len(X_test):,}"
        )

        print(
            f"Features: "
            f"{X_train.shape[1]}"
        )


        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        model = RandomForestClassifier(
            n_estimators=150,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced_subsample",
            max_features="sqrt"
        )


        print(
            "\nTraining Random Forest..."
        )


        model.fit(
            X_train,
            y_train
        )


        print(
            "Testing..."
        )


        y_pred = model.predict(
            X_test
        )


        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        precision = precision_score(
            y_test,
            y_pred,
            pos_label="ATTACK",
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            pos_label="ATTACK",
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            pos_label="ATTACK",
            zero_division=0
        )


        fp = (
            (y_test == "BENIGN") &
            (y_pred == "ATTACK")
        ).sum()


        fn = (
            (y_test == "ATTACK") &
            (y_pred == "BENIGN")
        ).sum()


        tp = (
            (y_test == "ATTACK") &
            (y_pred == "ATTACK")
        ).sum()


        tn = (
            (y_test == "BENIGN") &
            (y_pred == "BENIGN")
        ).sum()


        print("\nRESULT")

        print(
            f"Accuracy : {accuracy:.4f}"
        )

        print(
            f"Precision: {precision:.4f}"
        )

        print(
            f"Recall   : {recall:.4f}"
        )

        print(
            f"F1 Score : {f1:.4f}"
        )

        print(
            f"False Positives: {fp:,}"
        )

        print(
            f"False Negatives: {fn:,}"
        )


        results.append({

            "Train_Attack_Family":
                train_family,

            "Test_Attack_Family":
                test_family,

            "Training_Samples":
                len(X_train),

            "Testing_Samples":
                len(X_test),

            "Accuracy":
                accuracy,

            "Precision":
                precision,

            "Recall":
                recall,

            "F1":
                f1,

            "False_Positive":
                fp,

            "False_Negative":
                fn,

            "True_Positive":
                tp,

            "True_Negative":
                tn

        })


# ============================================================
# SAVE MATRIX
# ============================================================

results_df = pd.DataFrame(
    results
)


results_file = (
    RESULTS_DIR /
    "experiment4f_attack_generalization_matrix.csv"
)


results_df.to_csv(
    results_file,
    index=False
)


# ============================================================
# TEXT REPORT
# ============================================================

report_file = (
    RESULTS_DIR /
    "experiment4f_results.txt"
)


with open(
    report_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "EXPERIMENT 4F - CROSS ATTACK-TYPE GENERALIZATION\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        "Dataset: CIC-IDS2017\n"
    )

    f.write(
        "Attack families: DDoS, DoS, Patator\n"
    )

    f.write(
        "Maximum attack samples per family: 50,000\n"
    )

    f.write(
        "Maximum benign samples per family: 50,000\n"
    )

    f.write(
        "Random state: 42\n\n"
    )


    f.write(
        results_df.to_string(
            index=False
        )
    )


    f.write(
        "\n\n"
    )

    f.write(
        "INTERPRETATION\n"
    )

    f.write("=" * 70 + "\n")

    f.write(
        "\nEach row represents training on one "
        "attack family and testing on a different "
        "attack family.\n"
    )

    f.write(
        "\nThe diagonal cases are intentionally "
        "excluded because this experiment focuses "
        "on unseen attack-type generalization.\n"
    )


print("\n")
print("=" * 70)
print("EXPERIMENT 4F COMPLETED")
print("=" * 70)

print("\nRESULTS")

print(
    results_df.to_string(
        index=False
    )
)

print("\nMatrix saved to:")
print(results_file)

print("\nFull report saved to:")
print(report_file)