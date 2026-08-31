# Machine Learning Based Network Intrusion Detection System

A machine learning based Network Intrusion Detection System (IDS) developed using Python and the CIC-IDS2017 network traffic dataset.

The project investigates whether supervised machine learning can distinguish benign network traffic from malicious DDoS traffic using network-flow characteristics.

> **Project status:** Baseline experiments completed. Cross-day, multi-attack, and unseen-attack evaluation are planned as the next stage.

---

## 1. Project Overview

Network Intrusion Detection Systems monitor network traffic and identify potentially malicious activity.

Traditional IDS solutions commonly rely on predefined signatures or rules. Machine learning provides an alternative approach by learning patterns from previously observed network traffic.

This project implements a supervised machine learning IDS using the **Random Forest** algorithm.

The initial objective is:

> **Can a machine learning model accurately distinguish benign network flows from DDoS attack traffic using statistical network-flow features?**

The project is being developed incrementally, beginning with a binary DDoS detection baseline and progressing toward multi-attack and generalization testing.

---

## 2. Objectives

The main objectives of this project are:

* Analyze network-flow data from CIC-IDS2017.
* Develop a reproducible preprocessing pipeline.
* Remove identifiers that could introduce unwanted learning behavior.
* Convert network traffic labels into a binary classification problem.
* Train a Random Forest intrusion detection model.
* Evaluate the model using multiple classification metrics.
* Analyze the network features used by the model.
* Investigate the effect of individual features.
* Test whether the model generalizes beyond randomly mixed traffic.
* Extend the system toward multi-class intrusion detection.

---

## 3. Dataset

### CIC-IDS2017

The project currently uses the **CIC-IDS2017** dataset.

The first experiments use:

**Friday-WorkingHours-Afternoon-DDos**

The dataset contains labeled network-flow records representing benign traffic and DDoS attack traffic.

For the initial binary classification experiment:

```text
BENIGN → 0
DDoS   → 1
```

The raw dataset is intentionally excluded from the GitHub repository because of its size.

---

## 4. Technology Stack

| Component            | Technology    |
| -------------------- | ------------- |
| Operating System     | Windows       |
| Programming Language | Python        |
| Machine Learning     | Scikit-learn  |
| Data Processing      | Pandas        |
| Numerical Processing | NumPy         |
| Visualization        | Matplotlib    |
| Model                | Random Forest |
| Model Serialization  | Joblib        |
| Version Control      | Git           |
| Repository           | GitHub        |

---

## 5. Project Structure

```text
ml-network-ids/
│
├── data/
│   └── CIC-IDS2017 datasets
│
├── models/
│   └── trained machine learning models
│
├── results/
│   ├── baseline_results.txt
│   ├── model_results.txt
│   ├── model_results_no_port.txt
│   ├── confusion_matrix.png
│   ├── confusion_matrix_no_port.png
│   └── feature_importance.png
│
├── src/
│   ├── 01_explore_dataset.py
│   ├── 02_preprocess.py
│   ├── 03_train_model.py
│   ├── 04_evaluate_model.py
│   └── 05_feature_importance.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

# 6. Data Preprocessing

The preprocessing pipeline performs the following operations:

### Column name normalization

Whitespace is removed from dataset column names to make feature handling consistent.

### Identifier removal

The following fields are removed:

```text
Flow ID
Source IP
Source Port
Destination IP
Timestamp
```

These fields can act as identifiers rather than generalizable behavioral characteristics.

### Infinite-value handling

Positive and negative infinite values are converted to missing values.

### Missing-value removal

Records containing missing values are removed for the initial experiments.

### Label conversion

The original labels are converted to binary classes:

```text
BENIGN → 0
DDoS   → 1
```

### Duplicate removal

Duplicate network-flow records are removed.

---

# 7. Baseline Model

The first machine learning model is a **Random Forest Classifier**.

Configuration:

```text
n_estimators = 100
random_state = 42
class_weight = balanced
n_jobs = -1
```

The dataset is divided into:

```text
80% → Training
20% → Testing
```

A fixed random state of `42` is used to make the experiment reproducible.

---

# 8. Experiment 1: Baseline DDoS Detection

### Objective

Determine whether a Random Forest classifier can distinguish benign traffic from DDoS traffic using the selected network-flow features.

### Evaluation method

A stratified 80/20 random train-test split was used.

### Test set

```text
44,617 network flows
```

### Results

| Metric    |  Result |
| --------- | ------: |
| Accuracy  |  99.99% |
| Precision | 100.00% |
| Recall    |  99.99% |
| F1 Score  |  99.99% |
| ROC-AUC   | 100.00% |

### Confusion Matrix

```text
                  Predicted
                BENIGN  ATTACK

Actual BENIGN     19014       0
Actual ATTACK         3   25600
```

Therefore:

```text
True Negatives  = 19,014
False Positives = 0
False Negatives = 3
True Positives  = 25,600
```

The model produced only three false negatives and no false positives on the test set.

---

# 9. Experiment 2: Removing Destination Port

The second experiment investigated whether the model's performance depended significantly on the `Destination Port` feature.

The same preprocessing and Random Forest configuration were used, but `Destination Port` was removed.

### Objective

Determine whether the model maintains its performance without destination-port information.

### Results

| Metric          | Experiment 1 | Experiment 2 |
| --------------- | -----------: | -----------: |
| Accuracy        |       99.99% |       99.99% |
| Precision       |      100.00% |      100.00% |
| Recall          |       99.99% |       99.99% |
| F1 Score        |       99.99% |       99.99% |
| ROC-AUC         |      100.00% |      100.00% |
| False Positives |            0 |            0 |
| False Negatives |            3 |            3 |

### Interpretation

Removing `Destination Port` produced no measurable change in the evaluation results.

This indicates that the model was not dependent on destination-port information for this particular DDoS classification task.

However, this experiment does **not** prove that the model generalizes to different networks or attack scenarios.

---

# 10. Feature Importance Analysis

Random Forest feature importance was used to investigate which network-flow characteristics contributed most strongly to the classification.

The top features from the baseline model were:

| Rank | Feature                     | Importance |
| ---: | --------------------------- | ---------: |
|    1 | Init_Win_bytes_forward      |   0.074361 |
|    2 | Fwd Packet Length Max       |   0.065205 |
|    3 | Bwd Packet Length Max       |   0.057110 |
|    4 | Fwd Packet Length Mean      |   0.051349 |
|    5 | Avg Bwd Segment Size        |   0.045695 |
|    6 | Avg Fwd Segment Size        |   0.042871 |
|    7 | Subflow Fwd Bytes           |   0.042477 |
|    8 | Total Length of Fwd Packets |   0.038788 |
|    9 | Bwd Header Length           |   0.037347 |
|   10 | act_data_pkt_fwd            |   0.031733 |

Other highly ranked features included:

```text
Bwd Packet Length Min
Bwd Packet Length Std
Subflow Fwd Packets
Average Packet Size
Destination Port
Bwd Packet Length Mean
Fwd IAT Std
Fwd IAT Total
Fwd Header Length
Fwd Packet Length Std
```

### Interpretation

The important features are primarily related to network-flow behavior, including:

* Forward and backward packet sizes
* Packet counts
* Flow byte volumes
* TCP characteristics
* Segment sizes
* Inter-arrival times

This suggests that the Random Forest is learning characteristics of network traffic behavior rather than relying solely on explicit IP-based identifiers.

---

# 11. Important Limitation of Current Results

The current results should **not** be interpreted as evidence that the IDS can detect attacks in arbitrary real-world networks.

The primary reason is the evaluation methodology.

The current experiments use a random train-test split:

```text
CIC-IDS2017 DDoS flows
        ↓
Random 80/20 split
        ↓
Training + Testing
```

Network flows originating from the same controlled capture environment can have similar characteristics in both the training and testing sets.

Consequently, the very high performance may partly reflect dataset-specific patterns.

Therefore:

> **99.99% accuracy is currently a baseline result, not a claim of real-world IDS performance.**

This limitation is an intentional part of the project and will be investigated through subsequent experiments.

---

# 12. Planned Experiments

The next development phase will focus on model generalization.

## Experiment 3: Full CIC-IDS2017 Dataset

Expand beyond the Friday DDoS subset and incorporate multiple attack categories.

Potential classes include:

```text
BENIGN
DDoS
DoS Hulk
DoS GoldenEye
DoS slowloris
DoS Slowhttptest
PortScan
FTP-Patator
SSH-Patator
Bot
Web Attack
Infiltration
Heartbleed
```

---

## Experiment 4: Model Comparison

Compare several machine learning algorithms:

```text
Logistic Regression
Random Forest
MLP Neural Network
```

The models will be evaluated using the same dataset and evaluation methodology.

---

## Experiment 5: Cross-Day Validation

Instead of randomly mixing all traffic, training and testing data will be separated according to capture days.

Example:

```text
Training
   ↓
Monday + Tuesday + Wednesday

Testing
   ↓
Thursday + Friday
```

The objective is to determine whether the model generalizes to traffic captured under different conditions.

---

## Experiment 6: Unseen Attack Detection

The model will be trained using selected attack categories and evaluated against an attack type that was not present during training.

This experiment will investigate whether the model can identify previously unseen malicious traffic.

---

## Experiment 7: Error Analysis

False positives and false negatives will be investigated individually.

Particular attention will be given to:

* Missed attacks
* False alarms
* Attack-specific performance
* Class imbalance
* Feature dependence

---

# 13. Research Questions

The project is ultimately intended to investigate the following questions:

### RQ1

How accurately can supervised machine learning distinguish benign network traffic from malicious traffic using flow-level characteristics?

### RQ2

Which network-flow features contribute most strongly to intrusion classification?

### RQ3

Does removing potentially influential features significantly affect detection performance?

### RQ4

Does a model trained on one network-traffic distribution generalize to traffic collected under different conditions?

### RQ5

Can machine learning identify attack types that were not represented during training?

---

# 14. Current Findings

The first two experiments demonstrate that Random Forest can achieve near-perfect classification performance on the selected CIC-IDS2017 DDoS subset under random holdout validation.

The removal of `Destination Port` produced no measurable change in the results.

Feature-importance analysis indicates that the model primarily uses network-flow characteristics such as packet lengths, segment sizes, flow volumes, TCP-related characteristics, and packet timing.

However, the current evaluation methodology does not establish real-world generalization.

The next phase therefore focuses on **cross-day evaluation, multiple attack categories, unseen attacks, and comparative model evaluation**.

---

# 15. Security Perspective

The project is designed as an educational and research-oriented IDS prototype.

It does not currently inspect live network packets or automatically block malicious traffic.

The current architecture is:

```text
Network Flow Dataset
        ↓
Preprocessing
        ↓
Feature Extraction
        ↓
Machine Learning Model
        ↓
BENIGN / ATTACK
        ↓
Evaluation
```

Future development may extend the system toward near-real-time traffic classification.

---

# 16. Reproducibility

The project uses fixed random seeds and scripted preprocessing/training/evaluation steps.

Example:

```bash
python src/02_preprocess.py
python src/03_train_model.py
python src/04_evaluate_model.py
python src/05_feature_importance.py
```

The raw datasets and trained model binaries are excluded from version control through `.gitignore`.

---

# 17. Current Status

```text
[x] Dataset exploration
[x] Data preprocessing
[x] Binary DDoS classification
[x] Random Forest baseline
[x] Model evaluation
[x] Confusion matrix
[x] Feature importance analysis
[x] Destination-port ablation experiment

[ ] Full CIC-IDS2017 integration
[ ] Multi-class classification
[ ] Cross-day validation
[ ] Unseen-attack evaluation
[ ] Logistic Regression comparison
[ ] MLP comparison
[ ] Advanced error analysis
[ ] Final research report
```

---

## 18. Author

Developed as a cybersecurity and machine learning research project to investigate practical applications of artificial intelligence in network intrusion detection.

---

## 19. Disclaimer

This project is intended for educational, research, and defensive cybersecurity purposes.

The reported results are specific to the datasets and experimental methodology used. High performance on a benchmark dataset should not be interpreted as proof of equivalent performance in production networks.

---

## 20. Future Direction

The long-term goal is to develop a more rigorous ML-based IDS evaluation framework that emphasizes not only classification accuracy but also:

```text
Generalization
↓
Attack coverage
↓
False-positive reduction
↓
Unseen attack detection
↓
Interpretability
↓
Practical deployment
```

The emphasis of the project is therefore shifting from simply achieving high accuracy toward understanding **whether the model actually generalizes to new and previously unseen network traffic**.
