# Machine Learning Based Network Intrusion Detection System

A research-oriented machine learning Network Intrusion Detection System (NIDS) developed using Python and the CIC-IDS2017 network traffic dataset.

The project investigates not only whether machine learning can classify network attacks, but more importantly **whether a model trained on one traffic distribution can generalize to different days, attack families, and distribution conditions**.

The main model used throughout the experiments is a Random Forest classifier.

> **Project status: Completed experimental study and final research paper.**

---

## 1. Project Overview

Traditional Network Intrusion Detection Systems commonly depend on predefined signatures, rules, or manually engineered detection logic.

Machine learning provides an alternative by learning statistical patterns from network traffic. However, high performance on a benchmark dataset does not necessarily mean that a model will perform well when exposed to traffic generated under different conditions or previously unseen attack families.

This project therefore follows two stages:

1. Establish a high-performance benchmark using conventional random train/test validation.
2. Investigate whether that performance survives more demanding generalization and distribution-shift experiments.

The central research question is:

> **Does a machine learning based NIDS trained on CIC-IDS2017 traffic generalize reliably to different traffic distributions and attack scenarios?**

---

# 2. Research Objectives

The project investigates:

* Network-flow based intrusion detection using machine learning
* Binary and multiclass attack classification
* Feature importance and feature engineering
* Cross-day generalization
* DDoS generalization
* False-negative behavior
* Classification threshold sensitivity
* Target-domain exposure
* Unseen attack-family detection
* Cross-family attack generalization
* Statistical distribution shift
* The relationship between benchmark performance and generalization

The broader objective is to move beyond accuracy alone and evaluate whether an IDS model learns **generalizable attack behavior**.

---

# 3. Dataset

## CIC-IDS2017

The experiments use the **CIC-IDS2017** network intrusion detection dataset.

The dataset contains labeled network flows representing benign traffic and multiple categories of attacks.

The repository does **not** contain the raw CIC-IDS2017 CSV files because of their size and licensing/distribution considerations.

Dataset files should be placed locally under:

```text
data/
```

The repository provides:

```text
data/README.md
```

with information about the expected dataset structure.

---

# 4. Technology Stack

| Component            | Technology    |
| -------------------- | ------------- |
| Operating System     | Windows       |
| Programming Language | Python        |
| Machine Learning     | Scikit-learn  |
| Data Processing      | Pandas        |
| Numerical Processing | NumPy         |
| Visualization        | Matplotlib    |
| Classification Model | Random Forest |
| Model Serialization  | Joblib        |
| Version Control      | Git           |
| Repository           | GitHub        |

---

# 5. Repository Structure

```text
ml-network-ids/
│
├── data/
│   └── README.md
│
├── models/
│   └── README.md
│
├── paper/
│   ├── Evaluating Machine Learning-Based Network Intrusion Detection.pdf
│   └── Evaluating NIDS Generalization Under Distribution Shift.docx
│
├── results/
│   ├── baseline_results.txt
│   ├── dataset_inventory.csv
│   ├── experiment3a_*
│   ├── experiment3b_*
│   ├── experiment3c_*
│   ├── experiment3d_*
│   ├── experiment4a_*
│   ├── experiment4b_*
│   ├── experiment4c_*
│   ├── experiment4d_*
│   ├── experiment4e_*
│   ├── experiment4f_*
│   ├── experiment4g_*
│   ├── figure1_generalization.png
│   ├── figure2_domain_adaptation.png
│   ├── figure3_attack_family_generalization.png
│   ├── figure4_distribution_shift.png
│   ├── figure5_ddos_confusion_matrix.png
│   ├── figure6_feature_importance.png
│   ├── figure7_false_negative_features.png
│   ├── figure8_threshold_analysis.png
│   ├── master_experiment_results.csv
│   └── master_experiment_results.txt
│
├── src/
│   ├── 01_explore_dataset.py
│   ├── 02_preprocess.py
│   ├── 03_train_model.py
│   ├── 04_evaluate_model.py
│   ├── 05_feature_importance.py
│   ├── 06_dataset_inventory.py
│   ├── 07_build_multiclass_dataset.py
│   ├── 08_train_multiclass.py
│   ├── 09_experiment3b_cross_day.py
│   ├── 10_experiment3c_dos_generalization.py
│   ├── 11_experiment3d_false_negative_analysis.py
│   ├── 12_experiment4a_threshold_analysis.py
│   ├── 13_experiment4b_feature_engineering.py
│   ├── 14_experiment4c_domain_adaptation.py
│   ├── 15_experiment4d_cross_day_generalization.py
│   ├── 16_experiment4e_reverse_attack_generalization.py
│   ├── 17_experiment4f_attack_generalization_matrix.py
│   ├── 18_experiment4g_distribution_shift.py
│   ├── 19_build_master_results.py
│   ├── 20_create_final_figures.py
│   ├── 21_create_final_analysis_figures.py
│   └── 22_create_master_results.py
│
├── .gitignore
├── LICENSE.txt
├── README.md
└── requirements.txt
```

---

# 6. Data Preprocessing

The preprocessing pipeline includes:

### Column normalization

Whitespace is removed from feature names to provide consistent feature handling.

### Identifier removal

The following fields are removed where applicable:

```text
Flow ID
Source IP
Source Port
Destination IP
Timestamp
```

These fields can behave as identifiers or environment-specific information rather than generalizable traffic behavior.

### Infinite-value handling

Positive and negative infinite values are converted to missing values.

### Missing-value handling

Records containing invalid or missing feature values are removed according to the requirements of each experiment.

### Duplicate handling

Duplicate network-flow records are removed where applicable.

### Label processing

For binary experiments, traffic is represented as:

```text
BENIGN → 0
ATTACK → 1
```

For multiclass experiments, the original attack categories are retained.

---

# 7. Baseline Model

The primary classifier is a Random Forest model.

Typical configuration:

```text
n_estimators = 100
random_state = 42
class_weight = balanced
n_jobs = -1
```

Where applicable, experiments use:

```text
80% → Training
20% → Testing
```

with:

```text
random_state = 42
```

Stratification is used for random holdout experiments where appropriate.

---

# 8. Baseline Experiments

The initial baseline produced near-perfect performance on the selected CIC-IDS2017 DDoS traffic under random train/test validation.

The baseline results were:

| Metric    |  Result |
| --------- | ------: |
| Accuracy  |  99.99% |
| Precision | 100.00% |
| Recall    |  99.99% |
| F1 Score  |  99.99% |
| ROC-AUC   | 100.00% |

This result demonstrates that the Random Forest can separate the selected benchmark traffic extremely well under conventional random validation.

However, the later experiments demonstrate why this result alone is insufficient to establish real-world NIDS effectiveness.

---

# 9. Experiment 3A: Multiclass Classification

Experiment 3A expanded the evaluation from binary DDoS detection to multiclass classification.

The experiment used 10 traffic classes with a maximum sampling limit of 100,000 records per class.

Classes:

```text
BENIGN
Bot
DDoS
DoS GoldenEye
DoS Hulk
DoS Slowhttptest
DoS slowloris
FTP-Patator
PortScan
SSH-Patator
```

The data was divided using an 80/20 stratified split with random state 42.

### Results

| Metric    | Result |
| --------- | -----: |
| Accuracy  | 99.90% |
| Precision | 99.90% |
| Recall    | 99.90% |
| F1 Score  | 99.90% |

The multiclass model achieved very strong performance under random stratified validation.

However, subsequent experiments demonstrate that randomly mixed benchmark validation can substantially overestimate generalization performance.

---

# 10. Experiment 3B: Cross-Day Generalization

Experiment 3B tested whether the model trained on one group of capture days could generalize to traffic from another day.

### Results

| Metric    | Result |
| --------- | -----: |
| Accuracy  | 63.98% |
| Precision | 99.52% |
| Recall    | 28.10% |
| F1 Score  | 43.82% |

The dramatic reduction in recall is important.

The model maintained extremely high precision but failed to detect a large proportion of attacks in the target distribution.

This demonstrates a major weakness of random train/test evaluation: **a model can perform almost perfectly on a benchmark split while failing to generalize across traffic distributions.**

---

# 11. Experiment 3C: DDoS Generalization

Experiment 3C focused specifically on DDoS generalization.

### Results

| Metric    | Result |
| --------- | -----: |
| Accuracy  | 81.49% |
| Precision | 99.86% |
| Recall    | 63.51% |
| F1 Score  | 77.64% |

The result shows that precision remained extremely high while recall fell substantially.

Therefore, the principal problem was not false alarms but **missed attacks**.

---

# 12. Experiment 3D: DDoS False-Negative Analysis

Experiment 3D investigated why the model failed to detect a significant portion of DDoS traffic.

Out of:

```text
100,000 DDoS attacks
```

the model detected:

```text
63,507
```

and missed:

```text
36,493
```

Result:

```text
Attack recall     = 63.51%
False-negative rate = 36.49%
```

The analysis identified substantial differences between detected and missed attacks.

Important differences included:

```text
URG Flag Count
Min Packet Length
Fwd Packet Length Min
Flow IAT Min
Bwd Packet Length Min
Fwd IAT Min
Fwd Packets/s
Flow Packets/s
Bwd Packets/s
ACK Flag Count
```

This suggests that the missed attacks occupy a substantially different region of feature space from the attacks that the model detects successfully.

---

# 13. Experiment 4A: Classification Threshold Analysis

Experiment 4A examined whether changing the Random Forest decision threshold could improve attack detection.

Selected results:

| Threshold | Accuracy | Precision | Recall |     F1 |
| --------: | -------: | --------: | -----: | -----: |
|      0.90 |   49.40% |     0.00% |  0.00% |  0.00% |
|      0.80 |   50.36% |    98.15% |  1.91% |  3.75% |
|      0.70 |   65.78% |    99.85% | 32.41% | 48.93% |
|      0.60 |   81.50% |    99.91% | 63.49% | 77.64% |
|      0.50 |   81.49% |    99.86% | 63.51% | 77.64% |
|      0.40 |   81.49% |    99.81% | 63.53% | 77.64% |
|      0.30 |   81.46% |    99.70% | 63.55% | 77.62% |
|      0.20 |   80.93% |    97.93% | 63.64% | 77.15% |
|      0.10 |   78.10% |    89.93% | 63.86% | 74.68% |

The best F1 score occurred around a threshold of:

```text
0.60
```

with:

```text
Accuracy  = 81.50%
Precision = 99.91%
Recall    = 63.49%
F1        = 77.64%
```

The experiment shows that threshold adjustment alone does not solve the underlying generalization problem.

---

# 14. Experiment 4B: Feature Engineering

Experiment 4B introduced engineered traffic-flow features intended to capture relationships between packet counts, bytes, durations, and traffic direction.

Examples include:

```text
Bwd_Packets_Per_Duration
Fwd_Bwd_Packet_Length_Ratio
Log_Bwd_Packets_per_s
Fwd_Byte_Proportion
Bwd_Bytes_Per_Packet
Fwd_Bwd_Byte_Ratio
Bwd_Fwd_Byte_Ratio
Fwd_Bytes_Per_Duration
Bwd_Byte_Proportion
```

### Results

| Metric    | Result |
| --------- | -----: |
| Accuracy  | 81.47% |
| Precision | 99.86% |
| Recall    | 63.46% |
| F1 Score  | 77.60% |

Feature engineering produced only a negligible change in overall detection performance.

This indicates that the main limitation was not simply a lack of derived features. The problem is more strongly associated with **distributional differences between training and testing traffic**.

---

# 15. Experiment 4C: Target-Domain Exposure

Experiment 4C investigated whether exposing the training process to a portion of the target-domain DDoS traffic could improve generalization.

| Target-Domain Exposure | Training Samples | Accuracy | Precision | Recall |     F1 |
| ---------------------: | ---------------: | -------: | --------: | -----: | -----: |
|                     0% |          200,000 |   79.27% |    99.87% | 63.52% | 77.65% |
|                     5% |          203,200 |   99.85% |    99.92% | 99.81% | 99.86% |
|                    10% |          206,401 |   99.88% |    99.92% | 99.86% | 99.89% |
|                    25% |          216,003 |   99.91% |    99.92% | 99.93% | 99.92% |
|                    50% |          232,007 |   99.92% |    99.92% | 99.95% | 99.93% |

The sharp improvement after even limited target-domain exposure provides strong evidence that the target traffic distribution differs substantially from the original training distribution.

This result should not be interpreted as ordinary unseen-domain generalization because the target-domain data is deliberately introduced into training.

Instead, it demonstrates the potential value of **domain adaptation or limited target-domain calibration**.

---

# 16. Experiment 4D: Unseen Attack-Family Testing

Experiment 4D tested the model against an attack family not represented in the training data.

### Results

| Metric    | Result |
| --------- | -----: |
| Accuracy  | 87.10% |
| Precision |  0.35% |
| Recall    |  0.02% |
| F1 Score  |  0.04% |

Confusion statistics:

```text
False Positives = 853
False Negatives = 13,832
True Positives  = 3
True Negatives  = 99,147
```

Although accuracy appears relatively high, the attack detection performance is extremely poor.

Only:

```text
3 / 13,835
```

attacks were detected.

This is a critical finding because it demonstrates why accuracy can be misleading in imbalanced intrusion detection problems.

The model essentially classified the unseen attack family as benign.

---

# 17. Experiment 4E: Cross-Family DDoS Testing

Experiment 4E evaluated cross-family generalization using a DDoS-focused test configuration.

### Results

| Metric    | Result |
| --------- | -----: |
| Accuracy  | 81.43% |
| Precision | 99.89% |
| Recall    | 63.35% |
| F1 Score  | 77.53% |

Confusion statistics:

```text
False Positives = 69
False Negatives = 36,651
True Positives  = 63,349
True Negatives  = 97,649
```

The result is consistent with the earlier DDoS generalization experiments.

The model produces very few false positives but misses a large number of attacks.

---

# 18. Experiment 4F: Attack-Family Generalization Matrix

Experiment 4F evaluated whether a model trained on one attack family could recognize traffic from another family.

The results show strong asymmetry and poor generalization in several train/test family combinations.

| Train Family | Test Family | Accuracy | Precision | Recall |     F1 |
| ------------ | ----------- | -------: | --------: | -----: | -----: |
| DDoS         | DoS         |   49.99% |    38.46% |  0.04% |  0.08% |
| DDoS         | Patator     |   78.33% |     0.00% |  0.00% |  0.00% |
| DoS          | DDoS        |   81.69% |    99.84% | 63.48% | 77.61% |
| DoS          | Patator     |   77.47% |     0.55% |  0.02% |  0.04% |
| Patator      | DDoS        |   50.00% |     0.00% |  0.00% |  0.00% |
| Patator      | DoS         |   50.01% |    81.82% |  0.04% |  0.07% |

These results demonstrate that high performance within one attack family does not imply reliable transfer to another attack family.

---

# 19. Experiment 4G: Distribution Shift Analysis

Experiment 4G statistically examined the differences between attack-family feature distributions.

The Kolmogorov-Smirnov statistic was used to measure distribution differences.

### Mean KS statistics

| Attack Families | Mean KS |
| --------------- | ------: |
| DDoS vs Patator |   0.487 |
| DoS vs DDoS     |   0.435 |
| DoS vs Patator  |   0.522 |

The largest observed feature-level distribution shift included:

```text
DoS vs Patator
Bwd Packets/s
KS = 0.797790
```

Other substantial differences included packet size, packet timing, packet rate, and byte-volume features.

These findings provide statistical evidence supporting the poor cross-family generalization observed in Experiment 4F.

---

# 20. Master Experimental Findings

The complete experimental sequence reveals a clear pattern.

| Experiment | Purpose                   | Accuracy | Precision | Recall |     F1 |
| ---------- | ------------------------- | -------: | --------: | -----: | -----: |
| 3A         | Multiclass classification |   99.90% |    99.90% | 99.90% | 99.90% |
| 3B         | Cross-day generalization  |   63.98% |    99.52% | 28.10% | 43.82% |
| 3C         | DDoS generalization       |   81.49% |    99.86% | 63.51% | 77.64% |
| 4A         | Threshold analysis        |   81.50% |    99.91% | 63.49% | 77.64% |
| 4B         | Feature engineering       |   81.47% |    99.86% | 63.46% | 77.60% |
| 4C         | Target-domain exposure    |   99.92% |    99.92% | 99.95% | 99.93% |
| 4D         | Unseen attack family      |   87.10% |     0.35% |  0.02% |  0.04% |
| 4E         | Cross-family DDoS         |   81.43% |    99.89% | 63.35% | 77.53% |

The table should be interpreted together with Experiments 4F and 4G, which provide additional evidence about attack-family transfer and feature distribution shift.

---

# 21. Main Research Findings

The experiments support several important findings.

### Finding 1: Random holdout performance can be misleading

The model achieved approximately 99.9% performance in conventional random validation.

However, cross-day evaluation reduced recall to 28.1%.

Therefore, random train/test performance should not be treated as evidence of real-world generalization.

### Finding 2: False negatives are the dominant problem under distribution shift

Across the DDoS generalization experiments, precision remained close to 100%, while recall remained around 63%.

The model therefore rarely generated false alarms but failed to detect a substantial proportion of attacks.

### Finding 3: Feature engineering alone did not solve generalization

Experiment 4B produced results almost identical to the original DDoS generalization experiment.

Adding derived packet and flow features therefore did not address the core problem.

### Finding 4: Limited target-domain exposure dramatically improves performance

Experiment 4C showed that exposing the model to a small amount of target-domain traffic substantially increased performance.

This suggests that domain adaptation or target-domain calibration may be an important strategy for practical deployment.

### Finding 5: Unseen attack families are extremely difficult

The unseen attack-family experiment produced:

```text
Recall = 0.02%
F1     = 0.04%
```

This demonstrates that the model does not automatically learn a general concept of maliciousness simply because it has been trained on other attack families.

### Finding 6: Attack families occupy different feature distributions

Experiment 4G identified substantial statistical differences between attack families.

This provides evidence that distribution shift is a major factor behind the poor cross-family generalization observed in the experiments.

---

# 22. Research Questions and Answers

## RQ1

**How accurately can supervised machine learning distinguish benign network traffic from malicious traffic using flow-level characteristics?**

Under random holdout validation, the Random Forest achieved near-perfect performance, including approximately 99.9% accuracy in the multiclass experiment.

However, this performance decreased substantially under more realistic generalization scenarios.

## RQ2

**Which network-flow features contribute most strongly to intrusion classification?**

Important features included packet lengths, packet rates, flow timing, TCP characteristics, byte volumes, and forward/backward traffic relationships.

Feature importance was not identical across experiments, reinforcing the importance of traffic distribution.

## RQ3

**Does removing or engineering features significantly affect detection performance?**

Removing Destination Port had little effect on the original benchmark performance.

Likewise, engineered features produced only a marginal change during DDoS generalization.

## RQ4

**Does a model trained on one network-traffic distribution generalize to different conditions?**

The experiments indicate that generalization is limited.

Cross-day and cross-family testing produced substantially lower recall than random validation.

## RQ5

**Can the model detect attack types that were not represented during training?**

The unseen attack-family experiment indicates that the model performs extremely poorly when confronted with an attack family absent from training.

---

# 23. Practical Security Implications

The findings have direct implications for machine learning based IDS development.

A model that achieves 99%+ benchmark accuracy may still be unsuitable for deployment if its evaluation does not account for:

```text
Temporal distribution shift
Attack-family differences
Network-environment differences
Unseen attacks
False negatives
```

For an intrusion detection system, missed attacks can be considerably more important than a small number of false alarms.

Consequently, evaluation should include:

```text
Random holdout validation
Cross-day validation
Cross-family validation
Unseen attack testing
False-negative analysis
Distribution-shift analysis
```

rather than relying solely on overall accuracy.

---

# 24. Limitations

Several limitations should be considered.

### Dataset limitation

The study relies on CIC-IDS2017, a benchmark dataset generated under controlled experimental conditions.

Its traffic characteristics may not represent modern production networks.

### Dataset-specific behavior

The very high random-split performance may partly reflect dataset-specific characteristics shared between training and testing records.

### Limited model comparison

The study primarily focuses on Random Forest rather than comparing a large range of machine learning architectures.

### Unseen attacks

The unseen attack-family experiment demonstrates a severe limitation of conventional supervised learning, but broader evaluation across additional datasets would be necessary to generalize this conclusion.

### No live deployment

The current system operates on extracted network-flow records rather than directly processing live network traffic.

### Generalization remains dataset dependent

The conclusions should be understood as evidence from the experimental setup rather than universal claims about all machine learning IDS systems.

---

# 25. Future Work

Future research can extend the study in several directions.

### Cross-dataset validation

Evaluate the trained models using independent intrusion-detection datasets rather than only CIC-IDS2017.

### Domain adaptation

Investigate methods that adapt a trained IDS to a new network environment using limited target-domain data.

### Online learning

Study whether the model can update itself as network traffic changes over time.

### Novel attack detection

Combine supervised classification with anomaly detection or open-set recognition to identify attacks absent from the training set.

### Model comparison

Compare Random Forest against algorithms such as:

```text
Logistic Regression
XGBoost
MLP
Gradient Boosting
Support Vector Machine
Deep Neural Networks
```

### Explainability

Investigate SHAP or related explainability methods to understand individual predictions and false negatives.

### Real-time deployment

Integrate the trained model with a network-flow extraction system for near-real-time intrusion detection.

### Multi-dataset evaluation

Test whether conclusions remain consistent across CIC-IDS2017 and other publicly available datasets.

---

# 26. Reproducibility

The repository contains the preprocessing, training, evaluation, analysis, and figure-generation scripts used throughout the study.

The main workflow includes:

```bash
python src/01_explore_dataset.py
python src/02_preprocess.py
python src/03_train_model.py
python src/04_evaluate_model.py
python src/05_feature_importance.py
```

The later experiments can be executed using the corresponding scripts under:

```text
src/
```

For example:

```bash
python src/07_build_multiclass_dataset.py
python src/08_train_multiclass.py
python src/09_experiment3b_cross_day.py
python src/10_experiment3c_dos_generalization.py
python src/11_experiment3d_false_negative_analysis.py
python src/12_experiment4a_threshold_analysis.py
python src/13_experiment4b_feature_engineering.py
python src/14_experiment4c_domain_adaptation.py
python src/15_experiment4d_cross_day_generalization.py
python src/16_experiment4e_reverse_attack_generalization.py
python src/17_experiment4f_attack_generalization_matrix.py
python src/18_experiment4g_distribution_shift.py
```

Final result consolidation and visualization scripts are also included.

Random state:

```text
42
```

Raw datasets and trained model binaries are excluded from version control through `.gitignore`.

---

# 27. Research Outputs

The repository includes the final research paper in:

```text
paper/
```

Available formats:

```text
Evaluating Machine Learning-Based Network Intrusion Detection.pdf

Evaluating NIDS Generalization Under Distribution Shift.docx
```

The `results/` directory contains:

* Experimental result files
* Confusion matrices
* Feature importance tables
* False-negative analysis
* Distribution-shift analysis
* Threshold analysis
* Attack-family generalization results
* Final research figures
* Master experiment results

---

# 28. Final Research Figures

The final analysis includes eight figures:

```text
Figure 1 → Generalization performance
Figure 2 → Domain adaptation
Figure 3 → Attack-family generalization
Figure 4 → Distribution shift
Figure 5 → DDoS confusion matrix
Figure 6 → Feature importance
Figure 7 → False-negative feature analysis
Figure 8 → Classification threshold analysis
```

These figures are available under:

```text
results/
```

---

# 29. Conclusion

This project began with a conventional question:

> Can a machine learning model accurately detect malicious network traffic?

The initial experiments produced an apparently simple answer: yes.

The deeper experiments produced a more important conclusion.

A Random Forest can achieve near-perfect performance when training and testing traffic are randomly mixed. However, when the evaluation separates traffic by day, attack family, or distribution, performance can deteriorate dramatically, particularly in attack recall.

The experiments show that:

```text
High benchmark accuracy
        ≠
Reliable generalization
```

The most significant weakness observed was false negatives under distribution shift.

The model frequently maintained very high precision while failing to detect a substantial proportion of attacks. Unseen attack-family testing was even more challenging, with almost all previously unseen attacks being classified as benign.

At the same time, the target-domain exposure experiment showed that relatively limited exposure to the target distribution can dramatically improve performance. This points toward domain adaptation and continual calibration as promising directions for practical IDS development.

The central conclusion of this research is therefore:

> **Evaluating an ML-based IDS requires more than measuring accuracy on a random holdout set. Generalization across time, attack families, and traffic distributions must be treated as a central part of IDS evaluation.**

---

# 30. Author

Developed as a cybersecurity and machine learning research project investigating practical applications of supervised learning for network intrusion detection and the challenges of model generalization.

---

# 31. License

See:

```text
LICENSE.txt
```

---

# 32. Disclaimer

This project is intended for educational, research, and defensive cybersecurity purposes.

The reported results are specific to the dataset, preprocessing pipeline, model configuration, and experimental methodology used in this study.

High performance on a benchmark dataset should not be interpreted as proof of equivalent performance in production networks.

---

## Project Status

```text
[x] Dataset exploration
[x] Data preprocessing
[x] Binary DDoS classification
[x] Random Forest baseline
[x] Model evaluation
[x] Confusion matrix analysis
[x] Feature importance analysis
[x] Destination-port analysis

[x] Multiclass classification
[x] Cross-day generalization
[x] DDoS generalization
[x] DDoS false-negative analysis
[x] Classification threshold analysis
[x] Feature engineering
[x] Target-domain exposure
[x] Unseen attack-family testing
[x] Cross-family attack generalization
[x] Distribution-shift analysis

[x] Final research figures
[x] Master experiment results
[x] Final research paper
[x] PDF research paper
[x] DOCX research paper
[x] Repository documentation
```

---

## Core Research Message

```text
Benchmark Accuracy
        ↓
Generalization Testing
        ↓
False-Negative Analysis
        ↓
Attack-Family Testing
        ↓
Distribution-Shift Analysis
        ↓
Domain Adaptation
        ↓
More Realistic IDS Evaluation
```

The project therefore focuses not simply on building a highly accurate classifier, but on understanding **when, why, and where an ML-based intrusion detection model fails to generalize**.
