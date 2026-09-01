# Trained Models

Trained model files are intentionally not included in the GitHub repository because of their size.

The experiment scripts generate the required models automatically.

The main models produced during the experiments include:

* Experiment 3A multiclass Random Forest model
* Experiment 3B cross-day generalization model
* Experiment 3C DDoS generalization model
* Experiment 4B feature-engineered model
* Experiment 4B feature-engineered Random Forest model

Generated model files use formats such as `.joblib`.

To reproduce the models, install the dependencies listed in `requirements.txt`, download the CIC-IDS2017 dataset, place the CSV files in `data/`, and execute the corresponding experiment scripts under `src/`.
