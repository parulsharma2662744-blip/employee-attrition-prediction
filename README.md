# Employee Attrition Prediction

<p align="center">
  <strong>Machine Learning • HR Analytics • Predictive Dashboard</strong><br>
  A machine-learning project that estimates employee attrition risk and presents predictions through an interactive Streamlit dashboard.
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white">
  <img alt="Scikit-learn" src="https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikitlearn&logoColor=white">
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white">
  <img alt="Status" src="https://img.shields.io/badge/Project-Academic%20Project-informational">
</p>

## Overview

Employee attrition can affect workforce planning, recruitment costs, and organizational continuity. This project uses historical HR data to train and compare classification models, then predicts whether new employee records may be associated with attrition.

The application accepts a CSV file, generates an attrition prediction and probability estimate for each employee, summarizes the results with charts, and allows the results to be downloaded.

> **Important:** Predictions are statistical estimates, not certainties or judgments about individual employees. They should not be used as the sole basis for employment decisions.

## Project objectives

- Explore and prepare employee HR data for machine learning.
- Train and compare multiple classification algorithms.
- Select a model using Average Precision as the comparison metric.
- Predict attrition for new employee records.
- Present results in an interactive dashboard.
- Communicate model outputs with summary metrics and visualizations.

## Models compared

The project compares four approaches:

1. Decision Tree
2. Random Forest
3. Gradient Boosting
4. Stacking Classifier

### Model comparison

Results recorded during the current project run:

| Model | Accuracy | Precision | Recall | F1-score | Average Precision |
|---|---:|---:|---:|---:|---:|
| Decision Tree | 0.7857 | 0.3788 | 0.5319 | 0.4425 | 0.3282 |
| Random Forest | 0.8435 | 0.5385 | 0.1489 | 0.2333 | 0.4578 |
| Gradient Boosting | 0.8469 | 0.6000 | 0.1277 | 0.2105 | 0.4565 |
| Stacking Classifier | 0.8401 | 0.5000 | 0.1702 | 0.2540 | **0.4674** |

**Best model in this run:** Stacking Classifier, based on the highest Average Precision (0.4674).

Metrics depend on the dataset, preprocessing, train/test split, and model settings. In particular, attrition is often an imbalanced classification problem, so accuracy alone does not describe model performance.

## Interactive dashboard

The Streamlit dashboard includes:

- CSV upload for new employee records
- Predicted attrition / non-attrition labels
- Estimated attrition probabilities
- Summary cards for total employees, predicted to leave, and predicted to stay
- A chart comparing predicted classes
- A probability distribution chart
- Downloadable prediction results

## Project workflow

```text
HR Dataset (CSV)
      |
      v
Data preparation and preprocessing
      |
      v
Train and compare classification models
      |
      v
Save selected model
      |
      v
Upload new employee data
      |
      v
Predict attrition and estimate probabilities
      |
      v
Interactive Streamlit dashboard + downloadable results
```

## Repository structure

```text
employee-attrition-prediction/
├── app.py                       # Streamlit dashboard
├── train.py                     # Model training and comparison
├── predict.py                   # Batch predictions from a CSV
├── create_sample.py             # Creates a sample input CSV
├── requirements.txt             # Python dependencies
├── models/
│   └── best_attrition_model.pkl # Saved trained model
├── outputs/
│   └── model_comparison.csv      # Model evaluation results
├── .gitignore
└── README.md
```

The raw dataset, generated employee input file, prediction output, local environments, and backup script are intentionally excluded from version control.

## Run locally

### 1. Clone the repository

```bash
git clone https://github.com/parulsharma2662744-blip/employee-attrition-prediction.git
cd employee-attrition-prediction
```

### 2. Create and activate a virtual environment (recommended)

**Windows PowerShell**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the dashboard

```bash
python -m streamlit run app.py
```

Streamlit will provide a local address in the terminal (usually http://localhost:8501). Open it in your browser and upload a CSV containing the required employee input columns.

### Train or refresh the model

To train the model again, place the appropriately licensed HR dataset in the project root using the filename expected by `train.py` (currently `dataset.csv`), then run:

```bash
python train.py
```

To create a sample prediction input and run batch predictions:

```bash
python create_sample.py
python predict.py
```

The prediction script expects `new_employees.csv` in the project root.

## Data and responsible use

The dataset is not included in this repository. Obtain the data from its original source and review its terms of use before downloading or redistributing it. Do not upload real employee personal data, confidential HR records, or identifiable information to a public repository or an unapproved service.

Use this project for learning and demonstration. A model's estimated probability is not proof that an employee will leave. Human review, fairness assessment, privacy safeguards, and validation on appropriate data are essential before any real-world use.

## Future improvements

- Add model explainability and feature-importance visualizations.
- Evaluate precision-recall curves and calibration.
- Validate performance with cross-validation and a separate holdout set.
- Add fairness checks across relevant groups where legally and ethically appropriate.
- Deploy the dashboard so it can be accessed through a public demo link.

## Author

**Parul Sharma**

Academic machine-learning project focused on employee attrition prediction and interactive HR analytics.
