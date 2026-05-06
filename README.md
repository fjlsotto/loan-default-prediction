# Loan Default Prediction

Predicting the probability of loan default to enable proactive credit risk management. Built using gradient-boosted classification models with feature engineering and threshold optimisation tuned for business-specific recall targets.

## Business Problem

Credit institutions need to identify high-risk loan applicants before disbursement. Missing a true default (false negative) is far more costly than incorrectly flagging a good applicant. This project builds a model that maximises recall on the default class while maintaining acceptable precision — enabling risk teams to intervene early.

## Dataset

**Source:** [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data) (Kaggle)

Download and place the following files in the `data/` directory:
- `application_train.csv`
- `application_test.csv`

```bash
# Using Kaggle CLI
kaggle competitions download -c home-credit-default-risk -f application_train.csv
kaggle competitions download -c home-credit-default-risk -f application_test.csv
```

## Project Structure

```
loan-default-prediction/
├── data/                        # Raw data (gitignored)
├── notebooks/
│   ├── 01_eda.ipynb             # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_recommendations.ipynb
├── src/
│   ├── features.py              # Feature engineering functions
│   └── evaluate.py              # Model evaluation utilities
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
jupyter notebook
```

## Key Findings

> Populated after modeling is complete.

## Results

| Model | ROC-AUC | Recall (Default) | Precision (Default) |
|---|---|---|---|
| Logistic Regression (baseline) | — | — | — |
| XGBoost | — | — | — |
| XGBoost (tuned threshold) | — | — | — |

## Tech Stack

Python · pandas · scikit-learn · XGBoost · LightGBM · SHAP · Matplotlib · Seaborn
