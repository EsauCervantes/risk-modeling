# Credit Risk Modeling

This project builds a small, reproducible credit risk modeling workflow using the
Give Me Some Credit dataset. The goal is to estimate borrower probability of
default and compare an interpretable baseline model against a stronger tabular
machine learning benchmark.

The project is intentionally practical rather than research-level. It is meant
to show a clean modeling process, sensible evaluation metrics, and plots that
are relevant for credit risk model development.

## Models

- **Logistic regression with regularization** as the interpretable benchmark.
  This model is useful because the fitted coefficients can be inspected and
  explained in terms of borrower risk drivers.
- **XGBoost** as the nonlinear benchmark for tabular predictive performance.
  This model is included to test whether a more flexible algorithm improves
  default ranking and classification quality.

## Evaluation

The planned model comparison focuses on out-of-sample performance and credit
risk interpretability:

- ROC curve and ROC-AUC
- Precision-recall curve and PR-AUC
- Predicted probability of default distribution
- Calibration analysis
- Default rate by predicted risk decile

## Project Structure

```text
risk-modeling/
├── data/                 # local dataset files, ignored by Git
├── notebooks/            # polished analysis notebooks
├── reports/              # exported figures and tables
├── src/
│   ├── evaluate.py
│   ├── load_data.py
│   ├── logistic_model.py
│   ├── plots.py
│   └── xgboost_model.py
├── README.md
└── requirements.txt
```

## Reproducibility

The raw Give Me Some Credit files are not committed to this repository. Download
the competition files from Kaggle and place them under:

```text
data/GiveMeSomeCredit/
├── cs-training.csv
├── cs-test.csv
└── sampleEntry.csv
```

The expected source is:

https://www.kaggle.com/c/GiveMeSomeCredit

XGBoost runs on CPU by default so the project works on a clean checkout without
GPU hardware. If a compatible NVIDIA/CUDA environment is available, GPU training
can be enabled manually by constructing the model with `device="cuda"`.

## Current Status

The repository contains an applied, reproducible workflow for data exploration,
probability-of-default model training, and model comparison. It is intended as a
portfolio case study, not as a production-ready risk system.

## Future Extensions

Possible later extensions include scorecard-style reporting, Tableau dashboard
exports, model monitoring summaries, and a small German/English GenAI reporting
layer for business-facing model commentary.
