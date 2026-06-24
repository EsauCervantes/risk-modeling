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
└── README.md
```

The raw Give Me Some Credit files are not committed to the repository. They can
be downloaded from Kaggle:

https://www.kaggle.com/c/GiveMeSomeCredit

## Current Status

The repository currently contains early supervised modeling notebooks and Python
modules. The next development step is to clean the XGBoost implementation, add a
regularized logistic regression baseline, and organize the notebooks around a
clear model comparison workflow.

## Future Extensions

Possible later extensions include scorecard-style reporting, Tableau dashboard
exports, stress testing, and portfolio-level risk summaries. Research-oriented
methods such as Restricted Boltzmann Machines may be revisited later, but they
are outside the scope of the first practical version.
