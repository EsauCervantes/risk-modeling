# Model Card: Credit Risk Probability-of-Default Model

## Intended Use

This project is a portfolio case study for estimating the probability of serious
delinquency using the public Give Me Some Credit dataset. It is intended to
demonstrate a clean applied machine learning workflow for credit-risk style
analytics. It is not intended to make real lending, insurance, pricing, or
underwriting decisions.

## Dataset

The dataset is the public Give Me Some Credit dataset. It contains tabular
borrower-level features such as income, age, debt ratio, credit utilization, and
delinquency history. The target is serious delinquency within two years.

The raw data is not committed to this repository. Users are expected to download
the data separately and place it under `data/GiveMeSomeCredit/`.

## Models

Two supervised binary classification models are compared:

- **Regularized logistic regression** as an interpretable baseline. It provides
  coefficients and odds ratios that can be reviewed as borrower risk drivers.
- **XGBoost** as a nonlinear tabular benchmark. It is included to test whether a
  more flexible model improves predictive performance.

## Evaluation Metrics

The model comparison uses metrics that are relevant for probability-of-default
modelling:

- **ROC-AUC:** ranking quality across classification thresholds.
- **PR-AUC:** ranking quality with emphasis on the minority default class.
- **Log loss:** quality of predicted probabilities.
- **Brier score:** squared error of predicted probabilities.
- **Calibration:** agreement between predicted PDs and observed default rates.
- **Risk deciles:** observed default rates across predicted-risk segments.

## Main Findings

See `reports/tables/model_metrics.csv` after running
`notebooks/02_model_training.ipynb`. This model card intentionally does not
hard-code performance numbers, so results remain tied to the reproducible
training outputs.

## Interpretability

The logistic regression model supports direct interpretation through fitted
coefficients and odds ratios per standardized feature. These outputs are saved
to `reports/tables/logistic_coefficients.csv` after running the training
notebook.

XGBoost feature importance is a useful future or optional extension. SHAP could
also be added later for more detailed local and global explanations, but it is
not included in the current lightweight case study.

## Monitoring Considerations

If this type of model were used in a real analytical workflow, monitoring would
need to include:

- input feature drift,
- predicted PD distribution drift,
- calibration drift,
- performance degradation over time,
- periodic validation on recent outcomes.

## Limitations

- The dataset is public and does not represent a current production portfolio.
- There is no external validation dataset.
- Real production monitoring is not implemented.
- Fairness analysis is not included.
- No business-specific cost matrix or risk appetite threshold is included.
- Regulatory model validation is not included.

## Relevance to Applied AI / Insurance Analytics

The workflow is relevant to risk segmentation, underwriting-style analytics,
model monitoring, and business-facing machine learning reporting. The case study
shows how interpretable and nonlinear models can be compared using probability,
ranking, calibration, and segmentation diagnostics that are understandable to
both technical and non-technical stakeholders.
