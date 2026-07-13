# Lightweight MLOps Workflow

This folder adds a small MLOps-style layer around the credit risk case study. It is intentionally modest: the notebooks remain the main explanatory artifact, while these scripts make the main train/evaluate steps easier to rerun from the command line.

This is not a production deployment. It is a reproducible workflow for an applied ML portfolio project.

## Structure

- `configs/model_config.json`: data paths, train/validation split, model settings, and output folders.
- `scripts/train.py`: trains logistic regression and XGBoost, then writes validation predictions and metrics.
- `scripts/evaluate.py`: reads saved validation predictions and generates evaluation tables and figures.
- `scripts/run_checks.py`: runs lightweight import/config/metric checks before training.

## Commands

From the project root:

```bash
python mlops/scripts/run_checks.py --config mlops/configs/model_config.json
python mlops/scripts/train.py --config mlops/configs/model_config.json
python mlops/scripts/evaluate.py --config mlops/configs/model_config.json
```

## Data

The raw Kaggle files are not committed. The default config expects:

```text
data/GiveMeSomeCredit/cs-training.csv
```

## Outputs

The scripts write reusable artifacts to:

- `reports/tables/`
- `reports/figures/`
- `reports/monitoring/` when monitoring artifacts are added later

## CPU and GPU

XGBoost runs on CPU by default for portability. If CUDA is available, the config can be changed manually:

```json
"device": "cuda"
```

## Possible Next Steps

- Add a small monitoring report for feature drift and predicted PD drift.
- Add simple model/version metadata.
- Add a CI check that runs `run_checks.py`.
- Add a model card update step after each training run.
