"""Lightweight checks for the credit risk case study workflow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluate import (  # noqa: E402
    compute_binary_classification_metrics,
    make_calibration_table,
    make_decile_table,
)
from load_data import load_training_data  # noqa: E402
from logistic_model import LogisticPDModel  # noqa: E402
from xgboost_model import XGBoostPDModel  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lightweight workflow checks.")
    parser.add_argument(
        "--config",
        default=PROJECT_ROOT / "mlops" / "configs" / "model_config.json",
        type=Path,
        help="Path to the JSON config file.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    path = config_path if config_path.is_absolute() else PROJECT_ROOT / config_path
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def run_metric_smoke_checks() -> None:
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_score = np.array([0.02, 0.15, 0.82, 0.65, 0.30, 0.91])

    metrics = compute_binary_classification_metrics(y_true, y_score)
    deciles = make_decile_table(y_true, y_score, n_bins=3)
    calibration = make_calibration_table(y_true, y_score, n_bins=3)

    expected_metrics = {
        "roc_auc",
        "pr_auc",
        "log_loss",
        "brier_score",
        "default_rate",
        "mean_predicted_pd",
        "n_samples",
        "positive_count",
    }
    missing_metrics = expected_metrics - set(metrics)
    if missing_metrics:
        raise AssertionError(f"Missing metric keys: {sorted(missing_metrics)}")

    if deciles.empty or calibration.empty:
        raise AssertionError("Decile and calibration smoke tables must not be empty.")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    run_metric_smoke_checks()

    logistic_model = LogisticPDModel(**config.get("logistic_regression", {}))
    xgboost_model = XGBoostPDModel(**config.get("xgboost", {}))

    data_path = project_path(config["data"]["training_path"])
    data_status = "available"
    if data_path.exists():
        load_training_data(data_path)
    else:
        data_status = "missing; training will require the Kaggle CSV locally"

    print("Checks passed:")
    print("- Config can be loaded")
    print("- Existing src modules can be imported")
    print("- Metric/table utilities pass smoke checks")
    print(f"- Logistic model configured with penalty={logistic_model.penalty}")
    print(f"- XGBoost model configured with device={xgboost_model.device}")
    print(f"- Training data status: {data_status}")


if __name__ == "__main__":
    main()
