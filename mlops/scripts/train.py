"""Command-line training workflow for the credit risk case study.

This script wraps the existing project modules without changing their public
APIs. It is meant as a lightweight reproducibility layer, not a production
training pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluate import compare_model_metrics, save_table  # noqa: E402
from load_data import (  # noqa: E402
    fit_missing_value_preprocessor,
    load_training_data,
    make_train_val_split,
    prepare_features_target,
)
from logistic_model import LogisticPDModel  # noqa: E402
from xgboost_model import XGBoostPDModel  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train credit risk PD models.")
    parser.add_argument(
        "--config",
        default=PROJECT_ROOT / "mlops" / "configs" / "model_config.json",
        type=Path,
        help="Path to the JSON config file.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    path = config_path if config_path.is_absolute() else PROJECT_ROOT / config_path
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def build_logistic_model(config: dict) -> LogisticPDModel:
    logistic_config = dict(config.get("logistic_regression", {}))
    clip_quantiles = logistic_config.get("clip_quantiles")
    if clip_quantiles is not None:
        logistic_config["clip_quantiles"] = tuple(clip_quantiles)
    return LogisticPDModel(**logistic_config)


def build_xgboost_model(config: dict) -> XGBoostPDModel:
    xgboost_config = dict(config.get("xgboost", {}))
    return XGBoostPDModel(**xgboost_config)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    tables_dir = project_path(config["outputs"]["tables_dir"])
    training_path = project_path(config["data"]["training_path"])
    split_config = config.get("split", {})

    df = load_training_data(training_path)
    X, y = prepare_features_target(df)
    X_train, X_val, y_train, y_val = make_train_val_split(
        X,
        y,
        test_size=split_config.get("test_size", 0.2),
        random_state=split_config.get("random_state", 42),
    )

    missing_preprocessor = fit_missing_value_preprocessor(X_train)
    X_train_clean = missing_preprocessor.transform(X_train)
    X_val_clean = missing_preprocessor.transform(X_val)

    logistic_model = build_logistic_model(config)
    xgboost_model = build_xgboost_model(config)

    logistic_model.fit(X_train_clean, y_train)
    xgboost_model.fit(X_train_clean, y_train)

    logistic_pd = logistic_model.predict_proba(X_val_clean)
    xgboost_pd = xgboost_model.predict_proba(X_val_clean)

    metrics = compare_model_metrics(
        {
            "logistic_l2_clipped": (y_val, logistic_pd),
            "xgboost": (y_val, xgboost_pd),
        }
    )
    predictions = pd.DataFrame(
        {
            "row_id": X_val.index,
            "y_true": y_val.to_numpy(),
            "logistic_l2_clipped_pd": logistic_pd,
            "xgboost_pd": xgboost_pd,
        }
    )
    coefficients = logistic_model.coefficients()

    saved_paths = [
        save_table(metrics, tables_dir / "model_metrics.csv"),
        save_table(predictions, tables_dir / "validation_predictions.csv"),
        save_table(coefficients, tables_dir / "logistic_coefficients.csv"),
    ]

    print("Saved training artifacts:")
    for path in saved_paths:
        print(f"- {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
