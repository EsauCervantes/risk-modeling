"""Command-line evaluation workflow for saved credit risk predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib


matplotlib.use("Agg")


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd  # noqa: E402
from evaluate import make_calibration_table, make_decile_table, save_table  # noqa: E402
from plots import (  # noqa: E402
    plot_calibration_curve,
    plot_decile_default_rate,
    plot_logistic_coefficients,
    plot_pd_distribution,
    plot_precision_recall_curve,
    plot_roc_curve,
    save_figure,
)


MODEL_SCORE_COLUMNS = {
    "Logistic Regression": "logistic_l2_clipped_pd",
    "XGBoost": "xgboost_pd",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved credit risk PD scores.")
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


def slugify_model_name(model_name: str) -> str:
    return model_name.lower().replace(" ", "_")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    tables_dir = project_path(config["outputs"]["tables_dir"])
    figures_dir = project_path(config["outputs"]["figures_dir"])
    results_dir = PROJECT_ROOT / "results"
    predictions_path = tables_dir / "validation_predictions.csv"
    coefficients_path = tables_dir / "logistic_coefficients.csv"

    if not predictions_path.exists():
        raise FileNotFoundError(
            "Validation predictions are missing. Run "
            "`python mlops/scripts/train.py --config mlops/configs/model_config.json` "
            "first."
        )
    if not coefficients_path.exists():
        raise FileNotFoundError(
            "Logistic coefficients are missing. Run "
            "`python mlops/scripts/train.py --config mlops/configs/model_config.json` "
            "first; it writes reports/tables/logistic_coefficients.csv from the "
            "fitted logistic pipeline."
        )

    predictions = pd.read_csv(predictions_path)
    logistic_coefficients = pd.read_csv(coefficients_path)
    missing_columns = {"y_true", *MODEL_SCORE_COLUMNS.values()} - set(predictions.columns)
    if missing_columns:
        raise ValueError(
            f"{predictions_path} is missing required columns: {sorted(missing_columns)}"
        )

    y_true = predictions["y_true"]
    model_scores = {
        model_name: predictions[column]
        for model_name, column in MODEL_SCORE_COLUMNS.items()
    }

    calibration_tables = {
        model_name: make_calibration_table(y_true, scores, n_bins=10)
        for model_name, scores in model_scores.items()
    }
    decile_tables = {
        model_name: make_decile_table(y_true, scores, n_bins=10)
        for model_name, scores in model_scores.items()
    }

    saved_paths = []
    for model_name, table in calibration_tables.items():
        saved_paths.append(
            save_table(
                table,
                tables_dir / f"calibration_{slugify_model_name(model_name)}.csv",
            )
        )

    for model_name, table in decile_tables.items():
        saved_paths.append(
            save_table(
                table,
                tables_dir / f"deciles_{slugify_model_name(model_name)}.csv",
            )
        )

    fig, _ = plot_roc_curve(y_true, model_scores)
    saved_paths.append(save_figure(fig, figures_dir / "roc_auc_curve.png"))

    fig, _ = plot_precision_recall_curve(y_true, model_scores)
    saved_paths.append(save_figure(fig, figures_dir / "pr_auc_curve.png"))

    fig, _ = plot_calibration_curve(calibration_tables, log_scale=True)
    saved_paths.append(save_figure(fig, figures_dir / "calibration_curve.png"))

    fig, _ = plot_pd_distribution(model_scores)
    saved_paths.append(save_figure(fig, figures_dir / "pd_distribution.png"))

    fig, _ = plot_pd_distribution(model_scores, log_x=True)
    saved_paths.append(save_figure(fig, figures_dir / "pd_distribution_log_scale.png"))

    fig, _ = plot_logistic_coefficients(logistic_coefficients, top_n=12)
    saved_paths.append(
        save_figure(
            fig,
            results_dir / "logistic_regression_interpretability.png",
            dpi=180,
        )
    )

    for model_name, table in decile_tables.items():
        fig, _ = plot_decile_default_rate(table, model_name=model_name)
        saved_paths.append(
            save_figure(
                fig,
                figures_dir / f"decile_default_rate_{slugify_model_name(model_name)}.png",
            )
        )

    print("Saved evaluation artifacts:")
    for path in saved_paths:
        print(f"- {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
