"""Create interview-ready figures from saved validation predictions.

This script reads existing prediction CSV files only. It does not train or
retrain any model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluate import make_calibration_table  # noqa: E402


EXPERIMENT_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = EXPERIMENT_DIR / "outputs"
CONVENTIONAL_PREDICTIONS_PATH = (
    PROJECT_ROOT / "reports" / "tables" / "validation_predictions.csv"
)
SELF_SUPERVISED_PREDICTIONS_PATH = OUTPUTS_DIR / "validation_predictions.csv"
SUMMARY_FIGURE_PATH = PROJECT_ROOT / "results" / "model_comparison_summary.png"
PD_DIAGNOSTIC_PATH = PROJECT_ROOT / "results" / "pd_distribution_diagnostic.png"

MODEL_SCORE_COLUMNS = {
    "Logistic regression": "logistic_l2_clipped_pd",
    "XGBoost": "xgboost_pd",
    "XGBoost + self-supervised embeddings": "xgboost_original_plus_embeddings_pd",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the final credit-risk model comparison figure."
    )
    parser.add_argument(
        "--conventional-predictions-path",
        type=Path,
        default=CONVENTIONAL_PREDICTIONS_PATH,
        help="Path to reports/tables/validation_predictions.csv.",
    )
    parser.add_argument(
        "--self-supervised-predictions-path",
        type=Path,
        default=SELF_SUPERVISED_PREDICTIONS_PATH,
        help="Path to self-supervised validation_predictions.csv.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=SUMMARY_FIGURE_PATH,
        help="Path where the three-model summary figure should be saved.",
    )
    parser.add_argument(
        "--pd-diagnostic-path",
        type=Path,
        default=PD_DIAGNOSTIC_PATH,
        help="Path where the supplementary PD-distribution diagnostic is saved.",
    )
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def _read_predictions(path: Path, required_columns: set[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    predictions = pd.read_csv(path)
    missing_columns = required_columns - set(predictions.columns)
    if missing_columns:
        raise ValueError(f"{path} is missing columns: {sorted(missing_columns)}")
    if predictions["row_id"].duplicated().any():
        raise ValueError(f"{path} contains duplicate row_id values.")

    return predictions


def load_merged_predictions(
    conventional_predictions_path: str | Path = CONVENTIONAL_PREDICTIONS_PATH,
    self_supervised_predictions_path: str | Path = SELF_SUPERVISED_PREDICTIONS_PATH,
) -> pd.DataFrame:
    """Merge saved validation predictions and verify target consistency."""
    conventional_path = Path(conventional_predictions_path)
    self_supervised_path = Path(self_supervised_predictions_path)

    conventional = _read_predictions(
        conventional_path,
        {"row_id", "y_true", "logistic_l2_clipped_pd", "xgboost_pd"},
    )
    self_supervised = _read_predictions(
        self_supervised_path,
        {"row_id", "y_true", "xgboost_original_plus_embeddings_pd"},
    )

    merged = conventional.merge(
        self_supervised,
        on="row_id",
        how="inner",
        suffixes=("_conventional", "_self_supervised"),
        validate="one_to_one",
    )
    if len(merged) != len(conventional) or len(merged) != len(self_supervised):
        raise ValueError(
            "Prediction files do not contain the same validation row_id values."
        )

    if not np.array_equal(
        merged["y_true_conventional"].to_numpy(),
        merged["y_true_self_supervised"].to_numpy(),
    ):
        raise ValueError("y_true values are not identical after merging on row_id.")

    merged["y_true"] = merged["y_true_conventional"].astype(int)
    return merged


def create_interview_figure(
    conventional_predictions_path: str | Path = CONVENTIONAL_PREDICTIONS_PATH,
    self_supervised_predictions_path: str | Path = SELF_SUPERVISED_PREDICTIONS_PATH,
    output_path: str | Path = SUMMARY_FIGURE_PATH,
    pd_diagnostic_path: str | Path = PD_DIAGNOSTIC_PATH,
    dpi: int = 180,
) -> tuple[Path, Path]:
    merged = load_merged_predictions(
        conventional_predictions_path=conventional_predictions_path,
        self_supervised_predictions_path=self_supervised_predictions_path,
    )

    y_true = merged["y_true"].to_numpy()
    scores = {
        model_name: merged[column].to_numpy(dtype=float)
        for model_name, column in MODEL_SCORE_COLUMNS.items()
    }

    summary_path = create_model_comparison_summary(y_true, scores, output_path, dpi)
    diagnostic_path = create_pd_distribution_diagnostic(scores, pd_diagnostic_path, dpi)

    return summary_path, diagnostic_path


def create_model_comparison_summary(
    y_true: np.ndarray,
    scores: dict[str, np.ndarray],
    output_path: str | Path = SUMMARY_FIGURE_PATH,
    dpi: int = 180,
) -> Path:
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16, 5.2),
        gridspec_kw={"width_ratios": [1.15, 1.05, 1.05]},
    )

    _plot_capture_by_risk_band_panel(axes[0], y_true, scores)
    _plot_log_calibration_panel(axes[1], y_true, scores)
    _plot_precision_recall_panel(axes[2], y_true, scores)

    fig.suptitle(
        "Credit Risk Model Comparison",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    return _save_figure(fig, output_path, dpi=dpi)


def create_pd_distribution_diagnostic(
    scores: dict[str, np.ndarray],
    output_path: str | Path = PD_DIAGNOSTIC_PATH,
    dpi: int = 180,
) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.8))
    _plot_pd_distributions(ax, scores)
    ax.set_title("Predicted PD Distribution Diagnostic")
    fig.tight_layout()
    return _save_figure(fig, output_path, dpi=dpi)


def _save_figure(fig, output_path: str | Path, dpi: int = 180) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.12,
        facecolor="white",
        edgecolor="none",
        transparent=False,
    )
    plt.close(fig)
    return output_path


def _plot_precision_recall_panel(
    ax,
    y_true: np.ndarray,
    scores: dict[str, np.ndarray],
) -> None:
    default_rate = float(np.mean(y_true))

    for model_name, model_scores in scores.items():
        precision, recall, _ = precision_recall_curve(y_true, model_scores)
        pr_auc = average_precision_score(y_true, model_scores)
        roc_auc = roc_auc_score(y_true, model_scores)
        ax.plot(
            recall,
            precision,
            linewidth=2,
            label=f"{model_name} (PR-AUC={pr_auc:.3f}, ROC-AUC={roc_auc:.3f})",
        )

    ax.axhline(
        default_rate,
        linestyle="--",
        color="gray",
        linewidth=1,
        label=f"Default rate ({default_rate:.1%})",
    )
    ax.set_title("Precision-Recall")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=7)


def _plot_log_calibration_panel(
    ax,
    y_true: np.ndarray,
    scores: dict[str, np.ndarray],
) -> None:
    calibration_data = []
    axis_values = []

    for model_name, model_scores in scores.items():
        table = make_calibration_table(y_true, model_scores, n_bins=10)
        mean_pd = table["mean_predicted_pd"].to_numpy(dtype=float)
        observed_rate = table["observed_default_rate"].to_numpy(dtype=float)
        brier = brier_score_loss(y_true, model_scores)
        calibration_data.append((model_name, mean_pd, observed_rate, brier))
        axis_values.extend(mean_pd[mean_pd > 0].tolist())
        axis_values.extend(observed_rate[observed_rate > 0].tolist())

    if not axis_values:
        raise ValueError("Calibration plot requires at least one positive axis value.")

    min_axis = max(float(min(axis_values)) / 1.2, 1e-5)
    max_axis = min(1.0, float(max(axis_values)) * 1.2)
    max_axis = max(max_axis, min_axis * 10)

    for model_name, mean_pd, observed_rate, brier in calibration_data:
        ax.plot(
            np.clip(mean_pd, min_axis, max_axis),
            np.clip(observed_rate, min_axis, max_axis),
            marker="o",
            linewidth=2,
            label=f"{model_name} (Brier={brier:.4f})",
        )

    ax.plot(
        [min_axis, max_axis],
        [min_axis, max_axis],
        linestyle="--",
        color="gray",
        linewidth=1,
        label="Perfect calibration",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Calibration")
    ax.set_xlabel("Mean predicted PD")
    ax.set_ylabel("Observed default rate")
    ax.set_xlim(min_axis, max_axis)
    ax.set_ylim(min_axis, max_axis)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(frameon=False, fontsize=8)


def _plot_capture_by_risk_band_panel(
    ax,
    y_true: np.ndarray,
    scores: dict[str, np.ndarray],
) -> None:
    risk_bands = [0.01, 0.05, 0.10]
    band_labels = ["Top 1%", "Top 5%", "Top 10%"]
    model_names = list(scores.keys())
    x = np.arange(len(band_labels))
    width = 0.24
    colors = ["#4C78A8", "#F58518", "#54A24B"]

    max_capture = 0.0
    for model_index, model_name in enumerate(model_names):
        capture_rates = [
            _defaults_captured_in_top_fraction(y_true, scores[model_name], fraction)
            for fraction in risk_bands
        ]
        max_capture = max(max_capture, max(capture_rates))
        offset = (model_index - (len(model_names) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            capture_rates,
            width=width,
            label=model_name,
            color=colors[model_index % len(colors)],
        )
        for bar, value in zip(bars, capture_rates):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.0,
                f"{value:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_title("Defaults Captured In Highest-Risk Bands")
    ax.set_ylabel("Share of all defaults")
    ax.set_xticks(x)
    ax.set_xticklabels(band_labels)
    ax.set_ylim(0, min(100, max_capture * 1.2 + 4))
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(frameon=False, fontsize=8)


def _defaults_captured_in_top_fraction(
    y_true: np.ndarray,
    model_scores: np.ndarray,
    fraction: float,
) -> float:
    y_true = np.asarray(y_true).astype(int)
    model_scores = np.asarray(model_scores, dtype=float)
    n_top = max(1, int(np.ceil(len(model_scores) * fraction)))
    top_indices = np.argsort(model_scores)[::-1][:n_top]
    total_defaults = int(y_true.sum())
    if total_defaults == 0:
        return np.nan
    return float(y_true[top_indices].sum() / total_defaults * 100)


def _plot_pd_distributions(ax, scores: dict[str, np.ndarray]) -> None:
    positive_scores = {}
    for model_name, model_scores in scores.items():
        model_scores = np.asarray(model_scores, dtype=float)
        model_scores = model_scores[np.isfinite(model_scores)]
        positive_scores[model_name] = model_scores[model_scores > 0]

    combined_positive = np.concatenate(list(positive_scores.values()))
    if len(combined_positive) == 0:
        raise ValueError("PD distribution requires at least one positive score.")

    lower = float(combined_positive.min())
    upper = float(combined_positive.max())
    if lower <= 0:
        lower = np.finfo(float).tiny
    if upper <= lower:
        lower = max(lower / 10.0, np.finfo(float).tiny)
        upper = upper * 10.0

    bins = np.geomspace(lower, upper, 36)
    colors = {
        "Logistic regression": "#4C78A8",
        "XGBoost": "#F58518",
        "XGBoost + self-supervised embeddings": "#54A24B",
    }

    for model_name, model_scores in positive_scores.items():
        if model_name == "XGBoost + self-supervised embeddings":
            ax.hist(
                model_scores,
                bins=bins,
                density=True,
                histtype="step",
                linewidth=2.2,
                linestyle="--",
                color=colors[model_name],
                label=model_name,
                zorder=4,
            )
        else:
            ax.hist(
                model_scores,
                bins=bins,
                density=True,
                histtype="bar",
                alpha=0.32,
                color=colors[model_name],
                edgecolor=colors[model_name],
                linewidth=0.7,
                label=model_name,
                zorder=2,
            )

    ax.set_title("Predicted PD Distribution")
    ax.set_xlabel("Predicted probability of default")
    ax.set_ylabel("Density")
    ax.set_xscale("log")
    ax.set_xlim(lower, upper)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)


def main() -> None:
    args = parse_args()
    summary_path, diagnostic_path = create_interview_figure(
        conventional_predictions_path=args.conventional_predictions_path,
        self_supervised_predictions_path=args.self_supervised_predictions_path,
        output_path=args.output_path,
        pd_diagnostic_path=args.pd_diagnostic_path,
        dpi=args.dpi,
    )
    print(f"Saved summary figure: {summary_path}")
    print(f"Saved PD diagnostic figure: {diagnostic_path}")


if __name__ == "__main__":
    main()
