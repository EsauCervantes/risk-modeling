"""Plotting utilities for credit risk model evaluation."""

from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def _get_fig_ax(ax=None, figsize: tuple[float, float] = (7, 5)):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    return fig, ax


def _prepare_binary_inputs(y_true, y_score) -> tuple[np.ndarray, np.ndarray]:
    y_true_array = np.asarray(y_true).ravel().astype(int)
    y_score_array = np.asarray(y_score, dtype=float).ravel()

    if len(y_true_array) != len(y_score_array):
        raise ValueError(
            "y_true and y_score must have the same length: "
            f"{len(y_true_array)} != {len(y_score_array)}"
        )

    if len(y_true_array) == 0:
        raise ValueError("y_true and y_score must not be empty.")

    if not np.isfinite(y_score_array).all():
        raise ValueError("y_score contains NaN or infinite values.")

    return y_true_array, np.clip(y_score_array, 0.0, 1.0)


def _apply_common_style(ax):
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)


def _validate_model_scores(model_scores: Mapping[str, object]):
    if not model_scores:
        raise ValueError("model_scores must contain at least one model.")


def plot_roc_curve(
    y_true,
    model_scores: Mapping[str, object],
    ax=None,
):
    """Plot ROC curves for one or more probability-of-default models."""
    _validate_model_scores(model_scores)
    fig, ax = _get_fig_ax(ax)

    for model_name, y_score in model_scores.items():
        y_true_array, y_score_array = _prepare_binary_inputs(y_true, y_score)
        if len(np.unique(y_true_array)) < 2:
            raise ValueError("ROC curve requires both positive and negative labels.")
        fpr, tpr, _ = roc_curve(y_true_array, y_score_array)
        auc = roc_auc_score(y_true_array, y_score_array)
        ax.plot(fpr, tpr, linewidth=2, label=f"{model_name} (ROC-AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Random")
    ax.set_title("ROC-AUC Curve")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _apply_common_style(ax)

    return fig, ax


def plot_precision_recall_curve(
    y_true,
    model_scores: Mapping[str, object],
    ax=None,
):
    """Plot precision-recall curves for one or more PD models."""
    _validate_model_scores(model_scores)
    fig, ax = _get_fig_ax(ax)
    y_true_array = np.asarray(y_true).ravel().astype(int)
    default_rate = float(y_true_array.mean())

    for model_name, y_score in model_scores.items():
        y_true_model, y_score_array = _prepare_binary_inputs(y_true, y_score)
        precision, recall, _ = precision_recall_curve(y_true_model, y_score_array)
        pr_auc = average_precision_score(y_true_model, y_score_array)
        ax.plot(
            recall,
            precision,
            linewidth=2,
            label=f"{model_name} (PR-AUC={pr_auc:.3f})",
        )

    ax.axhline(
        default_rate,
        linestyle="--",
        color="gray",
        linewidth=1,
        label=f"Default rate ({default_rate:.1%})",
    )
    ax.set_title("Precision-Recall Curve (PR-AUC)")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _apply_common_style(ax)

    return fig, ax


def plot_calibration_curve(
    calibration_tables: Mapping[str, pd.DataFrame],
    ax=None,
    max_axis: float | None = None,
    log_scale: bool = False,
):
    """Plot mean predicted PD against observed default rate by calibration bin."""
    if not calibration_tables:
        raise ValueError("calibration_tables must contain at least one model.")

    fig, ax = _get_fig_ax(ax)
    axis_values = []
    plot_data = []

    for model_name, table in calibration_tables.items():
        required_columns = {"mean_predicted_pd", "observed_default_rate"}
        missing_columns = required_columns - set(table.columns)
        if missing_columns:
            raise ValueError(
                f"Calibration table for {model_name} is missing columns: "
                f"{sorted(missing_columns)}"
            )

        mean_pd = table["mean_predicted_pd"].to_numpy()
        observed_rate = table["observed_default_rate"].to_numpy()
        plot_data.append((model_name, mean_pd, observed_rate))
        axis_values.extend(mean_pd.tolist())
        axis_values.extend(observed_rate.tolist())

    if max_axis is None:
        max_axis = min(1.0, max(axis_values) * 1.15)
        max_axis = max(max_axis, 0.05)

    min_axis = 0.0
    if log_scale:
        positive_axis_values = [value for value in axis_values if value > 0]
        if not positive_axis_values:
            raise ValueError("Log-scale calibration requires positive bin values.")

        min_axis = max(min(positive_axis_values) / 1.15, 1e-6)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title("Calibration Curve (Log Scale)")
    else:
        ax.set_title("Calibration Curve")

    for model_name, mean_pd, observed_rate in plot_data:
        if log_scale:
            mean_pd = np.clip(mean_pd, min_axis, max_axis)
            observed_rate = np.clip(observed_rate, min_axis, max_axis)

        ax.plot(
            mean_pd,
            observed_rate,
            marker="o",
            linewidth=2,
            label=model_name,
        )

    ax.plot(
        [min_axis, max_axis],
        [min_axis, max_axis],
        linestyle="--",
        color="gray",
        linewidth=1,
        label="Perfect",
    )
    ax.set_xlabel("Mean predicted PD")
    ax.set_ylabel("Observed default rate")
    ax.set_xlim(min_axis, max_axis)
    ax.set_ylim(min_axis, max_axis)
    _apply_common_style(ax)

    return fig, ax


def plot_pd_distribution(
    model_scores: Mapping[str, object],
    ax=None,
    log_x: bool = False,
    bins: int = 40,
):
    """Plot predicted probability-of-default distributions by model."""
    _validate_model_scores(model_scores)
    fig, ax = _get_fig_ax(ax)
    prepared_scores = {}

    for model_name, y_score in model_scores.items():
        y_score_array = np.asarray(y_score, dtype=float).ravel()
        if len(y_score_array) == 0:
            raise ValueError(f"Scores for {model_name} must not be empty.")
        if not np.isfinite(y_score_array).all():
            raise ValueError(f"Scores for {model_name} contain NaN or infinite values.")

        prepared_scores[model_name] = np.clip(y_score_array, 0.0, 1.0)

    if log_x:
        positive_scores = np.concatenate(
            [scores[scores > 0] for scores in prepared_scores.values()]
        )
        if len(positive_scores) == 0:
            raise ValueError("At least one positive score is required for log scaling.")

        lower = max(positive_scores.min(), 1e-6)
        upper = max(positive_scores.max(), lower * 10)
        histogram_bins = np.geomspace(lower, upper, bins + 1)
    else:
        histogram_bins = bins

    for model_name, y_score_array in prepared_scores.items():
        scores = np.clip(y_score_array, lower, upper) if log_x else y_score_array

        ax.hist(
            scores,
            bins=histogram_bins,
            range=None if log_x else (0, 1),
            density=True,
            alpha=0.35,
            label=model_name,
        )

    ax.set_title("Predicted PD Distribution")
    ax.set_xlabel("Predicted probability of default")
    ax.set_ylabel("Density")
    if log_x:
        ax.set_xscale("log")
        ax.set_xlim(lower, upper)
    else:
        ax.set_xlim(0, 1)
    _apply_common_style(ax)

    return fig, ax


def plot_decile_default_rate(
    decile_table: pd.DataFrame,
    model_name: str | None = None,
    ax=None,
):
    """Plot observed default rate by risk decile.

    The expected decile convention is that decile 1 is the highest-risk group.
    """
    required_columns = {"decile", "observed_default_rate"}
    missing_columns = required_columns - set(decile_table.columns)
    if missing_columns:
        raise ValueError(f"Decile table is missing columns: {sorted(missing_columns)}")

    fig, ax = _get_fig_ax(ax)
    table = decile_table.sort_values("decile")

    ax.bar(
        table["decile"].astype(str),
        table["observed_default_rate"],
        color="#4C78A8",
    )
    ax.set_title(
        "Observed Default Rate By Risk Decile"
        if model_name is None
        else f"Observed Default Rate By Risk Decile: {model_name}"
    )
    ax.set_xlabel("Risk decile (1 = highest predicted risk)")
    ax.set_ylabel("Observed default rate")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    ax.annotate(
        "Highest risk",
        xy=(0, table["observed_default_rate"].iloc[0]),
        xytext=(0, table["observed_default_rate"].max() * 1.08),
        ha="center",
        arrowprops={"arrowstyle": "->", "color": "gray", "lw": 1},
    )
    ax.set_ylim(0, max(table["observed_default_rate"].max() * 1.2, 0.01))
    ax.grid(True, axis="y", alpha=0.3)

    return fig, ax


def save_figure(fig, path: str | Path, dpi: int = 160) -> Path:
    """Save a matplotlib figure, creating parent directories if needed."""
    output_path = Path(path)
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
    return output_path
