"""Evaluation utilities for binary probability-of-default models."""

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


EPSILON = 1e-15


def _prepare_binary_inputs(y_true, y_score) -> tuple[np.ndarray, np.ndarray]:
    y_true_array = np.asarray(y_true).ravel()
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

    unique_targets = set(pd.Series(y_true_array).dropna().unique())
    if not unique_targets.issubset({0, 1, False, True}):
        raise ValueError("y_true must contain binary labels encoded as 0/1.")

    if pd.Series(y_true_array).isna().any():
        raise ValueError("y_true contains missing values.")

    if ((y_score_array < -EPSILON) | (y_score_array > 1 + EPSILON)).any():
        raise ValueError("y_score must contain predicted probabilities in [0, 1].")

    return y_true_array.astype(int), np.clip(y_score_array, 0.0, 1.0)


def _validate_n_bins(n_bins: int) -> int:
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1.")
    return int(n_bins)


def _assign_quantile_bins(
    y_score: np.ndarray,
    n_bins: int,
    *,
    high_score_first: bool,
) -> pd.Series:
    """Assign quantile bins robustly, including when many scores are tied."""
    n_bins = min(_validate_n_bins(n_bins), len(y_score))
    ranked_scores = pd.Series(y_score).rank(
        method="first",
        ascending=not high_score_first,
    )
    return pd.qcut(ranked_scores, q=n_bins, labels=False, duplicates="drop") + 1


def compute_binary_classification_metrics(
    y_true,
    y_score,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    """Compute core validation metrics for a binary PD model.

    Parameters
    ----------
    y_true:
        Binary observed outcomes where 1 indicates default.
    y_score:
        Predicted probability of default.
    threshold:
        Reserved for threshold-based extensions. The returned metrics are
        probability/ranking metrics and do not depend on this value.
    """
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")

    y_true_array, y_score_array = _prepare_binary_inputs(y_true, y_score)
    y_score_for_log_loss = np.clip(y_score_array, EPSILON, 1 - EPSILON)
    positive_count = int(y_true_array.sum())
    has_two_classes = len(np.unique(y_true_array)) == 2

    return {
        "roc_auc": roc_auc_score(y_true_array, y_score_array)
        if has_two_classes
        else np.nan,
        "pr_auc": average_precision_score(y_true_array, y_score_array)
        if has_two_classes
        else np.nan,
        "log_loss": log_loss(y_true_array, y_score_for_log_loss, labels=[0, 1]),
        "brier_score": brier_score_loss(y_true_array, y_score_array),
        "default_rate": float(y_true_array.mean()),
        "mean_predicted_pd": float(y_score_array.mean()),
        "n_samples": int(len(y_true_array)),
        "positive_count": positive_count,
    }


def make_decile_table(y_true, y_score, n_bins: int = 10) -> pd.DataFrame:
    """Create a risk-decile table from predicted probabilities of default.

    Decile 1 is the highest-risk group. Quantile assignment is based on ranked
    predicted probabilities, which avoids failures when many borrowers have the
    same score.
    """
    y_true_array, y_score_array = _prepare_binary_inputs(y_true, y_score)
    portfolio_default_rate = float(y_true_array.mean())

    df = pd.DataFrame(
        {
            "y_true": y_true_array,
            "predicted_pd": y_score_array,
        }
    )
    df["decile"] = _assign_quantile_bins(
        y_score_array,
        n_bins,
        high_score_first=True,
    )

    decile_table = (
        df.groupby("decile", as_index=False)
        .agg(
            n_obs=("y_true", "size"),
            mean_predicted_pd=("predicted_pd", "mean"),
            observed_default_rate=("y_true", "mean"),
            default_count=("y_true", "sum"),
            min_predicted_pd=("predicted_pd", "min"),
            max_predicted_pd=("predicted_pd", "max"),
        )
        .sort_values("decile")
        .reset_index(drop=True)
    )

    if portfolio_default_rate > 0:
        decile_table["lift_vs_portfolio_default_rate"] = (
            decile_table["observed_default_rate"] / portfolio_default_rate
        )
    else:
        decile_table["lift_vs_portfolio_default_rate"] = np.nan

    decile_table["n_obs"] = decile_table["n_obs"].astype(int)
    decile_table["default_count"] = decile_table["default_count"].astype(int)

    return decile_table[
        [
            "decile",
            "n_obs",
            "mean_predicted_pd",
            "observed_default_rate",
            "default_count",
            "min_predicted_pd",
            "max_predicted_pd",
            "lift_vs_portfolio_default_rate",
        ]
    ]


def make_calibration_table(y_true, y_score, n_bins: int = 10) -> pd.DataFrame:
    """Create a quantile-binned calibration table for predicted PDs.

    Bins are ordered from lowest predicted PD to highest predicted PD.
    ``calibration_error`` is signed: observed default rate minus mean predicted
    PD.
    """
    y_true_array, y_score_array = _prepare_binary_inputs(y_true, y_score)

    df = pd.DataFrame(
        {
            "y_true": y_true_array,
            "predicted_pd": y_score_array,
        }
    )
    df["bin"] = _assign_quantile_bins(
        y_score_array,
        n_bins,
        high_score_first=False,
    )

    calibration_table = (
        df.groupby("bin", as_index=False)
        .agg(
            n_obs=("y_true", "size"),
            mean_predicted_pd=("predicted_pd", "mean"),
            observed_default_rate=("y_true", "mean"),
        )
        .sort_values("bin")
        .reset_index(drop=True)
    )
    calibration_table["calibration_error"] = (
        calibration_table["observed_default_rate"]
        - calibration_table["mean_predicted_pd"]
    )
    calibration_table["n_obs"] = calibration_table["n_obs"].astype(int)

    return calibration_table[
        [
            "bin",
            "n_obs",
            "mean_predicted_pd",
            "observed_default_rate",
            "calibration_error",
        ]
    ]


def compare_model_metrics(
    results: Mapping[str, tuple[object, object]],
) -> pd.DataFrame:
    """Compare binary PD metrics for multiple model score vectors."""
    if not results:
        raise ValueError("results must contain at least one model.")

    rows = []
    for model_name, (y_true, y_score) in results.items():
        metrics = compute_binary_classification_metrics(y_true, y_score)
        metrics["model"] = model_name
        rows.append(metrics)

    return pd.DataFrame(rows)[
        [
            "model",
            "roc_auc",
            "pr_auc",
            "log_loss",
            "brier_score",
            "default_rate",
            "mean_predicted_pd",
            "n_samples",
            "positive_count",
        ]
    ]


def save_table(df: pd.DataFrame, path: str | Path) -> Path:
    """Save a dataframe as CSV, creating parent directories if needed."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path
