"""Focused tests for reusable plotting utilities."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib


matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from plots import (  # noqa: E402
    _prepare_logistic_coefficient_plot_data,
    plot_logistic_coefficients,
)


def test_logistic_coefficient_plot_validates_required_columns() -> None:
    coefficients = pd.DataFrame(
        {
            "feature": ["feature_a"],
            "coefficient": [0.2],
        }
    )

    try:
        plot_logistic_coefficients(coefficients)
    except ValueError as error:
        assert "odds_ratio_per_std" in str(error)
    else:
        raise AssertionError("Expected missing required columns to raise ValueError.")


def test_logistic_coefficient_plot_orders_by_absolute_coefficient() -> None:
    coefficients = pd.DataFrame(
        {
            "feature": [
                "tiny",
                "medium_positive",
                "large_negative",
                "largest_positive",
            ],
            "coefficient": [0.01, 0.2, -0.6, 0.9],
            "odds_ratio_per_std": np.exp([0.01, 0.2, -0.6, 0.9]),
        }
    )

    plot_data = _prepare_logistic_coefficient_plot_data(coefficients, top_n=3)

    assert plot_data["feature"].tolist() == [
        "medium_positive",
        "large_negative",
        "largest_positive",
    ]
    assert plot_data["abs_coefficient"].is_monotonic_increasing


def test_logistic_coefficient_plot_creates_horizontal_bars() -> None:
    coefficients = pd.DataFrame(
        {
            "feature": ["feature_a", "feature_b"],
            "coefficient": [0.4, -0.2],
            "odds_ratio_per_std": np.exp([0.4, -0.2]),
        }
    )

    fig, ax = plot_logistic_coefficients(coefficients, top_n=2)

    assert len(ax.patches) == 2
    assert "Standardized logistic regression coefficients" in ax.get_title()
    plt.close(fig)


def main() -> None:
    test_logistic_coefficient_plot_validates_required_columns()
    test_logistic_coefficient_plot_orders_by_absolute_coefficient()
    test_logistic_coefficient_plot_creates_horizontal_bars()
    print("Plot tests passed.")


if __name__ == "__main__":
    main()
