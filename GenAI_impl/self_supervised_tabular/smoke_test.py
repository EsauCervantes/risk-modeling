"""Small smoke tests for the self-supervised tabular experiment code."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from make_interview_figure import create_interview_figure
from masked_autoencoder import (
    AutoencoderTrainingConfig,
    create_feature_mask,
    train_masked_autoencoder,
)


def test_mask_has_at_least_one_masked_feature_per_row() -> None:
    x = torch.ones((20, 6))
    mask = create_feature_mask(x, mask_fraction=0.25)
    assert mask.shape == x.shape
    assert torch.all(mask.sum(dim=1) >= 1)


def test_autoencoder_training_and_embedding_shape() -> None:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(96, 6)).astype(np.float32)
    config = AutoencoderTrainingConfig(
        latent_dim=3,
        batch_size=16,
        max_epochs=2,
        patience=2,
        internal_val_fraction=0.2,
        seed=42,
        device="cpu",
    )
    model, history, device = train_masked_autoencoder(X, config)
    embeddings = model.transform(X[:10], device=device, batch_size=5)
    assert len(history) >= 1
    assert embeddings.shape == (10, 3)


def test_interview_figure_from_saved_predictions() -> None:
    rng = np.random.default_rng(42)
    y_true = rng.binomial(1, 0.25, size=80)
    logistic = np.clip(0.12 + 0.45 * y_true + rng.normal(0, 0.16, size=80), 0, 1)
    original = np.clip(0.15 + 0.55 * y_true + rng.normal(0, 0.12, size=80), 0, 1)
    augmented = np.clip(original + rng.normal(0, 0.03, size=80), 0, 1)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        conventional_predictions_path = tmp_path / "conventional_predictions.csv"
        self_supervised_predictions_path = tmp_path / "self_supervised_predictions.csv"
        output_path = tmp_path / "model_comparison_summary.png"
        pd_diagnostic_path = tmp_path / "pd_distribution_diagnostic.png"
        pd.DataFrame(
            {
                "row_id": np.arange(len(y_true)),
                "y_true": y_true,
                "logistic_l2_clipped_pd": logistic,
                "xgboost_pd": original,
            }
        ).to_csv(conventional_predictions_path, index=False)
        pd.DataFrame(
            {
                "row_id": np.arange(len(y_true)),
                "y_true": y_true,
                "xgboost_original_pd": original,
                "xgboost_embeddings_pd": np.clip(rng.random(size=80), 0, 1),
                "xgboost_original_plus_embeddings_pd": augmented,
            }
        ).to_csv(self_supervised_predictions_path, index=False)

        create_interview_figure(
            conventional_predictions_path=conventional_predictions_path,
            self_supervised_predictions_path=self_supervised_predictions_path,
            output_path=output_path,
            pd_diagnostic_path=pd_diagnostic_path,
        )
        assert output_path.exists()
        assert output_path.stat().st_size > 0
        assert pd_diagnostic_path.exists()
        assert pd_diagnostic_path.stat().st_size > 0


def main() -> None:
    test_mask_has_at_least_one_masked_feature_per_row()
    test_autoencoder_training_and_embedding_shape()
    test_interview_figure_from_saved_predictions()
    print("Smoke tests passed.")


if __name__ == "__main__":
    main()
