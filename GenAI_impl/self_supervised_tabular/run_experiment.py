"""Run the self-supervised tabular representation experiment.

This is an exploratory portfolio experiment: a masked autoencoder learns
borrower representations without default labels, then XGBoost is trained in the
ordinary supervised way on three feature sets.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib


matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluate import compare_model_metrics, save_table  # noqa: E402
from load_data import (  # noqa: E402
    TRAIN_DATA_PATH,
    fit_missing_value_preprocessor,
    load_training_data,
    make_train_val_split,
    prepare_features_target,
)
from xgboost_model import XGBoostPDModel  # noqa: E402

from make_interview_figure import create_interview_figure  # noqa: E402
from masked_autoencoder import (  # noqa: E402
    AutoencoderTrainingConfig,
    train_masked_autoencoder,
)


EXPERIMENT_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = EXPERIMENT_DIR / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run GenAI-inspired self-supervised masked-feature pretraining "
            "for tabular credit-risk data."
        )
    )
    parser.add_argument("--data-path", type=Path, default=TRAIN_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--mask-fraction", type=float, default=0.25)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--internal-val-fraction", type=float, default=0.15)
    parser.add_argument(
        "--device",
        default="auto",
        help='Autoencoder device: "auto", "cpu", or "cuda".',
    )

    parser.add_argument("--xgb-device", default="cpu")
    parser.add_argument("--xgb-n-estimators", type=int, default=500)
    parser.add_argument("--xgb-learning-rate", type=float, default=0.05)
    parser.add_argument("--xgb-max-depth", type=int, default=5)
    parser.add_argument("--xgb-min-child-weight", type=float, default=1.0)
    parser.add_argument("--xgb-subsample", type=float, default=0.8)
    parser.add_argument("--xgb-colsample-bytree", type=float, default=0.8)
    parser.add_argument("--xgb-reg-lambda", type=float, default=1.0)
    parser.add_argument("--xgb-reg-alpha", type=float, default=0.0)
    return parser.parse_args()


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def make_embedding_frame(
    embeddings: np.ndarray,
    index: pd.Index,
    prefix: str = "mae_embedding",
) -> pd.DataFrame:
    return pd.DataFrame(
        embeddings,
        index=index,
        columns=[f"{prefix}_{i + 1}" for i in range(embeddings.shape[1])],
    )


def make_xgboost_model(args: argparse.Namespace) -> XGBoostPDModel:
    return XGBoostPDModel(
        learning_rate=args.xgb_learning_rate,
        max_depth=args.xgb_max_depth,
        min_child_weight=args.xgb_min_child_weight,
        subsample=args.xgb_subsample,
        colsample_bytree=args.xgb_colsample_bytree,
        reg_lambda=args.xgb_reg_lambda,
        reg_alpha=args.xgb_reg_alpha,
        n_estimators=args.xgb_n_estimators,
        random_state=args.seed,
        device=args.xgb_device,
    )


def train_and_score_xgboost(
    X_train: pd.DataFrame,
    y_train,
    X_val: pd.DataFrame,
    args: argparse.Namespace,
) -> np.ndarray:
    model = make_xgboost_model(args)
    model.fit(X_train, y_train)
    return model.predict_proba(X_val)


def save_reconstruction_plot(history: pd.DataFrame, output_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(
        history["epoch"],
        history["train_reconstruction_loss"],
        label="Internal train",
        linewidth=2,
    )
    ax.plot(
        history["epoch"],
        history["internal_val_reconstruction_loss"],
        label="Internal reconstruction validation",
        linewidth=2,
    )
    ax.set_title("Masked Autoencoder Reconstruction Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Masked-feature MSE")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_autoencoder_checkpoint(
    model,
    scaler: StandardScaler,
    feature_names: list[str],
    history: pd.DataFrame,
    device: torch.device,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            },
            "input_dim": model.input_dim,
            "latent_dim": model.latent_dim,
            "feature_names": feature_names,
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "best_internal_val_reconstruction_loss": float(
                history["internal_val_reconstruction_loss"].min()
            ),
            "device_used": str(device),
        },
        output_path,
    )
    return output_path


def _format_metric_delta(
    metrics: pd.DataFrame,
    metric: str,
    higher_is_better: bool,
) -> str:
    base = float(metrics.loc["xgboost_original", metric])
    augmented = float(metrics.loc["xgboost_original_plus_embeddings", metric])
    delta = augmented - base
    improved = delta > 0 if higher_is_better else delta < 0
    worsened = delta < 0 if higher_is_better else delta > 0

    if improved:
        direction = "improved"
    elif worsened:
        direction = "worsened"
    else:
        direction = "was unchanged"

    return (
        f"- `{metric}` {direction}: original={base:.6f}, "
        f"original_plus_embeddings={augmented:.6f}, delta={delta:.6f}."
    )


def write_experiment_summary(metrics: pd.DataFrame, output_path: Path) -> Path:
    indexed = metrics.set_index("model")
    default_rate = float(indexed.loc["xgboost_original", "default_rate"])

    mean_pd_original = float(indexed.loc["xgboost_original", "mean_predicted_pd"])
    mean_pd_augmented = float(
        indexed.loc["xgboost_original_plus_embeddings", "mean_predicted_pd"]
    )
    gap_original = abs(mean_pd_original - default_rate)
    gap_augmented = abs(mean_pd_augmented - default_rate)
    if gap_augmented < gap_original:
        mean_pd_text = "improved"
    elif gap_augmented > gap_original:
        mean_pd_text = "worsened"
    else:
        mean_pd_text = "was unchanged"

    lines = [
        "# Experiment Summary",
        "",
        "This experiment tests **GenAI-inspired self-supervised masked-feature "
        "pretraining for tabular data** in the existing credit-risk project.",
        "",
        "The neural network was pretrained without default labels. Its training "
        "targets were created by randomly masking standardized borrower features "
        "and reconstructing the original feature values.",
        "",
        "XGBoost was subsequently trained with default labels for probability-of-"
        "default estimation. XGBoost itself was not pretrained.",
        "",
        "## Metric Comparison",
        "",
        _format_metric_delta(indexed, "roc_auc", higher_is_better=True),
        _format_metric_delta(indexed, "pr_auc", higher_is_better=True),
        _format_metric_delta(indexed, "log_loss", higher_is_better=False),
        _format_metric_delta(indexed, "brier_score", higher_is_better=False),
        (
            f"- `mean_predicted_pd` calibration gap versus the observed default "
            f"rate {mean_pd_text}: original_gap={gap_original:.6f}, "
            f"original_plus_embeddings_gap={gap_augmented:.6f}, "
            f"observed_default_rate={default_rate:.6f}."
        ),
        "",
        "Neutral or negative results remain informative: XGBoost is already a "
        "strong model for small structured tabular datasets, so learned dense "
        "representations do not automatically improve downstream performance.",
        "",
        "This demonstrates self-supervised representation learning inspired by "
        "GenAI pretraining patterns. It is not a GenAI application, and the "
        "autoencoder itself is not described as a GenAI model.",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def main() -> None:
    args = parse_args()
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_training_data(project_path(args.data_path))
    X, y = prepare_features_target(df)

    X_train, X_val, y_train, y_val = make_train_val_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
    )

    missing_preprocessor = fit_missing_value_preprocessor(X_train)
    X_train_clean = missing_preprocessor.transform(X_train)
    X_val_clean = missing_preprocessor.transform(X_val)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_clean)
    X_val_scaled = scaler.transform(X_val_clean)

    autoencoder_config = AutoencoderTrainingConfig(
        latent_dim=args.latent_dim,
        mask_fraction=args.mask_fraction,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        internal_val_fraction=args.internal_val_fraction,
        seed=args.seed,
        device=args.device,
    )
    autoencoder, history, device = train_masked_autoencoder(
        X_train_scaled,
        autoencoder_config,
    )

    train_embeddings = autoencoder.transform(
        X_train_scaled,
        device=device,
        batch_size=args.batch_size,
    )
    val_embeddings = autoencoder.transform(
        X_val_scaled,
        device=device,
        batch_size=args.batch_size,
    )
    train_embedding_df = make_embedding_frame(train_embeddings, X_train_clean.index)
    val_embedding_df = make_embedding_frame(val_embeddings, X_val_clean.index)

    X_train_plus_embeddings = pd.concat([X_train_clean, train_embedding_df], axis=1)
    X_val_plus_embeddings = pd.concat([X_val_clean, val_embedding_df], axis=1)

    predictions = {
        "xgboost_original": train_and_score_xgboost(
            X_train_clean,
            y_train,
            X_val_clean,
            args,
        ),
        "xgboost_embeddings": train_and_score_xgboost(
            train_embedding_df,
            y_train,
            val_embedding_df,
            args,
        ),
        "xgboost_original_plus_embeddings": train_and_score_xgboost(
            X_train_plus_embeddings,
            y_train,
            X_val_plus_embeddings,
            args,
        ),
    }

    metrics = compare_model_metrics(
        {model_name: (y_val, scores) for model_name, scores in predictions.items()}
    )

    validation_predictions = pd.DataFrame(
        {
            "row_id": X_val.index,
            "y_true": y_val.to_numpy(),
            "xgboost_original_pd": predictions["xgboost_original"],
            "xgboost_embeddings_pd": predictions["xgboost_embeddings"],
            "xgboost_original_plus_embeddings_pd": predictions[
                "xgboost_original_plus_embeddings"
            ],
        }
    )

    saved_paths = [
        save_table(metrics, output_dir / "comparison_metrics.csv"),
        save_table(history, output_dir / "reconstruction_history.csv"),
        save_table(validation_predictions, output_dir / "validation_predictions.csv"),
        save_reconstruction_plot(history, output_dir / "reconstruction_loss.png"),
        save_autoencoder_checkpoint(
            autoencoder,
            scaler,
            X_train_clean.columns.tolist(),
            history,
            device,
            output_dir / "masked_autoencoder.pt",
        ),
        write_experiment_summary(metrics, output_dir / "experiment_summary.md"),
        create_interview_figure(
            predictions_path=output_dir / "validation_predictions.csv",
            output_path=output_dir / "interview_comparison.png",
        ),
    ]

    print("Saved self-supervised tabular experiment outputs:")
    for path in saved_paths:
        print(f"- {Path(path).relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
