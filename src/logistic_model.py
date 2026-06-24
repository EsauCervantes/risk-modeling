"""Regularized logistic regression model utilities."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class QuantileClipper:
    lower_bounds: pd.Series
    upper_bounds: pd.Series

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_clipped = X.copy()
        clip_columns = [col for col in self.lower_bounds.index if col in X_clipped.columns]

        X_clipped[clip_columns] = X_clipped[clip_columns].astype(float).clip(
            lower=self.lower_bounds[clip_columns],
            upper=self.upper_bounds[clip_columns],
            axis="columns",
        )

        return X_clipped


class LogisticPDModel:
    def __init__(
        self,
        C: float = 1.0,
        penalty: str = "l2",
        solver: str | None = None,
        l1_ratio: float | None = None,
        class_weight: str | dict[int, float] | None = None,
        clip_quantiles: tuple[float, float] | None = (0.01, 0.99),
        max_iter: int = 5000,
        random_state: int = 42,
    ):
        self.C = C
        self.penalty = penalty
        self.solver = solver
        self.l1_ratio = l1_ratio
        self.class_weight = class_weight
        self.clip_quantiles = clip_quantiles
        self.max_iter = max_iter
        self.random_state = random_state

        self.clipper = None
        self.feature_names_ = None
        self.model = None

    def _resolve_solver(self) -> str:
        if self.solver is not None:
            return self.solver
        if self.penalty in {"l1", "elasticnet"}:
            return "saga"
        return "lbfgs"

    def _validate_clip_quantiles(self):
        if self.clip_quantiles is None:
            return

        lower, upper = self.clip_quantiles
        if not 0 <= lower < upper <= 1:
            raise ValueError("clip_quantiles must satisfy 0 <= lower < upper <= 1.")

    def _to_dataframe(self, X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.copy()

        if self.feature_names_ is None:
            columns = [f"feature_{i}" for i in range(np.asarray(X).shape[1])]
        else:
            columns = self.feature_names_

        return pd.DataFrame(X, columns=columns)

    def _fit_clipper(self, X_train: pd.DataFrame):
        self._validate_clip_quantiles()
        if self.clip_quantiles is None:
            return None

        lower, upper = self.clip_quantiles
        numeric_columns = X_train.select_dtypes(include="number").columns

        return QuantileClipper(
            lower_bounds=X_train[numeric_columns].quantile(lower),
            upper_bounds=X_train[numeric_columns].quantile(upper),
        )

    def _prepare_features(self, X) -> pd.DataFrame:
        X_prepared = self._to_dataframe(X)

        if self.clipper is not None:
            X_prepared = self.clipper.transform(X_prepared)

        missing_columns = X_prepared.columns[X_prepared.isna().any()].tolist()
        if missing_columns:
            raise ValueError(
                "Missing values remain in features. "
                f"Run missing-value preprocessing first: {missing_columns}"
            )

        return X_prepared

    def _build_model(self) -> Pipeline:
        if self.penalty == "elasticnet" and self.l1_ratio is None:
            raise ValueError("l1_ratio must be set when penalty='elasticnet'.")

        classifier = LogisticRegression(
            C=self.C,
            penalty=self.penalty,
            solver=self._resolve_solver(),
            l1_ratio=self.l1_ratio,
            class_weight=self.class_weight,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )

        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", classifier),
            ]
        )

    def fit(self, X_train, y_train):
        X_train = self._to_dataframe(X_train)
        y_train = np.asarray(y_train).astype(int)

        self.feature_names_ = X_train.columns.tolist()
        self.clipper = self._fit_clipper(X_train)
        X_train_prepared = self._prepare_features(X_train)

        self.model = self._build_model()
        self.model.fit(X_train_prepared, y_train)

        return self

    def predict_proba(self, X) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model must be fitted before prediction.")

        X_prepared = self._prepare_features(X)
        return self.model.predict_proba(X_prepared)[:, 1]

    def evaluate(self, X, y) -> dict[str, float]:
        y = np.asarray(y).astype(int)
        pd_hat = self.predict_proba(X)

        return {
            "roc_auc": roc_auc_score(y, pd_hat),
            "pr_auc": average_precision_score(y, pd_hat),
            "log_loss": log_loss(y, pd_hat),
        }

    def coefficients(self) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Model must be fitted before reading coefficients.")

        coefficients = self.model.named_steps["model"].coef_[0]

        return (
            pd.DataFrame(
                {
                    "feature": self.feature_names_,
                    "coefficient": coefficients,
                    "odds_ratio_per_std": np.exp(coefficients),
                    "abs_coefficient": np.abs(coefficients),
                }
            )
            .sort_values("abs_coefficient", ascending=False)
            .drop(columns="abs_coefficient")
            .reset_index(drop=True)
        )
