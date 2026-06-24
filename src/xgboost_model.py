# src/xgboost_model.py

import numpy as np
import xgboost as xgb

from sklearn.metrics import roc_auc_score, average_precision_score, log_loss


class XGBoostPDModel:
    def __init__(
        self,
        learning_rate: float = 0.05,
        max_depth: int = 5,
        min_child_weight: float = 1.0,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_lambda: float = 1.0,
        reg_alpha: float = 0.0,
        n_estimators: int = 500,
        random_state: int = 42,
        device: str = "cuda",
    ):
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.device = device
        self.model = None

    def _build_model(self, scale_pos_weight: float):
        return xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_child_weight=self.min_child_weight,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda,
            reg_alpha=self.reg_alpha,
            n_estimators=self.n_estimators,
            scale_pos_weight=scale_pos_weight,
            tree_method="hist",
            device=self.device,
            random_state=self.random_state,
        )

    def fit(self, X_train, y_train):
        y_train = np.asarray(y_train).astype(int)

        n_pos = int((y_train == 1).sum())
        n_neg = int((y_train == 0).sum())
        scale_pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0

        self.model = self._build_model(scale_pos_weight=scale_pos_weight)
        self.model.fit(X_train, y_train)

        return self

    def predict_proba(self, X):
        if self.model is None:
            raise RuntimeError("Model must be fitted before prediction.")

        dmatrix = xgb.DMatrix(X)
        return self.model.get_booster().predict(dmatrix)

    def evaluate(self, X, y):
        y = np.asarray(y).astype(int)
        pd_hat = self.predict_proba(X)

        return {
            "roc_auc": roc_auc_score(y, pd_hat),
            "pr_auc": average_precision_score(y, pd_hat),
            "log_loss": log_loss(y, pd_hat),
        }
