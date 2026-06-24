# src/load_data.py

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "GiveMeSomeCredit"
TRAIN_DATA_PATH = DATA_DIR / "cs-training.csv"
TEST_DATA_PATH = DATA_DIR / "cs-test.csv"
TARGET_COL = "SeriousDlqin2yrs"
DROP_COLS = ["Unnamed: 0"]


@dataclass(frozen=True)
class MissingValuePreprocessor:
    impute_values: pd.Series
    indicator_columns: list[str]
    add_indicators: bool = True

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_clean = X.copy()

        if self.add_indicators:
            for col in self.indicator_columns:
                if col in X_clean.columns:
                    X_clean[f"{col}_missing"] = X_clean[col].isna().astype(int)

        impute_columns = [c for c in self.impute_values.index if c in X_clean.columns]
        X_clean[impute_columns] = X_clean[impute_columns].fillna(
            self.impute_values[impute_columns]
        )

        missing_columns = X_clean.columns[X_clean.isna().any()].tolist()
        if missing_columns:
            raise ValueError(f"Missing values remain in columns: {missing_columns}")

        return X_clean


def _resolve_data_path(data_path: str | Path) -> Path:
    path = Path(data_path)
    if path.is_absolute() or path.exists():
        return path
    return PROJECT_ROOT / path


def load_training_data(data_path: str | Path = TRAIN_DATA_PATH) -> pd.DataFrame:
    data_path = _resolve_data_path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")

    df = pd.read_csv(data_path)
    return df


def summarize_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    missing_count = df.isna().sum()
    missing_rate = missing_count / len(df)

    summary = pd.DataFrame(
        {
            "missing_count": missing_count,
            "missing_rate": missing_rate,
        }
    )
    summary = summary[summary["missing_count"] > 0]
    return summary.sort_values("missing_count", ascending=False)


def prepare_features_target(df: pd.DataFrame):
    df = df.copy()

    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)

    return X, y


def fit_missing_value_preprocessor(
    X_train: pd.DataFrame,
    add_indicators: bool = True,
) -> MissingValuePreprocessor:
    numeric_columns = X_train.select_dtypes(include="number").columns
    impute_values = X_train[numeric_columns].median()
    indicator_columns = X_train.columns[X_train.isna().any()].tolist()

    return MissingValuePreprocessor(
        impute_values=impute_values,
        indicator_columns=indicator_columns,
        add_indicators=add_indicators,
    )


def make_train_val_split(
    X,
    y,
    test_size: float = 0.2,
    random_state: int = 42,
):
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_val, y_train, y_val
