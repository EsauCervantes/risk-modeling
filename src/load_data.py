# src/data/loader.py

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


TARGET_COL = "SeriousDlqin2yrs"
DROP_COLS = ["Unnamed: 0"]


def load_training_data(data_path: str | Path) -> pd.DataFrame:
    data_path = Path(data_path)
    df = pd.read_csv(data_path)
    return df


def prepare_features_target(df: pd.DataFrame):
    df = df.copy()

    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)

    return X, y


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
