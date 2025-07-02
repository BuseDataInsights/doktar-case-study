from __future__ import annotations

from pathlib import Path
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

def load_dataset(path: str | Path, feature: str, target: str) -> pd.DataFrame:
    """Read the Excel file and return the two required columns only."""
    df = pd.read_excel(path, engine="openpyxl")
    df = df[[feature, target]].dropna()
    return df

def split(df: pd.DataFrame, feature: str, target: str, *, test_size: float, random_state: int):
    X = df[[feature]].values  # 2‑D array (n, 1)
    y = df[target].values
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
