"""Train all candidate models, keep the best, and save to disk."""
from __future__ import annotations

import yaml
from pathlib import Path
import joblib
from sklearn.model_selection import KFold
import numpy as np

from .data import load_dataset, split
from .models import build_models, evaluate, select_best, _fit_predict


def train_best(cfg_path: str | Path = "config.yaml", *, output: str | Path = "model.joblib") -> None:
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    df = load_dataset(cfg["data_file"], cfg["feature_col"], cfg["target_col"])
    X_train, X_val, y_train, y_val = split(
        df,
        cfg["feature_col"],
        cfg["target_col"],
        test_size=cfg["test_size"],
        random_state=cfg["random_state"],
    )

    models = build_models(cfg["models"])
    results = evaluate(models, X_train, X_val, y_train, y_val)
    best_name, best_rmse = select_best(results)
    best_model = models[best_name]

    # Retrain on the *full* data
    _fit_predict(best_model, df[[cfg["feature_col"]]].values, df[cfg["target_col"]].values, df[[cfg["feature_col"]]].values)  # type: ignore[arg-type]
    joblib.dump(best_model, output)
    print(f"Saved {best_name} with CV‑RMSE {best_rmse:.3f} → {output}")


if __name__ == "__main__":
    train_best()
