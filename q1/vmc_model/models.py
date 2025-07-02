from __future__ import annotations

import joblib
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from typing import Dict, Tuple

__all__ = ["build_models", "evaluate", "select_best"]

def build_models(cfg: dict) -> Dict[str, BaseEstimator]:
    """Instantiate candidate models from the YAML config."""
    models = {
        "linear": LinearRegression(),
        "poly2": Pipeline([
            ("poly", PolynomialFeatures(degree=cfg.get("poly2", {}).get("degree", 2), include_bias=False)),
            ("lin", LinearRegression()),
        ]),
        "isotonic": IsotonicRegression(out_of_bounds=cfg.get("isotonic", {}).get("out_of_bounds", "clip")),
        "gbdt": GradientBoostingRegressor(**cfg.get("gbdt", {})),
    }
    return models


def _fit_predict(model: BaseEstimator, X_train, y_train, X_val):
    """Helper because IsotonicRegression expects 1‑D arrays."""
    if isinstance(model, IsotonicRegression):
        model.fit(X_train.ravel(), y_train)
        preds = model.predict(X_val.ravel())
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
    return preds


def evaluate(models: Dict[str, BaseEstimator], X_train, X_val, y_train, y_val) -> Dict[str, Dict[str, float]]:
    """Return a metrics dict keyed by model name."""
    results: Dict[str, Dict[str, float]] = {}
    for name, mdl in models.items():
        preds = _fit_predict(mdl, X_train, y_train, X_val)
        # Calculate RMSE manually for older sklearn versions
        mse = mean_squared_error(y_val, preds)
        rmse = np.sqrt(mse)
        
        results[name] = {
            "rmse": rmse,
            "mae": mean_absolute_error(y_val, preds),
            "r2": r2_score(y_val, preds),
        }
    return results


def select_best(results: Dict[str, Dict[str, float]]) -> Tuple[str, float]:
    best = min(results.items(), key=lambda kv: kv[1]["rmse"])
    return best[0], best[1]["rmse"]
