"""Small prediction adapter for the saved Random Forest artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from src.ui.data import BASELINE_MODEL, FEATURE_COLUMNS, OPTIMIZED_MODEL

MODEL_ARTIFACTS = {
    "selected": BASELINE_MODEL,
    "candidate": OPTIMIZED_MODEL,
}


class Predictor(Protocol):
    """The estimator surface needed for one-row inference."""

    n_features_in_: int

    def predict(self, features: np.ndarray) -> np.ndarray: ...


def load_saved_model(variant: str = "selected") -> RandomForestRegressor:
    """Load and validate one of the existing local Random Forest artifacts."""
    try:
        artifact_path = MODEL_ARTIFACTS[variant]
    except KeyError as error:
        raise ValueError(
            f"Unknown model variant {variant!r}; expected one of "
            f"{sorted(MODEL_ARTIFACTS)}"
        ) from error
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {artifact_path}")
    model = joblib.load(artifact_path)
    if not isinstance(model, RandomForestRegressor):
        raise TypeError(f"Model artifact is not a RandomForestRegressor: {artifact_path}")
    if model.n_features_in_ != len(FEATURE_COLUMNS):
        raise ValueError(
            f"Expected a {len(FEATURE_COLUMNS)}-feature model, "
            f"found {model.n_features_in_}: {artifact_path}"
        )
    return model


def predict_soc_proxy(model: Predictor, features: Mapping[str, float]) -> float:
    """Predict one proxy percentage with explicit feature ordering and checks."""
    try:
        ordered = [float(features[column]) for column in FEATURE_COLUMNS]
    except KeyError as error:
        raise ValueError(f"Missing model input feature: {error.args[0]}") from error
    except (TypeError, ValueError) as error:
        raise ValueError("All model inputs must be numeric") from error
    values = np.asarray(ordered, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("All model inputs must be finite")
    if model.n_features_in_ != len(FEATURE_COLUMNS):
        raise ValueError(
            f"Expected a {len(FEATURE_COLUMNS)}-feature model, "
            f"found {model.n_features_in_}"
        )
    prediction = np.asarray(model.predict(values.reshape(1, -1)), dtype=float)
    if prediction.shape != (1,) or not np.isfinite(prediction[0]):
        raise ValueError("Model must return exactly one finite proxy prediction")
    return float(prediction[0])
