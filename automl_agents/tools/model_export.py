"""
Day-10 model persistence tools.

Produces a self-contained inference bundle — fitted model + frozen preprocessing
artifacts — so downstream consumers can call predict() on raw CSV rows without
knowing anything about how the training data was cleaned.

Design decisions (mirroring the Day-3 leakage contract):
  - fit_final_model refits the WINNING model once on the FULL selected/preprocessed
    dataset AFTER CV and Optuna have both completed.  CV folds are for honest
    evaluation only; the fold estimators are intentionally discarded by clone().
    Refitting on all available data maximises the deployed model's training signal.
  - The bundle persists PrepArtifacts alongside the model.  PrepArtifacts stores
    the fit-time statistics (imputer fills, encoder mappings, scaler parameters)
    produced by fit_preprocessor on the training data.  Applying those same
    parameters at inference time (via transform_preprocessor) guarantees
    train/inference feature parity without refitting on new data.
  - joblib (not pickle) is used for persistence because scikit-learn's docs
    recommend it for fitted estimators, and it handles the large internal numpy
    arrays inside tree ensembles (LightGBM, XGBoost, RandomForest) more
    efficiently than pickle's default serialisation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from automl_agents.tools.preprocessor import PrepArtifacts

logger = logging.getLogger(__name__)

# Bundle key names — kept as constants so callers can reference them rather than
# hard-coding string literals scattered across the codebase.
BUNDLE_KEYS = frozenset(
    {"model", "prep_artifacts", "selected_features", "target_column", "problem_type", "model_id"}
)


def fit_final_model(
    df: pd.DataFrame,
    target: str,
    model_id: str,
    problem_type: Literal["classification", "regression"],
    best_params: dict | None = None,
):
    """Fit the winning model once on the full (non-CV-split) dataset for deployment.

    Parameters
    ----------
    df:
        The *already-preprocessed and already-feature-selected* DataFrame.
        Must include the target column; ``target`` is dropped before fitting.
    target:
        Name of the target column inside ``df``.
    model_id:
        Canonical model identifier string (e.g. "LightGBM", "LightGBM (Tuned)").
        The ``(Tuned)`` suffix is stripped before looking up the model class.
    problem_type:
        "classification" or "regression".
    best_params:
        Optuna best-params dict (or None for default hyperparameters).

    Returns
    -------
    A *fitted* scikit-learn-compatible estimator.
    """
    # Strip the "(Tuned)" suffix that trainer_node appends to tuned results.
    base_model_id = model_id.replace(" (Tuned)", "").strip()

    from automl_agents.tools.training import _build_estimator  # local import avoids circular

    estimator = _build_estimator(base_model_id, problem_type, params=best_params)

    # Encode y for classification (mirrors the LabelEncoder used in run_model_battery)
    X = df.drop(columns=[target])
    y = df[target]

    if problem_type == "classification":
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(y)

    logger.info(
        f"Fitting final {base_model_id} on full dataset "
        f"({len(df)} rows, {len(X.columns)} features)..."
    )
    estimator.fit(X, y)
    logger.info(f"Final {base_model_id} fitted successfully.")
    return estimator


def save_model_bundle(
    estimator,
    prep_artifacts: PrepArtifacts,
    selected_features: list[str],
    target_column: str,
    problem_type: str,
    model_id: str,
    output_path: str | Path,
) -> Path:
    """Persist a self-contained {model + preprocessing} inference bundle via joblib.

    The bundle is a plain dict with the keys in ``BUNDLE_KEYS``.  It is
    intentionally NOT a custom class so that callers can load it in any
    environment that has joblib installed, without importing this module.

    Parameters
    ----------
    estimator:
        Fitted scikit-learn-compatible estimator (from fit_final_model).
    prep_artifacts:
        The PrepArtifacts object produced by fit_preprocessor on the training
        data.  Contains all fit-time statistics needed to reproduce the same
        transforms at inference time.
    selected_features:
        Ordered list of feature column names the model was trained on.
    target_column:
        Name of the target column (excluded from inference transforms).
    problem_type:
        "classification" or "regression".
    model_id:
        Human-readable model identifier (stored for provenance).
    output_path:
        Destination path for the .pkl file.  Parent directories are created
        if they don't already exist.

    Returns
    -------
    Resolved ``Path`` to the written bundle file.
    """
    import joblib  # pip install joblib (explicit dep in pyproject.toml from Day 10)

    bundle = {
        "model": estimator,
        "prep_artifacts": prep_artifacts,          # frozen fit-time statistics
        "selected_features": selected_features,
        "target_column": target_column,
        "problem_type": problem_type,
        "model_id": model_id,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_path)
    logger.info(f"Model bundle written to {output_path} ({output_path.stat().st_size:,} bytes)")
    return output_path.resolve()


def load_model_bundle(path: str | Path) -> dict:
    """Load a bundle produced by save_model_bundle.

    Returns the dict directly — no custom class needed.  Callers should treat
    all keys in ``BUNDLE_KEYS`` as present and access them directly.
    """
    import joblib

    bundle = joblib.load(path)
    missing = BUNDLE_KEYS - bundle.keys()
    if missing:
        raise ValueError(
            f"Loaded bundle from {path} is missing expected keys: {sorted(missing)}.  "
            "It may have been produced by an older version of Nimbus."
        )
    return bundle


def predict_from_bundle(bundle: dict, df_raw: pd.DataFrame) -> np.ndarray:
    """Raw CSV rows in → model predictions out.

    Applies the same fit-time preprocessing as the original training run
    (via ``transform_preprocessor`` with the bundle's frozen ``PrepArtifacts``)
    before calling ``model.predict()``.

    This is the single function an external caller needs in order to use a
    downloaded model.pkl without knowing anything about how the data was cleaned.

    Parameters
    ----------
    bundle:
        Dict produced by load_model_bundle (or save_model_bundle directly).
    df_raw:
        Raw, unpreprocessed DataFrame rows.  Column names must match the
        original training CSV (before any of Nimbus's preprocessing).

    Returns
    -------
    numpy.ndarray of predictions (class labels for classification, floats for
    regression).  Shape: (n_rows,).
    """
    from automl_agents.tools.preprocessor import transform_preprocessor

    prep_artifacts = bundle["prep_artifacts"]
    selected_features = bundle["selected_features"]

    # Apply the frozen fit-time transforms (no refitting, no data leakage).
    # target_column=None because the raw inference rows don't carry labels.
    df_prepped = transform_preprocessor(df_raw, prep_artifacts)

    # Restrict to the features the model was trained on (same order).
    available = [f for f in selected_features if f in df_prepped.columns]
    if len(available) < len(selected_features):
        missing_feats = set(selected_features) - set(df_prepped.columns)
        logger.warning(
            f"predict_from_bundle: {len(missing_feats)} selected feature(s) not found "
            f"after preprocessing: {sorted(missing_feats)}.  Proceeding with available features."
        )

    X_inference = df_prepped[available]
    return bundle["model"].predict(X_inference)
