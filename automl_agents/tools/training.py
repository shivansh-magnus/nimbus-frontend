"""
Day-4 deterministic model battery and training tools.

Supports stratified/k-fold CV across a battery of standard classification
and regression algorithms. Preprocessed inputs are assumed to be numeric.
"""

from __future__ import annotations

import logging
import warnings
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC, SVR


# Suppress warnings from models (like ConvergenceWarning in LogisticRegression)
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# Import gradient boosting packages safely
try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:
    XGBClassifier, XGBRegressor = None, None

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError:
    LGBMClassifier, LGBMRegressor = None, None


def _get_classification_models() -> dict:
    """Return dictionary of classification model constructors/instances."""
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(random_state=42, n_estimators=100, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(random_state=42, n_estimators=100),
        "SVM": CalibratedClassifierCV(SVC(random_state=42), ensemble=False),
        "KNN": KNeighborsClassifier(),
    }
    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(random_state=42, eval_metric="logloss", n_jobs=-1)
    if LGBMClassifier is not None:
        models["LightGBM"] = LGBMClassifier(random_state=42, verbose=-1, n_jobs=-1)
    return models


def _get_regression_models() -> dict:
    """Return dictionary of regression model constructors/instances."""
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(random_state=42, n_estimators=100, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(random_state=42, n_estimators=100),
        "SVR": SVR(),
        "KNN": KNeighborsRegressor(),
    }
    if XGBRegressor is not None:
        models["XGBoost"] = XGBRegressor(random_state=42, n_jobs=-1)
    if LGBMRegressor is not None:
        models["LightGBM"] = LGBMRegressor(random_state=42, verbose=-1, n_jobs=-1)
    return models


def _build_estimator(
    model_id: str,
    problem_type: Literal["classification", "regression"],
    params: dict | None = None,
):
    """Construct an unfitted estimator for *model_id* using the given params.

    This is the single authoritative model-name-to-class mapping for the whole
    project.  Both ``tune_model`` (which builds candidates per Optuna trial) and
    ``model_export.fit_final_model`` (which builds the winning model once for
    deployment) call this helper so the mapping is never duplicated.

    ``params`` mirrors the key names returned by Optuna's ``study.best_params``.
    If ``params`` is None or missing a key, sensible defaults are used.

    Returns a scikit-learn-compatible unfitted estimator (``BaseEstimator`` API).
    Raises ``ValueError`` for unknown ``model_id`` strings.
    """
    p = params or {}

    if model_id in ("LogisticRegression",):
        return LogisticRegression(C=p.get("C", 1.0), max_iter=1000, random_state=42)

    if model_id == "LinearRegression":
        return LinearRegression(fit_intercept=p.get("fit_intercept", True))

    if model_id == "RandomForest":
        kw = dict(
            n_estimators=p.get("n_estimators", 100),
            max_depth=p.get("max_depth", None),
            random_state=42,
            n_jobs=-1,
        )
        if problem_type == "classification":
            return RandomForestClassifier(**kw)
        return RandomForestRegressor(**kw)

    if model_id == "GradientBoosting":
        kw = dict(
            learning_rate=p.get("learning_rate", 0.1),
            n_estimators=p.get("n_estimators", 100),
            max_depth=p.get("max_depth", 3),
            random_state=42,
        )
        if problem_type == "classification":
            return GradientBoostingClassifier(**kw)
        return GradientBoostingRegressor(**kw)

    if model_id == "XGBoost":
        kw = dict(
            learning_rate=p.get("learning_rate", 0.1),
            n_estimators=p.get("n_estimators", 100),
            max_depth=p.get("max_depth", 6),
            random_state=42,
            n_jobs=-1,
        )
        if problem_type == "classification":
            if XGBClassifier is None:
                raise ImportError("xgboost is not installed.")
            return XGBClassifier(eval_metric="logloss", **kw)
        if XGBRegressor is None:
            raise ImportError("xgboost is not installed.")
        return XGBRegressor(**kw)

    if model_id == "LightGBM":
        kw = dict(
            learning_rate=p.get("learning_rate", 0.1),
            n_estimators=p.get("n_estimators", 100),
            max_depth=p.get("max_depth", -1),
            num_leaves=p.get("num_leaves", 31),
            random_state=42,
            verbose=-1,
            n_jobs=-1,
        )
        if problem_type == "classification":
            if LGBMClassifier is None:
                raise ImportError("lightgbm is not installed.")
            return LGBMClassifier(**kw)
        if LGBMRegressor is None:
            raise ImportError("lightgbm is not installed.")
        return LGBMRegressor(**kw)

    if model_id in ("SVM", "SVR"):
        kw = dict(C=p.get("C", 1.0), gamma=p.get("gamma", "scale"))
        if problem_type == "classification":
            return CalibratedClassifierCV(SVC(random_state=42, **kw), ensemble=False)
        return SVR(**kw)

    if model_id == "KNN":
        kw = dict(
            n_neighbors=p.get("n_neighbors", 5),
            weights=p.get("weights", "uniform"),
        )
        if problem_type == "classification":
            return KNeighborsClassifier(**kw)
        return KNeighborsRegressor(**kw)

    raise ValueError(
        f"Unknown model_id '{model_id}'. Valid values: LogisticRegression, LinearRegression, "
        "RandomForest, GradientBoosting, XGBoost, LightGBM, SVM, SVR, KNN."
    )


def run_model_battery(
    df: pd.DataFrame,
    target: str,
    problem_type: Literal["classification", "regression"],
    cv: int = 5,
) -> list[dict]:
    """Run CV across the model battery; return score metrics for each model.

    Target column is dropped from features. Remaining features are assumed to
    be numeric. Any residual missing values (which shouldn't be present after
    preprocessor.py, but could exist) are temporarily imputed using SimpleImputer.
    """
    features = [c for c in df.columns if c != target]
    if not features:
        raise ValueError("No features available to train the model battery.")

    # Convert features to numeric, impute any NaN placeholders
    X = df[features].copy()
    X_numeric = X.select_dtypes(include=[np.number])
    if X_numeric.shape[1] < X.shape[1]:
        logger.warning("Non-numeric features detected in training battery; keeping numeric features only.")
        X = X_numeric

    y = df[target]

    # Ensure clean numeric data for models
    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Initialize CV splitters
    if problem_type == "classification":
        label_encoder = LabelEncoder()
        y = pd.Series(label_encoder.fit_transform(y), index=y.index, name=target)
        splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        models = _get_classification_models()
        # Determine average parameter for f1_score based on classes
        unique_classes = np.unique(y)
        f1_avg = "binary" if len(unique_classes) <= 2 else "macro"
    else:
        splitter = KFold(n_splits=cv, shuffle=True, random_state=42)
        models = _get_regression_models()

    results = []

    for model_name, model in models.items():
        logger.info(f"Training {model_name}...")
        
        fold_scores = []
        try:
            for train_idx, val_idx in splitter.split(X_imputed, y):
                X_train, X_val = X_imputed.iloc[train_idx], X_imputed.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                # Train model
                # Make a clean clone/new instance of the estimator
                from sklearn.base import clone
                estimator = clone(model)
                estimator.fit(X_train, y_train)

                # Predict & Evaluate
                y_pred = estimator.predict(X_val)

                if problem_type == "classification":
                    acc = float(accuracy_score(y_val, y_pred))
                    f1 = float(f1_score(y_val, y_pred, average=f1_avg))
                    fold_scores.append({"accuracy": acc, "f1": f1})
                else:
                    mae = float(mean_absolute_error(y_val, y_pred))
                    mse = float(mean_squared_error(y_val, y_pred))
                    rmse = float(np.sqrt(mse))
                    r2 = float(r2_score(y_val, y_pred))
                    fold_scores.append({"mae": mae, "rmse": rmse, "r2": r2})

            # Aggregate scores across folds
            metrics = list(fold_scores[0].keys())
            model_metrics = {"scores": {}, "mean_scores": {}, "std_scores": {}}

            for m in metrics:
                vals = [f[m] for f in fold_scores]
                model_metrics["scores"][m] = vals
                model_metrics["mean_scores"][m] = float(np.mean(vals))
                model_metrics["std_scores"][m] = float(np.std(vals))

            results.append({
                "model_id": model_name,
                "scores": model_metrics["scores"],
                "mean_scores": model_metrics["mean_scores"],
                "std_scores": model_metrics["std_scores"],
            })

        except Exception as e:
            logger.error(f"Failed to run model battery for {model_name}: {e}", exc_info=True)
            # Do not crash the entire run, continue to the next model

    return results


def tune_model(
    df: pd.DataFrame,
    target: str,
    model_id: str,
    problem_type: Literal["classification", "regression"],
    metric: str,
    cv: int = 3,
    n_trials: int = 10,
) -> dict | None:
    """Tune hyperparameters of a single model using Optuna inside CV training folds."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    features = [c for c in df.columns if c != target]
    if not features:
        return None

    X = df[features].copy()
    y = df[target]

    # Impute missing values just in case
    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    if problem_type == "classification":
        label_encoder = LabelEncoder()
        y = pd.Series(label_encoder.fit_transform(y), index=y.index, name=target)
        splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        unique_classes = np.unique(y)
        f1_avg = "binary" if len(unique_classes) <= 2 else "macro"
    else:
        splitter = KFold(n_splits=cv, shuffle=True, random_state=42)

    def objective(trial):
        if model_id == "LogisticRegression":
            C = trial.suggest_float("C", 1e-4, 1e2, log=True)
            model = LogisticRegression(C=C, max_iter=1000, random_state=42)
        elif model_id == "LinearRegression":
            fit_intercept = trial.suggest_categorical("fit_intercept", [True, False])
            model = LinearRegression(fit_intercept=fit_intercept)
        elif model_id == "RandomForest":
            n_estimators = trial.suggest_int("n_estimators", 50, 200)
            max_depth = trial.suggest_int("max_depth", 3, 12)
            if problem_type == "classification":
                model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
            else:
                model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
        elif model_id == "GradientBoosting":
            learning_rate = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
            n_estimators = trial.suggest_int("n_estimators", 50, 200)
            max_depth = trial.suggest_int("max_depth", 3, 8)
            if problem_type == "classification":
                model = GradientBoostingClassifier(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            else:
                model = GradientBoostingRegressor(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        elif model_id == "XGBoost":
            if XGBClassifier is None:
                return 0.0
            learning_rate = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
            n_estimators = trial.suggest_int("n_estimators", 50, 200)
            max_depth = trial.suggest_int("max_depth", 3, 8)
            if problem_type == "classification":
                model = XGBClassifier(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth, random_state=42, eval_metric="logloss", n_jobs=-1)
            else:
                model = XGBRegressor(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
        elif model_id == "LightGBM":
            if LGBMClassifier is None:
                return 0.0
            learning_rate = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
            n_estimators = trial.suggest_int("n_estimators", 50, 200)
            max_depth = trial.suggest_int("max_depth", 3, 10)
            num_leaves = trial.suggest_int("num_leaves", 15, 127)
            if problem_type == "classification":
                model = LGBMClassifier(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth, num_leaves=num_leaves, random_state=42, verbose=-1, n_jobs=-1)
            else:
                model = LGBMRegressor(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth, num_leaves=num_leaves, random_state=42, verbose=-1, n_jobs=-1)
        elif model_id == "SVM" or model_id == "SVR":
            C = trial.suggest_float("C", 1e-3, 1e2, log=True)
            gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
            if problem_type == "classification":
                model = CalibratedClassifierCV(SVC(C=C, gamma=gamma, random_state=42), ensemble=False)
            else:
                model = SVR(C=C, gamma=gamma)
        elif model_id == "KNN":
            n_neighbors = trial.suggest_int("n_neighbors", 3, 15)
            weights = trial.suggest_categorical("weights", ["uniform", "distance"])
            if problem_type == "classification":
                model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
            else:
                model = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights)
        else:
            return 0.0

        fold_scores = []
        try:
            for train_idx, val_idx in splitter.split(X_imputed, y):
                X_train, X_val = X_imputed.iloc[train_idx], X_imputed.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                from sklearn.base import clone
                estimator = clone(model)
                estimator.fit(X_train, y_train)
                y_pred = estimator.predict(X_val)

                if problem_type == "classification":
                    if metric == "accuracy":
                        fold_scores.append(float(accuracy_score(y_val, y_pred)))
                    else:
                        fold_scores.append(float(f1_score(y_val, y_pred, average=f1_avg)))
                else:
                    if metric == "mae":
                        fold_scores.append(float(mean_absolute_error(y_val, y_pred)))
                    elif metric == "rmse":
                        fold_scores.append(float(np.sqrt(mean_squared_error(y_val, y_pred))))
                    else:
                        fold_scores.append(float(r2_score(y_val, y_pred)))
            return float(np.mean(fold_scores))
        except Exception:
            return float("inf") if metric in ["rmse", "mae"] else float("-inf")

    is_smaller_better = metric in ["rmse", "mae"]
    direction = "minimize" if is_smaller_better else "maximize"

    study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    logger.info(f"Optuna completed tuning for {model_id}. Best params: {best_params}")

    # Re-evaluate with the best model configuration to collect all scores
    if model_id == "LogisticRegression":
        best_model = LogisticRegression(C=best_params["C"], max_iter=1000, random_state=42)
    elif model_id == "LinearRegression":
        best_model = LinearRegression(fit_intercept=best_params["fit_intercept"])
    elif model_id == "RandomForest":
        if problem_type == "classification":
            best_model = RandomForestClassifier(n_estimators=best_params["n_estimators"], max_depth=best_params["max_depth"], random_state=42, n_jobs=-1)
        else:
            best_model = RandomForestRegressor(n_estimators=best_params["n_estimators"], max_depth=best_params["max_depth"], random_state=42, n_jobs=-1)
    elif model_id == "GradientBoosting":
        if problem_type == "classification":
            best_model = GradientBoostingClassifier(learning_rate=best_params["learning_rate"], n_estimators=best_params["n_estimators"], max_depth=best_params["max_depth"], random_state=42)
        else:
            best_model = GradientBoostingRegressor(learning_rate=best_params["learning_rate"], n_estimators=best_params["n_estimators"], max_depth=best_params["max_depth"], random_state=42)
    elif model_id == "XGBoost":
        if problem_type == "classification":
            best_model = XGBClassifier(learning_rate=best_params["learning_rate"], n_estimators=best_params["n_estimators"], max_depth=best_params["max_depth"], random_state=42, eval_metric="logloss", n_jobs=-1)
        else:
            best_model = XGBRegressor(learning_rate=best_params["learning_rate"], n_estimators=best_params["n_estimators"], max_depth=best_params["max_depth"], random_state=42, n_jobs=-1)
    elif model_id == "LightGBM":
        if problem_type == "classification":
            best_model = LGBMClassifier(learning_rate=best_params["learning_rate"], n_estimators=best_params["n_estimators"], max_depth=best_params["max_depth"], num_leaves=best_params["num_leaves"], random_state=42, verbose=-1, n_jobs=-1)
        else:
            best_model = LGBMRegressor(learning_rate=best_params["learning_rate"], n_estimators=best_params["n_estimators"], max_depth=best_params["max_depth"], num_leaves=best_params["num_leaves"], random_state=42, verbose=-1, n_jobs=-1)
    elif model_id == "SVM" or model_id == "SVR":
        if problem_type == "classification":
            best_model = CalibratedClassifierCV(SVC(C=best_params["C"], gamma=best_params["gamma"], random_state=42), ensemble=False)
        else:
            best_model = SVR(C=best_params["C"], gamma=best_params["gamma"])
    elif model_id == "KNN":
        if problem_type == "classification":
            best_model = KNeighborsClassifier(n_neighbors=best_params["n_neighbors"], weights=best_params["weights"])
        else:
            best_model = KNeighborsRegressor(n_neighbors=best_params["n_neighbors"], weights=best_params["weights"])
    else:
        return None

    # Perform full evaluation
    full_fold_scores = []
    for train_idx, val_idx in splitter.split(X_imputed, y):
        X_train, X_val = X_imputed.iloc[train_idx], X_imputed.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        from sklearn.base import clone
        estimator = clone(best_model)
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_val)

        if problem_type == "classification":
            acc = float(accuracy_score(y_val, y_pred))
            f1 = float(f1_score(y_val, y_pred, average=f1_avg))
            full_fold_scores.append({"accuracy": acc, "f1": f1})
        else:
            mae = float(mean_absolute_error(y_val, y_pred))
            mse = float(mean_squared_error(y_val, y_pred))
            rmse = float(np.sqrt(mse))
            r2 = float(r2_score(y_val, y_pred))
            full_fold_scores.append({"mae": mae, "rmse": rmse, "r2": r2})

    metrics = list(full_fold_scores[0].keys())
    model_metrics = {"scores": {}, "mean_scores": {}, "std_scores": {}}
    for m in metrics:
        vals = [f[m] for f in full_fold_scores]
        model_metrics["scores"][m] = vals
        model_metrics["mean_scores"][m] = float(np.mean(vals))
        model_metrics["std_scores"][m] = float(np.std(vals))

    return {
        "model_id": f"{model_id} (Tuned)",
        "scores": model_metrics["scores"],
        "mean_scores": model_metrics["mean_scores"],
        "std_scores": model_metrics["std_scores"],
        "best_params": best_params,
    }

