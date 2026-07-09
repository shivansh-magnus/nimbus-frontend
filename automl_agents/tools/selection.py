"""
Day-4 deterministic feature selection tools.

Pure functions that rank or prune features based on variance, correlation,
mutual information, or random forest importance. No LLM calls, no graph state.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.impute import SimpleImputer


def _determine_k(n_features: int, k: int | float) -> int:
    """Helper to convert absolute or fractional k value to an integer count."""
    if isinstance(k, float) and 0.0 < k < 1.0:
        return max(1, int(round(k * n_features)))
    elif isinstance(k, (int, float)) and k >= 1:
        return min(int(k), n_features)
    else:
        return n_features


def variance_threshold_pruning(
    df: pd.DataFrame,
    target: str,
    threshold: float = 0.0,
) -> list[str]:
    """Keep columns with variance strictly greater than threshold.

    Only applies to numeric columns; non-numeric columns are kept.
    """
    features = [c for c in df.columns if c != target]
    numeric_cols = df[features].select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        return features

    variances = df[numeric_cols].var()
    selected_numeric = variances[variances > threshold].index.tolist()
    non_numeric = [c for c in features if c not in numeric_cols]
    
    return selected_numeric + non_numeric


def correlation_pruning(
    df: pd.DataFrame,
    target: str,
    threshold: float = 0.9,
) -> list[str]:
    """Drop highly correlated feature pairs to reduce collinearity.

    For any pair of numeric columns with absolute correlation > threshold,
    the feature with the higher column index (later in the dataframe) is dropped.
    """
    features = [c for c in df.columns if c != target]
    numeric_cols = df[features].select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) <= 1:
        return features

    corr_matrix = df[numeric_cols].corr().abs()
    dropped = set()
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            col_i = corr_matrix.columns[i]
            col_j = corr_matrix.columns[j]
            if col_i in dropped or col_j in dropped:
                continue
            if corr_matrix.iloc[i, j] > threshold:
                dropped.add(col_i)

    selected_numeric = [c for c in numeric_cols if c not in dropped]
    non_numeric = [c for c in features if c not in numeric_cols]
    
    return selected_numeric + non_numeric


def mutual_info_pruning(
    df: pd.DataFrame,
    target: str,
    problem_type: str,
    k: int | float = 0.5,
) -> list[str]:
    """Select the top k (or top fraction k) features based on Mutual Information."""
    features = [c for c in df.columns if c != target]
    if not features:
        return []

    numeric_cols = df[features].select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return features

    X = df[numeric_cols].copy()
    y = df[target]

    # Temporarily impute NaNs for sklearn compatibility
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    if problem_type == "classification":
        mi_scores = mutual_info_classif(X_imputed, y, random_state=42)
    else:
        mi_scores = mutual_info_regression(X_imputed, y, random_state=42)

    feat_scores = pd.Series(mi_scores, index=numeric_cols).sort_values(ascending=False)
    n_select = _determine_k(len(numeric_cols), k)
    selected_numeric = feat_scores.index[:n_select].tolist()
    
    non_numeric = [c for c in features if c not in numeric_cols]
    
    return selected_numeric + non_numeric


def rf_importance_pruning(
    df: pd.DataFrame,
    target: str,
    problem_type: str,
    k: int | float = 0.5,
) -> list[str]:
    """Select the top k (or top fraction k) features based on Random Forest feature importances."""
    features = [c for c in df.columns if c != target]
    if not features:
        return []

    numeric_cols = df[features].select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return features

    X = df[numeric_cols].copy()
    y = df[target]

    # Temporarily impute NaNs for sklearn compatibility
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    if problem_type == "classification":
        rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    else:
        rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)

    rf.fit(X_imputed, y)
    importances = rf.feature_importances_

    feat_scores = pd.Series(importances, index=numeric_cols).sort_values(ascending=False)
    n_select = _determine_k(len(numeric_cols), k)
    selected_numeric = feat_scores.index[:n_select].tolist()

    non_numeric = [c for c in features if c not in numeric_cols]

    return selected_numeric + non_numeric


def run_selection(
    df: pd.DataFrame,
    target: str,
    method: Literal["variance", "correlation", "mutual_info", "rf_importance", "none"],
    problem_type: str = "classification",
    **params,
) -> list[str]:
    """Orchestrate the feature selection phase and return selected feature names."""
    if method == "variance":
        threshold = params.get("threshold", 0.0)
        return variance_threshold_pruning(df, target, threshold)
    elif method == "correlation":
        threshold = params.get("threshold", 0.9)
        return correlation_pruning(df, target, threshold)
    elif method == "mutual_info":
        k = params.get("k", 0.5)
        return mutual_info_pruning(df, target, problem_type, k)
    elif method == "rf_importance":
        k = params.get("k", 0.5)
        return rf_importance_pruning(df, target, problem_type, k)
    elif method == "none":
        return [c for c in df.columns if c != target]
    else:
        raise ValueError(f"Unknown feature selection method: '{method}'")
