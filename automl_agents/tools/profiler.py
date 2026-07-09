"""
Day-2 deterministic profiling tools.

Pure functions that measure dataset structure. No LLM calls, no graph state
mutations. Agents will reason over the returned EDAReport on Day 6.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from automl_agents.schemas import (
    ClassProportion,
    ColumnProfile,
    CorrelationPair,
    EDAReport,
)

CLASSIFICATION_MAX_UNIQUE = 20
CLASSIFICATION_MAX_UNIQUE_FRACTION = 0.05


def load_csv(path: str | Path) -> pd.DataFrame:
    """Read a CSV and validate it is non-empty."""
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Dataset at {path} is empty.")
    if len(df.columns) == 0:
        raise ValueError(f"Dataset at {path} has no columns.")
    return df


def infer_dtypes(df: pd.DataFrame) -> dict[str, str]:
    """Per-column semantic dtype labels, including mixed-type detection."""
    return {col: _classify_column_dtype(df[col]) for col in df.columns}


def _classify_column_dtype(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "bool"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime64"
    if pd.api.types.is_numeric_dtype(series):
        return str(series.dtype)

    non_null = series.dropna().astype(str)
    if non_null.empty:
        return "object"

    as_num = pd.to_numeric(non_null, errors="coerce")
    numeric_frac = float(as_num.notna().mean())

    if 0.0 < numeric_frac < 1.0:
        return "mixed_numeric"
    if numeric_frac >= 1.0:
        return "numeric_string"

    parsed_dates = pd.to_datetime(non_null, errors="coerce", format="mixed")
    if float(parsed_dates.notna().mean()) >= 0.9:
        return "datetime_string"

    return "object"


def compute_missingness(df: pd.DataFrame) -> dict[str, float]:
    """Column -> fraction of rows that are missing (0.0-1.0)."""
    n = len(df)
    if n == 0:
        return {col: 0.0 for col in df.columns}
    return {col: float(df[col].isna().mean()) for col in df.columns}


def compute_cardinality(df: pd.DataFrame) -> dict[str, int]:
    """Column -> number of unique non-null values."""
    return {col: int(df[col].nunique(dropna=True)) for col in df.columns}


def detect_problem_type(
    df: pd.DataFrame,
    target_column: str,
    *,
    max_unique: int = CLASSIFICATION_MAX_UNIQUE,
    max_unique_fraction: float = CLASSIFICATION_MAX_UNIQUE_FRACTION,
) -> Literal["classification", "regression"]:
    """Infer supervised problem type from the target column."""
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' not found in dataset.")

    target = df[target_column].dropna()
    if target.empty:
        raise ValueError(f"Target column '{target_column}' is entirely missing.")

    n = len(target)
    n_unique = target.nunique(dropna=True)

    if not pd.api.types.is_numeric_dtype(target):
        return "classification"

    if n_unique <= max_unique or n_unique / n <= max_unique_fraction:
        return "classification"

    return "regression"


def compute_target_balance(
    df: pd.DataFrame,
    target_column: str,
    problem_type: Literal["classification", "regression"] | None = None,
) -> list[ClassProportion] | None:
    """Class label proportions for classification; None for regression."""
    if problem_type is None:
        problem_type = detect_problem_type(df, target_column)
    if problem_type == "regression":
        return None

    counts = df[target_column].value_counts(dropna=True)
    total = float(counts.sum())
    if total == 0:
        return []

    return [
        ClassProportion(label=str(label), proportion=float(count / total))
        for label, count in counts.items()
    ]


def _numeric_series_for_correlation(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    return pd.to_numeric(series, errors="coerce")


def compute_correlation_flags(
    df: pd.DataFrame,
    target_column: str,
    *,
    threshold: float = 0.9,
) -> list[CorrelationPair]:
    """Flag column pairs (including vs target) with |correlation| >= threshold."""
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' not found in dataset.")

    numeric_cols: list[str] = []
    numeric_data: dict[str, pd.Series] = {}

    for col in df.columns:
        coerced = _numeric_series_for_correlation(df[col])
        if coerced.notna().sum() >= 2:
            numeric_cols.append(col)
            numeric_data[col] = coerced

    if target_column not in numeric_cols:
        return []

    matrix = pd.DataFrame(numeric_data).corr()
    flagged: list[CorrelationPair] = []
    seen: set[tuple[str, str]] = set()

    for col_a in numeric_cols:
        for col_b in numeric_cols:
            if col_a >= col_b:
                continue
            corr = matrix.loc[col_a, col_b]
            if pd.isna(corr) or abs(corr) < threshold:
                continue
            pair = (col_a, col_b)
            if pair in seen:
                continue
            seen.add(pair)
            flagged.append(
                CorrelationPair(col_a=col_a, col_b=col_b, corr=round(float(corr), 6))
            )

    flagged.sort(key=lambda p: abs(p.corr), reverse=True)
    return flagged


def detect_outliers_iqr(
    df: pd.DataFrame,
    column: str,
    *,
    k: float = 1.5,
) -> dict[str, float | int | str]:
    """IQR outlier bounds and count for one numeric column."""
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in dataset.")

    series = _numeric_series_for_correlation(df[column]).dropna()
    if series.empty:
        return {
            "column": column,
            "lower_bound": float("nan"),
            "upper_bound": float("nan"),
            "n_outliers": 0,
            "outlier_fraction": 0.0,
        }

    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr

    mask = (series < lower) | (series > upper)
    n_outliers = int(mask.sum())
    n_rows = len(df)

    return {
        "column": column,
        "lower_bound": lower,
        "upper_bound": upper,
        "n_outliers": n_outliers,
        "outlier_fraction": n_outliers / n_rows if n_rows else 0.0,
    }


def profile_dataframe(df: pd.DataFrame, target_column: str) -> EDAReport:
    """Assemble a full EDAReport from an in-memory dataframe."""
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' not found in dataset.")

    dtypes = infer_dtypes(df)
    missingness = compute_missingness(df)
    cardinality = compute_cardinality(df)
    problem_type = detect_problem_type(df, target_column)

    columns = [
        ColumnProfile(
            column=col,
            dtype=dtypes[col],
            missing_fraction=missingness[col],
            cardinality=cardinality[col],
        )
        for col in df.columns
    ]

    return EDAReport(
        n_rows=len(df),
        n_cols=len(df.columns),
        columns=columns,
        problem_type=problem_type,
        target_balance=compute_target_balance(df, target_column, problem_type),
        correlations_flagged=compute_correlation_flags(df, target_column),
        concerns=[],
    )


def profile_dataset(path: str | Path, target_column: str) -> EDAReport:
    """Load a CSV from disk and return a structured EDA report."""
    return profile_dataframe(load_csv(path), target_column)


def _json_default(obj: object) -> object:
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def main() -> int:
    """CLI entrypoint: profile a CSV and print the EDAReport as JSON."""
    import sys

    if len(sys.argv) != 3:
        print(f"Usage: python -m automl_agents.tools.profiler <csv_path> <target_column>")
        return 1

    report = profile_dataset(sys.argv[1], sys.argv[2])
    print(json.dumps(report.model_dump(), indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
