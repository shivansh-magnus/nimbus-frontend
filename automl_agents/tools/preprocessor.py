"""
Day-3 deterministic data prep tools.

Pure, stateless helper functions plus a fit/transform pair that stores all
learned statistics in ``PrepArtifacts`` so that Day-5 graph nodes can call
``fit_preprocessor`` on the training split and ``transform_preprocessor`` on
the test split without any data leakage.

Leakage contract (stated explicitly per roadmap §2.3–2.4):
  - ``fit_preprocessor`` / ``fit_encoders`` / ``fit_scalers`` MUST be called on
    the training split only.  They compute and store statistics inside
    ``PrepArtifacts``; they never look at the held-out data.
  - ``transform_preprocessor`` / ``transform_encoders`` / ``transform_scalers``
    NEVER refit.  They only apply the statistics stored in ``PrepArtifacts``.
  - The target column is sacred: it is never imputed, encoded, or scaled.
    Only rows where the target is null are dropped.
  - ``PrepArtifacts`` stores only small statistics objects (dicts of numbers /
    strings), never raw dataframes.  No raw dataframes flow through
    ``PipelineState`` (Day-5 rule); parquet paths are used instead.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# PrepArtifacts — the serialisable output of the fit phase
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class PrepArtifacts:
    """Stores all statistics learned during the fit phase.

    Only small scalar / dict-of-scalars objects live here; never raw
    dataframes.  Day-5 graph state will carry a path to a parquet snapshot,
    not this object directly — but the object can be pickled for convenience
    during local testing.
    """

    # Columns that were dropped during fit (so transform can skip them)
    dropped_columns: list[str] = dataclasses.field(default_factory=list)

    # Datetime columns that were parsed + expanded (original col dropped)
    datetime_columns_expanded: list[str] = dataclasses.field(default_factory=list)

    # Imputation fill values: {col: fill_value}
    imputer_fills: dict[str, float | str] = dataclasses.field(default_factory=dict)

    # Encoder state: {col: {"strategy": str, "mapping": dict}}
    #   onehot   → mapping unused; new cols added with pd.get_dummies fit categories
    #   ordinal  → {category: int_code}
    #   target   → {category: float mean-target}
    encoder_state: dict[str, dict] = dataclasses.field(default_factory=dict)

    # OHE column order learned during fit (so transform produces same schema)
    ohe_feature_names: list[str] = dataclasses.field(default_factory=list)

    # Scaler params: {col: {"strategy": str, "center": float, "scale": float}}
    scaler_params: dict[str, dict] = dataclasses.field(default_factory=dict)

    # Outlier clip bounds learned during fit: {col: {"lower": float, "upper": float}}
    clip_bounds: dict[str, dict] = dataclasses.field(default_factory=dict)

    # The target column used during fit (for leakage checks in transform)
    target_column: str = ""


# ---------------------------------------------------------------------------
# Column / row removal
# ---------------------------------------------------------------------------

def drop_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Drop an explicit list of columns (silently skips missing ones)."""
    to_drop = [c for c in cols if c in df.columns]
    return df.drop(columns=to_drop)


def drop_all_null_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop any column whose missing fraction is exactly 1.0."""
    mask = df.isna().all()
    return df.drop(columns=df.columns[mask].tolist())


def drop_single_category_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop any column with ≤ 1 distinct non-null value (zero-variance)."""
    to_drop = [col for col in df.columns if df[col].nunique(dropna=True) <= 1]
    return df.drop(columns=to_drop)


def drop_rows_with_null_target(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Drop rows where the target column is null.

    The target is sacred — this is the only operation we perform on it
    before modelling.  We never impute, encode, or scale the target.
    """
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' not found.")
    return df.dropna(subset=[target_column]).reset_index(drop=True)


def dedupe_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows (keep first occurrence)."""
    return df.drop_duplicates().reset_index(drop=True)


# ---------------------------------------------------------------------------
# Dtype coercion
# ---------------------------------------------------------------------------

def coerce_mixed_numeric(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Coerce a mixed-type column to float; non-parseable tokens become NaN.

    Use for columns like ``credit_score_text`` that contain mostly numeric
    strings with occasional invalid tokens (e.g. "N/A", "unknown").
    """
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found.")
    df = df.copy()
    df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def coerce_numeric_string(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Coerce an all-numeric string column to float.

    Assumes every non-null value is parseable as a number.  Use
    ``coerce_mixed_numeric`` when invalid tokens are possible.
    """
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found.")
    df = df.copy()
    df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def parse_datetime_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Parse a string column into a proper datetime64 column in-place."""
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found.")
    df = df.copy()
    df[col] = pd.to_datetime(df[col], errors="coerce", format="mixed")
    return df


def add_datetime_features(
    df: pd.DataFrame,
    col: str,
    *,
    drop_original: bool = True,
) -> pd.DataFrame:
    """Extract year / month / day / day-of-week features from a datetime column.

    The original column is dropped by default (pass ``drop_original=False`` to
    keep it).  If the column is a string it is parsed first.
    """
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found.")
    df = df.copy()

    # Parse strings automatically
    if not pd.api.types.is_datetime64_any_dtype(df[col]):
        df[col] = pd.to_datetime(df[col], errors="coerce", format="mixed")

    df[f"{col}_year"] = df[col].dt.year.astype("float64")
    df[f"{col}_month"] = df[col].dt.month.astype("float64")
    df[f"{col}_day"] = df[col].dt.day.astype("float64")
    df[f"{col}_dayofweek"] = df[col].dt.dayofweek.astype("float64")

    if drop_original:
        df = df.drop(columns=[col])
    return df


# ---------------------------------------------------------------------------
# Imputation
# ---------------------------------------------------------------------------

ImputeStrategy = Literal["mean", "median", "mode", "constant"]


def impute_column(
    df: pd.DataFrame,
    col: str,
    strategy: ImputeStrategy = "median",
    *,
    fill_value: float | str | None = None,
) -> tuple[pd.DataFrame, float | str]:
    """Impute nulls in *col* using *strategy*; returns (new_df, fill_value_used).

    Leakage note: call this on the TRAINING split only.  The returned
    ``fill_value`` should be stored in ``PrepArtifacts.imputer_fills`` and
    used verbatim in ``transform_preprocessor``.

    Strategies:
        mean     — numeric mean of non-null values
        median   — numeric median of non-null values
        mode     — most frequent non-null value (works for object/numeric)
        constant — use the provided ``fill_value``
    """
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found.")

    df = df.copy()
    series = df[col]

    if strategy == "mean":
        fv: float | str = float(series.mean())
    elif strategy == "median":
        fv = float(series.median())
    elif strategy == "mode":
        mode_vals = series.mode(dropna=True)
        if mode_vals.empty:
            fv = 0.0
        else:
            raw = mode_vals.iloc[0]
            fv = float(raw) if pd.api.types.is_numeric_dtype(series) else str(raw)
    elif strategy == "constant":
        if fill_value is None:
            raise ValueError("fill_value must be provided when strategy='constant'.")
        fv = fill_value
    else:
        raise ValueError(f"Unknown imputation strategy: '{strategy}'")

    df[col] = series.fillna(fv)
    return df, fv


def apply_imputer_fill(
    df: pd.DataFrame,
    col: str,
    fill_value: float | str,
) -> pd.DataFrame:
    """Apply a pre-computed fill value (from PrepArtifacts) without refitting."""
    if col not in df.columns:
        return df
    df = df.copy()
    df[col] = df[col].fillna(fill_value)
    return df


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

EncoderStrategy = Literal["onehot", "ordinal", "target"]


def fit_encoders(
    df: pd.DataFrame,
    cols: list[str],
    strategy: EncoderStrategy,
    *,
    target_col: str | None = None,
) -> dict[str, dict]:
    """Fit encoding mappings from training data.

    Leakage note: call on the TRAINING split only.  The returned state dict
    must be stored in ``PrepArtifacts.encoder_state`` and passed to
    ``transform_encoders``.

    Returns a dict:  {col: {"strategy": str, "mapping": dict | None}}
    """
    if strategy == "target" and target_col is None:
        raise ValueError("target_col is required for strategy='target'.")

    state: dict[str, dict] = {}

    for col in cols:
        if col not in df.columns:
            continue

        if strategy == "onehot":
            # Store the unique categories seen during training
            categories = sorted(df[col].dropna().unique().tolist())
            state[col] = {"strategy": "onehot", "categories": [str(c) for c in categories]}

        elif strategy == "ordinal":
            categories = sorted(df[col].dropna().unique().tolist())
            mapping = {str(cat): idx for idx, cat in enumerate(categories)}
            state[col] = {"strategy": "ordinal", "mapping": mapping}

        elif strategy == "target":
            assert target_col is not None  # already checked above
            if target_col not in df.columns:
                raise KeyError(f"Target column '{target_col}' not found.")
            # Mean-target encoding — encode with training mean per category
            group_means = (
                df.groupby(col, observed=True)[target_col]
                .mean()
                .to_dict()
            )
            global_mean = float(df[target_col].mean())
            mapping = {str(k): float(v) for k, v in group_means.items()}
            state[col] = {
                "strategy": "target",
                "mapping": mapping,
                "global_mean": global_mean,  # fallback for unseen categories
            }

    return state


def transform_encoders(
    df: pd.DataFrame,
    encoder_state: dict[str, dict],
    *,
    ohe_feature_names: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Apply pre-fitted encodings; never refits.

    Returns (transformed_df, updated_ohe_feature_names).
    ``ohe_feature_names`` is passed in (from PrepArtifacts) so OHE columns
    are always aligned with the training schema.
    """
    df = df.copy()
    new_ohe_names: list[str] = list(ohe_feature_names or [])

    for col, info in encoder_state.items():
        if col not in df.columns:
            continue
        strat = info["strategy"]

        if strat == "onehot":
            categories = info["categories"]
            dummies = pd.get_dummies(df[col].astype(str), prefix=col)
            # Align to training schema: add missing cols, drop extra cols
            # Use pd.concat (not per-column insert) to avoid DataFrame fragmentation.
            missing = {
                f"{col}_{cat}": np.zeros(len(dummies), dtype=np.uint8)
                for cat in categories
                if f"{col}_{cat}" not in dummies.columns
            }
            if missing:
                dummies = pd.concat(
                    [dummies, pd.DataFrame(missing, index=dummies.index)], axis=1
                )
            # Keep only the columns seen during training
            train_ohe_cols = [f"{col}_{cat}" for cat in categories]
            dummies = dummies[[c for c in train_ohe_cols if c in dummies.columns]]
            dummies = dummies.astype(np.uint8)
            df = df.drop(columns=[col])
            df = pd.concat([df, dummies], axis=1)
            for c in dummies.columns:
                if c not in new_ohe_names:
                    new_ohe_names.append(c)

        elif strat == "ordinal":
            mapping = info["mapping"]
            # Unknown categories get -1
            df[col] = df[col].astype(str).map(mapping).fillna(-1).astype(float)

        elif strat == "target":
            mapping = info["mapping"]
            global_mean = info.get("global_mean", 0.0)
            df[col] = (
                df[col].astype(str).map(mapping).fillna(global_mean).astype(float)
            )

    return df, new_ohe_names


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------

ScalerStrategy = Literal["standard", "minmax", "robust"]


def fit_scalers(
    df: pd.DataFrame,
    cols: list[str],
    strategy: ScalerStrategy = "standard",
) -> dict[str, dict]:
    """Compute scaling parameters from training data.

    Leakage note: call on the TRAINING split only.  Store the returned dict
    in ``PrepArtifacts.scaler_params`` and pass to ``transform_scalers``.

    Returns: {col: {"strategy": str, "center": float, "scale": float}}
    """
    params: dict[str, dict] = {}

    for col in cols:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            continue

        if strategy == "standard":
            center = float(series.mean())
            scale = float(series.std(ddof=1))
            if scale == 0.0:
                scale = 1.0
        elif strategy == "minmax":
            center = float(series.min())
            scale = float(series.max() - series.min())
            if scale == 0.0:
                scale = 1.0
        elif strategy == "robust":
            center = float(series.median())
            q1, q3 = float(series.quantile(0.25)), float(series.quantile(0.75))
            scale = q3 - q1
            if scale == 0.0:
                scale = 1.0
        else:
            raise ValueError(f"Unknown scaler strategy: '{strategy}'")

        params[col] = {"strategy": strategy, "center": center, "scale": scale}

    return params


def transform_scalers(
    df: pd.DataFrame,
    scaler_params: dict[str, dict],
) -> pd.DataFrame:
    """Apply pre-fitted scaler parameters; never refits."""
    df = df.copy()
    for col, params in scaler_params.items():
        if col not in df.columns:
            continue
        center = params["center"]
        scale = params["scale"]
        df[col] = (pd.to_numeric(df[col], errors="coerce").astype("float64") - center) / scale
    return df


# ---------------------------------------------------------------------------
# Outlier clipping
# ---------------------------------------------------------------------------

def clip_outliers_iqr(
    df: pd.DataFrame,
    col: str,
    *,
    k: float = 1.5,
) -> tuple[pd.DataFrame, dict]:
    """Clip *col* to [Q1 − k·IQR, Q3 + k·IQR]; returns (df, bounds_dict).

    Leakage note: compute bounds on training data only; store in
    ``PrepArtifacts.clip_bounds`` and pass to ``apply_clip_bounds`` for
    the test split.
    """
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found.")
    df = df.copy()
    # Cast to float64 explicitly — bool/uint8 dtypes cause numpy boolean-
    # subtract TypeError inside Series.quantile on some NumPy versions.
    series = pd.to_numeric(df[col], errors="coerce").astype("float64")
    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    df[col] = series.clip(lower=lower, upper=upper)
    return df, {"lower": lower, "upper": upper}


def clip_outliers_percentile(
    df: pd.DataFrame,
    col: str,
    *,
    lower_pct: float = 0.01,
    upper_pct: float = 0.99,
) -> tuple[pd.DataFrame, dict]:
    """Clip *col* to [lower_pct, upper_pct] percentiles; returns (df, bounds_dict).

    Leakage note: same as ``clip_outliers_iqr`` — compute on training data
    only and store the returned bounds.
    """
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found.")
    df = df.copy()
    series = pd.to_numeric(df[col], errors="coerce").astype("float64")
    lower = float(series.quantile(lower_pct))
    upper = float(series.quantile(upper_pct))
    df[col] = series.clip(lower=lower, upper=upper)
    return df, {"lower": lower, "upper": upper}


def apply_clip_bounds(
    df: pd.DataFrame,
    col: str,
    bounds: dict,
) -> pd.DataFrame:
    """Apply pre-computed clip bounds (from PrepArtifacts) without refitting."""
    if col not in df.columns:
        return df
    df = df.copy()
    df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64").clip(
        lower=bounds["lower"], upper=bounds["upper"]
    )
    return df


# ---------------------------------------------------------------------------
# Orchestration — default pipeline
# ---------------------------------------------------------------------------

#: Maximum number of unique categories before a column is dropped instead of
#: one-hot encoded.  Prevents encoding high-cardinality ID columns.
_MAX_OHE_CARDINALITY: int = 50

@dataclasses.dataclass
class PrepConfig:
    """Optional configuration for the default ``fit_preprocessor`` pipeline.

    Day-6 agents will emit a structured PrepConfig to override these defaults.
    Today we provide sensible defaults so Day-3 integration tests can run.
    """

    # Columns to drop before any other step (e.g. leaky columns, IDs)
    drop_cols: list[str] = dataclasses.field(default_factory=list)

    # Columns known to be datetime strings that need parsing + expansion
    datetime_cols: list[str] = dataclasses.field(default_factory=list)

    # Columns to coerce from mixed-type string to numeric
    mixed_numeric_cols: list[str] = dataclasses.field(default_factory=list)

    # Imputation: {col: strategy} — "mean"/"median"/"mode"/"constant"
    impute: dict[str, ImputeStrategy] = dataclasses.field(default_factory=dict)

    # Encoding: {col: strategy} — "onehot"/"ordinal"/"target"
    encode: dict[str, EncoderStrategy] = dataclasses.field(default_factory=dict)

    # Scaling strategy applied to all remaining numeric cols after encoding
    scale_strategy: ScalerStrategy = "standard"

    # IQR clip multiplier for outlier treatment (None = skip clipping)
    iqr_k: float | None = 1.5


def _infer_default_config(df: pd.DataFrame, target_column: str) -> PrepConfig:
    """Build a sensible default PrepConfig by inspecting the dataframe.

    This is used by ``fit_preprocessor`` when no explicit config is given.
    Agents will override this on Day 6 via structured output.
    """
    drop_cols: list[str] = []
    datetime_cols: list[str] = []
    mixed_numeric_cols: list[str] = []
    impute: dict[str, ImputeStrategy] = {}
    encode: dict[str, EncoderStrategy] = {}

    for col in df.columns:
        if col == target_column:
            continue

        series = df[col]
        missing_frac = float(series.isna().mean())

        # Detect dtype category
        if pd.api.types.is_datetime64_any_dtype(series):
            datetime_cols.append(col)
            continue

        if pd.api.types.is_numeric_dtype(series):
            if missing_frac > 0:
                impute[col] = "median"
            continue

        # Object / string column
        non_null = series.dropna().astype(str)
        if non_null.empty:
            drop_cols.append(col)
            continue

        as_num = pd.to_numeric(non_null, errors="coerce")
        numeric_frac = float(as_num.notna().mean())

        if numeric_frac > 0:
            # Mixed or fully-numeric string → coerce
            mixed_numeric_cols.append(col)
            if missing_frac > 0 or numeric_frac < 1.0:
                impute[col] = "median"
            continue

        # Try datetime
        parsed = pd.to_datetime(non_null, errors="coerce", format="mixed")
        if float(parsed.notna().mean()) >= 0.9:
            datetime_cols.append(col)
            continue

        # Categorical — guard against high-cardinality ID-like columns
        n_unique = series.nunique(dropna=True)
        if n_unique <= 1:
            drop_cols.append(col)
        elif n_unique > _MAX_OHE_CARDINALITY:
            # Too many categories for OHE (e.g. customer IDs) — drop them.
            drop_cols.append(col)
        else:
            encode[col] = "onehot"
            if missing_frac > 0:
                impute[col] = "mode"

    return PrepConfig(
        drop_cols=drop_cols,
        datetime_cols=datetime_cols,
        mixed_numeric_cols=mixed_numeric_cols,
        impute=impute,
        encode=encode,
        scale_strategy="standard",
        iqr_k=1.5,
    )


def fit_preprocessor(
    df: pd.DataFrame,
    target_column: str,
    *,
    config: PrepConfig | None = None,
) -> PrepArtifacts:
    """Fit the preprocessing pipeline on training data only.

    Leakage contract: this function MUST be called on the training split only.
    All statistics it computes (imputer fills, encoder mappings, scaler params,
    clip bounds) are stored in the returned ``PrepArtifacts`` object and must
    be used verbatim when transforming the held-out (test / validation) data.

    The target column is never imputed, encoded, or scaled.
    """
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' not found.")

    arts = PrepArtifacts(target_column=target_column)

    # --- step 0: drop rows with null target (so stats aren't polluted) ---
    df = drop_rows_with_null_target(df, target_column)

    # --- step 1: auto-infer config if not provided ---
    if config is None:
        config = _infer_default_config(df, target_column)

    # --- step 2: structural drops (all-null, single-category, explicit) ---
    all_null_cols = df.columns[df.isna().all()].tolist()
    single_cat_cols = [
        c for c in df.columns
        if c != target_column and df[c].nunique(dropna=True) <= 1
    ]
    arts.dropped_columns = list(
        dict.fromkeys(all_null_cols + single_cat_cols + config.drop_cols)
    )
    df = drop_columns(df, arts.dropped_columns)

    # --- step 3: deduplicate ---
    df = dedupe_rows(df)

    # --- step 4: datetime coercion & feature extraction ---
    for col in config.datetime_cols:
        if col in df.columns:
            df = add_datetime_features(df, col, drop_original=True)
            arts.datetime_columns_expanded.append(col)

    # Auto-register datetime-derived feature cols for imputation so NaNs
    # inherited from the original null datetimes are filled in step 6.
    for src_col in arts.datetime_columns_expanded:
        for feat in [
            f"{src_col}_year", f"{src_col}_month",
            f"{src_col}_day", f"{src_col}_dayofweek",
        ]:
            if feat in df.columns and feat not in config.impute:
                config.impute[feat] = "median"

    # --- step 5: mixed-numeric coercion ---
    for col in config.mixed_numeric_cols:
        if col in df.columns:
            df = coerce_mixed_numeric(df, col)

    # --- step 6: imputation (learn fills from training data) ---
    for col, strategy in config.impute.items():
        if col not in df.columns or col == target_column:
            continue
        df, fv = impute_column(df, col, strategy)
        arts.imputer_fills[col] = fv

    # --- step 7: encoding (learn mappings from training data) ---
    ohe_cols = [c for c, s in config.encode.items() if s == "onehot" and c in df.columns]
    ordinal_cols = [c for c, s in config.encode.items() if s == "ordinal" and c in df.columns]
    target_enc_cols = [c for c, s in config.encode.items() if s == "target" and c in df.columns]

    if ohe_cols:
        arts.encoder_state.update(fit_encoders(df, ohe_cols, "onehot"))
    if ordinal_cols:
        arts.encoder_state.update(fit_encoders(df, ordinal_cols, "ordinal"))
    if target_enc_cols:
        arts.encoder_state.update(
            fit_encoders(df, target_enc_cols, "target", target_col=target_column)
        )

    # Apply encoders to the training df so scaler stats are correct
    if arts.encoder_state:
        df, arts.ohe_feature_names = transform_encoders(
            df, arts.encoder_state, ohe_feature_names=[]
        )

    # --- step 8: outlier clipping (learn bounds from training data) ---
    # Skip bool columns (e.g. OHE uint8 are fine; pure bool would cause
    # numpy boolean-subtract TypeError in quantile).
    numeric_cols = [
        c for c in df.columns
        if c != target_column
        and pd.api.types.is_numeric_dtype(df[c])
        and not pd.api.types.is_bool_dtype(df[c])
    ]
    if config.iqr_k is not None:
        for col in numeric_cols:
            df, bounds = clip_outliers_iqr(df, col, k=config.iqr_k)
            arts.clip_bounds[col] = bounds

    # --- step 9: scaling (learn params from post-clip training data) ---
    scale_cols = [
        c for c in df.columns
        if c != target_column
        and pd.api.types.is_numeric_dtype(df[c])
        and not pd.api.types.is_bool_dtype(df[c])
    ]
    arts.scaler_params = fit_scalers(df, scale_cols, config.scale_strategy)

    return arts


def transform_preprocessor(
    df: pd.DataFrame,
    artifacts: PrepArtifacts,
) -> pd.DataFrame:
    """Apply the preprocessing pipeline using pre-fitted artifacts.

    Never refits any statistics — all values come from ``artifacts`` which
    were computed on the training split only.  Safe to call on the test split.

    The target column is left completely untouched.
    """
    target_column = artifacts.target_column

    # --- step 1: drop null-target rows ---
    if target_column and target_column in df.columns:
        df = drop_rows_with_null_target(df, target_column)

    # --- step 2: drop columns learned during fit ---
    df = drop_columns(df, artifacts.dropped_columns)

    # --- step 3: deduplicate ---
    df = dedupe_rows(df)

    # --- step 4: datetime expansion ---
    for col in artifacts.datetime_columns_expanded:
        if col in df.columns:
            df = add_datetime_features(df, col, drop_original=True)

    # --- step 5: mixed-numeric coercion (re-applied; no leakage since it's
    # just pd.to_numeric) ---
    for col in list(df.columns):
        if col == target_column:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            non_null = df[col].dropna().astype(str)
            if not non_null.empty:
                as_num = pd.to_numeric(non_null, errors="coerce")
                if float(as_num.notna().mean()) > 0:
                    df = coerce_mixed_numeric(df, col)

    # --- step 6: apply imputer fills (no refitting) ---
    for col, fv in artifacts.imputer_fills.items():
        df = apply_imputer_fill(df, col, fv)

    # --- step 7: apply encoders (no refitting) ---
    if artifacts.encoder_state:
        df, _ = transform_encoders(
            df, artifacts.encoder_state,
            ohe_feature_names=artifacts.ohe_feature_names,
        )

    # --- step 8: apply clip bounds (no refitting) ---
    for col, bounds in artifacts.clip_bounds.items():
        df = apply_clip_bounds(df, col, bounds)

    # --- step 9: apply scaler params (no refitting) ---
    df = transform_scalers(df, artifacts.scaler_params)

    return df


def prep_dataframe(
    df: pd.DataFrame,
    target_column: str,
    *,
    config: PrepConfig | None = None,
) -> pd.DataFrame:
    """Convenience: fit on *df* and immediately transform it.

    Use for smoke tests and single-split workflows only.  In Day-5 graph nodes,
    always use ``fit_preprocessor`` on the training split and
    ``transform_preprocessor`` on the test split to avoid leakage.
    """
    artifacts = fit_preprocessor(df, target_column, config=config)
    return transform_preprocessor(df, artifacts)


# ---------------------------------------------------------------------------
# Parquet snapshot helpers (Day-5 graph state integration)
# ---------------------------------------------------------------------------

def save_parquet_snapshot(df: pd.DataFrame, path: str | Path) -> Path:
    """Save *df* to a Parquet file.

    Converts any remaining object columns to strings before writing so that
    pyarrow does not choke on mixed Python objects.

    Returns the resolved path that was written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert object columns that aren't already string to string to ensure
    # round-trip compatibility.
    df = df.copy()
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str)

    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path)
    return path.resolve()


def load_parquet_snapshot(path: str | Path) -> pd.DataFrame:
    """Load a Parquet file written by ``save_parquet_snapshot``."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet snapshot not found: {path}")
    return pq.read_table(path).to_pandas()
