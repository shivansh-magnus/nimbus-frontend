"""
Day-6 LangGraph node for agentic dataset preprocessing using LLM structured prep plans.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field
from langgraph.runtime import Runtime

from automl_agents.schemas import PipelineState, RunConfig, StageLogEntry
from automl_agents.tools.profiler import load_csv
from automl_agents.tools.preprocessor import (
    fit_preprocessor,
    transform_preprocessor,
    save_parquet_snapshot,
    PrepConfig,
)
from automl_agents.llm_client import get_llm, llm_retry_decorator
from automl_agents.llm_util import record_token_usage

logger = logging.getLogger(__name__)


class ColumnPrepAction(BaseModel):
    """Preprocessing action choice for a single column."""

    column: str = Field(description="Name of the column to configure.")
    impute: Literal["mean", "median", "mode", "constant", "none"] = Field(
        default="none",
        description="Imputation strategy if the column has missing values.",
    )
    encode: Literal["onehot", "ordinal", "target", "none"] = Field(
        default="none",
        description="Encoding strategy for categorical/object columns.",
    )


class PrepPlanSchema(BaseModel):
    """Structured LLM output for the global preprocessing plan."""

    drop_cols: list[str] = Field(
        default_factory=list,
        description="Columns that should be completely dropped from the dataset (e.g. IDs, target duplicates, leaky features).",
    )
    datetime_cols: list[str] = Field(
        default_factory=list,
        description="Columns containing date/time strings that need parsing and feature extraction.",
    )
    mixed_numeric_cols: list[str] = Field(
        default_factory=list,
        description="Columns with mixed text/numeric types (e.g., '100USD' or 'credit_score_600') to coerce to numeric.",
    )
    column_actions: list[ColumnPrepAction] = Field(
        default_factory=list,
        description="Per-column actions (imputation, encoding) for all other features. Do not include target column.",
    )
    scale_strategy: Literal["standard", "minmax", "robust"] = Field(
        default="standard",
        description="Scaling strategy for all remaining numeric features.",
    )
    iqr_k: float | None = Field(
        default=1.5,
        description="IQR outlier clipping multiplier (e.g., 1.5). Use None to skip outlier clipping.",
    )
    custom_code: str | None = Field(
        default=None,
        description="Optional Python code snippet using pandas (dataframe variable is 'df') to perform custom transformations. Use ONLY when standard tools (impute, encode, scale) are insufficient.",
    )


def prep_node(state: PipelineState, runtime: Runtime[RunConfig]) -> dict:
    """Agentic preprocessing node: gets a custom prep plan from LLM, then applies it."""
    logger.info("Starting dataset preprocessing...")
    csv_path = state["dataset_path"]
    target_column = state["target_column"]
    eda_report = state["eda_report"]

    if not eda_report:
        log_entry: StageLogEntry = {
            "stage": "data_prep",
            "status": "failed",
            "message": "Preprocessing skipped: upstream 'eda_report' is missing (profiler likely failed).",
        }
        return {"stage_log": [log_entry]}

    # Get runtime config for LLM
    context = runtime.context
    provider = context.get("llm_provider", "gemini")
    model = context.get("model_name")
    run_id = context.get("run_id", "default_run")

    try:
        # Step 1: Load original CSV
        df = load_csv(csv_path)

        # Step 2: Query LLM for custom prep plan based on EDA Report & concerns
        system_prompt = (
            "You are a Lead Data Scientist designing a dataset cleaning and preprocessing pipeline.\n"
            "Review the statistical profile of columns, target distribution, and highly correlated pairs.\n"
            "Pay close attention to the list of concerns (especially potential target leakage or target duplicates, which MUST be dropped).\n"
            "Determine which columns to drop completely, which columns are datetime or mixed numeric types, "
            "and the imputation / encoding action for all other feature columns.\n"
            "Do NOT apply any actions to the target column.\n\n"
            "IMPORTANT — execution order: drop_cols, datetime_cols, mixed_numeric_cols, impute, encode, and "
            "scaling are all applied BEFORE custom_code runs. By the time custom_code executes, any column "
            "listed in datetime_cols has already been parsed into <col>_year/_month/_day/_dayofweek and the "
            "raw column has already been DROPPED. Any column in mixed_numeric_cols has already been coerced "
            "to numeric. Never reference a column inside custom_code that you already declared in drop_cols, "
            "datetime_cols, or mixed_numeric_cols — choose exactly one mechanism per column. Only use "
            "custom_code for logic the structured fields above cannot express (e.g. combining two columns "
            "into a ratio, or parsing a format not covered by the standard tools)."
        )
        user_prompt = (
            f"Target Column: {target_column}\n\n"
            f"EDA Report Concerns:\n{chr(10).join(f'- {c}' for c in eda_report.concerns) if eda_report.concerns else 'None'}\n\n"
            f"EDA Report Columns:\n"
        )
        for col in eda_report.columns:
            if col.column != target_column:
                user_prompt += f"- `{col.column}`: type={col.dtype}, missingness={col.missing_fraction:.2%}, cardinality={col.cardinality}\n"

        llm = get_llm(provider=provider, model=model)
        structured_llm = llm.with_structured_output(PrepPlanSchema, include_raw=True)

        logger.info(f"Querying DataPrep Agent using provider={provider}, model={model}...")
        response = llm_retry_decorator(structured_llm.invoke)([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ])

        plan = response["parsed"]  # type: ignore[index]
        raw_msg = response["raw"]  # type: ignore[index]

        # Step 3: Map PrepPlanSchema back into PrepConfig format
        impute_dict = {}
        encode_dict = {}
        for action in plan.column_actions:
            if action.impute != "none" and action.column != target_column:
                impute_dict[action.column] = action.impute
            if action.encode != "none" and action.column != target_column:
                encode_dict[action.column] = action.encode

        config = PrepConfig(
            drop_cols=plan.drop_cols,
            datetime_cols=plan.datetime_cols,
            mixed_numeric_cols=plan.mixed_numeric_cols,
            impute=impute_dict,
            encode=encode_dict,
            scale_strategy=plan.scale_strategy,
            iqr_k=plan.iqr_k,
        )

        logger.info(f"DataPrep Agent chose config: {config}")

        # Step 4: Run fit/transform
        artifacts = fit_preprocessor(df, target_column, config=config)
        df_prepped = transform_preprocessor(df, artifacts)

        # Apply custom code sandbox if provided by agent
        if plan.custom_code:
            logger.info(f"Custom code detected from agent:\n{plan.custom_code}")
            from automl_agents.tools.custom_transform import run_custom_transform_sandboxed
            df_prepped = run_custom_transform_sandboxed(plan.custom_code, df_prepped)

        # Step 5: Save Parquet snapshot
        runs_dir = Path("runs") / run_id
        runs_dir.mkdir(parents=True, exist_ok=True)
        cleaned_parquet_path = runs_dir / "02_cleaned.parquet"
        save_parquet_snapshot(df_prepped, cleaned_parquet_path)

        # Step 6: Record token usage
        token_entry = record_token_usage("data_prep", provider, model or "default", raw_msg)

        log_entry: StageLogEntry = {
            "stage": "data_prep",
            "status": "ok",
            "message": f"Preprocessed successfully. Dropped {len(config.drop_cols)} columns. Saved parquet snapshot. Shape is {df_prepped.shape}.",
        }

        # prep_plan in PipelineState expects a dict (serialize dataclass)
        serialized_plan = dataclasses.asdict(config)
        serialized_plan["custom_code"] = plan.custom_code

        return {
            "cleaned_data_path": str(cleaned_parquet_path.resolve()),
            "prep_plan": serialized_plan,
            "stage_log": [log_entry],
            "token_usage": [token_entry],
        }

    except Exception as e:
        logger.error(f"Error during preprocessing node: {e}", exc_info=True)
        log_entry: StageLogEntry = {
            "stage": "data_prep",
            "status": "failed",
            "message": f"Preprocessing failed: {str(e)}",
        }
        return {
            "stage_log": [log_entry],
        }
