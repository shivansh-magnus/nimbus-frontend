"""
Day-6 LangGraph node for agentic feature selection using LLM structured decisions.
"""

from __future__ import annotations

import logging
from typing import Literal
from pydantic import BaseModel, Field
from langgraph.runtime import Runtime

from automl_agents.schemas import PipelineState, RunConfig, StageLogEntry
from automl_agents.tools.preprocessor import load_parquet_snapshot
from automl_agents.tools.selection import run_selection
from automl_agents.llm_client import get_llm, llm_retry_decorator
from automl_agents.llm_util import record_token_usage

logger = logging.getLogger(__name__)


class SelectorDecision(BaseModel):
    """Structured LLM output for the feature selection strategy."""

    method: Literal["variance", "correlation", "mutual_info", "rf_importance", "none"] = Field(
        description="Feature selection method to apply to the preprocessed dataset.",
    )
    k: float = Field(
        default=0.8,
        description="Fraction of top features to keep (between 0.0 and 1.0) for mutual_info or rf_importance.",
    )
    threshold: float = Field(
        default=0.0,
        description="Threshold for variance (strict minimum variance) or correlation (maximum absolute correlation threshold).",
    )
    rationale: str = Field(
        description="Detailed explanation of why this feature selection method and parameters were chosen based on the dataset.",
    )


def selector_node(state: PipelineState, runtime: Runtime[RunConfig]) -> dict:
    """Agentic selector node: queries the LLM for a selection strategy, then runs it."""
    logger.info("Starting feature selection...")
    cleaned_path = state["cleaned_data_path"]
    target_column = state["target_column"]
    eda_report = state["eda_report"]

    if not cleaned_path or not eda_report:
        missing = "cleaned_data_path" if not cleaned_path else "eda_report"
        log_entry: StageLogEntry = {
            "stage": "selector",
            "status": "failed",
            "message": f"Feature selection skipped: upstream '{missing}' is missing (data_prep likely failed).",
        }
        return {"stage_log": [log_entry]}

    # Get runtime config for LLM
    context = runtime.context
    provider = context.get("llm_provider", "gemini")
    model = context.get("model_name")

    try:
        # Load preprocessed snapshot
        df = load_parquet_snapshot(cleaned_path)
        problem_type = eda_report.problem_type
        prepped_features = [col for col in df.columns if col != target_column]

        # Query LLM for selection decision
        system_prompt = (
            "You are a Lead Data Scientist choosing a feature selection strategy.\n"
            "Review the list of preprocessed feature columns, the target column, and the problem type.\n"
            "Choose from the following methods:\n"
            "- 'variance': drop zero-variance / low-variance features (configure threshold).\n"
            "- 'correlation': drop collinear feature pairs (configure threshold).\n"
            "- 'mutual_info': keep top k fraction based on Mutual Information (configure k).\n"
            "- 'rf_importance': keep top k fraction based on Random Forest feature importances (configure k).\n"
            "- 'none': keep all features.\n"
            "Prefer 'rf_importance' or 'mutual_info' with a k around 0.7-0.9 if there are many features, or 'none' if features are few and high-quality."
        )
        user_prompt = (
            f"Problem Type: {problem_type}\n"
            f"Target Column: {target_column}\n"
            f"Preprocessed Feature Columns ({len(prepped_features)} total):\n"
            f"{', '.join(prepped_features)}"
        )

        llm = get_llm(provider=provider, model=model)
        structured_llm = llm.with_structured_output(SelectorDecision, include_raw=True)

        logger.info(f"Querying Feature Selector Agent using provider={provider}, model={model}...")
        response = llm_retry_decorator(structured_llm.invoke)([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ])

        decision = response["parsed"]  # type: ignore[index]
        raw_msg = response["raw"]  # type: ignore[index]

        # Run feature selection
        selected_features = run_selection(
            df,
            target_column,
            method=decision.method,
            problem_type=problem_type,
            k=decision.k,
            threshold=decision.threshold,
        )

        logger.info(f"Feature Selector Agent selected {len(selected_features)} features: {selected_features}")

        # Record token usage
        token_entry = record_token_usage("selector", provider, model or "default", raw_msg)

        log_entry: StageLogEntry = {
            "stage": "selector",
            "status": "ok",
            "message": f"Selected {len(selected_features)} features using {decision.method}. Rationale: {decision.rationale}",
        }

        return {
            "selected_features": selected_features,
            "selection_rationale": decision.rationale,
            "stage_log": [log_entry],
            "token_usage": [token_entry],
        }

    except Exception as e:
        logger.error(f"Error during feature selection: {e}", exc_info=True)
        log_entry: StageLogEntry = {
            "stage": "selector",
            "status": "failed",
            "message": f"Feature selection failed: {str(e)}",
        }
        return {
            "stage_log": [log_entry],
        }
