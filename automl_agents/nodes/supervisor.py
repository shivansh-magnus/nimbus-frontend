"""
Day-7 Retry Supervisor node to handle training-time validation failures.
"""

from __future__ import annotations

import logging
from pydantic import BaseModel, Field
from langgraph.runtime import Runtime

from automl_agents.schemas import PipelineState, RunConfig, StageLogEntry
from automl_agents.llm_client import get_llm, llm_retry_decorator
from automl_agents.llm_util import record_token_usage

logger = logging.getLogger(__name__)


class SupervisorDecision(BaseModel):
    """Structured LLM output for the Retry Supervisor."""

    columns_to_drop: list[str] = Field(
        default_factory=list,
        description="Leaky feature columns to drop from the dataset to resolve the target leakage/validation errors.",
    )
    explanation: str = Field(
        description="Detailed explanation explaining how the target leakage/validation error was diagnosed and why these columns are being dropped.",
    )


def retry_supervisor_node(state: PipelineState, runtime: Runtime[RunConfig]) -> dict:
    """Agentic Retry Supervisor: analyzes training-time validation errors to resolve target leakage."""
    logger.info("Starting Retry Supervisor analysis...")

    validation_errors = state.get("validation_errors") or []
    target_column = state["target_column"]
    selected_features = state.get("selected_features") or []
    eda_report = state.get("eda_report")

    if not validation_errors:
        logger.info("No validation errors to resolve. Skipping supervisor.")
        return {}

    # Get runtime config for LLM
    context = runtime.context
    provider = context.get("llm_provider", "gemini")
    model = context.get("model_name")

    # Increment retry_count for "data_prep"
    retry_count = dict(state.get("retry_count") or {})
    retry_count["data_prep"] = retry_count.get("data_prep", 0) + 1

    try:
        system_prompt = (
            "You are a Lead Data Scientist acting as a Retry Supervisor in a machine learning pipeline.\n"
            "Your job is to analyze training-time validation failures (such as target leakage or perfect cross-validation scores) "
            "and determine which feature(s) should be dropped to resolve the leakage.\n"
            "Review the list of validation errors, the target column, the selected features, and historical concerns.\n"
            "Identify the offending feature column(s) and output them in the columns_to_drop list, along with a detailed explanation."
        )

        user_prompt = (
            f"Target Column: {target_column}\n"
            f"Selected Features: {selected_features}\n"
            f"Validation Errors:\n"
            f"{chr(10).join(f'- {err}' for err in validation_errors)}\n\n"
            f"Historical Concerns:\n"
            f"{chr(10).join(f'- {c}' for c in eda_report.concerns) if eda_report and eda_report.concerns else 'None'}"
        )

        llm = get_llm(provider=provider, model=model)
        structured_llm = llm.with_structured_output(SupervisorDecision, include_raw=True)

        logger.info(f"Querying Retry Supervisor Agent using provider={provider}, model={model}...")
        response = llm_retry_decorator(structured_llm.invoke)([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ])

        decision = response["parsed"]  # type: ignore[index]
        raw_msg = response["raw"]  # type: ignore[index]

        # Record token usage
        token_entry = record_token_usage("retry_supervisor", provider, model or "default", raw_msg)

        # Update eda_report concerns
        updated_concerns = list(eda_report.concerns) if eda_report else []
        for col in decision.columns_to_drop:
            # Add a concern that explicitly tells the DataPrep Agent to drop this column
            updated_concerns.append(f"Leaky feature '{col}' detected. Drop this column.")
        
        # Also append supervisor's explanation to the concerns so the prep agent can reason about it
        updated_concerns.append(f"Supervisor explanation: {decision.explanation}")

        if eda_report:
            new_report = eda_report.model_copy(update={"concerns": updated_concerns})
        else:
            new_report = None

        log_entry: StageLogEntry = {
            "stage": "retry_supervisor",
            "status": "retried",
            "message": f"Supervisor flagged leaky columns {decision.columns_to_drop}. Explanation: {decision.explanation}",
        }

        logger.info(f"Retry Supervisor decision: drop={decision.columns_to_drop}, concerns={updated_concerns}")

        return {
            "eda_report": new_report,
            "retry_count": retry_count,
            "validation_errors": [],  # Clear validation errors to reset state for the next run
            "stage_log": [log_entry],
            "token_usage": [token_entry],
        }

    except Exception as e:
        logger.error(f"Error during Retry Supervisor execution: {e}", exc_info=True)
        log_entry: StageLogEntry = {
            "stage": "retry_supervisor",
            "status": "failed",
            "message": f"Retry Supervisor failed: {str(e)}",
        }
        return {
            "retry_count": retry_count,
            "validation_errors": [],  # Avoid infinite loops by clearing error state even on supervisor crash
            "stage_log": [log_entry],
        }
