"""
Day-6 LangGraph node for agentic dataset profiling with LLM concerns.
"""

from __future__ import annotations

import logging
from pydantic import BaseModel, Field
from langgraph.runtime import Runtime

from automl_agents.schemas import PipelineState, RunConfig, StageLogEntry
from automl_agents.tools.profiler import profile_dataset
from automl_agents.llm_client import get_llm, llm_retry_decorator
from automl_agents.llm_util import record_token_usage

logger = logging.getLogger(__name__)


class ProfilerAnalysis(BaseModel):
    """Structured LLM output for dataset analysis concerns."""

    concerns: list[str] = Field(
        default_factory=list,
        description="A list of specific, narrative concerns about data quality, target leakage, features to drop, or class imbalance.",
    )


def _format_eda_summary(report) -> str:
    """Format the statistical EDAReport into a readable summary for the LLM."""
    summary = []
    summary.append("Dataset Overview:")
    summary.append(f"- Total Rows: {report.n_rows}")
    summary.append(f"- Total Columns: {report.n_cols}")
    summary.append(f"- Inferred Problem Type: {report.problem_type}")

    summary.append("\nColumns Profile:")
    for col in report.columns:
        summary.append(
            f"  - `{col.column}`: type={col.dtype}, missingness={col.missing_fraction:.2%}, cardinality={col.cardinality}"
        )

    if report.target_balance:
        summary.append("\nTarget Class Proportions:")
        for label_prop in report.target_balance:
            summary.append(f"  - Class `{label_prop.label}`: {label_prop.proportion:.2%}")

    if report.correlations_flagged:
        summary.append("\nHigh Correlation Pairs (|correlation| >= 0.90):")
        for pair in report.correlations_flagged:
            summary.append(f"  - `{pair.col_a}` and `{pair.col_b}`: correlation={pair.corr:.4f}")

    return "\n".join(summary)


def profiler_node(state: PipelineState, runtime: Runtime[RunConfig]) -> dict:
    """Agentic profiler node: runs profiling and calls LLM to identify concerns."""
    logger.info("Starting dataset profiling...")
    csv_path = state["dataset_path"]
    target_column = state["target_column"]

    # Get runtime config for LLM
    context = runtime.context
    provider = context.get("llm_provider", "gemini")
    model = context.get("model_name")

    try:
        # Step 1: Run deterministic statistical profiler
        report = profile_dataset(csv_path, target_column)
        logger.info(f"Deterministic profiling completed: {report.n_rows} rows, {report.n_cols} columns.")

        # Step 2: Format summary and query the LLM for schema concerns / leakage
        summary_text = _format_eda_summary(report)
        system_prompt = (
            "You are a Lead Data Scientist analyzing a dataset profile for a machine learning task.\n"
            "Review the statistical summary of columns, target distribution, and highly correlated pairs.\n"
            "Flag any potential target leakage (e.g. columns that are duplicate copies of target, or correlate at 1.0 with the target).\n"
            "Also flag any features that should be dropped (such as ID columns with unique categories per row, zero-variance columns, or columns that are all null)."
        )
        user_prompt = f"Target Column: {target_column}\n\nDataset Profile:\n{summary_text}"

        llm = get_llm(provider=provider, model=model)
        structured_llm = llm.with_structured_output(ProfilerAnalysis, include_raw=True)

        logger.info(f"Querying Profiler Agent using provider={provider}, model={model}...")
        response = llm_retry_decorator(structured_llm.invoke)([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ])

        analysis = response["parsed"]  # type: ignore[index]
        raw_msg = response["raw"]  # type: ignore[index]

        # Save concerns to report
        report.concerns = analysis.concerns
        logger.info(f"Profiler Agent identified {len(report.concerns)} concerns: {report.concerns}")

        # Step 3: Record token usage
        token_entry = record_token_usage("profiler", provider, model or "default", raw_msg)

        log_entry: StageLogEntry = {
            "stage": "profiler",
            "status": "ok",
            "message": f"Inferred problem type: {report.problem_type}. Identified {len(report.concerns)} concerns.",
        }

        return {
            "eda_report": report,
            "stage_log": [log_entry],
            "token_usage": [token_entry],
        }

    except Exception as e:
        logger.error(f"Error during dataset profiling: {e}", exc_info=True)
        log_entry: StageLogEntry = {
            "stage": "profiler",
            "status": "failed",
            "message": f"Profiling failed: {str(e)}",
        }
        return {
            "stage_log": [log_entry],
        }
