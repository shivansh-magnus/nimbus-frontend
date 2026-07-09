"""
Day-6 LangGraph node for generating execution report with LLM summary and token usage.
"""

from __future__ import annotations

import logging
from pathlib import Path
from pydantic import BaseModel, Field
from langgraph.runtime import Runtime

from automl_agents.schemas import PipelineState, RunConfig, StageLogEntry
from automl_agents.llm_client import get_llm, llm_retry_decorator
from automl_agents.llm_util import record_token_usage

logger = logging.getLogger(__name__)


class ReportExecutiveSummary(BaseModel):
    """Structured LLM output for the report's narrative review and recommendations."""

    executive_summary: str = Field(
        description="High-level narrative review summarizing the dataset characteristics, key preprocessing/leakage decisions, and trained model results.",
    )
    recommendations: list[str] = Field(
        description="Recommended next steps for production deployment or future iteration.",
    )


def reporter_node(state: PipelineState, runtime: Runtime[RunConfig]) -> dict:
    """Agentic reporter node: queries the LLM for a narrative summary, then generates report.md."""
    logger.info("Generating markdown report...")
    eda_report = state["eda_report"]
    selected_features = state["selected_features"]
    model_results = state["model_results"]
    best_model_id = state["best_model_id"]
    prep_plan = state["prep_plan"]
    stage_log = state["stage_log"]
    token_usage = list(state["token_usage"])
    model_path = state.get("model_path")

    # Get runtime config for LLM
    context = runtime.context
    provider = context.get("llm_provider", "gemini")
    model = context.get("model_name")
    run_id = context.get("run_id", "default_run")

    try:
        # Step 1: Query LLM for Executive Summary & Recommendations
        system_prompt = (
            "You are a Lead Data Scientist writing an executive run report for an AutoML pipeline.\n"
            "Analyze the results of the dataset profiling, the preprocessing cleaning plan, the selected features, "
            "and the candidate model validation scores.\n"
            "Highlight any major actions taken (e.g. dropping target leakage features) and describe the performance "
            "of the best model."
        )

        user_prompt = (
            f"Dataset Columns: {eda_report.n_cols if eda_report else 'N/A'}\n"
            f"Preprocessed Drop Columns: {prep_plan.get('drop_cols', []) if prep_plan else 'N/A'}\n"
            f"Selected Features: {selected_features}\n"
            f"Best Model ID: {best_model_id}\n"
            f"Model Scores:\n"
        )
        if model_results:
            for res in model_results:
                user_prompt += f"- {res['model_id']}: mean_scores={res['mean_scores']}\n"

        llm = get_llm(provider=provider, model=model)
        structured_llm = llm.with_structured_output(ReportExecutiveSummary, include_raw=True)

        logger.info(f"Querying Reporter Agent using provider={provider}, model={model}...")
        response = llm_retry_decorator(structured_llm.invoke)([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ])

        summary_data = response["parsed"]  # type: ignore[index]
        raw_msg = response["raw"]  # type: ignore[index]

        # Record reporter stage token usage
        reporter_token_entry = record_token_usage("reporter", provider, model or "default", raw_msg)
        token_usage.append(reporter_token_entry)

        # Step 2: Build Markdown Report
        report_content = []
        report_content.append(f"# AutoML Pipeline Run Report: `{run_id}`\n")

        # LLM Executive Summary
        report_content.append("## Executive Summary")
        report_content.append(summary_data.executive_summary)
        report_content.append("\n### Recommendations")
        for rec in summary_data.recommendations:
            report_content.append(f"- {rec}")
        report_content.append("\n")

        # 1. Dataset Overview
        if eda_report:
            report_content.append("## 1. Dataset Overview")
            report_content.append(f"- **Problem Type**: {eda_report.problem_type}")
            report_content.append(f"- **Rows**: {eda_report.n_rows}")
            report_content.append(f"- **Columns**: {eda_report.n_cols}")
            if eda_report.target_balance:
                report_content.append("\n### Class Proportions")
                for item in eda_report.target_balance:
                    report_content.append(f"  - `{item.label}`: {item.proportion:.2%}")
            report_content.append("\n")
        else:
            report_content.append("## 1. Dataset Overview\nNo dataset profiling information available.\n")

        # 2. Preprocessing & Cleaning Plan
        if prep_plan:
            report_content.append("## 2. Preprocessing & Cleaning Plan")
            report_content.append(f"- **Drop Columns**: {prep_plan.get('drop_cols', [])}")
            report_content.append(f"- **Datetime Columns**: {prep_plan.get('datetime_cols', [])}")
            report_content.append(f"- **Mixed Numeric Columns**: {prep_plan.get('mixed_numeric_cols', [])}")
            report_content.append(f"- **Scaling Strategy**: {prep_plan.get('scale_strategy', 'standard')}")
            report_content.append("\n")

        # 3. Feature Selection
        if selected_features:
            report_content.append("## 3. Feature Selection")
            report_content.append(f"- **Selected {len(selected_features)} features**: {', '.join(selected_features)}")
            report_content.append(f"- **Selection Rationale**: {state.get('selection_rationale', 'N/A')}")
            report_content.append("\n")

        # 4. Model Battery Results
        if model_results:
            report_content.append("## 4. Model Battery Results")
            report_content.append(f"### Best Model: **{best_model_id}**\n")
            if model_path:
                report_content.append(f"- **Saved model bundle**: `{model_path}`")
                report_content.append(
                    "  *(Load with `load_model_bundle(path)` + call `predict_from_bundle(bundle, df_raw)` for zero-skew inference)*"
                )
            else:
                report_content.append("- **Saved model bundle**: not available (export failed or skipped).")
            report_content.append("")
            report_content.append("| Model ID | Metric | Mean Validation Score | Std Dev |")
            report_content.append("| --- | --- | --- | --- |")
            for res in model_results:
                model_name = res["model_id"]
                for metric, val in res["mean_scores"].items():
                    std_val = res["std_scores"].get(metric, 0.0)
                    report_content.append(f"| {model_name} | {metric} | {val:.4f} | {std_val:.4f} |")
            report_content.append("\n")

        # 5. Token Usage Metrics
        report_content.append("## 5. Token Usage & Costs")
        report_content.append("| Stage | Provider | Model | Input Tokens | Output Tokens | Total Tokens |")
        report_content.append("| --- | --- | --- | --- | --- | --- |")
        total_in, total_out = 0, 0
        for token_entry in token_usage:
            stage_name = token_entry["stage"]
            prov = token_entry["provider"]
            m_name = token_entry["model"]
            in_t = token_entry["input_tokens"]
            out_t = token_entry["output_tokens"]
            tot = in_t + out_t
            total_in += in_t
            total_out += out_t
            report_content.append(f"| {stage_name} | {prov} | {m_name} | {in_t} | {out_t} | {tot} |")
        report_content.append(f"| **Total** | | | **{total_in}** | **{total_out}** | **{total_in + total_out}** |")
        report_content.append("\n")

        # 6. Execution Logs
        report_content.append("## 6. Execution Steps Log")
        report_content.append("| Stage | Status | Message |")
        report_content.append("| --- | --- | --- |")
        for entry in stage_log:
            report_content.append(f"| {entry['stage']} | {entry['status']} | {entry['message']} |")
        report_content.append(f"| reporter | ok | Report generated successfully. |")
        report_content.append("\n")

        # Write to file
        runs_dir = Path("runs") / run_id
        runs_dir.mkdir(parents=True, exist_ok=True)
        report_path = runs_dir / "report.md"

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_content))

        logger.info(f"Report written to {report_path}")

        # MLflow local experiment tracking
        try:
            import os
            os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
            import mlflow
            mlflow.set_experiment("nimbus-automl")
            
            with mlflow.start_run(run_name=run_id) as run:
                # Log general parameters
                mlflow.log_param("target_column", state.get("target_column"))
                mlflow.log_param("dataset_path", state.get("dataset_path"))
                mlflow.log_param("best_model_id", best_model_id)
                mlflow.log_param("llm_provider", provider)
                mlflow.log_param("llm_model", model)
                
                # Log preprocessing plan details
                if prep_plan:
                    mlflow.log_param("scale_strategy", prep_plan.get("scale_strategy"))
                    mlflow.log_param("iqr_k", prep_plan.get("iqr_k"))
                    mlflow.log_param("dropped_columns_count", len(prep_plan.get("drop_cols", [])))
                    mlflow.log_param("selected_features_count", len(selected_features))
                
                # Log metrics
                total_tokens = sum((t.get("input_tokens", 0) or 0) + (t.get("output_tokens", 0) or 0) for t in token_usage)
                mlflow.log_metric("total_tokens_used", total_tokens)
                
                prep_retries = state.get("retry_count", {}).get("data_prep", 0)
                mlflow.log_metric("prep_retries", prep_retries)
                
                # Log validation scores for each trained model candidate
                for res in model_results:
                    m_clean_id = res["model_id"].replace(" ", "_").replace("(", "").replace(")", "")
                    for metric, mean_val in res.get("mean_scores", {}).items():
                        mlflow.log_metric(f"{m_clean_id}_{metric}_mean", mean_val)
                    for metric, std_val in res.get("std_scores", {}).items():
                        mlflow.log_metric(f"{m_clean_id}_{metric}_std", std_val)
                
                # Log artifacts
                report_path_str = str(report_path.resolve())
                if os.path.exists(report_path_str):
                    mlflow.log_artifact(report_path_str, artifact_path="reports")

                cleaned_path_val = state.get("cleaned_data_path")
                if cleaned_path_val and os.path.exists(cleaned_path_val):
                    mlflow.log_artifact(cleaned_path_val, artifact_path="datasets")

                # Day-10: log the model bundle alongside report and parquet
                if model_path and os.path.exists(model_path):
                    mlflow.log_artifact(model_path, artifact_path="models")
                    logger.info(f"Logged model bundle to MLflow: {model_path}")
                    
            logger.info("Successfully tracked experiment run in MLflow.")
        except Exception as mlflow_e:
            logger.warning(f"Failed to log experiment to MLflow: {mlflow_e}")

        log_entry: StageLogEntry = {
            "stage": "reporter",
            "status": "ok",
            "message": f"Report generated successfully and saved to {report_path}.",
        }

        return {
            "report_path": str(report_path.resolve()),
            "stage_log": [log_entry],
            "token_usage": [reporter_token_entry],
        }

    except Exception as e:
        logger.error(f"Error during report generation: {e}", exc_info=True)
        log_entry: StageLogEntry = {
            "stage": "reporter",
            "status": "failed",
            "message": f"Report generation failed: {str(e)}",
        }
        return {
            "stage_log": [log_entry],
        }
