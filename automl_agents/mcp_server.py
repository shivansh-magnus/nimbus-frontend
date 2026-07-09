"""
Day-10 MCP server — exposes the compiled LangGraph pipeline as callable tools
over streamable-http (default) or stdio.

Architecture note (from DAY10_NEXT_STEPS.md §1):
  MCP sits at a different layer than the internal multi-agent system.  The
  internal nodes (profiler, prep, selector, trainer, reporter, supervisor)
  collaborate inside one LangGraph run.  This server exposes the *whole
  compiled graph* as a single tool to an *external* agent — Claude Desktop,
  Claude Code, or another orchestrator.

Blocking calls:
  FastMCP tool handlers are async def functions on an asyncio event loop.
  graph.invoke() is fully synchronous and can run for minutes (CV folds,
  Optuna trials, sequential LLM calls).  Calling it directly inside an
  async def would stall the event loop.  We use asyncio.to_thread() to hand
  the blocking call to a worker thread and keep the loop free for other
  concurrent MCP traffic.

Security:
  run_automl_pipeline validates dataset_path against an allow-list directory
  (data/raw/) before the graph ever runs.  This matters especially since the
  default transport is streamable-http (network-reachable), not local-only stdio.

Transport:
  Default: streamable-http on port 8000.
  Override: set --transport stdio (e.g. for Claude Desktop subprocess config).

Usage:
    uv run python src/automl_agents/mcp_server.py          # streamable-http:8000
    uv run nimbus serve-mcp                                 # same via CLI
    uv run nimbus serve-mcp --transport stdio               # stdio for local desktop
"""

from __future__ import annotations

import asyncio
import datetime
import logging
import re
import sys
from pathlib import Path

# Ensure the src/ package is importable when running as __main__
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcp.server.fastmcp import FastMCP  # noqa: E402 -- after sys.path fixup

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Server instance
# ---------------------------------------------------------------------------

mcp = FastMCP("nimbus-automl")

# Allow-listed directory for dataset_path arguments.  Arbitrary filesystem
# paths are rejected before the graph ever runs.
ALLOWED_DATA_DIR = (ROOT / "data" / "raw").resolve()


# ---------------------------------------------------------------------------
# Path validation helper
# ---------------------------------------------------------------------------

def _validate_dataset_path(dataset_path: str) -> Path:
    """Resolve dataset_path and verify it lives under ALLOWED_DATA_DIR.

    Raises ValueError for paths outside the allow-list (path traversal
    attempts, absolute paths to unrelated directories, etc.).
    """
    resolved = Path(dataset_path).resolve()
    # Accept the path if it's directly inside ALLOWED_DATA_DIR or a subdir
    if ALLOWED_DATA_DIR not in resolved.parents and resolved.parent != ALLOWED_DATA_DIR:
        raise ValueError(
            f"dataset_path must live under {ALLOWED_DATA_DIR}.  "
            f"Received: {resolved}"
        )
    if not resolved.exists():
        raise ValueError(f"Dataset file not found: {resolved}")
    return resolved


# ---------------------------------------------------------------------------
# Tool: run_automl_pipeline
# ---------------------------------------------------------------------------

@mcp.tool()
async def run_automl_pipeline(
    dataset_path: str,
    target_column: str,
    llm_provider: str = "gemini",
) -> dict:
    """Run the full multi-agent AutoML pipeline on a CSV and return a run summary.

    The pipeline runs profiling, preprocessing, feature selection, model
    training (with Optuna tuning), leakage detection, and report generation.
    It is entirely blocking internally; this tool offloads it to a thread so
    the MCP event loop stays responsive.

    Parameters
    ----------
    dataset_path:
        Path to a CSV file under data/raw/.  Arbitrary filesystem paths are
        rejected for security (the server defaults to streamable-http, making
        this input attacker-controllable).
    target_column:
        Name of the column to predict.
    llm_provider:
        LLM provider to use: "gemini" (default), "groq", or "ollama".

    Returns
    -------
    Dict with keys: best_model_id, model_path, report_path, selected_features.
    """
    csv_path = _validate_dataset_path(dataset_path)

    # Build run_id — same pattern as run_pipeline.py
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"mcp_{csv_path.stem}_{timestamp}"

    initial_state = {
        "dataset_path": str(csv_path),
        "target_column": target_column,
        "eda_report": None,
        "cleaned_data_path": None,
        "prep_plan": None,
        "selected_features": [],
        "selection_rationale": "",
        "model_results": [],
        "best_model_id": None,
        "model_path": None,
        "report_path": None,
        "stage_log": [],
        "retry_count": {},
        "token_usage": [],
        "validation_errors": None,
    }
    context = {
        "run_id": run_id,
        "llm_provider": llm_provider,
        "model_name": None,  # picks up provider default from llm_client.py
        "max_retries": 2,
        "token_budget": None,
    }

    # CRITICAL: offload the blocking graph.invoke() call to a worker thread.
    # Calling it directly inside an async def would stall the whole event loop.
    from automl_agents.graph import graph  # lazy import; heavy but only once

    final_state = await asyncio.to_thread(graph.invoke, initial_state, context=context)

    return {
        "run_id": run_id,
        "best_model_id": final_state.get("best_model_id"),
        "model_path": final_state.get("model_path"),
        "report_path": final_state.get("report_path"),
        "selected_features": final_state.get("selected_features", []),
    }


# ---------------------------------------------------------------------------
# Tool: get_run_report
# ---------------------------------------------------------------------------

@mcp.tool()
async def get_run_report(run_id: str) -> str:
    """Return the contents of a past run's report.md.

    Parameters
    ----------
    run_id:
        The run identifier (e.g. "run_20260709_082521" or "mcp_synthetic_ground_truth_20260709_082521").
        Must match an existing subdirectory under runs/.

    Returns
    -------
    Full markdown text of the report, or an error message if not found.
    """
    # Sanitise: run_id must look like a simple identifier, no path separators
    if not re.fullmatch(r"[\w\-]+", run_id):
        return f"Error: run_id '{run_id}' contains invalid characters."

    report_path = ROOT / "runs" / run_id / "report.md"
    if not report_path.exists():
        return f"Error: No report found for run_id '{run_id}'.  Expected: {report_path}"

    return report_path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Tool: list_local_datasets
# ---------------------------------------------------------------------------

@mcp.tool()
async def list_local_datasets() -> list[dict]:
    """List all CSV datasets available under data/raw/.

    Returns the manifest.json entries (if present) plus the synthetic dataset
    entry.  An MCP caller can pass any of the returned paths to
    run_automl_pipeline.dataset_path.

    Returns
    -------
    List of dicts, each with at least {"id", "path", "target_column"}.
    """
    import json

    manifest_path = ROOT / "data" / "raw" / "manifest.json"
    datasets: list[dict] = []

    if manifest_path.exists():
        try:
            datasets = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Failed to read manifest.json: {e}")

    # Always include the synthetic ground truth if present
    synthetic_path = ROOT / "data" / "raw" / "synthetic_ground_truth.csv"
    has_synthetic = any(d.get("id") in ("synthetic", "synthetic_ground_truth") for d in datasets)
    if not has_synthetic and synthetic_path.exists():
        datasets.append({
            "id": "synthetic_ground_truth",
            "path": str(synthetic_path.relative_to(ROOT)).replace("\\", "/"),
            "target_column": "churn",
            "problem_type": "classification",
            "description": "Controlled synthetic set with nulls, outliers, mixed dtypes, and a leaky decoy column.",
        })

    return datasets


# ---------------------------------------------------------------------------
# Tool: run_stress_test
# ---------------------------------------------------------------------------

@mcp.tool()
async def run_stress_test() -> dict:
    """Run the AutoML pipeline against all local datasets (stress test).

    Executes synchronously via asyncio.to_thread so the event loop stays free.
    This is a long-running call (minutes per dataset).  For v1 this is
    synchronous-via-thread; a background-job + polling design is deferred.

    Returns
    -------
    Dict with keys: results (list), report_path (str), all_passed (bool).
    """
    from scripts.stress_test import run_stress_test as _run_stress_test  # noqa: PLC0415

    return await asyncio.to_thread(_run_stress_test)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Nimbus MCP server.")
    parser.add_argument(
        "--transport",
        choices=["streamable-http", "stdio"],
        default="streamable-http",
        help="MCP transport (default: streamable-http on port 8000)",
    )
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.transport == "streamable-http":
        mcp.settings.port = args.port
        mcp.run(transport="streamable-http")
    else:
        mcp.run(transport="stdio")
