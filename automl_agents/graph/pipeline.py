"""
Day-7 LangGraph pipeline graph topology with conditional routing and retry loops.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from automl_agents.nodes import (
    profiler_node,
    prep_node,
    selector_node,
    trainer_node,
    reporter_node,
    retry_supervisor_node,
)
from automl_agents.schemas import PipelineState, RunConfig


def route_after_profiler(state: PipelineState) -> str:
    """Route to classification or regression preprocessing based on problem type."""
    report = state.get("eda_report")
    if report and report.problem_type == "regression":
        return "regression_prep"
    return "classification_prep"


def route_after_trainer(state: PipelineState) -> str:
    """Route to supervisor for leakage resolution if validation errors occur, otherwise to reporter."""
    errors = state.get("validation_errors")
    retries = (state.get("retry_count") or {}).get("data_prep", 0)
    if errors and len(errors) > 0:
        if retries < 2:
            return "retry_supervisor"
    return "reporter"


def route_after_supervisor(state: PipelineState) -> str:
    """Route back to preprocessing based on problem type after supervisor updates state."""
    report = state.get("eda_report")
    if report and report.problem_type == "regression":
        return "regression_prep"
    return "classification_prep"


# Build the workflow StateGraph
builder = StateGraph(PipelineState, context_schema=RunConfig)

# Add nodes
builder.add_node("profiler", profiler_node)
builder.add_node("classification_prep", prep_node)
builder.add_node("regression_prep", prep_node)
builder.add_node("classification_selector", selector_node)
builder.add_node("regression_selector", selector_node)
builder.add_node("classification_trainer", trainer_node)
builder.add_node("regression_trainer", trainer_node)
builder.add_node("retry_supervisor", retry_supervisor_node)
builder.add_node("reporter", reporter_node)

# Add static edges
builder.add_edge(START, "profiler")
builder.add_edge("classification_prep", "classification_selector")
builder.add_edge("classification_selector", "classification_trainer")
builder.add_edge("regression_prep", "regression_selector")
builder.add_edge("regression_selector", "regression_trainer")
builder.add_edge("reporter", END)

# Add conditional edges
builder.add_conditional_edges(
    "profiler",
    route_after_profiler,
    {
        "classification_prep": "classification_prep",
        "regression_prep": "regression_prep",
    },
)
builder.add_conditional_edges(
    "classification_trainer",
    route_after_trainer,
    {
        "retry_supervisor": "retry_supervisor",
        "reporter": "reporter",
    },
)
builder.add_conditional_edges(
    "regression_trainer",
    route_after_trainer,
    {
        "retry_supervisor": "retry_supervisor",
        "reporter": "reporter",
    },
)
builder.add_conditional_edges(
    "retry_supervisor",
    route_after_supervisor,
    {
        "classification_prep": "classification_prep",
        "regression_prep": "regression_prep",
    },
)

# Compile graph
graph = builder.compile()

