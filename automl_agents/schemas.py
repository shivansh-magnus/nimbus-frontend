"""
Core state schemas for the pipeline graph.

Day-1 rule (see roadmap Section 2.4): the raw dataframe NEVER lives in this
state. Only paths to parquet snapshots and statistical summaries do. Agents
reason over EDAReport / small samples, never over full tabular data pushed
through the graph.

Schema fix (Day 1, before Day 6 wiring): the original EDAReport used
dict[str, X] fields keyed by column name (dtypes, missingness, cardinality,
target_balance) and a fixed-length tuple (correlations_flagged). Gemini's
structured-output schema is a restricted subset of OpenAPI 3.0:
  - OpenAPI 3.0 has no tuple/prefixItems concept, so a fixed-length,
    mixed-type tuple like tuple[str, str, float] cannot be represented.
    (A variable-length, single-type tuple like tuple[int, ...] is fine --
    it's specifically the fixed + mixed-type combination that breaks.)
  - Open-ended dict[str, X] maps compile to "additionalProperties", which
    has historically been a gap in some structured-output client libraries.
    Untested against the pinned langchain-google-genai version here, so
    treat as medium risk rather than confirmed-broken.
Every LLM-facing field below is restructured into a list of small typed
records instead of a dict keyed by unknown column names, and the one tuple
is now a named model. Verified locally: EDAReport.model_json_schema() below
contains no "prefixItems" and no "additionalProperties" anywhere.

Convenience methods on EDAReport turn the list-of-records shape back into
plain dicts for the deterministic tool code (profiler.py, preprocessor.py,
etc. -- Days 2-5), so nothing downstream has to change its calling
convention just because the LLM-facing shape changed.
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field


class ColumnProfile(BaseModel):
    """One row of the Profiler's per-column findings. Replaces the old
    dtypes / missingness / cardinality dicts -- same information, shaped as
    a list of records instead of three parallel dicts keyed by column name."""

    column: str
    dtype: str
    missing_fraction: float = Field(description="fraction missing, 0.0-1.0")
    cardinality: int = Field(description="number of unique values in this column")


class ClassProportion(BaseModel):
    """One class label's share of the target column. Classification only."""

    label: str
    proportion: float = Field(description="fraction of rows with this label, 0.0-1.0")


class CorrelationPair(BaseModel):
    """A pair of columns with a suspicious correlation."""
    col_a: str
    col_b: str
    corr: float = Field(description="correlation strength, -1.0 to 1.0")


class EDAReport(BaseModel):
    """Structured output of the Profiler node. Filled in on Day 6."""

    n_rows: int
    n_cols: int
    columns: list[ColumnProfile]
    problem_type: Literal["classification", "regression"]
    target_balance: list[ClassProportion] | None = Field(
        default=None, description="class label -> proportion, classification only"
    )
    correlations_flagged: list[CorrelationPair] = Field(
        default_factory=list,
        description="column pairs above a suspicion threshold",
    )
    concerns: list[str] = Field(
        default_factory=list,
        description="LLM-written narrative flags, e.g. 'col X looks like a leaked target'",
    )

    # --- convenience accessors for the deterministic tool code (Days 2-5) ---
    # These exist so profiler.py / preprocessor.py etc. can still do
    # eda_report.missingness_by_column()["income"] instead of looping over
    # eda_report.columns every time. The LLM never sees these methods --
    # they only run on the already-validated Python object, after the fact.

    def dtypes_by_column(self) -> dict[str, str]:
        return {c.column: c.dtype for c in self.columns}

    def missingness_by_column(self) -> dict[str, float]:
        return {c.column: c.missing_fraction for c in self.columns}

    def cardinality_by_column(self) -> dict[str, int]:
        return {c.column: c.cardinality for c in self.columns}

    def target_balance_by_label(self) -> dict[str, float] | None:
        if self.target_balance is None:
            return None
        return {c.label: c.proportion for c in self.target_balance}


class StageLogEntry(TypedDict):
    stage: str
    status: Literal["ok", "retried", "failed"]
    message: str


class TokenUsageEntry(TypedDict):
    stage: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int


class PipelineState(TypedDict):
    # --- input ---
    dataset_path: str  # path to the ORIGINAL uploaded csv, read-only
    target_column: str

    # --- profiler stage ---
    eda_report: EDAReport | None

    # --- data prep stage (cleaning + feature engineering, merged) ---
    cleaned_data_path: str | None  # parquet snapshot, not a dataframe
    prep_plan: dict | None  # column -> chosen strategy, LLM structured output
    # NOTE: when you design the Data Prep structured-output schema on Day 6,
    # apply the same list-of-records principle used above for EDAReport --
    # prep_plan is inherently a column -> strategy mapping with unknown
    # keys ahead of time, so it carries the same additionalProperties risk.

    # --- feature selection stage ---
    selected_features: list[str]
    selection_rationale: str

    # --- training stage ---
    model_results: list[dict]  # per-candidate CV scores
    best_model_id: str | None
    validation_errors: list[str] | None
    # Path to the joblib bundle: runs/{run_id}/model.pkl
    # Contains {model, prep_artifacts, selected_features, target_column, problem_type, model_id}
    model_path: str | None

    # --- reporting stage ---
    report_path: str | None

    # --- control / observability (use operator.add reducers: nodes append, never overwrite) ---
    stage_log: Annotated[list[StageLogEntry], operator.add]
    retry_count: dict[str, int]
    token_usage: Annotated[list[TokenUsageEntry], operator.add]


class RunConfig(TypedDict):
    """Read-only run configuration, injected via LangGraph's context_schema.
    Never put this inside PipelineState -- it doesn't change during a run."""

    run_id: str
    llm_provider: Literal["gemini", "groq", "ollama"]
    model_name: str  # e.g. "gemini-3.1-flash-lite" or "llama-3.3-70b-versatile"
    max_retries: int
    token_budget: int | None