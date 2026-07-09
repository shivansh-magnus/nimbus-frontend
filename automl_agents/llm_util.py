"""
Utilities for tracking token usage across different LLM providers.
"""

from __future__ import annotations

from typing import Any
from automl_agents.schemas import TokenUsageEntry


def extract_token_usage(response_metadata: dict, usage_metadata: dict | None = None) -> dict[str, int]:
    """Standardize token usage across providers."""
    if usage_metadata:
        return {
            "input_tokens": usage_metadata.get("input_tokens", 0),
            "output_tokens": usage_metadata.get("output_tokens", 0),
        }

    # Check response_metadata for gemini or groq/openai styles
    gemini_usage = response_metadata.get("usage_metadata")
    if gemini_usage:
        return {
            "input_tokens": gemini_usage.get("prompt_tokens", 0),
            "output_tokens": gemini_usage.get("candidates_tokens", 0),
        }

    groq_usage = response_metadata.get("token_usage")
    if groq_usage:
        return {
            "input_tokens": groq_usage.get("prompt_tokens", 0),
            "output_tokens": groq_usage.get("completion_tokens", 0),
        }

    return {"input_tokens": 0, "output_tokens": 0}


def record_token_usage(
    stage: str,
    provider: str,
    model: str,
    raw_message: Any,
) -> TokenUsageEntry:
    """Create a standardized TokenUsageEntry from a LangChain raw response message."""
    metadata = getattr(raw_message, "response_metadata", {})
    usage = getattr(raw_message, "usage_metadata", None)

    usage_dict = extract_token_usage(metadata, usage)

    return {
        "stage": stage,
        "provider": provider,
        "model": model,
        "input_tokens": usage_dict["input_tokens"],
        "output_tokens": usage_dict["output_tokens"],
    }
