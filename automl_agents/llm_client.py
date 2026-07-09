"""
Provider-agnostic LLM client factory.

switching providers is a one-line env var change, *before* any agent logic exists. 
Dev against Groq/Ollama to save Gemini
free-tier quota; switch to Gemini for real agent-reasoning tests and for
the final demo (see roadmap §4).

Usage:
    from automl_agents.llm_client import get_llm
    llm = get_llm()                 # reads LLM_PROVIDER from .env
    llm = get_llm(provider="groq")  # explicit override
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

_DEFAULT_MODELS = {
    "gemini": "gemini-3.1-flash-lite",
    "groq": "llama-3.3-70b-versatile",
    "ollama": "llama3.1",
}


def get_llm(provider: str | None = None, model: str | None = None, temperature: float = 0.0):
    """Return a LangChain chat model instance for the given provider.

    provider defaults to the LLM_PROVIDER env var (falls back to 'gemini').
    Raises a clear error if the required API key is missing, rather than
    failing deep inside a graph node.
    """
    provider = (provider or os.getenv("LLM_PROVIDER", "gemini")).lower()
    model = model or os.getenv(f"{provider.upper()}_MODEL") or _DEFAULT_MODELS.get(provider, "gemini-3.1-flash-lite")

    if provider == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY not set. Get a free key at https://aistudio.google.com/apikey "
                "and put it in your .env file."
            )
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model, temperature=temperature, api_key=api_key)

    if provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY not set. Get a free key at https://console.groq.com/keys "
                "and put it in your .env file."
            )
        from langchain_groq import ChatGroq
        from pydantic import SecretStr

        return ChatGroq(model=model, temperature=temperature, api_key=SecretStr(api_key))

    if provider == "ollama":
        from langchain_ollama import ChatOllama  # pip install langchain-ollama if you use this path

        return ChatOllama(model=model, temperature=temperature)
    raise ValueError(f"Unknown provider '{provider}'. Use one of: gemini, groq, ollama.")


llm_retry_decorator = retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=4, max=30),
    reraise=True,
)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=20))
def ping(provider: str | None = None) -> str:
    """Smallest possible round trip to confirm a provider actually works.
    Retries with exponential backoff on transient/rate-limit errors."""
    llm = get_llm(provider=provider)
    response = llm.invoke("Reply with exactly one word: OK")
    return str(response.content)
