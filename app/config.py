"""
config.py — Centralized Configuration

WHY THIS EXISTS:
    Instead of calling os.getenv() in every file, we load all environment
    variables once into a single dataclass. This gives us:
    1. One place to see every config value
    2. Type safety (temperature is a float, not a string)
    3. Fail-fast validation (missing API key = crash at startup, not mid-request)

INTERVIEW POINT:
    "I use a dataclass instead of Pydantic BaseSettings to keep dependencies
    minimal. For a production app with complex validation, I'd switch to Pydantic."
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load .env file into os.environ
# This runs once when this module is first imported
load_dotenv()


@dataclass
class Settings:
    """Application settings loaded from environment variables."""

    # --- API Keys ---
    groq_api_key: str

    # --- LLM ---
    llm_model: str
    llm_temperature: float

    # --- Vector Store ---
    vector_store_path: str
    embedding_model: str


def _load_settings() -> Settings:
    """
    Read environment variables and return a validated Settings object.
    Raises ValueError immediately if required keys are missing.
    """
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY is missing from .env file")

    return Settings(
        groq_api_key=groq_api_key,
        llm_model=os.getenv("LLM_MODEL", "llama-3.1-8b-instant"),
        llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
        vector_store_path=os.getenv("VECTOR_STORE_PATH", "./chroma_data"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
    )


# --- Singleton: created once, imported everywhere ---
settings = _load_settings()
