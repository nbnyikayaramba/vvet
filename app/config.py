"""Runtime configuration loaded from the environment (and local .env file)."""

import os

from dotenv import load_dotenv

load_dotenv()

# Required for real extraction. Never log this value anywhere.
LLM_API_KEY: str | None = os.getenv("LLM_API_KEY")

# Optional overrides with sensible defaults.
LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-6-luna")
LLM_TIMEOUT_SECONDS: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "20"))
LLM_BASE_URL: str | None = os.getenv("LLM_BASE_URL")  # e.g. an OpenAI-compatible proxy
