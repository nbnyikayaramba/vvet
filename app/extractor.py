"""Extractor interface and default placeholder.

A real LLM-backed extractor is wired in a later step. Everything else in the
app depends only on the ``Extractor`` protocol, so providers stay swappable
and mockable through FastAPI dependency injection.
"""

from typing import Protocol

from .schemas import Finding


class ExtractionError(Exception):
    """Extraction is unavailable or the provider returned unusable output."""


class Extractor(Protocol):
    async def extract(self, message: str) -> list[Finding]:
        """Extract findings from a farmer message.

        Implementations must return findings only: no escalation decisions,
        no human-readable reasons. Raises ``ExtractionError`` on any failure
        so callers never receive a falsely reassuring empty result.
        """
        ...


class UnconfiguredExtractor:
    """Placeholder until a real LLM extractor is configured.

    Fails closed: extraction is not available, so the endpoint returns a
    controlled error instead of pretending nothing was found.
    """

    async def extract(self, message: str) -> list[Finding]:
        raise ExtractionError(
            "No symptom extractor is configured. Set LLM provider settings "
            "in .env and restart the app."
        )
