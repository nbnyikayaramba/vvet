"""OpenAI-backed implementation of the ``Extractor`` protocol.

Uses the provider's structured-output mode (JSON schema generated from a
Pydantic model), then re-validates the raw JSON locally in ``validation.py``.
The API key is read from the environment and is never logged or embedded in
code.
"""

import json

from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI
from pydantic import BaseModel

from .config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_TIMEOUT_SECONDS
from .extractor import ExtractionError, Extractor, UnconfiguredExtractor
from .schemas import Finding
from .validation import validate_findings

SYSTEM_PROMPT = """You extract mentions of clinical signs from livestock health
messages written by farmers in English, Shona, or a mix of both.

Signs you may report (use these exact IDs, and no others):
- reduced_appetite: not eating, poor appetite, e.g. Shona "haisi kudya"
- weakness: weak, lethargic, no energy
- swelling: any swelling, including "mbandabvu" (swelling near the ear, jaw or shoulder)
- rapid_breathing: breathing fast or with difficulty
- cloudy_eye: cloudy, white or bluish eye
- froth_nose_mouth: foam or froth from the nose or mouth
- red_urine: red, dark or tea-coloured urine
- unable_to_stand: cannot stand or get up
- ticks: ticks visible on the animal

Rules:
1. Report ONLY signs explicitly mentioned in the message. Never add signs
   that were not mentioned, and do not report a sign at all if it is not
   discussed.
2. An explicit negative ("no ticks", "she has no swelling") has value "no".
   Mention without negation has value "yes".
3. Doubt or uncertainty ("I'm not sure", "maybe", "I think") has value
   "not_sure".
4. Each finding's "quote" must be an exact verbatim substring of the original
   message, copied character for character in its ORIGINAL language. Never
   translate, transliterate, rephrase, or fix the wording of a quote — even
   when the message is in Shona or mixes languages, quote the words exactly
   as the farmer wrote them.
5. Extract findings only. Do not decide escalation, do not give advice, and
   do not add explanations outside the structured output."""


class FindingsResult(BaseModel):
    """Structured-output schema handed to the provider."""

    findings: list[Finding]


class OpenAIExtractor:
    """Extracts findings via OpenAI chat completions with structured output."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout_seconds: float | None = None,
        base_url: str | None = None,
        client: AsyncOpenAI | None = None,
    ) -> None:
        # ``client`` is the test seam: tests inject a fake, no network needed.
        self._model = model or LLM_MODEL
        self._client = client or AsyncOpenAI(
            api_key=api_key if api_key is not None else LLM_API_KEY,
            base_url=base_url if base_url is not None else LLM_BASE_URL,
            timeout=timeout_seconds if timeout_seconds is not None else LLM_TIMEOUT_SECONDS,
        )

    async def extract(self, message: str) -> list[Finding]:
        try:
            response = await self._client.chat.completions.parse(
                model=self._model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": message},
                ],
                response_format=FindingsResult,
            )
        except (APITimeoutError, APIConnectionError, APIStatusError) as exc:
            raise ExtractionError(f"LLM provider error: {type(exc).__name__}") from exc

        # Validate from the raw JSON text, not the SDK's parsed object, so
        # nothing the provider did not schema-check slips through.
        content = response.choices[0].message.content
        if not content:
            # Provider refusal or no structured output at all.
            raise ExtractionError("LLM returned no parseable structured output.")
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ExtractionError("LLM returned invalid JSON.") from exc
        return validate_findings(payload, message)


def build_extractor() -> Extractor:
    """Factory used by the app's dependency-injection point."""
    if not LLM_API_KEY:
        return UnconfiguredExtractor()
    return OpenAIExtractor()
