"""OpenAI-backed implementation of the ``Extractor`` protocol.

Uses the provider's structured-output mode (JSON schema generated from a
Pydantic model), then re-validates the raw JSON locally in ``validation.py``.
The API key is read from the environment and is never logged or embedded in
code.
"""

import json

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    ContentFilterFinishReasonError,
    LengthFinishReasonError,
)
from pydantic import BaseModel

from .config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_TIMEOUT_SECONDS
from .extractor import ExtractionError, Extractor, UnconfiguredExtractor
from .schemas import Finding
from .validation import validate_findings

SYSTEM_PROMPT = """You extract mentions of clinical signs from livestock health
messages written by farmers in English, Shona, or a mix of both.

Signs you may report (use these exact IDs, and no others):
- reduced_appetite: the animal is eating less than usual, or has stopped
  eating or grazing
- weakness: dull, weak, or lying down more than usual
- swelling: a lump below the ear, under the jaw, or in front of the
  shoulder (farmers may call this "mbandabvu" in Shona)
- rapid_breathing: breathing fast while resting
- cloudy_eye: the eye looks white, grey or milky
- froth_nose_mouth: foam or froth from the nose or mouth
- red_urine: urine that is red, brown, or the colour of strong tea
- unable_to_stand: cannot get up or stay standing
- ticks: ticks can be seen on the animal

Rules:
1. Report ONLY signs explicitly mentioned in the message. Never add signs
   that were not mentioned, and do not report a sign at all if it is not
   discussed.
2. An explicit negative ("no ticks", "she has no swelling") has value "no".
   Mention without negation has value "yes".
3. Doubt or uncertainty ("I'm not sure", "maybe", "I think") has value
   "not_sure".
4. Swelling: report ONLY when the message states one of the locations in
   the definition (below the ear, under the jaw, in front of the shoulder)
   or uses the word "mbandabvu". A swollen leg or any other body part is NOT
   this sign unless one of those locations is also stated.
5. Rapid breathing: report ONLY when the farmer describes the animal
   breathing fast (for example "breathing very fast"). General breathing
   difficulty without fast breathing is NOT this sign.
6. Each finding's "quote" must be an exact verbatim substring of the original
   message, copied character for character in its ORIGINAL language. Never
   translate, transliterate, rephrase, or fix the wording of a quote — even
   when the message is in Shona or mixes languages, quote the words exactly
   as the farmer wrote them.
7. Extract findings only. Do not decide escalation, do not give advice, and
   do not add explanations outside the structured output.

Example of correct output for a Shona message:
Message: "Mombe yangu haisi kudya."
Correct output: {"findings": [{"sign_id": "reduced_appetite", "value": "yes", "quote": "haisi kudya"}]}
The quote keeps the farmer's original words — never an English translation."""


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
        # ``LengthFinishReasonError``/``ContentFilterFinishReasonError`` are
        # raised by ``parse`` on truncated/filtered structured output; they
        # are OpenAIError subclasses, not APIStatusError subclasses, so they
        # need explicit handling — otherwise they would surface as HTTP 500.
        except (
            APITimeoutError,
            APIConnectionError,
            APIStatusError,
            LengthFinishReasonError,
            ContentFilterFinishReasonError,
        ) as exc:
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
