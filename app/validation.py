"""Strict local validation of LLM findings before they reach the response.

The LLM's structured output is untrusted: even with provider-side schema
enforcement, we re-validate everything locally. Anything invalid raises
``ExtractionError`` so the endpoint returns a controlled error instead of
silently changing, discarding or fabricating findings.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from .extractor import ExtractionError
from .schemas import Finding


class RawFindings(BaseModel):
    """Expected shape of the LLM's structured output.

    ``extra="forbid"`` rejects any field the model added that we did not
    ask for (e.g. an attempted escalation decision or advice).
    """

    model_config = ConfigDict(extra="forbid")

    findings: list[Finding]


def validate_findings(payload: Any, message: str) -> list[Finding]:
    """Validate raw LLM output against the contract.

    Checks, in order:
      1. Payload matches the schema: a ``findings`` list of objects with
         exactly ``sign_id``, ``value``, ``quote`` — no extra fields.
      2. ``sign_id`` is one of the nine allowed IDs (enforced by the enum).
      3. ``value`` is one of ``yes``, ``no``, ``not_sure`` (enforced by the enum).
      4. No duplicate sign IDs.
      5. Each quote is a non-empty exact substring of the original message.

    Raises ``ExtractionError`` on the first problem found.
    """
    try:
        result = RawFindings.model_validate(payload)
    except ValidationError as exc:
        raise ExtractionError(f"LLM output failed schema validation: {exc.errors()[0]['msg']}") from exc

    seen: set[str] = set()
    for finding in result.findings:
        sign = finding.sign_id.value
        if sign in seen:
            raise ExtractionError(f"LLM output contains duplicate sign_id: {sign}")
        seen.add(sign)
        if finding.quote not in message:
            raise ExtractionError(
                f"Quote for '{sign}' is not an exact substring of the message"
            )
    return result.findings
