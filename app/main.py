"""FastAPI application exposing POST /extract."""

from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException

from .escalation import decide_escalation
from .extractor import Extractor, ExtractionError
from .providers import build_extractor
from .schemas import ExtractRequest, ExtractResponse

app = FastAPI(title="V-Vet Symptom Extractor")


@lru_cache(maxsize=1)
def _extractor() -> Extractor:
    return build_extractor()


def get_extractor() -> Extractor:
    """Dependency-injection point: swap or mock the extractor in tests."""
    return _extractor()


@app.post("/extract", response_model=ExtractResponse)
async def extract_symptoms(
    request: ExtractRequest,
    extractor: Extractor = Depends(get_extractor),
) -> ExtractResponse:
    try:
        findings = await extractor.extract(request.message)
    except ExtractionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    escalate, reason = decide_escalation(findings)
    return ExtractResponse(findings=findings, escalate=escalate, reason=reason)
