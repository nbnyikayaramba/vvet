"""Endpoint-level failure-handling tests: every failure is a controlled non-200."""

import pytest

from app.extractor import ExtractionError
from app.main import app, get_extractor


class RaisingExtractor:
    def __init__(self, error: Exception):
        self._error = error

    async def extract(self, message: str):
        raise self._error


@pytest.fixture(autouse=True)
def clear_overrides():
    yield
    app.dependency_overrides.clear()


@pytest.mark.parametrize(
    "error",
    [
        ExtractionError("LLM provider error: APITimeoutError"),
        ExtractionError("LLM returned invalid JSON."),
        ExtractionError("LLM output contains duplicate sign_id: ticks"),
        ExtractionError("Quote for 'weakness' is not an exact substring of the message"),
    ],
    ids=["timeout", "malformed-json", "duplicate-sign", "fabricated-quote"],
)
def test_extractor_failure_returns_503_not_empty_ok(client, error):
    app.dependency_overrides[get_extractor] = lambda: RaisingExtractor(error)
    response = client.post("/extract", json={"message": "my cow is sick"})
    assert response.status_code == 503
    # must NOT be a falsely reassuring empty result
    assert response.json() != {"findings": [], "escalate": False, "reason": "No escalation-triggering sign was reported."}


def test_error_detail_is_human_readable(client):
    app.dependency_overrides[get_extractor] = lambda: RaisingExtractor(
        ExtractionError("LLM provider error: APIConnectionError")
    )
    response = client.post("/extract", json={"message": "urgent"})
    assert "LLM" in response.json()["detail"]
