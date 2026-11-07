import pytest
from fastapi.testclient import TestClient

from app.extractor import UnconfiguredExtractor
from app.main import app, get_extractor
from app.schemas import Finding, FindingValue, SignId


class FakeExtractor:
    """Test double standing in for the LLM extractor."""

    def __init__(self, findings: list[Finding]):
        self._findings = findings

    async def extract(self, message: str) -> list[Finding]:
        return self._findings


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def clear_overrides():
    yield
    app.dependency_overrides.clear()


def test_extract_rejects_empty_message(client):
    response = client.post("/extract", json={"message": ""})
    assert response.status_code == 422


def test_extract_rejects_blank_message(client):
    response = client.post("/extract", json={"message": "   "})
    assert response.status_code == 422


def test_extract_rejects_non_string_message(client):
    response = client.post("/extract", json={"message": 123})
    assert response.status_code == 422


def test_unconfigured_extractor_fails_closed_with_503(client):
    app.dependency_overrides[get_extractor] = UnconfiguredExtractor
    response = client.post("/extract", json={"message": "My cow is sick"})
    assert response.status_code == 503


def test_injected_extractor_returns_contract_shape(client):
    findings = [
        Finding(
            sign_id=SignId.REDUCED_APPETITE,
            value=FindingValue.YES,
            quote="not been eating",
        ),
        Finding(
            sign_id=SignId.FROTH_NOSE_MOUTH,
            value=FindingValue.YES,
            quote="foam from her nose",
        ),
    ]
    app.dependency_overrides[get_extractor] = lambda: FakeExtractor(findings)
    response = client.post(
        "/extract",
        json={"message": "She has not been eating and there is foam from her nose"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["findings"][0] == {
        "sign_id": "reduced_appetite",
        "value": "yes",
        "quote": "not been eating",
    }
    assert body["escalate"] is True
    assert "froth_nose_mouth" in body["reason"]


def test_explicit_no_value_does_not_escalate(client):
    findings = [
        Finding(sign_id=SignId.TICKS, value=FindingValue.NO, quote="no ticks"),
    ]
    app.dependency_overrides[get_extractor] = lambda: FakeExtractor(findings)
    response = client.post("/extract", json={"message": "I see no ticks on her"})
    assert response.status_code == 200
    body = response.json()
    assert body["escalate"] is False
    assert body["reason"] == "No escalation-triggering sign was reported."
