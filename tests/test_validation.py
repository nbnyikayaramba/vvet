"""Unit tests for strict validation of raw LLM output."""

import pytest

from app.extractor import ExtractionError
from app.schemas import Finding, FindingValue, SignId
from app.validation import validate_findings

MESSAGE = "My cow has not been eating and there is foam from her nose"


def valid_payload():
    return {
        "findings": [
            {"sign_id": "reduced_appetite", "value": "yes", "quote": "not been eating"},
            {"sign_id": "froth_nose_mouth", "value": "yes", "quote": "foam from her nose"},
        ]
    }


def test_valid_payload_passes():
    findings = validate_findings(valid_payload(), MESSAGE)
    assert findings == [
        Finding(sign_id=SignId.REDUCED_APPETITE, value=FindingValue.YES, quote="not been eating"),
        Finding(sign_id=SignId.FROTH_NOSE_MOUTH, value=FindingValue.YES, quote="foam from her nose"),
    ]


def test_empty_findings_list_passes():
    assert validate_findings({"findings": []}, MESSAGE) == []


def test_unknown_sign_id_rejected():
    payload = {"findings": [{"sign_id": "coughing", "value": "yes", "quote": "coughing"}]}
    with pytest.raises(ExtractionError, match="schema validation"):
        validate_findings(payload, MESSAGE)


def test_invalid_value_rejected():
    payload = {"findings": [{"sign_id": "ticks", "value": "maybe", "quote": "maybe ticks"}]}
    with pytest.raises(ExtractionError, match="schema validation"):
        validate_findings(payload, MESSAGE)


def test_duplicate_sign_id_rejected():
    payload = {
        "findings": [
            {"sign_id": "ticks", "value": "yes", "quote": "not been eating"},
            {"sign_id": "ticks", "value": "no", "quote": "not been eating"},
        ]
    }
    with pytest.raises(ExtractionError, match="duplicate sign_id"):
        validate_findings(payload, MESSAGE)


def test_fabricated_quote_rejected():
    payload = {
        "findings": [{"sign_id": "weakness", "value": "yes", "quote": "she is very weak today"}]
    }
    with pytest.raises(ExtractionError, match="exact substring"):
        validate_findings(payload, MESSAGE)


def test_empty_quote_rejected():
    payload = {"findings": [{"sign_id": "weakness", "value": "yes", "quote": ""}]}
    with pytest.raises(ExtractionError):
        validate_findings(payload, MESSAGE)


def test_extra_top_level_field_rejected():
    payload = valid_payload()
    payload["escalate"] = False  # model tried to decide escalation itself
    with pytest.raises(ExtractionError, match="schema validation"):
        validate_findings(payload, MESSAGE)


def test_extra_field_on_finding_rejected():
    payload = valid_payload()
    payload["findings"][0]["advice"] = "call the vet immediately"
    with pytest.raises(ExtractionError, match="schema validation"):
        validate_findings(payload, MESSAGE)


def test_missing_findings_key_rejected():
    with pytest.raises(ExtractionError):
        validate_findings({"results": []}, MESSAGE)


def test_non_dict_payload_rejected():
    with pytest.raises(ExtractionError):
        validate_findings(["not", "a", "dict"], MESSAGE)
