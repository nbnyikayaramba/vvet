"""Pure unit tests for the deterministic escalation function.

No LLM, no HTTP — just the Python rule from the contract.
"""

import pytest

from app.escalation import decide_escalation
from app.schemas import Finding, FindingValue, SignId

ESCALATION_SIGNS = [SignId.FROTH_NOSE_MOUTH, SignId.UNABLE_TO_STAND, SignId.CLOUDY_EYE]
NON_ESCALATION_SIGNS = [
    SignId.REDUCED_APPETITE,
    SignId.WEAKNESS,
    SignId.SWELLING,
    SignId.RAPID_BREATHING,
    SignId.RED_URINE,
    SignId.TICKS,
]


def make(sign_id, value):
    return Finding(sign_id=sign_id, value=value, quote="quote text")


@pytest.mark.parametrize("sign", ESCALATION_SIGNS)
def test_yes_on_escalation_sign_escalates(sign):
    escalate, reason = decide_escalation([make(sign, FindingValue.YES)])
    assert escalate is True
    assert sign.value in reason


@pytest.mark.parametrize("sign", ESCALATION_SIGNS)
def test_not_sure_on_escalation_sign_escalates(sign):
    escalate, reason = decide_escalation([make(sign, FindingValue.NOT_SURE)])
    assert escalate is True


@pytest.mark.parametrize("sign", ESCALATION_SIGNS)
def test_no_on_escalation_sign_never_escalates(sign):
    escalate, reason = decide_escalation([make(sign, FindingValue.NO)])
    assert escalate is False
    assert reason == "No escalation-triggering sign was reported."


@pytest.mark.parametrize("sign", NON_ESCALATION_SIGNS)
@pytest.mark.parametrize("value", list(FindingValue))
def test_non_escalation_signs_never_escalate(sign, value):
    escalate, _ = decide_escalation([make(sign, value)])
    assert escalate is False


def test_empty_findings_no_escalation():
    escalate, reason = decide_escalation([])
    assert escalate is False
    assert reason == "No escalation-triggering sign was reported."


def test_mixed_findings_escalates_and_reason_lists_triggers():
    findings = [
        make(SignId.TICKS, FindingValue.YES),
        make(SignId.UNABLE_TO_STAND, FindingValue.NOT_SURE),
        make(SignId.FROTH_NOSE_MOUTH, FindingValue.YES),
        make(SignId.CLOUDY_EYE, FindingValue.NO),
    ]
    escalate, reason = decide_escalation(findings)
    assert escalate is True
    assert "unable_to_stand" in reason
    assert "froth_nose_mouth" in reason
    # 'no' findings are not escalation triggers
    assert "cloudy_eye" not in reason
