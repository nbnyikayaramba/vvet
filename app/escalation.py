"""Deterministic escalation decision, owned by Python — never by the LLM."""

from .schemas import Finding, FindingValue, SignId

ESCALATION_SIGNS: frozenset[SignId] = frozenset(
    {SignId.FROTH_NOSE_MOUTH, SignId.UNABLE_TO_STAND, SignId.CLOUDY_EYE}
)
ESCALATION_VALUES: frozenset[FindingValue] = frozenset(
    {FindingValue.YES, FindingValue.NOT_SURE}
)


def decide_escalation(findings: list[Finding]) -> tuple[bool, str]:
    """Return (escalate, reason) for validated findings.

    Escalates iff an escalation sign has value ``yes`` or ``not_sure``.
    A value of ``no`` never escalates.
    """
    triggers = [
        f.sign_id.value
        for f in findings
        if f.sign_id in ESCALATION_SIGNS and f.value in ESCALATION_VALUES
    ]
    if triggers:
        return True, "Escalation-triggering sign(s) reported: " + ", ".join(triggers) + "."
    return False, "No escalation-triggering sign was reported."
