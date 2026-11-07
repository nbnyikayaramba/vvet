"""API-level tests for V-Vet's five example messages.

The LLM is mocked: these tests verify the API contract, validation and
escalation plumbing — they do NOT prove real extraction accuracy.
"""

from app.schemas import Finding, FindingValue, SignId

EXAMPLE_1 = "My cow has not been eating since yesterday and she is breathing very fast."
EXAMPLE_2 = "There's foam coming from her nose and she can't get up."
EXAMPLE_3 = "I think the eyes look a bit cloudy but I'm not sure. She has mbandabvu under the ear."
EXAMPLE_4 = "The calf is fine, just lots of ticks on the ears."
EXAMPLE_5 = "Mombe yangu haisi kudya, and her urine is dark like tea."


def post_extract(client, message):
    return client.post("/extract", json={"message": message})


def test_example_1_appetite_and_rapid_breathing_no_escalation(client, inject_findings):
    inject_findings([
        Finding(sign_id=SignId.REDUCED_APPETITE, value=FindingValue.YES, quote="not been eating"),
        Finding(sign_id=SignId.RAPID_BREATHING, value=FindingValue.YES, quote="breathing very fast"),
    ])
    response = post_extract(client, EXAMPLE_1)
    assert response.status_code == 200
    body = response.json()
    assert body["findings"] == [
        {"sign_id": "reduced_appetite", "value": "yes", "quote": "not been eating"},
        {"sign_id": "rapid_breathing", "value": "yes", "quote": "breathing very fast"},
    ]
    assert body["escalate"] is False
    assert body["reason"] == "No escalation-triggering sign was reported."


def test_example_2_froth_and_unable_to_stand_escalates(client, inject_findings):
    inject_findings([
        Finding(sign_id=SignId.FROTH_NOSE_MOUTH, value=FindingValue.YES, quote="foam coming from her nose"),
        Finding(sign_id=SignId.UNABLE_TO_STAND, value=FindingValue.YES, quote="can't get up"),
    ])
    response = post_extract(client, EXAMPLE_2)
    assert response.status_code == 200
    body = response.json()
    assert body["escalate"] is True
    assert "froth_nose_mouth" in body["reason"]
    assert "unable_to_stand" in body["reason"]


def test_example_3_cloudy_eye_not_sure_and_mbandabvu_escalates(client, inject_findings):
    inject_findings([
        Finding(sign_id=SignId.CLOUDY_EYE, value=FindingValue.NOT_SURE, quote="I think the eyes look a bit cloudy"),
        Finding(sign_id=SignId.SWELLING, value=FindingValue.YES, quote="mbandabvu"),
    ])
    response = post_extract(client, EXAMPLE_3)
    assert response.status_code == 200
    body = response.json()
    assert body["findings"][0]["value"] == "not_sure"
    assert body["findings"][1]["sign_id"] == "swelling"
    # not_sure on an escalation sign must escalate
    assert body["escalate"] is True


def test_example_4_ticks_only_no_escalation(client, inject_findings):
    inject_findings([
        Finding(sign_id=SignId.TICKS, value=FindingValue.YES, quote="lots of ticks on the ears"),
    ])
    response = post_extract(client, EXAMPLE_4)
    assert response.status_code == 200
    body = response.json()
    assert [f["sign_id"] for f in body["findings"]] == ["ticks"]
    assert body["escalate"] is False


def test_example_5_shona_and_tea_urine_no_escalation(client, inject_findings):
    inject_findings([
        Finding(sign_id=SignId.REDUCED_APPETITE, value=FindingValue.YES, quote="haisi kudya"),
        Finding(sign_id=SignId.RED_URINE, value=FindingValue.YES, quote="dark like tea"),
    ])
    response = post_extract(client, EXAMPLE_5)
    assert response.status_code == 200
    body = response.json()
    assert {f["sign_id"] for f in body["findings"]} == {"reduced_appetite", "red_urine"}
    assert body["escalate"] is False


def test_explicit_negative_reported_as_no(client, inject_findings):
    inject_findings([
        Finding(sign_id=SignId.TICKS, value=FindingValue.NO, quote="no ticks"),
    ])
    response = post_extract(client, "I checked and she has no ticks at all")
    assert response.status_code == 200
    body = response.json()
    assert body["findings"] == [{"sign_id": "ticks", "value": "no", "quote": "no ticks"}]
    assert body["escalate"] is False


def test_empty_findings_is_valid_no_escalation(client, inject_findings):
    inject_findings([])
    response = post_extract(client, "The cow gave birth last week, all is well")
    assert response.status_code == 200
    body = response.json()
    assert body["findings"] == []
    assert body["escalate"] is False
    assert body["reason"] == "No escalation-triggering sign was reported."
