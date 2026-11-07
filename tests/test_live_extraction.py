"""Live-model regression tests — OPT-IN via VVET_RUN_LIVE=1.

These call the real OpenAI API and need LLM_API_KEY plus network access.
They are SKIPPED by default so ``pytest -q`` stays fully offline.

Unlike the mocked suite, these exercise the model's real extraction
semantics: the narrowed swelling/rapid-breathing definitions, negation and
uncertainty handling, verbatim quotes, and the five V-Vet contract examples.
Mocked tests validate application logic only — never model accuracy.
"""

import os

import pytest

from app.config import LLM_API_KEY
from app.escalation import decide_escalation
from app.providers import OpenAIExtractor
from app.schemas import FindingValue, SignId

pytestmark = [
    pytest.mark.skipif(
        os.getenv("VVET_RUN_LIVE") != "1" or not LLM_API_KEY,
        reason="live tests: set VVET_RUN_LIVE=1 with LLM_API_KEY configured",
    ),
    pytest.mark.anyio,
]


@pytest.fixture
async def extractor():
    # Function-scoped on purpose: anyio gives each test its own event loop,
    # and an AsyncOpenAI client must not be shared across closed loops.
    ex = OpenAIExtractor(timeout_seconds=60)
    yield ex
    # Close the httpx connection pool so loop shutdown stays clean; an
    # unclosed pool's cleanup callbacks fire after the loop closes and
    # anyio reports them as spurious test failures.
    await ex._client.close()


async def extract_with_escalation(extractor, message):
    findings = await extractor.extract(message)  # strict quote validation inside
    escalate, _ = decide_escalation(findings)
    return findings, escalate


def by_sign(findings, sign_id):
    return [f for f in findings if f.sign_id == sign_id]


# ---------------------------------------------------------------------------
# The five V-Vet contract examples
# ---------------------------------------------------------------------------

async def test_example_1(extractor):
    findings, escalate = await extract_with_escalation(
        extractor, "My cow has not been eating since yesterday and she is breathing very fast."
    )
    assert {f.sign_id for f in findings} == {SignId.REDUCED_APPETITE, SignId.RAPID_BREATHING}
    assert by_sign(findings, SignId.REDUCED_APPETITE)[0].value == FindingValue.YES
    assert by_sign(findings, SignId.RAPID_BREATHING)[0].value == FindingValue.YES
    assert escalate is False  # the tightened definition must not break this


async def test_example_2(extractor):
    findings, escalate = await extract_with_escalation(
        extractor, "There's foam coming from her nose and she can't get up."
    )
    assert {f.sign_id for f in findings} == {SignId.FROTH_NOSE_MOUTH, SignId.UNABLE_TO_STAND}
    assert escalate is True


async def test_example_3(extractor):
    findings, escalate = await extract_with_escalation(
        extractor,
        "I think the eyes look a bit cloudy but I'm not sure. She has mbandabvu under the ear.",
    )
    assert by_sign(findings, SignId.CLOUDY_EYE)[0].value == FindingValue.NOT_SURE
    assert by_sign(findings, SignId.SWELLING)[0].value == FindingValue.YES
    assert escalate is True


async def test_example_4(extractor):
    findings, escalate = await extract_with_escalation(
        extractor, "The calf is fine, just lots of ticks on the ears."
    )
    assert [f.sign_id for f in findings] == [SignId.TICKS]
    assert escalate is False


async def test_example_5(extractor):
    findings, escalate = await extract_with_escalation(
        extractor, "Mombe yangu haisi kudya, and her urine is dark like tea."
    )
    assert {f.sign_id for f in findings} == {SignId.REDUCED_APPETITE, SignId.RED_URINE}
    # The model may quote a longer span; require the Shona words to be inside it.
    appetite_quote = by_sign(findings, SignId.REDUCED_APPETITE)[0].quote
    assert "haisi kudya" in appetite_quote
    assert escalate is False


# ---------------------------------------------------------------------------
# Narrowed definitions (regression cases from review)
# ---------------------------------------------------------------------------

async def test_swollen_leg_only_is_not_swelling(extractor):
    findings, _ = await extract_with_escalation(
        extractor, "My cow has a swollen leg after the long walk."
    )
    assert by_sign(findings, SignId.SWELLING) == []


async def test_lump_under_jaw_is_swelling(extractor):
    findings, _ = await extract_with_escalation(
        extractor, "There is a lump under the jaw of my cow."
    )
    assert by_sign(findings, SignId.SWELLING)[0].value == FindingValue.YES


async def test_mbandabvu_is_swelling(extractor):
    findings, _ = await extract_with_escalation(
        extractor, "Mombe yangu ine mbandabvu."
    )
    swelling = by_sign(findings, SignId.SWELLING)
    assert swelling and swelling[0].value == FindingValue.YES
    assert "mbandabvu" in swelling[0].quote


async def test_breathing_difficulty_without_fast_is_not_rapid_breathing(extractor):
    findings, _ = await extract_with_escalation(
        extractor, "My cow is struggling to breathe."
    )
    assert by_sign(findings, SignId.RAPID_BREATHING) == []


async def test_explicitly_fast_breathing_is_rapid_breathing(extractor):
    findings, _ = await extract_with_escalation(
        extractor, "She is breathing very fast even while resting."
    )
    assert by_sign(findings, SignId.RAPID_BREATHING)[0].value == FindingValue.YES


async def test_uncertain_cloudy_eye_is_not_sure_and_escalates(extractor):
    findings, escalate = await extract_with_escalation(
        extractor, "I am not sure, but the eye looks milky."
    )
    cloudy = by_sign(findings, SignId.CLOUDY_EYE)
    assert cloudy and cloudy[0].value == FindingValue.NOT_SURE
    assert escalate is True


async def test_negated_froth_is_no_and_does_not_escalate(extractor):
    findings, escalate = await extract_with_escalation(
        extractor, "There is no froth from her nose."
    )
    froth = by_sign(findings, SignId.FROTH_NOSE_MOUTH)
    # If reported at all, an explicit negative must be value "no".
    assert all(f.value == FindingValue.NO for f in froth)
    assert escalate is False
