"""Regression tests pinning the extraction prompt to V-Vet's sign definitions.

The prompt is the contract with the model: these tests fail loudly if the
definitions are accidentally loosened (e.g. "any swelling") or narrowed
beyond V-Vet's examples. Model behaviour itself is covered by the live
tests in test_live_extraction.py.
"""

import re

from app.providers import SYSTEM_PROMPT

# Collapse whitespace so assertions do not depend on prompt line wrapping.
PROMPT = re.sub(r"\s+", " ", SYSTEM_PROMPT)


def test_sign_definitions_match_vvet_spec():
    for phrase in [
        # reduced_appetite: eating less, or stopped eating/grazing
        "eating less than usual, or has stopped eating or grazing",
        # weakness
        "dull, weak, or lying down more than usual",
        # swelling: specific locations + mbandabvu
        "a lump below the ear, under the jaw, or in front of the shoulder",
        "mbandabvu",
        # rapid_breathing
        "breathing fast while resting",
        # cloudy_eye
        "the eye looks white, grey or milky",
        # froth_nose_mouth
        "foam or froth from the nose or mouth",
        # red_urine
        "urine that is red, brown, or the colour of strong tea",
        # unable_to_stand
        "cannot get up or stay standing",
        # ticks
        "ticks can be seen on the animal",
    ]:
        assert phrase in PROMPT, f"missing definition phrase: {phrase!r}"


def test_swelling_not_broadened_beyond_specified_locations():
    # The prompt must NOT contain a blanket "any swelling" definition, and
    # must carry the explicit narrowing rule.
    assert "any swelling" not in PROMPT
    assert "swollen leg or any other body part is NOT this sign" in PROMPT


def test_rapid_breathing_not_broadened_to_any_difficulty():
    # "breathing fast or with difficulty" was the old, too-loose definition.
    assert "or with difficulty" not in PROMPT
    assert "General breathing difficulty without fast breathing is NOT this sign" in PROMPT


def test_negation_uncertainty_and_quote_rules_intact():
    for phrase in [
        'has value "no"',
        'has value "not_sure"',
        "exact verbatim substring",
        "Report ONLY signs explicitly mentioned",
    ]:
        assert phrase in PROMPT, f"missing rule phrase: {phrase!r}"
