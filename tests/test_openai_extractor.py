"""Offline tests for the OpenAI extractor using a fake SDK client."""

import json

import httpx
import pytest
from openai import (
    APIConnectionError,
    APITimeoutError,
    ContentFilterFinishReasonError,
    LengthFinishReasonError,
)
from openai.types.chat import ChatCompletion

from app.extractor import ExtractionError
from app.providers import FindingsResult, OpenAIExtractor


def make_finish_reason_error(error_type) -> Exception:
    """Build the exception the SDK raises from chat.completions.parse."""
    reason = "length" if error_type is LengthFinishReasonError else "content_filter"
    completion = ChatCompletion(
        id="chatcmpl-test",
        choices=[{
            "index": 0,
            "finish_reason": reason,
            "message": {"role": "assistant", "content": None},
        }],
        created=0,
        model="gpt-6-luna",
        object="chat.completion",
    )
    return error_type(completion=completion)


class FakeMessage:
    """Stand-in for the SDK's message object.

    ``content`` is the raw JSON string the app actually validates; ``parsed``
    mirrors real SDK behaviour but is intentionally not relied on.
    """

    def __init__(self, content):
        self.content = content
        self.parsed = None
        self.refusal = None if content else "refused"


class FakeChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeResponse:
    def __init__(self, content):
        self.choices = [FakeChoice(content)]


class FakeCompletions:
    """Stand-in for client.chat.completions."""

    def __init__(self, content=None, error: Exception | None = None):
        self._content = content
        self._error = error

    async def parse(self, **kwargs):
        assert kwargs["response_format"] is FindingsResult
        if self._error is not None:
            raise self._error
        return FakeResponse(self._content)


class FakeClient:
    def __init__(self, content=None, error: Exception | None = None):
        self.chat = type("Chat", (), {"completions": FakeCompletions(content, error)})()


def make_extractor(content=None, error=None) -> OpenAIExtractor:
    return OpenAIExtractor(api_key="test-key", client=FakeClient(content, error))


def payload(*findings: dict) -> str:
    return json.dumps({"findings": list(findings)})


MESSAGE = "My cow has not been eating since yesterday"

VALID_CONTENT = payload(
    {"sign_id": "reduced_appetite", "value": "yes", "quote": "not been eating"}
)


@pytest.mark.anyio
async def test_extract_returns_structured_findings():
    extractor = make_extractor(content=VALID_CONTENT)
    findings = await extractor.extract(MESSAGE)
    assert len(findings) == 1
    assert findings[0].sign_id.value == "reduced_appetite"
    assert findings[0].quote == "not been eating"


@pytest.mark.anyio
async def test_refusal_or_missing_output_fails_closed():
    extractor = make_extractor(content=None)
    with pytest.raises(ExtractionError, match="no parseable structured output"):
        await extractor.extract(MESSAGE)


@pytest.mark.anyio
async def test_malformed_json_fails_closed():
    extractor = make_extractor(content="{not valid json")
    with pytest.raises(ExtractionError, match="invalid JSON"):
        await extractor.extract(MESSAGE)


@pytest.mark.anyio
async def test_fabricated_quote_rejected_end_to_end():
    content = payload(
        {"sign_id": "weakness", "value": "yes", "quote": "she is extremely weak today"}
    )
    extractor = make_extractor(content=content)
    with pytest.raises(ExtractionError, match="exact substring"):
        await extractor.extract(MESSAGE)


@pytest.mark.anyio
async def test_provider_timeout_becomes_extraction_error():
    timeout = APITimeoutError(request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"))
    extractor = make_extractor(error=timeout)
    with pytest.raises(ExtractionError, match="LLM provider error"):
        await extractor.extract(MESSAGE)


@pytest.mark.anyio
async def test_provider_outage_becomes_extraction_error():
    outage = APIConnectionError(request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"))
    extractor = make_extractor(error=outage)
    with pytest.raises(ExtractionError, match="LLM provider error"):
        await extractor.extract(MESSAGE)


@pytest.mark.anyio
@pytest.mark.parametrize(
    "error_type",
    [LengthFinishReasonError, ContentFilterFinishReasonError],
)
async def test_parse_finish_reason_errors_become_extraction_error(error_type):
    # These are raised by the SDK's parse() on truncated/filtered structured
    # output. They are OpenAIError but NOT APIStatusError subclasses, so they
    # need explicit handling to avoid an unhandled HTTP 500.
    extractor = make_extractor(error=make_finish_reason_error(error_type))
    with pytest.raises(ExtractionError, match="LLM provider error"):
        await extractor.extract(MESSAGE)
