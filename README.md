# V-Vet Symptom Extractor

A small FastAPI service that extracts cattle clinical signs from farmer
messages (English, Shona, or mixed) and flags messages that need urgent
veterinary escalation.

`POST /extract` accepts `{"message": "..."}` and returns structured findings,
a deterministic escalation decision, and a machine-generated reason:

```json
{
  "findings": [{"sign_id": "reduced_appetite", "value": "yes", "quote": "not been eating"}],
  "escalate": false,
  "reason": "No escalation-triggering sign was reported."
}
```

## Prerequisites

- Python 3.11+
- An OpenAI API key (only needed for real extraction; tests run without one)

## Setup

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Windows Git Bash / Linux / macOS:
# source .venv/Scripts/activate  (or .venv/bin/activate)

pip install -r requirements.txt
```

## Configuration

Copy `.env.example` to `.env` and add your key:

```bash
cp .env.example .env
# then edit .env:
#   LLM_API_KEY=sk-...
```

`.env` is gitignored — never commit it. Optional overrides: `LLM_MODEL`
(default `gpt-6-luna`; see [MODEL_DECISIONS.md](MODEL_DECISIONS.md) for why),
`LLM_TIMEOUT_SECONDS` (default 20), `LLM_BASE_URL` (only for OpenAI-compatible
proxies).

Without a key the app still starts, but every extraction request fails closed
with HTTP 503 instead of returning a falsely reassuring empty result.

## Run

```bash
uvicorn app.main:app --reload
```

Try it:

```bash
curl -X POST http://127.0.0.1:8000/extract \
  -H "Content-Type: application/json" \
  -d '{"message": "My cow has not been eating since yesterday and she is breathing very fast."}'
```

## Test (offline, no key or network needed)

```bash
pytest -q
```

The suite mocks the LLM provider and covers the API contract, the five
assessment examples, pure escalation rules, strict output validation, and
failure handling. Mocked tests verify plumbing — they do **not** demonstrate
real extraction accuracy. During development the five example messages were
also run against the live model (`gpt-6-luna`; the `gpt-6-astra` comparison is
in [MODEL_DECISIONS.md](MODEL_DECISIONS.md)) and returned the expected
signs, values, quotes and escalation decisions; that is a smoke test, not an
accuracy evaluation.

## How it works

```
POST /extract ──► schemas.ExtractRequest (message: non-empty string)
      │
      ▼
providers.OpenAIExtractor ──► OpenAI structured output (JSON schema)
      │                          system prompt: extract signs only, verbatim
      │                          quotes, negation→no, doubt→not_sure,
      │                          mbandabvu→swelling, tea urine→red_urine
      ▼
validation.validate_findings ──► strict local re-validation of the raw JSON
      │                          extra fields forbidden, 9 sign IDs only,
      │                          yes|no|not_sure only, no duplicates,
      │                          every quote an exact substring of the input
      ▼
escalation.decide_escalation ──► pure Python, no LLM involved:
      │                          escalate iff froth_nose_mouth,
      │                          unable_to_stand or cloudy_eye is
      │                          yes|not_sure; reason generated here
      ▼
schemas.ExtractResponse (findings, escalate, reason)
```

Key design decisions:

- **The LLM extracts findings only.** It cannot set `escalate`, write the
  `reason`, or add fields — extra fields in its output are rejected.
- **Exact-quote policy.** Every quote must be a non-empty verbatim substring
  of the farmer's message; anything else is a controlled error. This keeps the
  evidence auditable by a vet reading the original message.
- **Python-only escalation.** The rule lives in `app/escalation.py` as a pure
  function, independently testable and editable without touching the LLM
  path. Changing the rule means editing one frozenset.
- **Fail-closed errors.** Provider timeouts, outages, refusals, malformed
  JSON, schema violations, duplicate signs, or fabricated quotes all produce
  HTTP 503 with a readable detail — never a silently empty `findings` list.

## Evaluating real accuracy

The bundled tests are mocked. To evaluate the real model I would:

1. Build a human-labelled set of ~100–300 real-style messages across English,
   Shona, and mixed code-switching, covering all nine signs plus negatives
   and uncertain phrasing.
2. Label each message with the correct sign IDs and values, and mark the
   exact span each quote should cover.
3. Score per sign: precision/recall (was the sign reported when present, and
   only when present?), value classification accuracy (yes/no/not_sure), and
   quote correctness (exact substring match against the labelled span).
4. Report **missed escalations** separately — a false negative on
   `froth_nose_mouth`, `unable_to_stand`, or `cloudy_eye` is the costliest
   error and deserves its own recall number.
5. Iterate on the system prompt (and model choice) against errors found,
   keeping a held-out split to check for overfitting.

## Limitations

- The five contract examples pass against the live model, but broader
  extraction quality on real farmer messages is unevaluated; the mocked
  tests say nothing about it.
- Single LLM provider (OpenAI); no retry policy beyond the client timeout.
- No per-farmer history, no message persistence, no authentication — this is
  an assessment-scoped extraction endpoint.
- The nine-sign vocabulary is fixed by contract; longer Shona-only messages
  have not been stress-tested.

With more time I would: gather and label real English/Shona data, run the
evaluation above, add provider retries with backoff, add a second provider
for comparison, and have a clinician review borderline extraction cases.

## AI assistance disclosure

This project was built with the help of an AI coding assistant (Kimi Code):
it scaffolded the project structure, implemented the OpenAI provider
integration, the validation and escalation modules, and the test suite, and
it reviewed the code for contract compliance. I directed the work prompt by
prompt, reviewed every change, ran the tests, and made the design decisions
(described above). I can explain and edit every module — including changing
an escalation rule or adding a sign — myself.

## Layout

```
app/
  main.py        FastAPI app, POST /extract, dependency injection
  schemas.py     Pydantic v2 models; the 9 sign IDs and 3 values as enums
  extractor.py   Extractor protocol + fail-closed unconfigured placeholder
  providers.py   OpenAI extractor (structured output) + factory
  validation.py  strict local validation of raw LLM JSON
  escalation.py  pure escalation rule and reason generation
  config.py      env/.env configuration
tests/           offline pytest suite (mocked LLM)
```
