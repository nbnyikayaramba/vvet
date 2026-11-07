# Model decisions (ADR)

This document records which OpenAI model the extractor uses and why, so the
choice can be defended and revised. Evaluation data below is from live runs
on 2026-11-08 against the five assessment example messages.

## Decision

**Use `gpt-6-luna` as the extraction model** (code default in
`app/config.py`; overridable via `LLM_MODEL` in `.env`).

## Context

Sign extraction from a farmer message is a bounded structured-output task:
identify zero or more of nine known signs, classify each as
`yes|no|not_sure`, and copy a verbatim quote. It needs solid instruction
following across English/Shona code-switching, not deep multi-step reasoning.
The deployment context is a WhatsApp-style triage flow: high message volume,
short messages, cost and latency matter, and every response passes through
the same strict local validation regardless of model.

## Alternatives considered

| Model | Role | Verdict |
|---|---|---|
| `gpt-6-luna` | Fast, low-cost GPT-6 chat model (1M-token context) | **Chosen** — passes all five examples at a fraction of frontier cost/latency |
| `gpt-6-astra` | Frontier reasoning model, strongest alignment per OpenAI | Evaluated — passes all five examples, but its reasoning depth buys nothing for a nine-class extraction task at materially higher cost/latency |
| `gpt-6.1-sol` / `gpt-6.1-terra` | Mid/high-tier GPT-6.1 | Not evaluated — no expected gain over Luna for this task class |
| `gpt-4o-mini` (previous default) | Legacy mini | Superseded by the GPT-6 family; Luna matches or exceeds it on our smoke test |

Sources: [Introducing GPT-6 Sol and Luna — OpenAI](https://openai.com/index/introducing-gpt-6-sol-and-luna/),
[GPT-6 family overview and API identifiers](https://blog.laozhang.ai/en/posts/gpt-6-sol-vs-terra-vs-luna-vs-astra),
[OpenRouter GPT-6 Astra vs Luna comparison](https://openrouter.ai/compare/openai/gpt-6-astra/openai/gpt-6-luna).

## Evaluation (live, 2026-11-08)

Both models were run through the real extraction pipeline
(`OpenAIExtractor` → strict validation → deterministic escalation) on the
five contract examples. Expected: ex. 1 appetite+rapid breathing (no
escalation); ex. 2 froth+unable to stand (escalate); ex. 3 cloudy eye
not_sure + mbandabvu→swelling (escalate); ex. 4 ticks only; ex. 5 Shona
"haisi kudya"→reduced appetite + tea urine→red_urine.

| Example | gpt-6-luna | gpt-6-astra |
|---|---|---|
| 1 | ✅ both signs, verbatim quotes, no escalation | ✅ both signs, verbatim quotes, no escalation |
| 2 | ✅ both signs, **escalation** | ✅ both signs, **escalation** |
| 3 | ✅ cloudy_eye `not_sure` + swelling, verbatim "mbandabvu", **escalation** | ✅ same, **escalation** |
| 4 | ✅ ticks only | ✅ ticks only |
| 5 | ✅ reduced_appetite (quote "haisi kudya") + red_urine | ✅ same |

Result: 5/5 on both. Luna was also observably faster per call.

Caveats: five messages are a smoke test, not an accuracy evaluation. A
serious comparison needs the labelled English/Shona set described in the
README, with missed-escalation recall reported separately.

## When to revisit

- Escalate to `gpt-6-astra` if labelled evaluation shows Luna missing
  subtle negation/uncertainty cues or making value-classification errors on
  real messages — Astra is the fallback quality lever.
- Re-run this comparison if OpenAI revises Luna/Astra or if a labelled set
  shows a cheaper model (e.g. a future nano) is sufficient.
- The model choice is a single environment variable (`LLM_MODEL`); no code
  change is needed to switch.
