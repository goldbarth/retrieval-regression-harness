<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/rrh-wordmark-brackets-dark.svg">
  <img alt="rrh — retrieval regression harness" src="assets/rrh-wordmark-brackets-light.svg" width="380">
</picture>

**A regression test for a system that gives different answers to the same question.**

[![Status](https://img.shields.io/badge/status-phase%202%20in%20progress-orange?style=flat-square)](#status)
![Python 3.14](https://img.shields.io/badge/Python-3.14-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)

</div>

A RAG pipeline can quietly get worse without a single test turning red: same shape of
answer, source attached, tests green. What changed is which passages that answer came
from, and nothing in a normal test suite looks at that.

The unit of work here is a recorded run: the configuration it ran under, plus every chunk
it retrieved, with score and rank. Running the same set of hand-written questions against
two configurations produces a diff, question by question, and the chunk list is where I
start: it shows which passage an answer was built from, so a change has a traceable place
to begin.

## Why this exists

Retrieval quality is what I check first when a RAG answer looks wrong, and it's the hardest
thing to see after the fact: the pipeline hands back an answer, not the passages it came
from, and by the time the next run finishes, the previous one is gone.

I used to compare two answers side by side and go on impression. That catches obvious
breakage, but not the case that actually matters: one configuration slightly better on
some questions and slightly worse on others.

This harness keeps enough of each run to turn that impression into a diff.

## Status

**Phase 2, in progress.** The harness itself does not exist: no chunking, no embeddings, no database, no runs, no diff.
What runs today is the layer underneath it, the service and the raw LLM calls the pipeline gets built into.

| Runs today                                                                                     | Where                                      |
|------------------------------------------------------------------------------------------------|--------------------------------------------|
| A FastAPI service with `/health`, `/version`, `/analyze`, `/analyze/stream` and `/rag/analyze` | `src/harness/api/`                         |
| `OpenAiLlmClient` behind five role protocols, wired through FastAPI's `Depends`                | `core/interfaces.py`                       |
| Answers streamed token by token over server-sent events                                        | `api/routers/analyze.py`, `api/sse.py`     |
| Structured outputs through the Responses API with `strict`, validated against pydantic         | `infrastructure/llm/client.py`             |
| Tool calling written by hand, the loop in the adapter and the tools provider-neutral           | `core/tools.py`                            |
| One error table both exits read, the HTTP handler and the SSE error frame                      | `api/errors.py`                            |
| Token usage per call, cached and reasoning tokens included                                     | `TokenUsage` in `core/interfaces.py`       |
| 13 gold questions against a vendored 24-document FastAPI docs subset                           | `data/gold_questions.json`, `data/corpus/` |
| 175 tests, `mypy --strict` and `pyright --strict` clean                                        | `tests/`                                   |

The five roles are `TextCompleter`, `TextStreamer`, `StructuredCompleter`, `ToolCompleter` and `StructuredToolCompleter`.
A caller depends on the single role it uses, so a method added for one of them cannot break a test double that never touches it.

Errors are classified by who has to act on them:

| Error class                             | Answers | Because                              |
|-----------------------------------------|---------|--------------------------------------|
| `LlmConfigurationError`, `LlmToolError` | 500     | Ours to fix                          |
| `LlmUnavailableError`                   | 502     | The provider's fault                 |
| `LlmResponseFormatError`                | 502     | A response arrives but can't be used |

Two things the table above does not say:

- `/rag/analyze` runs the whole tool loop, but the tool it calls searches a
  two-section stub, not the vendored corpus. The corpus itself is read only
  by the gold-question tests, which check that every expected source exists
  on disk.
- The provider is a configuration value, not an architectural decision: the
  OpenAI client speaks to any OpenAI-compatible endpoint through a different
  `base_url`.

Two gaps a green test suite cannot see, each checked with a script that
spends tokens against the real API:

| Gap                                        | Checked by                           | Finding                                                                                                                               |
|--------------------------------------------|--------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------|
| Schema *accepted* vs. constraint *honored* | `scripts/schema_constraint_probe.py` | Recorded in the docstring with model, date and SDK version - the answer belongs to the provider, not to pydantic                      |
| Prompt *passed on* vs. prompt *followed*   | `scripts/rag_smoke.py`               | Run against a corpus stating facts the model can't know, so an answer from memory reads as visibly wrong rather than merely unsourced |

Retrieval, the schema and the first diff are phase 3.
[ROADMAP.md](docs/ROADMAP.md) says what lands when, and [DECISIONS.md](docs/DECISIONS.md) says why the part that already runs looks the way it does.

## Running it

Python 3.14 and [uv](https://docs.astral.sh/uv/). Everything except the tests needs a key.

```bash
uv sync
echo "OPENAI_API_KEY=sk-..." > .env    # TIMEOUT=30.0 is optional
uv run fastapi dev src/harness/main.py
```

`/health` and `/version` answer without a key; every other endpoint calls the provider and spends tokens.

```bash
curl localhost:8000/health
curl -X POST localhost:8000/analyze -H 'content-type: application/json' -d '{"text": "..."}'
curl -N -X POST localhost:8000/analyze/stream -H 'content-type: application/json' -d '{"text": "..."}'
```

`-N` matters on the streaming call: without it curl buffers the response and prints it in one go, which is the behaviour streaming exists to avoid.

The suite fakes the provider throughout and needs no key:

```bash
uv run pytest
uv run mypy --strict && uv run pyright && uv run ruff check
```

What a fake model cannot answer is left to the scripts in `scripts/`, described above.

## Why there is nothing to assert

A unit test names an input and the output it expects.
Retrieval does not hand me that pair.
The same question against the same corpus returns a different set of passages once the embedding model or the chunk size changes, and there is no correct set I could write down beforehand.

What I can observe is the direction of a change.
If a question was answered from the section I expected before a configuration change, and from a different one after, that is a regression I can name, whether or not the generated text still reads well.

Reading that direction requires the earlier run to still be around in enough detail to
compare against. Without persisting it, the retrieved passages, their scores and their
ranks disappear once the request that produced them ends. Here they're written down, so
the diff can still read them later.

---

> discrimen (Latin) - "a dividing line, a decisive point". Which is exactly what a regression check looks for between two runs.

## License

Licensed under the [MIT License](LICENSE). © 2026 Felix Wahl.

Related: [Chartula](https://github.com/goldbarth/chartula), a grounded changelog CLI in .NET, and [goldbarth.dev](https://www.goldbarth.dev/), where the experiments behind both are written up while they run.
