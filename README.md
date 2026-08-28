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

Changing the chunk size of a retrieval pipeline changes which passages come back, and with them the answers.
The test suite still passes, because the pipeline is intact: it returns a paragraph with a source attached, the way it did before.
What the suite cannot tell me is whether that answer still comes from the part of the corpus it should.

The unit of work here is a recorded run.
Each run is stored together with the configuration it ran under and with the chunks it retrieved, including score and rank.
Running the same gold questions against two configurations then produces a diff, question by question.
The chunk list is the part I work with: it shows which passage an answer was assembled from, so a change has a traceable place to start.

## Why this exists

Retrieval quality is the first thing I look at when a RAG answer goes wrong, and it is the thing that is hardest to see afterwards.
The pipeline reports an answer, not the passages it was built from, and the previous run is gone by the time the new one finishes.

So the comparison usually happens by reading two answers side by side and forming an impression.
That works for obvious breakage and stops working at the point where it matters, when one configuration is slightly better on some questions and slightly worse on others.

This harness keeps enough of each run to replace that impression with a diff.
The [RAGGY paper](https://arxiv.org/abs/2504.13587) describes the same working pattern from the other direction: developers debugging RAG pipelines check retrieval first, want to see which chunks were and were not returned, and compare strategies against each other.

## Status

**Phase 2, in progress.** The harness itself does not exist: no chunking, no embeddings, no database, no runs, no diff.
What runs today is the layer underneath it, the service and the raw LLM calls the pipeline gets built into.

| Runs today | Where |
|---|---|
| A FastAPI service with `/health`, `/version`, `/analyze`, `/analyze/stream` and `/rag/analyze` | `src/harness/api/` |
| `OpenAiLlmClient` behind five role protocols, wired through FastAPI's `Depends` | `core/interfaces.py` |
| Answers streamed token by token over server-sent events | `api/routers/analyze.py`, `api/sse.py` |
| Structured outputs through the Responses API with `strict`, validated against pydantic | `infrastructure/llm/client.py` |
| Tool calling written by hand, the loop in the adapter and the tools provider-neutral | `core/tools.py` |
| One error table both exits read, the HTTP handler and the SSE error frame | `api/errors.py` |
| Token usage per call, cached and reasoning tokens included | `TokenUsage` in `core/interfaces.py` |
| 13 gold questions against a vendored 24-document FastAPI docs subset | `data/gold_questions.json`, `data/corpus/` |
| 175 tests, `mypy --strict` and `pyright --strict` clean | `tests/` |

The five roles are `TextCompleter`, `TextStreamer`, `StructuredCompleter`, `ToolCompleter` and `StructuredToolCompleter`.
A caller depends on the single role it uses, so a method added for one of them cannot break a test double that never touches it.

Errors are classified by who has to act on them.
`LlmConfigurationError` and `LlmToolError` are mine and answer 500; `LlmUnavailableError` and `LlmResponseFormatError` are the provider's and answer 502, the second one for a response that arrives but cannot be used.

Two things the list does not say. `/rag/analyze` runs the whole tool loop, but the tool it calls searches a two-section stub, not the vendored corpus: the corpus is read by the gold-question tests, which check that every expected source of every question exists on disk. And the provider is a configuration value rather than an architectural decision, because the OpenAI client speaks to any OpenAI-compatible endpoint through a different `base_url`.

`strict` guarantees structure, never meaning. `scripts/schema_constraint_probe.py` measures where that line runs against the real API, and its docstring records the result with model, date and SDK version, because the answer belongs to a provider rather than to pydantic.

The same gap sits between a prompt and what a model does with it.
Every test around the tool path fakes the model, so they show that a system prompt is passed on and never that it is followed.
`scripts/rag_smoke.py` asks the real one, against a corpus stating facts the model cannot know, so an answer from memory reads as visibly wrong rather than merely unsourced.

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

What a fake model cannot answer is left to the scripts in `scripts/`. Each one spends tokens, states in its docstring what it measures, and records its last result with model, date and SDK version.

## Why there is nothing to assert

A unit test names an input and the output it expects.
Retrieval does not hand me that pair.
The same question against the same corpus returns a different set of passages once the embedding model or the chunk size changes, and there is no correct set I could write down beforehand.

What I can observe is the direction of a change.
If a question was answered from the section I expected before a configuration change, and from a different one after, that is a regression I can name, whether or not the generated text still reads well.

Reading that direction requires the earlier run to still be around in enough detail to compare against.
The retrieved passages, their scores and their ranks usually do not outlive the request they were made for.
Here they are written down, and the diff reads them later.

---

> discrimen (Latin) - "a dividing line, a decisive point". Which is exactly what a regression check looks for between two runs.

## License

Licensed under the [MIT License](LICENSE). © 2026 Felix Wahl.

Related: [Chartula](https://github.com/goldbarth/chartula), a grounded changelog CLI in .NET, and [goldbarth.dev](https://www.goldbarth.dev/), where the experiments behind both are written up while they run.
