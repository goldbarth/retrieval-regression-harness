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

**Phase 2, in progress.** There is nothing to install yet and nothing to point at a corpus.
The harness itself does not exist: no documents, no chunks, no embeddings, no database, no runs, no diff.

What exists is the layer underneath it, the service skeleton and the raw LLM calls the pipeline gets built into:

- A FastAPI service with `/health`, `/version`, `/analyze` and `/rag/analyze`
- `OpenAiLlmClient` behind three role protocols, `TextCompleter`, `StructuredCompleter` and `ToolCompleter`, wired through FastAPI's `Depends`.
  A caller depends on the single role it uses, so a method added for one of them cannot break a test double that never touches it.
- Error classification split by who has to act on the failure.
  `LlmConfigurationError` (500) and `LlmToolError` (500) are mine, a wrong model name and a tool handler that raised.
  `LlmUnavailableError` (502) and `LlmResponseFormatError` (502) are the provider's, the second one for a response that arrives but cannot be used.
- `LlmConfig` as a pydantic model, passed into the pipeline rather than read from a global
- Token usage recorded per call, cached and reasoning tokens included, because they change the price and cannot be reconstructed afterwards.
  `max_output_tokens` caps the expensive side, and a truncated answer is marked rather than returned as if it were whole.
- Structured outputs through the Responses API with `strict`, validated against pydantic, with the judge as the first caller
- Tool calling built by hand once, the loop in the adapter and the tools provider-neutral in `core`.
  A run that used up its rounds or was cut off returns the partial answer with the reason attached, because those tokens were spent either way and a truncated answer has to be distinguishable from a finished one.
- 101 tests, `mypy --strict` and `pyright --strict` clean

The provider is a configuration value here, not an architectural decision.
The OpenAI client speaks to any OpenAI-compatible endpoint through a different `base_url`, so a fallback to Groq changes a setting rather than a layer.

`strict` guarantees structure, never meaning. `scripts/schema_constraint_probe.py` measures where that line runs against the real API, and its docstring records the result with model, date and SDK version, because the answer belongs to a provider rather than to pydantic.

The same gap sits between a prompt and what a model does with it.
Every test around the tool path fakes the model, so they show that a system prompt is passed on and never that it is followed.
`scripts/rag_smoke.py` asks the real one, against a corpus stating facts the model cannot know, so an answer from memory reads as visibly wrong rather than merely unsourced.

Retrieval, the schema and the first diff are phase 3.
The [roadmap](docs/ROADMAP.md) says what lands when.

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