# Decisions

Decisions that were made and that something already runs on.
The [README](../README.md) says what runs; [ROADMAP.md](ROADMAP.md) says what is intended.
This file says why the running part looks the way it does.

An entry is numbered because a number is an address: a code comment or a write-up can
point at `DECISIONS.md#3` without repeating the reasoning. Numbers are never reused.
The date is the day the decision was made, taken from the commit that carried it, not
the day it was written down here. Entries are appended, and a reversed decision keeps
its number and gets a note.

Intentions about parts that are not built yet live in [ROADMAP.md](ROADMAP.md) instead,
and move here once something runs against them.

## 1. Five role protocols, not one client interface

*Decided 2026-08-18, extended 2026-08-19 and 2026-08-24. `src/harness/core/interfaces.py`.*

Every capability of the LLM adapter is its own `Protocol`: `TextCompleter`,
`StructuredCompleter`, `TextStreamer`, `ToolCompleter`, `StructuredToolCompleter`.
`LlmClient` inherits all five, and exists for wiring only.

A caller declares the single role it uses, so a method added for one role cannot break a
test double that never touches it. That is the property being bought, and it is bought
against the alternative that keeps suggesting itself: one interface with an optional
argument that switches behaviour. `stream: bool` on `complete()`, `schema: type[T] | None`
on `complete_with_tools()`. Both make the return type depend on the *value* of an
argument, which neither a type checker nor a test double can express, and every caller of
the plain path starts paying for a field it never asked for.

The split is therefore not about small interfaces as a principle. It is about keeping the
return type a function of the type signature.

What it costs: five names to know instead of one, and a genuinely shared change touches
five places. `Protocol` has to stay in the bases of `LlmClient`, or it silently becomes a
plain ABC that adapters would have to inherit from.

## 2. One error ladder, one table, looked up along the MRO

*Decided 2026-08-27. `src/harness/api/errors.py`, `src/harness/main.py`.*

`LlmError` and its four subclasses are classified by who has to act on the failure, not by
what went wrong technically. `LlmConfigurationError` and `LlmToolError` are ours (500);
`LlmUnavailableError` and `LlmResponseFormatError` are the provider's (502).

Every exit reads one table, `LLM_ERROR_RESPONSES`, which carries status code, the wording
the caller sees, and the wording that goes into the log. The log message travels with the
other two deliberately: one handler now answers every `LlmError`, so the log wording is
the third thing that would otherwise be left behind next to the class.

The lookup walks `type(exc).__mro__` rather than testing `isinstance` against an ordered
list. With an ordered list, the order of rows decides which row a subclass gets, and a row
moved for readability could change an answer. The MRO puts the most specific class first
by construction, so a new subclass inherits its parent's answer until it is given a row.
`LlmError` itself has a row rather than a fallback branch, so the walk always terminates.

The reason there is a table at all: the SSE endpoint added a second exit. Before the first
byte a failure is still an ordinary 502; after it, it can only be an `event: error` frame.
The first draft handed `str(exc)` into that frame, so how much a caller learned about our
internals depended on *when* the run failed. One table removes that difference.

What it costs: an error class added without a row answers 500 silently rather than failing
loudly, and nothing catches that. `test_llm_error_answers_its_table_row` is parametrized
over the table itself, so it proves every row that exists is really delivered end to end,
and cannot notice a row that was never written.

## 3. Tools are defined provider-neutrally in `core`

*Decided 2026-08-19. `src/harness/core/tools.py`, `ToolSpec` in `interfaces.py`.*

A tool is a `ToolSpec`: a name, a pydantic model for its parameters, and a handler that
takes an instance of that model and returns a string. Nothing in it mentions OpenAI. The
adapter translates a `ToolSpec` into the provider's schema and runs the loop; `core`
neither knows nor cares which provider that is.

The tool loop was written by hand once rather than taken from a framework, because this
project's subject is the comparison between hand-built and framework-built pipelines. A
loop that was never built by hand has nothing to compare against.

What it costs: the translation into the provider's tool schema is ours to maintain, and
strict mode leaks one precondition upwards. Strict makes every field required, so an
optional parameter has to be typed `str | None` in the params model rather than given a
default, and `extra="forbid"` is mandatory. That precondition lives in `_to_tool_param`'s
docstring, not in the type: a model that breaks it still builds a schema, and the provider
rejects it as an `LlmConfigurationError` far from the tool that caused it.

## 4. `strict` guarantees structure, never meaning

*Decided 2026-08-18. `scripts/schema_constraint_probe.py`.*

Structured outputs go through the Responses API with `strict: True` and are validated
against pydantic afterwards. The validation is not redundant: `strict` guarantees that the
answer has the shape of the schema, and nothing about whether the values in it are true.

Where exactly that line runs is a property of the provider, not of pydantic, so it was
measured rather than assumed. `schema_constraint_probe.py` asks two questions per
constraint: was the schema *accepted*, and was the constraint *honored*. Only the first is
answered by the provider's error message, which is why every probe asks the model for a
value that violates its own constraint. A constraint that is accepted and then ignored is
the dangerous case, because the schema reads like a guarantee and is not one.

Measured 2026-08-18 against gpt-5.6-luna, openai SDK 2.53.0: `set[str]`, `tuple[str, int]`
and `dict[str, str]` are rejected outright; ten constraints from `max_length` to `Literal`
were accepted and enforced. The docstring carries the full table with model, date and SDK
version, because a result without those three is a claim rather than evidence.

The same gap sits one level up, between a prompt and what a model does with it. Every test
around the tool path fakes the model, so they prove that a system prompt is passed on and
never that it is followed. `scripts/rag_smoke.py` asks the real model against a corpus
stating facts it cannot know, so an answer from memory reads as visibly wrong rather than
merely unsourced.

What it costs: two scripts that spend tokens and cannot run in CI, and a table that is
only as current as its last run.
