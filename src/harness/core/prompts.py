"""System prompts for the evaluation pipeline.

Kept in one module so prompt wording stays diffable line by line.
E501 is disabled here, see per-file-ignores in pyproject.toml.
"""

JUDGE_SYSTEM_PROMPT = """You are a strict evaluator for retrieval QA regression tests.

Compare the actual answer against the expected answer for the given question.
Return "correct" only if the actual answer answers the question consistently with the expected answer.
Return "incorrect" if the actual answer is missing, contradictory, unsupported, or materially incomplete.
Return "unclear" only if the comparison itself cannot be decided, for example when the question is underspecified, when the expected answer is ambiguous or self-contradictory, or when the actual answer addresses a different but defensible reading of the question that the expected answer does not cover.

"unclear" judges the test case, never the quality of the actual answer. A weak, hedged, vague, or partially correct answer is "incorrect", not "unclear". If you can decide the comparison at all, decide it.

Always provide a concise reasoning. For "incorrect", state what is wrong. For "unclear", state which part of the question or the expected answer blocks the decision.
"""


RAG_SYSTEM_PROMPT = """Answer the question from the indexed corpus, not from memory.
Call search_sections first, then ground every claim in what it returned.
If nothing relevant comes back, set answer_status to "no_relevant_sections" and say so instead of guessing."""
"""System prompt for the tool-augmented RAG endpoint.

Moved here out of api/routers/rag.py. Prompt wording is a configuration
dimension of the harness: a diff between two runs has to be able to name the
prompt that produced each one, and a prompt defined inside a router is not
covered by that. Same reason JUDGE_SYSTEM_PROMPT sits here.

The line "Cite each claim as doc_id#section." is gone. RagAnswer now carries
the citations as a field, so the provider enforces their shape structurally and
a prompt line asking for the same thing would be a second, weaker source of
truth that can drift from the schema without anything failing.
"""
