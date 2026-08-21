"""Tests for the answer contract the model has to fill.

The cross-field rule is the subject here. Everything else about RagAnswer is
plain pydantic; the relation between answer_status and citations is the part
that encodes a decision and can therefore be broken by a later edit that looks
harmless.
"""

import pytest
from pydantic import ValidationError

from harness.core.gold import ExpectedSource
from harness.core.rag import RagAnswer

SOURCE = ExpectedSource(
    doc_id="tutorial/response-model", section="response-model-priority"
)


def test_an_answered_response_carries_its_sources() -> None:
    answer = RagAnswer(
        answer_status="answered", answer="response_model wins.", citations=[SOURCE]
    )

    assert answer.citations == [SOURCE]


def test_an_answered_response_without_a_citation_is_rejected() -> None:
    """An answer claiming to be grounded and naming no source is the case the
    field exists to prevent. Caught at parse time, because by the time the
    runner has turned it into a number the answer is gone."""
    with pytest.raises(ValidationError, match="must cite at least one section"):
        RagAnswer(answer_status="answered", answer="response_model wins.", citations=[])


def test_a_no_relevant_sections_response_may_carry_no_citation() -> None:
    """The case Field(min_length=1) would have made impossible.

    The prompt tells the model to say so when retrieval returns nothing usable.
    Under a hard minimum of one its only way to comply would be to invent a
    source, and a fabricated citation is indistinguishable from a real one
    downstream.
    """
    answer = RagAnswer(
        answer_status="no_relevant_sections",
        answer="Nothing in the corpus covers this.",
        citations=[],
    )

    assert answer.citations == []


def test_a_no_relevant_sections_response_with_a_citation_is_rejected() -> None:
    # The other direction of the same rule. Claiming there was nothing and
    # naming a section at the same time is a contradiction, and the runner
    # would have to guess which half to believe.
    with pytest.raises(ValidationError, match="must not cite any section"):
        RagAnswer(
            answer_status="no_relevant_sections", answer="Nothing.", citations=[SOURCE]
        )


def test_an_unknown_answer_status_is_rejected() -> None:
    # The Literal is what makes a per-status breakdown in phase 3 hold. A
    # typo'd third state would show up as its own bucket of size one.
    with pytest.raises(ValidationError):
        RagAnswer.model_validate(
            {"answer_status": "maybe", "answer": "x", "citations": []}
        )


def test_an_unknown_field_is_rejected() -> None:
    # extra="forbid" is also a precondition of strict mode, not only a
    # tidiness rule: the provider wants additionalProperties false.
    with pytest.raises(ValidationError):
        RagAnswer.model_validate(
            {
                "answer_status": "answered",
                "answer": "x",
                "sources": [{"doc_id": "d", "section": "s"}],
                "citations": [{"doc_id": "d", "section": "s"}],
            }
        )


def test_a_repeated_citation_is_kept() -> None:
    """Deliberately unlike GoldQuestion, which rejects a repeated source.

    The gold file is authored once, so a duplicate there is a typo. A repeated
    citation is the model's output and therefore a fact about the run; removing
    it here would hide from the runner that the model named one section twice.
    """
    answer = RagAnswer(answer_status="answered", answer="x", citations=[SOURCE, SOURCE])

    assert len(answer.citations) == 2
