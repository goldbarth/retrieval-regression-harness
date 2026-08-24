"""The test that keeps data/gold_questions.json alive.

The anchor test is the point of this file, not the model test. Without it the
gold file rots silently against a re-vendored corpus, and phase 3 reports a
regression that is in truth a dead anchor.
"""

import json
from pathlib import Path
from typing import get_args

import pytest

from harness.core.corpus import iter_sections
from harness.core.gold import GoldError, GoldQuestionSet, Hardness, load_gold_questions

# Anchored on the test file, not on the working directory, same as
# tests/test_corpus.py: repo data has to be found from anywhere pytest starts.
CORPUS_ROOT = Path(__file__).parent.parent / "data" / "corpus" / "fastapi"
GOLD_PATH = Path(__file__).parent.parent / "data" / "gold_questions.json"


@pytest.fixture(scope="module")
def corpus_keys() -> set[tuple[str, str]]:
    """Every (doc_id, section) the corpus actually offers."""
    return {(s.doc_id, s.section) for s in iter_sections(CORPUS_ROOT)}


@pytest.fixture(scope="module")
def gold() -> GoldQuestionSet:
    """The real file, not a fixture document.

    A test against a synthetic set would stay green while the shipped file is
    broken.
    """
    return load_gold_questions(GOLD_PATH)


def test_every_expected_source_exists_in_the_corpus(
    gold: GoldQuestionSet, corpus_keys: set[tuple[str, str]]
) -> None:
    """The core test of this file: a dangling (doc_id, section) is a dead
    anchor, and a dead anchor scores as a retrieval miss forever after.

    The misses are collected across all questions instead of asserted per
    question: the first dead anchor would hide the other nine, and after a
    corpus bump they usually come in batches.
    """
    misses = [
        (q.id, src.doc_id, src.section)
        for q in gold.questions
        for src in q.expected_sources
        if (src.doc_id, src.section) not in corpus_keys
    ]
    assert not misses, f"dead anchors (question, doc_id, section): {misses}"


def test_question_ids_are_unique(gold: GoldQuestionSet) -> None:
    """Guards the behavior, not the validator.

    If the check later moves out of the model, this test still has to hold.
    """
    ids = [q.id for q in gold.questions]
    assert len(ids) == len(set(ids))


def test_every_question_has_at_least_one_expected_source(
    gold: GoldQuestionSet,
) -> None:
    """Guards the behavior, not the validator.

    Same reasoning as the id-uniqueness test: `min_length=1` could move or get
    relaxed later, this stays the check that a source-less question never
    scores as a retrieval metric.
    """
    empty = [q.id for q in gold.questions if not q.expected_sources]
    assert not empty, f"questions without an expected source: {empty}"


def test_all_three_hardness_kinds_are_present(gold: GoldQuestionSet) -> None:
    """Every kind in the Literal has to occur at least once.

    Derived from Hardness itself, not written out again: a fourth kind added
    to the Literal has to make this test fail until a question covers it.
    """
    present = {q.hardness for q in gold.questions}
    expected: set[Hardness] = set(get_args(Hardness))
    missing = expected - present
    assert not missing, f"missing hardness kinds: {missing}"


def test_broken_json_raises(tmp_path: Path) -> None:
    """A truncated file has to raise, not load as an empty set.

    An empty set on a parse error would report as a green run over zero
    questions.
    """
    broken = tmp_path / "broken.json"
    broken.write_text('{"questions": [', encoding="utf-8")

    with pytest.raises(GoldError):
        load_gold_questions(broken)


def test_an_unknown_field_is_rejected(tmp_path: Path) -> None:
    """extra="forbid" earns its keep here: a renamed key, say `sources`
    instead of `expected_sources`, would otherwise load as a question with no
    sources at all.
    """
    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps(
            {
                "questions": [
                    {
                        "id": "q1",
                        "question": "irrelevant",
                        "expected_answer": "irrelevant",
                        "sources": [{"doc_id": "x", "section": "y"}],
                        "hardness": "term_collision",
                        "note": None,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(GoldError):
        load_gold_questions(bad)


def test_an_unknown_hardness_is_rejected(tmp_path: Path) -> None:
    """The Literal is the reason the phase 3 breakdown by hardness holds.

    A typo'd fourth kind would show up as its own bucket of size one.
    """
    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps(
            {
                "questions": [
                    {
                        "id": "q1",
                        "question": "irrelevant",
                        "expected_answer": "irrelevant",
                        "expected_sources": [{"doc_id": "x", "section": "y"}],
                        "hardness": "ambiguous",
                        "note": None,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(GoldError):
        load_gold_questions(bad)
