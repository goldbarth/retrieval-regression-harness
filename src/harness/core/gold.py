"""The gold question set: what a run is measured against.

An expected source is (doc_id, section), never a chunk id. Both are derivable
from a file on disk without any chunking, so the file stays valid when phase 3
starts varying chunk size and overlap. A chunk id would be void after the
first configuration change, and every question with it.
"""

from collections import Counter
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

Hardness = Literal["near_duplicate", "term_collision", "multi_section"]
"""Why this question is hard, as a closed set.

Free text here would block the interesting statement in phase 3: that a
regression hit only one sort of question. Three sorts, three retrieval
failure modes, each one a different fix.
"""


class GoldError(Exception):
    """The gold file on disk is not usable as a measurement baseline."""


class ExpectedSource(BaseModel):
    """One section that a correct answer has to be grounded in."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    """Path relative to the corpus root, without `.md`,
    e.g. `tutorial/dependencies/index`.
    """

    section: str
    """The anchor as authored in the doc, e.g. `first-steps`. Never slugified here."""


class GoldQuestion(BaseModel):
    """One question plus the sections its answer has to come from."""

    model_config = ConfigDict(extra="forbid")

    id: str
    question: str
    expected_answer: str

    expected_sources: list[ExpectedSource] = Field(min_length=1)
    """A list, never a set. A set renders as `uniqueItems`, which the provider's
    strict mode rejects outright, see scripts/schema_constraint_probe.py. The
    same shape is reused as a response schema in commit 4, so the constraint
    applies here too. min_length=1: a question without a source is not usable
    as a retrieval metric, only as a text comparison.
    """

    hardness: Hardness
    note: str | None
    """Why exactly this question is hard. `str | None`, not a pydantic default:
    the probe run shows the provider honors an optional field, and an explicit
    null in the JSON reads differently from a key someone forgot.
    """

    @model_validator(mode="after")
    def _expected_sources_are_distinct(self) -> Self:
        """Reject the same (doc_id, section) twice in one question.

        Not a matter of tidiness: the denominator of recall@k is |expected|,
        so the duplicate counts there, while the numerator can only hit it
        once. Such a question can never reach 1.0.
        """
        keys = [(src.doc_id, src.section) for src in self.expected_sources]
        counts = Counter(keys)
        duplicates = sorted(key for key, n in counts.items() if n > 1)
        if duplicates:
            raise ValueError(f"question has duplicate expected sources: {duplicates}")
        return self


class GoldQuestionSet(BaseModel):
    """The whole gold file, validated as one unit."""

    model_config = ConfigDict(extra="forbid")

    questions: list[GoldQuestion]

    @model_validator(mode="after")
    def _ids_are_unique(self) -> Self:
        """Reject a duplicate id at load time, not at diff time.

        Two questions sharing an id put two rows of a later diff on the same
        key: one silently overwrites the other, and the run reports fewer
        questions than it asked.
        """
        counts = Counter(q.id for q in self.questions)
        duplicates = sorted(id_ for id_, n in counts.items() if n > 1)
        if duplicates:
            raise ValueError(f"duplicate question ids: {duplicates}")
        return self


def load_gold_questions(path: Path) -> GoldQuestionSet:
    """Read and validate the gold file, or raise.

    Broken JSON and a schema violation both have to raise. A loader that
    returns an empty set on a parse error turns a corrupt file into a run with
    zero questions, which reports as a green run.
    """
    # missing file -> OSError, left unwrapped like corpus.py
    text = path.read_text(encoding="utf-8")

    try:
        return GoldQuestionSet.model_validate_json(text)
    except ValidationError as e:
        raise GoldError(f"invalid gold file {path}: {e}") from e
