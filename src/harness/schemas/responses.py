from typing import Literal

from pydantic import BaseModel

from harness.core.gold import ExpectedSource
from harness.core.interfaces import LlmIncompleteReason, LlmToolStopReason
from harness.core.rag import RagAnswerStatus


class TextResponse(BaseModel):
    result: str
    num_chars: int
    incomplete_reason: LlmIncompleteReason | None = None
    """Set when the provider stopped early. The result is then a partial answer."""


class HealthResponse(BaseModel):
    status: Literal["ok"]


class VersionResponse(BaseModel):
    version: str


class RagResponse(BaseModel):
    result: str
    num_chars: int
    stop_reason: LlmToolStopReason

    citations: list[ExpectedSource] = []
    """The sections the model says its answer rests on.

    Same shape as GoldQuestion.expected_sources, so the phase 3 runner can
    compare a citation against an expected source without converting either.
    Empty when the run produced no parsed answer at all, which stop_reason then
    explains, and empty when answer_status is "no_relevant_sections", which is a
    grounded statement rather than a missing one. The two cases are only
    distinguishable by reading both fields."""

    answer_status: RagAnswerStatus | None = None
    """None when the run did not get far enough to produce one, so stop_reason
    is the field that says why. Set otherwise, and then it separates "answered
    with sources" from "nothing relevant was found", which a caller cannot infer
    from an empty citation list alone."""
