from dataclasses import asdict
from typing import Literal

from pydantic import BaseModel

from harness.core.gold import ExpectedSource
from harness.core.interfaces import (
    LlmIncompleteReason,
    LlmToolStopReason,
    TokenUsage,
)
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


class TokenUsageResponse(BaseModel):
    """Wire shape of TokenUsage. A model, because the port's dataclass is not one.

    Deliberately a copy rather than a pydantic dataclass over the same fields:
    the port belongs to core and must not learn about the HTTP layer, and the
    duplication is six ints that a builder in one place keeps in step.
    """

    input_tokens: int
    output_tokens: int
    total_tokens: int
    cached_tokens: int
    cache_write_tokens: int
    reasoning_tokens: int

    @classmethod
    def from_usage(cls, usage: TokenUsage) -> TokenUsageResponse:
        """Copy the port's dataclass into the wire model.

        Python trick: dataclasses.asdict() flattens the whole thing into a dict
        of the same field names, so cls(**asdict(usage)) is the whole body and
        a field added on one side fails loudly on the other.
        """
        return cls(**asdict(usage))


class StreamDeltaEvent(BaseModel):
    """Payload of one `event: delta` frame. Mirrors LlmTextDelta.

    A model rather than a bare string, because the frame body is JSON and a
    top-level JSON string would leave no room for a second field later.
    """

    text: str


class StreamEndEvent(BaseModel):
    """Payload of the single `event: end` frame. Mirrors LlmStreamEnd.

    usage lands here and nowhere else: the provider only reports it in its
    terminal event, so a per-delta usage field would be None on every frame but
    the last one.
    """

    usage: TokenUsageResponse | None = None
    incomplete_reason: LlmIncompleteReason | None = None
    """Set when the provider stopped early. The deltas already sent are then a
    partial answer, and the frame is the only place that can say so - the
    status code was fixed at 200 before the first token went out."""


class StreamErrorEvent(BaseModel):
    """Payload of an `event: error` frame. Only reachable after the first byte.

    Before the first byte an LlmError still becomes a 502 through the handlers
    in main.py. Afterwards the status line is gone, so the error has to travel
    inside the stream itself.
    """

    detail: str
