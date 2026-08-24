from collections.abc import Iterator

from harness.core.config import LlmConfig
from harness.core.interfaces import (
    LlmStreamEnd,
    LlmStreamEvent,
    LlmTextDelta,
    TextStreamer,
    TokenUsage,
)


class FakeStreamer:
    def stream(
        self, system_prompt: str, user_message: str, config: LlmConfig
    ) -> Iterator[LlmStreamEvent]:
        yield LlmTextDelta("hi")
        yield LlmTextDelta(" there")

        yield LlmStreamEnd(
            TokenUsage(
                input_tokens=1,
                output_tokens=2,
                total_tokens=3,
                cached_tokens=0,
                cache_write_tokens=0,
                reasoning_tokens=0,
            ),
            None,
        )


def test_stream_events_satisfy_the_port() -> None:
    """Structural conformance, checked by mypy/pyright rather than at runtime.

    Protocol without @runtime_checkable has no isinstance. The annotated
    assignment is the assertion: the type checkers fail if the fake's signature
    drifts from the port, and pytest still runs the body so a broken import or
    a typo in the dataclass fields fails loudly too.
    """
    streamer: TextStreamer = FakeStreamer()
    events = list(streamer.stream("sys", "hi", LlmConfig(model_name="test-model")))

    ends = [e for e in events if isinstance(e, LlmStreamEnd)]
    assert len(ends) == 1
    assert isinstance(events[-1], LlmStreamEnd)
