import json
from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient

from harness.api.dependencies import get_llm_client
from harness.api.errors import LLM_ERROR_RESPONSES, LlmErrorResponse
from harness.api.routers.analyze import to_sse_frames
from harness.api.sse import SSE_MEDIA_TYPE
from harness.core.config import LlmConfig
from harness.core.interfaces import (
    LLM_INCOMPLETE_REASONS,
    LlmError,
    LlmIncompleteReason,
    LlmStreamEnd,
    LlmStreamEvent,
    LlmTextDelta,
    LlmUnavailableError,
    TokenUsage,
)
from harness.main import app


class ScriptedStreamer:
    """A streamer whose events and failure point the test decides.

    Records whether close() reached it, which is the only way a test can prove
    that a client hanging up releases the provider connection: the generator is
    suspended on a yield at that moment and cannot report anything itself.
    """

    def __init__(
        self,
        events: list[LlmStreamEvent],
        error: LlmError | None = None,
        error_after: int = 0,
    ) -> None:
        self.closed = False
        self._events = events
        self._error = error
        self._error_after = error_after

    def stream(
        self, system_prompt: str, user_message: str, config: LlmConfig
    ) -> Generator[LlmStreamEvent]:
        """Yield the scripted events, raise `error` after `error_after` of them.

        error_after=0 puts the raise before the first yield, which is the case
        the endpoint has to keep in front of the response so it can still be a
        502. Any higher value moves it behind the first byte.

        Python trick: set self.closed in a `finally` around the loop - it runs
        on GeneratorExit too, which is what close() throws in.
        """
        try:
            if self._error is not None and self._error_after == 0:
                raise self._error
            for i, event in enumerate(self._events, start=1):
                yield event
                if self._error is not None and i == self._error_after:
                    raise self._error
        finally:
            self.closed = True


DEFAULT_USAGE = TokenUsage(
    input_tokens=2,
    output_tokens=6,
    total_tokens=8,
    cached_tokens=0,
    cache_write_tokens=0,
    reasoning_tokens=0,
)


def hello_there_events(
    usage: TokenUsage | None = DEFAULT_USAGE,
    incomplete_reason: LlmIncompleteReason | None = None,
) -> list[LlmStreamEvent]:
    """Build the two-delta run the frame tests script.

    A frozen dataclass as a default argument, not the None-sentinel dance: the
    default is evaluated once at import and shared by every call, which is only
    a trap for mutable defaults (list, dict) and TokenUsage is neither.
    """
    return [
        LlmTextDelta("hi"),
        LlmTextDelta(" there"),
        LlmStreamEnd(usage, incomplete_reason),
    ]


def parse_sse_frames(body: str) -> list[tuple[str, dict[str, object]]]:
    """Split a raw SSE body into (event name, parsed data) pairs.

    Frames are separated by a blank line, so the split is on "\\n\\n". Doing it
    by hand rather than with a client library keeps the test honest about the
    bytes actually on the wire, including the trailing blank line.
    """
    frames = body.split("\n\n")
    result: list[tuple[str, dict[str, object]]] = []
    for frame in frames:
        if not frame.strip():
            continue

        event: str | None = None
        data_lines: list[str] = []
        for line in frame.split("\n"):
            if line.startswith("event:"):
                event = line.removeprefix("event:").strip()
            elif line.startswith("data:"):
                data_lines.append(line.removeprefix("data:").strip())

        if event is None:
            continue

        result.append((event, json.loads("\n".join(data_lines))))

    return result


def test_stream_sends_one_named_frame_per_event(client: TestClient) -> None:
    """Each port event becomes its own frame, under its own event name.

    The names are the client's dispatch table: EventSource fires listeners
    registered per name, so a delta arriving as "end" reaches nobody. Checking
    the sequence pins both at once, that nothing was merged and that each frame
    is labelled as what it is.

    Frame count and names only, not timing: starlette's TestClient runs the
    whole app into a BytesIO before httpx sees a byte, so a buffered response
    and a streamed one look identical from here.
    """
    streamer = ScriptedStreamer(events=hello_there_events())
    app.dependency_overrides[get_llm_client] = lambda: streamer

    response = client.post("/analyze/stream", json={"text": "hi"})

    frames = parse_sse_frames(response.text)
    assert [name for name, _ in frames] == ["delta", "delta", "end"]


def test_stream_reports_usage_only_in_the_last_frame(client: TestClient) -> None:
    """usage belongs to the run, not to a token, so exactly one frame carries it.

    Goes through the endpoint rather than the fake: the fake yields port
    dataclasses, and it is the router that decides which frame the usage is
    written into. client.post() is enough here, because the question is what
    the body contains, not when it arrives.
    """
    events = hello_there_events()
    streamer = ScriptedStreamer(events=events)
    app.dependency_overrides[get_llm_client] = lambda: streamer

    response = client.post("/analyze/stream", json={"text": "hi"})

    frames = parse_sse_frames(response.text)
    usage_frames = [frame for frame in frames if frame[1].get("usage") is not None]
    assert len(usage_frames) == 1
    assert frames[-1] is usage_frames[0]


def test_stream_answers_502_when_the_provider_fails_before_the_first_delta(
    client: TestClient,
) -> None:
    """The status code is still ours here, so this stays an ordinary 502.

    This is what pulling the first event in the endpoint buys, and the reason
    the adapter may raise LlmError before its first yield at all.
    """
    events = hello_there_events()
    streamer = ScriptedStreamer(
        events=events,
        error=LlmUnavailableError("Model gpt-5.6-luna did not answer (timeout)."),
        error_after=0,
    )
    app.dependency_overrides[get_llm_client] = lambda: streamer

    with client.stream("POST", "/analyze/stream", json={"text": "hi"}) as response:
        assert response.status_code == 502
        body = b"".join(response.iter_bytes())

    # Read from the table, like the endpoint does: an error before the first
    # byte is answered by the handler in main.py, so this is the same row the
    # error frame below would carry.
    expected = LLM_ERROR_RESPONSES[LlmUnavailableError]
    assert response.status_code == expected.status_code
    assert json.loads(body) == {"detail": expected.detail}


def test_stream_answers_200_and_an_error_frame_when_it_fails_mid_answer(
    client: TestClient,
) -> None:
    """After the first byte the 200 is spent, so the error travels as a frame.

    Tearing the connection down instead would reach the client as a stream that
    simply stopped, indistinguishable from a finished answer.
    """
    events = hello_there_events()
    # Worded like the adapter's own errors, model name and provider message
    # included: a fake that happens to say what the handlers say could not tell
    # the fixed wording from a leaked one.
    streamer = ScriptedStreamer(
        events=events,
        error=LlmUnavailableError(
            "Model gpt-5.6-luna reported a failed response "
            "(server_error: upstream exploded)."
        ),
        error_after=1,
    )
    app.dependency_overrides[get_llm_client] = lambda: streamer

    response = client.post("/analyze/stream", json={"text": "hi"})
    assert response.status_code == 200

    frames = parse_sse_frames(response.text)
    assert [name for name, _ in frames] == ["delta", "error"]
    assert frames[0][1]["text"] == "hi"

    # The fixed wording from the table, not the provider's own text: what a
    # caller learns must not depend on whether the run failed before or after
    # the first byte.
    assert frames[-1][1] == {"detail": LLM_ERROR_RESPONSES[LlmUnavailableError].detail}


def test_stream_survives_a_consumer_that_stops_reading(client: TestClient) -> None:
    """Leaving the with-block early produces no exception. That is all it shows.

    Not a hang-up test, however much it looks like one: the TestClient runs the
    app to completion before httpx sees a byte, so the generator is exhausted
    rather than abandoned by the time the block exits. The close() that the
    endpoint's finally performs is pinned one level down, in
    test_frames_close_the_provider_stream_when_the_consumer_stops.
    """
    streamer = ScriptedStreamer(events=hello_there_events())
    app.dependency_overrides[get_llm_client] = lambda: streamer

    with client.stream("POST", "/analyze/stream", json={"text": "hi"}) as response:
        lines = response.iter_lines()
        first_data_line = next(line for line in lines if line.startswith("data:"))

    assert first_data_line.startswith("data:")


def test_frames_close_the_provider_stream_when_the_consumer_stops() -> None:
    """The endpoint's finally, tested where a consumer can actually stop.

    Through the TestClient this is untestable: it runs the app to completion
    before the first byte reaches httpx, so the generator is exhausted rather
    than abandoned and `closed` says nothing. Here the close() is real -
    GeneratorExit lands on the suspended yield inside to_sse_frames, which is
    what releases the provider connection when a browser tab goes away.
    """
    streamer = ScriptedStreamer(events=hello_there_events())
    events = streamer.stream("sys", "hi", LlmConfig(model_name="test-model"))
    first = next(events)

    frames = to_sse_frames(first, events)
    next(frames)
    frames.close()

    assert streamer.closed


@pytest.mark.parametrize("reason", LLM_INCOMPLETE_REASONS)
def test_stream_reports_that_the_provider_cut_the_answer_short(
    client: TestClient, reason: LlmIncompleteReason
) -> None:
    """A truncated run is a 200 with real text, so the end frame has to say so.

    Same role the field plays in TextResponse, one transport further: the
    status line went out before the provider stopped, and every delta already
    sent is genuine text. Only the last frame can tell the caller that the
    answer stops there because it ran out, not because it was finished.
    """
    streamer = ScriptedStreamer(events=hello_there_events(incomplete_reason=reason))
    app.dependency_overrides[get_llm_client] = lambda: streamer

    response = client.post("/analyze/stream", json={"text": "tell me more"})

    name, data = parse_sse_frames(response.text)[-1]
    assert name == "end"
    assert data["incomplete_reason"] == reason


@pytest.mark.parametrize(
    ("payload", "error_type"),
    [
        ({"text": "  "}, "string_too_short"),
        ({"txet": "I love Donuts :)"}, "missing"),
    ],
)
def test_stream_rejects_an_invalid_request_before_the_first_byte(
    client: TestClient, payload: dict[str, str], error_type: str
) -> None:
    """The 422 has to arrive as a status code, not as an error frame.

    Dependencies are resolved first and the validation errors are collected
    after them, so get_llm_client does run. What never runs is the endpoint
    function, so stream() is never called and no byte goes out - the status
    line is still ours to set. Were it otherwise, a malformed request would
    reach the caller as a 200 carrying an error frame, indistinguishable from a
    provider that failed mid-answer.

    Uses the fixture's FakeLlmClient rather than a ScriptedStreamer: it is
    built, but nothing is supposed to be pulled from it.
    """
    response = client.post("/analyze/stream", json=payload)

    assert response.status_code == 422
    assert response.headers["content-type"] == "application/json"
    assert any(
        detail["type"] == error_type and detail["loc"] == ["body", "text"]
        for detail in response.json()["detail"]
    )


def test_stream_answers_502_when_the_run_yields_no_event_at_all(
    client: TestClient,
) -> None:
    """The empty run the port forbids, answered as the upstream fault it is.

    An adapter cannot reach this, a broken test double can. Without the guard
    the bare next() raises StopIteration through the threadpool, which reaches
    the caller as a 500 about ourselves rather than a 502 about the provider.
    """
    streamer = ScriptedStreamer(events=[])
    app.dependency_overrides[get_llm_client] = lambda: streamer

    response = client.post("/analyze/stream", json={"text": "hi"})

    expected = LLM_ERROR_RESPONSES[LlmUnavailableError]
    assert response.status_code == expected.status_code
    assert response.json() == {"detail": expected.detail}


def test_stream_answers_as_an_event_stream(client: TestClient) -> None:
    """The media type is what makes a browser read frames instead of a body.

    Runs against the fixture's FakeLlmClient rather than a ScriptedStreamer:
    this asks what the endpoint answers with, not what a scripted run puts in
    it, and it is the one test that exercises the fake's stream() at all.

    The two buffering headers are pinned here as well. They exist for hops the
    tests cannot see - a proxy, a cache - so a comment in the router would be
    the only record that they were ever a decision.
    """
    response = client.post("/analyze/stream", json={"text": "hi"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith(SSE_MEDIA_TYPE)
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"


@pytest.mark.parametrize(
    ("error_type", "expected"),
    [
        pytest.param(error_type, response, id=error_type.__name__)
        for error_type, response in LLM_ERROR_RESPONSES.items()
    ],
)
def test_error_frame_carries_the_table_row_for_every_error(
    client: TestClient, error_type: type[LlmError], expected: LlmErrorResponse
) -> None:
    """The streaming twin of test_llm_error_answers_its_table_row.

    Both test sets parametrize over the same table, which is what makes the
    wording provably single-sourced: there is no longer a copy on either side
    that could keep asserting what the table no longer says. Only the detail is
    compared - the status code on the row was spent when the first byte went
    out, and this response is a 200 whatever failed afterwards.
    """
    streamer = ScriptedStreamer(
        events=hello_there_events(),
        error=error_type("provider detail we are not supposed to repeat"),
        error_after=1,
    )
    app.dependency_overrides[get_llm_client] = lambda: streamer

    response = client.post("/analyze/stream", json={"text": "hi"})

    assert response.status_code == 200
    frames = parse_sse_frames(response.text)
    assert [name for name, _ in frames] == ["delta", "error"]
    assert frames[-1][1] == {"detail": expected.detail}
