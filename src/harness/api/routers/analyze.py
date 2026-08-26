import logging
from collections.abc import Generator
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from harness.api.dependencies import get_llm_client, get_llm_config
from harness.api.errors import llm_error_response
from harness.api.sse import (
    SSE_DELTA,
    SSE_END,
    SSE_ERROR,
    SSE_MEDIA_TYPE,
    format_sse_frame,
)
from harness.core.config import LlmConfig
from harness.core.interfaces import (
    LlmError,
    LlmStreamEvent,
    LlmTextDelta,
    LlmUnavailableError,
    TextCompleter,
    TextStreamer,
)
from harness.schemas.requests import TextRequest
from harness.schemas.responses import (
    StreamDeltaEvent,
    StreamEndEvent,
    StreamErrorEvent,
    TextResponse,
    TokenUsageResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["analyze"])

SYSTEM_PROMPT = "Analyze the given text and return a concise result."


@router.post("/analyze")
def analyze(
    request: TextRequest,
    llm: Annotated[TextCompleter, Depends(get_llm_client)],
    llm_config: Annotated[LlmConfig, Depends(get_llm_config)],
) -> TextResponse:
    result = llm.complete(
        system_prompt=SYSTEM_PROMPT,
        user_message=request.text,
        config=llm_config,
    )

    # incomplete_reason travels to the caller: a truncated answer is still a
    # 200, so the body is the only place that can say the text is partial.
    return TextResponse(
        result=result.text,
        num_chars=len(result.text),
        incomplete_reason=result.incomplete_reason,
    )


def to_sse_frames(
    first: LlmStreamEvent, events: Generator[LlmStreamEvent]
) -> Generator[str]:
    """Turn port events into SSE frames, starting from an already pulled one.

    Takes `first` separately because the endpoint had to pull it before it could
    commit to a status code, and that event must not be dropped.

    After the first byte the status line is gone: the response is a 200 and
    stays one. A raise from here would tear the connection down mid-frame, and
    a client cannot tell that apart from a finished answer, so an LlmError
    becomes an `event: error` frame instead and the generator returns.

    Returns a Generator rather than an Iterator, and is public rather than
    underscored, for the same reason: close() is part of what it promises, and
    the test that pins that promise has to be able to name it.
    """
    try:
        yield _frame(first)
        for event in events:
            yield _frame(event)
    except LlmError as exc:
        # The same row the handler in main.py would have answered with, since
        # the frame is that handler's streaming twin: the provider's own text
        # goes to the log, the caller gets the fixed wording. The status code
        # on the row is spent by now - this response has been a 200 since its
        # first byte - so only the detail travels.
        logger.exception("LLM stream failed after the response had started")
        detail = llm_error_response(exc).detail
        yield format_sse_frame(SSE_ERROR, StreamErrorEvent(detail=detail))
    finally:
        events.close()


def _frame(event: LlmStreamEvent) -> str:
    """Map one port event onto its wire frame.

    The SSE_* constants are event names, not templates: the bytes are built by
    format_sse_frame from a pydantic payload, so a token containing a newline
    cannot break the frame apart.

    No else-branch guard at the end: LlmStreamEvent is a closed union of two
    dataclasses, so the isinstance narrows it to LlmStreamEnd and a third
    member would fail type checking here rather than at runtime.
    """
    if isinstance(event, LlmTextDelta):
        return format_sse_frame(SSE_DELTA, StreamDeltaEvent(text=event.text))

    usage = (
        TokenUsageResponse.from_usage(event.usage) if event.usage is not None else None
    )
    return format_sse_frame(
        SSE_END,
        StreamEndEvent(usage=usage, incomplete_reason=event.incomplete_reason),
    )


@router.post("/analyze/stream")
def analyze_stream(
    request: TextRequest,
    llm: Annotated[TextStreamer, Depends(get_llm_client)],
    llm_config: Annotated[LlmConfig, Depends(get_llm_config)],
) -> StreamingResponse:
    """Stream the answer token by token instead of waiting for all of it.

    The whole difficulty is the status code. stream() is a generator function,
    so calling it runs nothing and a connection failure would otherwise surface
    inside StreamingResponse, long after FastAPI sent 200. Pulling one event
    here keeps that failure in front of the response, where the handlers in
    main.py still turn it into a 502 or a 500.
    """
    events = llm.stream(SYSTEM_PROMPT, request.text, llm_config)

    # The port rules an empty run out, so this enforces a contract rather than
    # handling a case. Without the default, a streamer that yields nothing
    # raises StopIteration through the threadpool and arrives as a bare 500;
    # the adapter already names the same fault itself when a stream ends
    # without a terminal event.
    first = next(events, None)
    if first is None:
        raise LlmUnavailableError(
            f"The {llm_config.model_name} stream ended before its first event."
        )

    # Headers against the layers between us and the client: nginx buffers a
    # proxied response by default, and a cache would happily hold on to one.
    # Either turns the stream back into a single block on arrival, which is the
    # same failure the blank line in the frame exists to prevent, one hop later.
    return StreamingResponse(
        to_sse_frames(first, events),
        media_type=SSE_MEDIA_TYPE,
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
