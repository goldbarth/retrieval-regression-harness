"""Manual smoke check for the streaming path. Needs a valid .env.

Answers what tests/test_streaming.py structurally cannot: whether the first
token really arrives before the last one. Every test fakes the stream, so they
prove the event order, never that the provider actually dribbles.

Usage:
    uv run python scripts/stream_smoke.py

Spends tokens: three runs, the third one abandoned midway. The provider bills
what it already sent, so an abandoned run is not a free one.

Measured on 2026-08-26 against gpt-5.6-luna, one run of each:

    complete()   6.70s to the full answer, 389 output tokens
    stream()     2.59s to the first token, 9.13s to the full answer,
                 280 deltas, 369 output tokens
    abandoned    5 deltas, outcome='abandoned', no usage on the line

Three readings, and the second one is the uncomfortable half:

1. Waiting before the first word drops from 6.70s to 2.59s. That is what
   streaming buys, and the only thing it buys.
2. The full answer took longer streamed than blocking, 9.13s against 6.70s.
   Two calls with different answers, so this is not a benchmark - but it is
   enough to say that streaming improves perceived latency and does not
   improve throughput. It can cost some.
3. 280 deltas carried 369 output tokens. A delta is not a token, so the
   `deltas` field in the cost line must not be read as a token estimate. It
   exists for the runs that never reach a usage number at all.
"""

import logging
import time
from dataclasses import dataclass

from harness.api.dependencies import get_llm_client, get_llm_config
from harness.core.interfaces import LlmTextDelta

_SYSTEM_PROMPT = "You're a helpful assistant."

# Long enough that the answer has something to spread out over. A one-sentence
# question finishes before the abandon run below can cut into it, and the two
# timings then measure the same moment twice.
_USER_MESSAGE = (
    "Explain in about 200 words what a retrieval regression harness is, "
    "why gold questions belong to it, and what it measures between two runs."
)

_ABANDON_AFTER_DELTAS = 5

# The adapter logs under its module name, and the cost line is the whole point
# of the third run.
_CLIENT_LOGGER_NAME = "harness.infrastructure.llm.client"


@dataclass(frozen=True)
class StreamTiming:
    """Both numbers from one run. Time to first token is why streaming exists."""

    ttft_seconds: float
    total_seconds: float
    deltas: int
    text: str


class _ExtraFormatter(logging.Formatter):
    """Append the adapter's structured fields to the rendered message.

    The default Formatter ignores `extra` entirely, so `outcome` and `deltas`
    never reach stdout. Naming them in a format string instead would fail on
    every other record in the process, which carries no such fields, so this
    reads whatever is present and leaves the rest alone.
    """

    _FIELDS = (
        "model_name",
        "outcome",
        "incomplete_reason",
        "deltas",
        "input_tokens",
        "output_tokens",
        "total_tokens",
    )

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        pairs = [
            f"{field}={getattr(record, field)!r}"
            for field in self._FIELDS
            if hasattr(record, field)
        ]
        return f"{base} {' '.join(pairs)}" if pairs else base


def configure_stream_logging() -> None:
    """Make the adapter's cost line visible, fields and all.

    Attached to the client logger rather than the root one: the format above is
    only correct for records that carry those fields, and propagate=False keeps
    the line from being printed a second time by a plain root handler.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(_ExtraFormatter("%(levelname)s %(name)s: %(message)s"))

    logger = logging.getLogger(_CLIENT_LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.propagate = False


def measure_stream(system_prompt: str, user_message: str) -> StreamTiming:
    """Pull the whole stream and clock the first delta against the last event.

    perf_counter, not time.time: it is monotonic and unaffected by a clock
    adjustment landing mid-run. The clock starts before stream() is called,
    since the generator body does not run until the first next() and the wait
    for the provider's headers is part of what a user waits for.
    """
    client = get_llm_client()
    config = get_llm_config()

    start = time.perf_counter()
    ttft: float | None = None
    deltas = 0
    chunks: list[str] = []

    for event in client.stream(system_prompt, user_message, config):
        if isinstance(event, LlmTextDelta):
            if ttft is None:
                ttft = time.perf_counter() - start
            deltas += 1
            chunks.append(event.text)
        else:
            # Narrowed to LlmStreamEnd, and the port promises it is the last
            # event, so there is nothing left to read after it.
            break

    total = time.perf_counter() - start
    if ttft is None:
        raise RuntimeError("The stream ended without a single delta.")

    return StreamTiming(
        ttft_seconds=ttft, total_seconds=total, deltas=deltas, text="".join(chunks)
    )


def measure_complete(system_prompt: str, user_message: str) -> float:
    """Time the same question through complete(), as the number to compare against.

    The blocking call has no time to first token: for the caller its first byte
    and its last byte are the same moment. That single number is what the two
    numbers above are read against.
    """
    client = get_llm_client()
    config = get_llm_config()

    start = time.perf_counter()
    client.complete(system_prompt, user_message, config)
    return time.perf_counter() - start


def abandon_midway(system_prompt: str, user_message: str, after_deltas: int) -> int:
    """Stop reading after N deltas and close the generator, as an SSE client would.

    close() throws GeneratorExit into the generator at its yield, which is the
    one exit `except LlmError` never sees: GeneratorExit is not an Exception
    subclass. Breaking out of the loop alone would get there too, since CPython
    closes an unreferenced generator, but that leaves the log line at the mercy
    of refcounting - so it is closed explicitly.

    Prints nothing itself. The proof is the adapter's line with
    outcome='abandoned' and no usage on it.
    """
    client = get_llm_client()
    config = get_llm_config()

    events = client.stream(system_prompt, user_message, config)
    seen = 0
    try:
        for event in events:
            if isinstance(event, LlmTextDelta):
                seen += 1
                if seen >= after_deltas:
                    break
    finally:
        events.close()

    return seen


if __name__ == "__main__":
    configure_stream_logging()

    print("--- blocking ---")
    blocking_seconds = measure_complete(_SYSTEM_PROMPT, _USER_MESSAGE)
    print(f"complete(): {blocking_seconds:.2f}s to the full answer\n")

    print("--- streaming ---")
    timing = measure_stream(_SYSTEM_PROMPT, _USER_MESSAGE)
    print(
        f"stream(): {timing.ttft_seconds:.2f}s to the first token, "
        f"{timing.total_seconds:.2f}s to the full answer, "
        f"{timing.deltas} deltas"
    )
    print(f"answer: {timing.text}\n")

    # Two provider calls with different answer lengths, so this is a rough
    # comparison and not a benchmark. It carries "first token after 0.4s
    # instead of 6s", it does not carry a percentage.
    print(
        f"waiting before the first word: {timing.ttft_seconds:.2f}s streamed "
        f"vs {blocking_seconds:.2f}s blocking\n"
    )

    print("--- abandoned ---")
    abandoned_deltas = abandon_midway(
        _SYSTEM_PROMPT, _USER_MESSAGE, _ABANDON_AFTER_DELTAS
    )
    print(f"stopped reading after {abandoned_deltas} deltas, cost line above")
