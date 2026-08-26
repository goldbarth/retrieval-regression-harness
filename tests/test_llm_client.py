import logging
from collections.abc import Callable, Generator, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import httpx2
import pytest
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    OpenAI,
    OpenAIError,
    PermissionDeniedError,
    RateLimitError,
    omit,
)
from pydantic import BaseModel, ValidationError

from harness.core.config import LlmConfig
from harness.core.interfaces import (
    LlmCompletion,
    LlmConfigurationError,
    LlmResponseFormatError,
    LlmStreamEnd,
    LlmStreamEvent,
    LlmStructuredToolCompletion,
    LlmTextDelta,
    LlmToolError,
    LlmUnavailableError,
    TokenUsage,
    ToolSpec,
)
from harness.core.tools import SectionSearchParams
from harness.infrastructure.llm.client import OpenAiLlmClient


@dataclass
class FakeInputTokensDetails:
    cached_tokens: int
    cache_write_tokens: int


@dataclass
class FakeOutputTokensDetails:
    reasoning_tokens: int


@dataclass
class FakeUsage:
    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_tokens_details: FakeInputTokensDetails
    output_tokens_details: FakeOutputTokensDetails


@dataclass
class FakeIncompleteDetails:
    reason: str | None


@dataclass
class FakeError:
    """The `error` object on a failed response: message plus code."""

    message: str
    code: str | None = None


@dataclass
class FakeResponse:
    output_text: str
    usage: FakeUsage | None = None
    incomplete_details: FakeIncompleteDetails | None = None
    error: FakeError | None = None
    """Only a "response.failed" event carries one. The non-stream paths never
    read it, so it stays optional and the existing fixtures keep working."""


def make_usage() -> FakeUsage:
    """Distinct numbers everywhere, so a swapped field cannot pass."""
    return FakeUsage(
        input_tokens=11,
        output_tokens=22,
        total_tokens=33,
        input_tokens_details=FakeInputTokensDetails(
            cached_tokens=44, cache_write_tokens=55
        ),
        output_tokens_details=FakeOutputTokensDetails(reasoning_tokens=66),
    )


class FakeSchema(BaseModel):
    value: str


@dataclass
class FakeParsedResponse:
    output_parsed: FakeSchema | None
    usage: FakeUsage | None = None
    incomplete_details: FakeIncompleteDetails | None = None


@dataclass
class FakeFunctionCall:
    name: str
    call_id: str
    arguments: str
    type: str = "function_call"


@dataclass
class FakeToolResponse:
    """A response the tool loop can walk: output items plus the final text."""

    output_text: str = ""
    output: list[FakeFunctionCall] = field(default_factory=list[FakeFunctionCall])
    usage: FakeUsage | None = None
    incomplete_details: FakeIncompleteDetails | None = None


class FakeStream:
    """Stands in for openai.Stream: iterable, context manager, closable.

    `closed` is the point of the class. Whether the adapter really closes the
    connection is not visible in any returned value, so the double has to
    record it.

    __iter__ has to be able to raise mid-iteration: transport failures arrive
    while the events are pulled, not at create(), which returns as soon as the
    headers are in. `raise_at` fires `error` in place of that event and buys
    the tests the case the except-ladder around the loop exists for. It is an
    index into `events`, so a value past the end would be a raise the adapter
    can never reach; the assert turns that from a silent no-op into a failure.
    """

    def __init__(
        self,
        events: list[Any],
        error: Exception | None = None,
        raise_at: int | None = None,
    ) -> None:
        assert raise_at is None or raise_at < len(events), (
            f"raise_at={raise_at} points past the last of {len(events)} events."
        )
        self.events = events
        self.closed = False
        self.error = error
        self.raise_at = raise_at

    def __enter__(self) -> FakeStream:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __iter__(self) -> Iterator[Any]:
        """Yield the events, and raise `error` once `raise_at` is reached.

        Python detail: a generator body here means the raise lands on the
        adapter's next() call, which is where a real transport failure lands
        too. Returning iter(self.events) instead could never model that.
        """
        for yielded, event in enumerate(self.events):
            if yielded == self.raise_at:
                assert self.error is not None
                raise self.error
            yield event

    def close(self) -> None:
        self.closed = True


@dataclass(frozen=True)
class FakeStreamEvent:
    """One event, duck-typed. The adapter branches on `type`, so the double
    needs nothing the branch it targets does not read."""

    type: str
    delta: str = ""
    response: FakeResponse | None = None
    message: str = ""
    code: str | None = None
    """The top-level "error" event carries its message on the event itself,
    unlike "response.failed", which carries it on response.error."""


def make_tool_call(arguments: str = '{"query": "x", "top_k": 1}') -> FakeFunctionCall:
    return FakeFunctionCall(name="search", call_id="call-1", arguments=arguments)


def make_tool_spec(handler: Callable[[SectionSearchParams], str]) -> ToolSpec[Any]:
    return ToolSpec(name="search", params=SectionSearchParams, handler=handler)


@dataclass
class FakeParsedToolResponse:
    """A tool round under a schema.

    Same walkable shape as FakeToolResponse plus output_parsed, because the
    adapter reads both from the same response object: the output items decide
    whether another round follows, and output_parsed only ever carries
    something on a round that called no tool.
    """

    output_text: str = ""
    output: list[FakeFunctionCall] = field(default_factory=list[FakeFunctionCall])
    output_parsed: FakeSchema | None = None
    usage: FakeUsage | None = None
    incomplete_details: FakeIncompleteDetails | None = None


@dataclass
class FakeResponses:
    result: FakeResponse | None = None
    parsed_result: FakeParsedResponse | None = None
    error: Exception | None = None
    calls: list[dict[str, object]] = field(default_factory=list[dict[str, object]])
    # One entry per round, consumed in order. A tool run makes several calls,
    # so a single result cannot express what the second round returns.
    tool_results: list[FakeToolResponse] = field(default_factory=list[FakeToolResponse])
    # Kept apart from tool_results rather than widened into a union: the two
    # paths call different SDK methods, and a shared queue would let a test
    # feed a create() response into a parse() run without anything noticing.
    parsed_tool_results: list[FakeParsedToolResponse] = field(
        default_factory=list[FakeParsedToolResponse]
    )
    # Kept apart from `result` for the same reason: a stream run and a plain
    # run go through the same create(), and a shared field would let a test
    # feed a non-stream response into a stream run without anything noticing.
    stream_result: FakeStream | None = None

    def create(self, **kwargs: object) -> Any:
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            if self.error is not None:
                raise self.error
            assert self.stream_result is not None
            return self.stream_result
        if self.error is not None:
            raise self.error
        if self.tool_results:
            return self.tool_results.pop(0)
        assert self.result is not None
        return self.result

    def parse(self, **kwargs: object) -> Any:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        if self.parsed_tool_results:
            return self.parsed_tool_results.pop(0)
        assert self.parsed_result is not None
        return self.parsed_result


@dataclass
class FakeOpenAI:
    responses: FakeResponses


def make_client(responses: FakeResponses) -> OpenAiLlmClient:
    return OpenAiLlmClient(
        client=cast(OpenAI, cast(object, FakeOpenAI(responses=responses))),
    )


def make_status_error(
    error_type: type[APIStatusError], status_code: int
) -> APIStatusError:
    """Build an openai status error without touching the network."""
    request = httpx2.Request("POST", "https://api.openai.com/v1/responses")
    response = httpx2.Response(status_code=status_code, request=request)
    return error_type("upstream said no", response=response, body=None)


def make_validation_error() -> ValidationError:
    """Build a real pydantic error, so the mapping is tested against the real type."""
    try:
        FakeSchema.model_validate({})
    except ValidationError as exc:
        return exc
    raise AssertionError("FakeSchema unexpectedly accepted an empty payload.")


def test_complete_returns_response_text() -> None:
    responses = FakeResponses(result=FakeResponse(output_text="Hello from the model."))
    client = make_client(responses)

    result = client.complete(
        system_prompt="You are helpful.",
        user_message="Say hello.",
        config=LlmConfig(model_name="test-model", temperature=0.2),
    )

    assert result == LlmCompletion(text="Hello from the model.")
    assert responses.calls == [
        {
            "model": "test-model",
            "instructions": "You are helpful.",
            "input": "Say hello.",
            "temperature": 0.2,
            "max_output_tokens": omit,
        }
    ]


def test_complete_uses_omit_for_temperature_when_not_configured() -> None:
    responses = FakeResponses(result=FakeResponse(output_text="Hello from the model."))
    client = make_client(responses)

    result = client.complete(
        system_prompt="You are helpful.",
        user_message="Say hello.",
        config=LlmConfig(model_name="reasoning-model", temperature=None),
    )

    assert result == LlmCompletion(text="Hello from the model.")
    assert responses.calls == [
        {
            "model": "reasoning-model",
            "instructions": "You are helpful.",
            "input": "Say hello.",
            "temperature": omit,
            "max_output_tokens": omit,
        }
    ]


@pytest.mark.parametrize(
    ("error_type", "status_code"),
    [
        (AuthenticationError, 401),
        (PermissionDeniedError, 403),
        (BadRequestError, 400),
        (NotFoundError, 404),
    ],
)
def test_complete_maps_rejected_requests_to_configuration_error(
    error_type: type[APIStatusError], status_code: int
) -> None:
    upstream_error = make_status_error(error_type, status_code)
    client = make_client(FakeResponses(error=upstream_error))

    with pytest.raises(LlmConfigurationError) as exc_info:
        client.complete(
            system_prompt="You are helpful.",
            user_message="Say hello.",
            config=LlmConfig(model_name="test-model", temperature=0.2),
        )

    assert exc_info.value.__cause__ is upstream_error


def test_complete_maps_rate_limit_to_unavailable_error() -> None:
    upstream_error = make_status_error(RateLimitError, 429)
    client = make_client(FakeResponses(error=upstream_error))

    with pytest.raises(LlmUnavailableError) as exc_info:
        client.complete(
            system_prompt="You are helpful.",
            user_message="Say hello.",
            config=LlmConfig(model_name="test-model", temperature=0.2),
        )

    assert exc_info.value.__cause__ is upstream_error


@pytest.mark.parametrize("error_type", [APIConnectionError, APITimeoutError])
def test_complete_maps_connection_error_to_unavailable_error(
    error_type: type[APIConnectionError],
) -> None:
    upstream_error = error_type(
        request=httpx2.Request("POST", "https://api.openai.com/v1/responses")
    )
    client = make_client(FakeResponses(error=upstream_error))

    with pytest.raises(LlmUnavailableError) as exc_info:
        client.complete(
            system_prompt="You are helpful.",
            user_message="Say hello.",
            config=LlmConfig(model_name="test-model", temperature=0.2),
        )

    assert exc_info.value.__cause__ is upstream_error


def test_complete_maps_unknown_openai_error_to_unavailable_error() -> None:
    upstream_error = OpenAIError("something else went wrong")
    client = make_client(FakeResponses(error=upstream_error))

    with pytest.raises(LlmUnavailableError) as exc_info:
        client.complete(
            system_prompt="You are helpful.",
            user_message="Say hello.",
            config=LlmConfig(model_name="test-model", temperature=0.2),
        )

    assert exc_info.value.__cause__ is upstream_error


@pytest.mark.parametrize("output_text", ["", "   \n\t "])
def test_complete_rejects_empty_response(output_text: str) -> None:
    client = make_client(FakeResponses(result=FakeResponse(output_text=output_text)))

    with pytest.raises(LlmUnavailableError):
        client.complete(
            system_prompt="You are helpful.",
            user_message="Say hello.",
            config=LlmConfig(model_name="test-model", temperature=0.2),
        )


def test_complete_passes_full_usage_through() -> None:
    client = make_client(
        FakeResponses(
            result=FakeResponse(output_text="Hello.", usage=make_usage()),
        )
    )

    result = client.complete(
        system_prompt="You are helpful.",
        user_message="Say hello.",
        config=LlmConfig(model_name="test-model", temperature=0.2),
    )

    assert result.usage == TokenUsage(
        input_tokens=11,
        output_tokens=22,
        total_tokens=33,
        cached_tokens=44,
        cache_write_tokens=55,
        reasoning_tokens=66,
    )


def test_complete_returns_no_usage_when_the_provider_omits_it() -> None:
    client = make_client(
        FakeResponses(result=FakeResponse(output_text="Hello.", usage=None))
    )

    result = client.complete(
        system_prompt="You are helpful.",
        user_message="Say hello.",
        config=LlmConfig(model_name="test-model", temperature=0.2),
    )

    assert result.usage is None


def test_complete_sends_max_output_tokens_when_configured() -> None:
    responses = FakeResponses(result=FakeResponse(output_text="Hello."))
    client = make_client(responses)

    client.complete(
        system_prompt="You are helpful.",
        user_message="Say hello.",
        config=LlmConfig(
            model_name="test-model", temperature=0.2, max_output_tokens=64
        ),
    )

    assert responses.calls == [
        {
            "model": "test-model",
            "instructions": "You are helpful.",
            "input": "Say hello.",
            "temperature": 0.2,
            "max_output_tokens": 64,
        }
    ]


@pytest.mark.parametrize("reason", ["max_output_tokens", "content_filter"])
def test_complete_reports_a_truncated_answer_instead_of_hiding_it(reason: str) -> None:
    client = make_client(
        FakeResponses(
            result=FakeResponse(
                output_text="Half an ans",
                usage=make_usage(),
                incomplete_details=FakeIncompleteDetails(reason=reason),
            )
        )
    )

    result = client.complete(
        system_prompt="You are helpful.",
        user_message="Say hello.",
        config=LlmConfig(model_name="test-model", max_output_tokens=8),
    )

    assert result.text == "Half an ans"
    assert result.incomplete_reason == reason


def test_complete_does_not_blame_the_provider_for_an_empty_truncated_answer() -> None:
    """The budget went into reasoning tokens, so there is no text but also no outage."""
    client = make_client(
        FakeResponses(
            result=FakeResponse(
                output_text="",
                usage=make_usage(),
                incomplete_details=FakeIncompleteDetails(reason="max_output_tokens"),
            )
        )
    )

    result = client.complete(
        system_prompt="You are helpful.",
        user_message="Say hello.",
        config=LlmConfig(model_name="test-model", max_output_tokens=8),
    )

    assert result.text == ""
    assert result.incomplete_reason == "max_output_tokens"


def test_complete_leaves_incomplete_reason_unset_for_a_full_answer() -> None:
    client = make_client(
        FakeResponses(result=FakeResponse(output_text="Hello.", usage=make_usage()))
    )

    result = client.complete(
        system_prompt="You are helpful.",
        user_message="Say hello.",
        config=LlmConfig(model_name="test-model", temperature=0.2),
    )

    assert result.incomplete_reason is None


def test_complete_structured_sends_the_configured_call_parameters() -> None:
    responses = FakeResponses(
        parsed_result=FakeParsedResponse(output_parsed=FakeSchema(value="parsed"))
    )
    client = make_client(responses)

    result = client.complete_structured(
        system_prompt="You are helpful.",
        user_message="Say hello.",
        config=LlmConfig(
            model_name="test-model", temperature=0.2, max_output_tokens=64
        ),
        schema=FakeSchema,
    )

    assert result.parsed == FakeSchema(value="parsed")
    assert responses.calls == [
        {
            "model": "test-model",
            "instructions": "You are helpful.",
            "input": "Say hello.",
            "text_format": FakeSchema,
            "temperature": 0.2,
            "max_output_tokens": 64,
        }
    ]


def test_complete_structured_uses_omit_for_unset_call_parameters() -> None:
    responses = FakeResponses(
        parsed_result=FakeParsedResponse(output_parsed=FakeSchema(value="parsed"))
    )
    client = make_client(responses)

    result = client.complete_structured(
        system_prompt="You are helpful.",
        user_message="Say hello.",
        config=LlmConfig(
            model_name="reasoning-model", temperature=None, max_output_tokens=None
        ),
        schema=FakeSchema,
    )

    assert result.parsed == FakeSchema(value="parsed")
    assert responses.calls == [
        {
            "model": "reasoning-model",
            "instructions": "You are helpful.",
            "input": "Say hello.",
            "text_format": FakeSchema,
            "temperature": omit,
            "max_output_tokens": omit,
        }
    ]


def call_structured(client: OpenAiLlmClient) -> None:
    client.complete_structured(
        system_prompt="You are helpful.",
        user_message="Say hello.",
        config=LlmConfig(model_name="test-model", temperature=0.2),
        schema=FakeSchema,
    )


@pytest.mark.parametrize(
    ("error_type", "status_code"),
    [
        (AuthenticationError, 401),
        (PermissionDeniedError, 403),
        (BadRequestError, 400),
        (NotFoundError, 404),
    ],
)
def test_complete_structured_maps_rejected_requests_to_configuration_error(
    error_type: type[APIStatusError], status_code: int
) -> None:
    upstream_error = make_status_error(error_type, status_code)
    client = make_client(FakeResponses(error=upstream_error))

    with pytest.raises(LlmConfigurationError) as exc_info:
        call_structured(client)

    assert exc_info.value.__cause__ is upstream_error


def test_complete_structured_maps_rate_limit_to_unavailable_error() -> None:
    upstream_error = make_status_error(RateLimitError, 429)
    client = make_client(FakeResponses(error=upstream_error))

    with pytest.raises(LlmUnavailableError) as exc_info:
        call_structured(client)

    assert exc_info.value.__cause__ is upstream_error


@pytest.mark.parametrize("error_type", [APIConnectionError, APITimeoutError])
def test_complete_structured_maps_connection_error_to_unavailable_error(
    error_type: type[APIConnectionError],
) -> None:
    upstream_error = error_type(
        request=httpx2.Request("POST", "https://api.openai.com/v1/responses")
    )
    client = make_client(FakeResponses(error=upstream_error))

    with pytest.raises(LlmUnavailableError) as exc_info:
        call_structured(client)

    assert exc_info.value.__cause__ is upstream_error


def test_complete_structured_maps_unknown_openai_error_to_unavailable_error() -> None:
    upstream_error = OpenAIError("something else went wrong")
    client = make_client(FakeResponses(error=upstream_error))

    with pytest.raises(LlmUnavailableError) as exc_info:
        call_structured(client)

    assert exc_info.value.__cause__ is upstream_error


def test_complete_structured_maps_schema_violation_to_response_format_error() -> None:
    upstream_error = make_validation_error()
    client = make_client(FakeResponses(error=upstream_error))

    with pytest.raises(LlmResponseFormatError) as exc_info:
        call_structured(client)

    assert exc_info.value.__cause__ is upstream_error
    assert "FakeSchema" in str(exc_info.value)


def test_complete_structured_rejects_a_missing_parse_result() -> None:
    client = make_client(
        FakeResponses(
            parsed_result=FakeParsedResponse(
                output_parsed=None,
                incomplete_details=FakeIncompleteDetails(reason="max_output_tokens"),
            )
        )
    )

    with pytest.raises(LlmResponseFormatError) as exc_info:
        call_structured(client)

    assert "max_output_tokens" in str(exc_info.value)


@pytest.mark.parametrize(
    "incomplete_details", [None, FakeIncompleteDetails(reason=None)]
)
def test_complete_structured_reports_a_refusal_when_no_reason_is_given(
    incomplete_details: FakeIncompleteDetails | None,
) -> None:
    """A missing result without a provider reason is a refusal, not a truncation."""
    client = make_client(
        FakeResponses(
            parsed_result=FakeParsedResponse(
                output_parsed=None, incomplete_details=incomplete_details
            )
        )
    )

    with pytest.raises(LlmResponseFormatError) as exc_info:
        call_structured(client)

    assert "refusal" in str(exc_info.value)


def test_complete_with_tools_rejects_duplicate_tool_names() -> None:
    # Two specs under one name: by_name keeps the last, but both would be sent
    # to the model, so one of the advertised tools is unreachable.
    responses = FakeResponses(result=FakeResponse(output_text="Hello from the model."))
    client = make_client(responses)
    spec = ToolSpec(name="search", params=SectionSearchParams, handler=lambda _: "")

    with pytest.raises(ValueError) as exc_info:
        client.complete_with_tools(
            system_prompt="You are helpful.",
            user_message="Say hello.",
            config=LlmConfig(model_name="test-model"),
            tools=[spec, spec],
        )

    assert "search" in str(exc_info.value)
    assert responses.calls == []


@pytest.mark.parametrize("max_rounds", [0, -1])
def test_complete_with_tools_rejects_a_budget_below_one(max_rounds: int) -> None:
    responses = FakeResponses(result=FakeResponse(output_text="Hello from the model."))
    client = make_client(responses)

    with pytest.raises(ValueError):
        client.complete_with_tools(
            system_prompt="You are helpful.",
            user_message="Say hello.",
            config=LlmConfig(model_name="test-model", temperature=0.2),
            tools=[],
            max_rounds=max_rounds,
        )

    assert responses.calls == []


def test_complete_with_tools_runs_a_handler_and_returns_the_final_answer() -> None:
    responses = FakeResponses(
        tool_results=[
            FakeToolResponse(output=[make_tool_call()], usage=make_usage()),
            FakeToolResponse(output_text="grounded answer", usage=make_usage()),
        ]
    )
    client = make_client(responses)
    seen: list[SectionSearchParams] = []

    def handler(params: SectionSearchParams) -> str:
        seen.append(params)
        return "doc#section: body"

    result = client.complete_with_tools(
        system_prompt="You are helpful.",
        user_message="Say hello.",
        config=LlmConfig(model_name="test-model"),
        tools=[make_tool_spec(handler)],
    )

    assert result.text == "grounded answer"
    assert result.stop_reason == "completed"
    assert [round_.tool_names for round_ in result.rounds] == [("search",), ()]
    assert [params.query for params in seen] == ["x"]
    # The second request has to carry the call and its output back, otherwise
    # the provider rejects it with a 400.
    assert len(responses.calls) == 2


def test_complete_with_tools_wraps_a_failing_handler() -> None:
    responses = FakeResponses(
        tool_results=[FakeToolResponse(output=[make_tool_call()], usage=make_usage())]
    )
    client = make_client(responses)

    def handler(params: SectionSearchParams) -> str:
        raise RuntimeError("the corpus is on fire")

    with pytest.raises(LlmToolError) as exc_info:
        client.complete_with_tools(
            system_prompt="You are helpful.",
            user_message="Say hello.",
            config=LlmConfig(model_name="test-model"),
            tools=[make_tool_spec(handler)],
        )

    # The tool name belongs in the message; the original stays reachable as the
    # cause so the traceback still points at the handler.
    assert "search" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_complete_with_tools_rejects_an_unexplained_empty_answer() -> None:
    responses = FakeResponses(
        tool_results=[FakeToolResponse(output_text="   ", usage=make_usage())]
    )
    client = make_client(responses)

    with pytest.raises(LlmUnavailableError):
        client.complete_with_tools(
            system_prompt="You are helpful.",
            user_message="Say hello.",
            config=LlmConfig(model_name="test-model"),
            tools=[],
        )


def test_complete_with_tools_keeps_an_empty_answer_that_was_cut_off() -> None:
    # Same empty text, but the provider explained it. That is a partial answer,
    # not a dead response, so the stop reason carries it instead of an error.
    responses = FakeResponses(
        tool_results=[
            FakeToolResponse(
                output_text="",
                incomplete_details=FakeIncompleteDetails(reason="max_output_tokens"),
                usage=make_usage(),
            )
        ]
    )
    client = make_client(responses)

    result = client.complete_with_tools(
        system_prompt="You are helpful.",
        user_message="Say hello.",
        config=LlmConfig(model_name="test-model"),
        tools=[],
    )

    assert result.text == ""
    assert result.stop_reason == "incomplete_details"


def test_complete_with_tools_stops_before_a_round_it_cannot_use() -> None:
    # The last round may still ask for a tool. Running the handler there would
    # spend the work and throw the result away, so the loop stops first.
    calls_made: list[str] = []

    def handler(params: SectionSearchParams) -> str:
        calls_made.append(params.query)
        return "never used"

    responses = FakeResponses(
        tool_results=[FakeToolResponse(output=[make_tool_call()], usage=make_usage())]
    )
    client = make_client(responses)

    result = client.complete_with_tools(
        system_prompt="You are helpful.",
        user_message="Say hello.",
        config=LlmConfig(model_name="test-model"),
        tools=[make_tool_spec(handler)],
        max_rounds=1,
    )

    assert result.stop_reason == "max_rounds"
    assert len(result.rounds) == 1
    assert calls_made == []


def test_complete_with_tools_rejects_arguments_that_miss_the_schema() -> None:
    responses = FakeResponses(
        tool_results=[
            FakeToolResponse(
                output=[make_tool_call(arguments='{"query": "x", "top_k": 99}')],
                usage=make_usage(),
            ),
            FakeToolResponse(output_text="unreachable"),
        ]
    )
    client = make_client(responses)

    with pytest.raises(LlmResponseFormatError):
        client.complete_with_tools(
            system_prompt="You are helpful.",
            user_message="Say hello.",
            config=LlmConfig(model_name="test-model"),
            tools=[make_tool_spec(lambda _: "")],
        )


def test_complete_with_tools_rejects_a_tool_the_model_invented() -> None:
    responses = FakeResponses(
        tool_results=[
            FakeToolResponse(
                output=[
                    FakeFunctionCall(
                        name="delete_everything", call_id="c", arguments="{}"
                    )
                ],
                usage=make_usage(),
            )
        ]
    )
    client = make_client(responses)

    with pytest.raises(LlmResponseFormatError) as exc_info:
        client.complete_with_tools(
            system_prompt="You are helpful.",
            user_message="Say hello.",
            config=LlmConfig(model_name="test-model"),
            tools=[make_tool_spec(lambda _: "")],
            max_rounds=3,
        )

    assert "delete_everything" in str(exc_info.value)


def test_complete_with_tools_logs_the_cost_of_a_run_that_failed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A run that raises spent the same tokens as one that returned. Without
    # this line the only runs missing from the cost log would be the broken
    # ones, which is the wrong half to lose.
    responses = FakeResponses(
        tool_results=[FakeToolResponse(output=[make_tool_call()], usage=make_usage())]
    )
    client = make_client(responses)

    def handler(params: SectionSearchParams) -> str:
        raise RuntimeError("the corpus is on fire")

    with (
        caplog.at_level(logging.INFO, logger="harness.infrastructure.llm.client"),
        pytest.raises(LlmToolError),
    ):
        client.complete_with_tools(
            system_prompt="You are helpful.",
            user_message="Say hello.",
            config=LlmConfig(model_name="test-model"),
            tools=[make_tool_spec(handler)],
        )

    runs = [r for r in caplog.records if r.message == "LLM tool run completed"]
    assert len(runs) == 1
    assert runs[0].outcome == "failed"  # type: ignore[attr-defined]
    assert runs[0].rounds == 1  # type: ignore[attr-defined]
    assert runs[0].total_tokens == 33  # type: ignore[attr-defined]
    assert runs[0].cache_write_tokens == 55  # type: ignore[attr-defined]


def _structured_client(
    responses: FakeResponses,
) -> Callable[..., LlmStructuredToolCompletion[FakeSchema]]:
    """Bind the boilerplate so each structured test shows only its own setup."""
    client = make_client(responses)

    def run(
        handler: Callable[[SectionSearchParams], str] = lambda params: "section text",
        max_rounds: int = 5,
    ) -> LlmStructuredToolCompletion[FakeSchema]:
        return client.complete_with_tools_structured(
            system_prompt="You are helpful.",
            user_message="Say hello.",
            config=LlmConfig(model_name="test-model"),
            tools=[make_tool_spec(handler)],
            schema=FakeSchema,
            max_rounds=max_rounds,
        )

    return run


def test_complete_with_tools_structured_runs_a_handler_and_parses_the_answer() -> None:
    """The whole point of the method: tools and a schema in one run.

    Two rounds, because a one-round run would not show that the tool result
    travelled back in. The schema applies to both rounds, but only the second
    one carries output_parsed, since the first produced a call and no answer.
    """
    responses = FakeResponses(
        parsed_tool_results=[
            FakeParsedToolResponse(output=[make_tool_call()]),
            FakeParsedToolResponse(
                output_text='{"value": "grounded"}',
                output_parsed=FakeSchema(value="grounded"),
            ),
        ]
    )
    seen: list[str] = []

    def handler(params: SectionSearchParams) -> str:
        seen.append(params.query)
        return "section text"

    result = _structured_client(responses)(handler)

    assert seen == ["x"]
    assert result.stop_reason == "completed"
    assert result.parsed == FakeSchema(value="grounded")
    assert result.text == '{"value": "grounded"}'
    assert len(result.rounds) == 2


def test_complete_with_tools_structured_sends_the_schema_as_text_format() -> None:
    # Without this the run would still pass its assertions while the provider
    # was never told to produce structured output at all.
    responses = FakeResponses(
        parsed_tool_results=[
            FakeParsedToolResponse(
                output_text='{"value": "x"}', output_parsed=FakeSchema(value="x")
            )
        ]
    )

    _structured_client(responses)()

    assert responses.calls[0]["text_format"] is FakeSchema
    assert responses.calls[0]["tools"]


def test_complete_with_tools_structured_leaves_parsed_empty_on_max_rounds() -> None:
    """The invariant: parsed is set exactly when stop_reason is "completed".

    An exhausted loop never reached a final answer, so there is nothing to
    parse. The partial text still travels, which is why this returns instead of
    raising: raising would throw away both the text and stop_reason.
    """
    responses = FakeResponses(
        parsed_tool_results=[
            FakeParsedToolResponse(output_text="partial", output=[make_tool_call()])
        ]
    )

    result = _structured_client(responses)(max_rounds=1)

    assert result.stop_reason == "max_rounds"
    assert result.parsed is None
    assert result.text == "partial"


def test_complete_with_tools_structured_leaves_parsed_empty_when_cut_off() -> None:
    # Same invariant from the other side. The provider stopped the round, so
    # any object built from it would be built from a truncated payload.
    responses = FakeResponses(
        parsed_tool_results=[
            FakeParsedToolResponse(
                output_text="cut",
                incomplete_details=FakeIncompleteDetails("max_output_tokens"),
            )
        ]
    )

    result = _structured_client(responses)()

    assert result.stop_reason == "incomplete_details"
    assert result.parsed is None
    assert result.text == "cut"


def test_complete_with_tools_structured_rejects_an_answer_it_cannot_parse() -> None:
    """A finished round with no object is a broken answer, not a broken provider.

    LlmResponseFormatError rather than LlmUnavailableError: the model answered,
    the answer just did not satisfy the schema, most likely a refusal.
    """
    responses = FakeResponses(
        parsed_tool_results=[FakeParsedToolResponse(output_text="", output_parsed=None)]
    )

    with pytest.raises(LlmResponseFormatError, match="could not be parsed"):
        _structured_client(responses)()


def test_complete_with_tools_structured_maps_a_validation_error() -> None:
    # The SDK validates inside parse(), so a schema violation surfaces as a
    # pydantic error from the call itself rather than as a bad return value.
    responses = FakeResponses(error=make_validation_error())

    with pytest.raises(LlmResponseFormatError, match="did not satisfy"):
        _structured_client(responses)()


def test_complete_with_tools_structured_still_rejects_a_duplicate_tool_name() -> None:
    # Guards the extraction itself: both public methods now share one loop, so
    # the checks that used to sit in complete_with_tools have to still fire on
    # the structured path.
    client = make_client(FakeResponses())
    spec = make_tool_spec(lambda params: "text")

    with pytest.raises(ValueError, match="Duplicate tool names"):
        client.complete_with_tools_structured(
            system_prompt="You are helpful.",
            user_message="Say hello.",
            config=LlmConfig(model_name="test-model"),
            tools=[spec, spec],
            schema=FakeSchema,
        )


# --- Streaming adapter -------------------------------------------------------
def make_stream_client(
    events: list[FakeStreamEvent],
    error: Exception | None = None,
    raise_at: int | None = None,
    create_error: Exception | None = None,
) -> tuple[OpenAiLlmClient, FakeStream, FakeResponses]:
    """Client plus both doubles, since the assertions need all three.

    Two error slots, because the two failure moments are different: create_error
    fires when the request is made and the caller sees it before the first
    event, `error` fires while the events are pulled and the caller has already
    received deltas by then.
    """
    stream = FakeStream(events=events, error=error, raise_at=raise_at)
    responses = FakeResponses(stream_result=stream, error=create_error)
    return make_client(responses), stream, responses


def make_completed_event(usage: FakeUsage | None = None) -> FakeStreamEvent:
    """A "response.completed" event wrapping a response with usage on it."""
    return FakeStreamEvent(
        type="response.completed",
        response=FakeResponse(output_text="", usage=usage),
    )


def make_incomplete_event(
    reason: str, usage: FakeUsage | None = None
) -> FakeStreamEvent:
    """A "response.incomplete" event: the terminal event of a run that was cut off."""
    return FakeStreamEvent(
        type="response.incomplete",
        response=FakeResponse(
            output_text="",
            usage=usage,
            incomplete_details=FakeIncompleteDetails(reason=reason),
        ),
    )


def delta_texts(chunks: Sequence[LlmStreamEvent]) -> list[str]:
    """The text of every chunk, and proof that every chunk carries text.

    Narrows the LlmStreamEvent union for the checkers, and the assert is the
    point rather than a formality: an LlmStreamEnd among the deltas would break
    the port's "end is last" promise.
    """
    texts: list[str] = []
    for chunk in chunks:
        assert isinstance(chunk, LlmTextDelta), f"expected a delta, got {chunk!r}"
        texts.append(chunk.text)
    return texts


def start_stream(client: OpenAiLlmClient) -> Generator[LlmStreamEvent]:
    """Start a run without consuming it, so a test can pull event by event.

    A Generator, like the port itself says: two tests need next() on a half-read
    run and close() on a suspended one, and only a generator offers close().
    """
    return client.stream(
        system_prompt="You are helpful.",
        user_message="Say hello.",
        config=LlmConfig(model_name="test-model", temperature=0.2),
    )


def stream_run_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.message == "LLM stream completed"]


def test_stream_yields_deltas_then_one_end() -> None:
    """The port's two guarantees in one test.

    Deltas in order, exactly one LlmStreamEnd, and it last. Usage travels on
    the end event only, so assert it there and nowhere else.
    """
    client, _stream, _responses = make_stream_client(
        [
            FakeStreamEvent(type="response.output_text.delta", delta="Hello "),
            FakeStreamEvent(type="response.output_text.delta", delta="world."),
            make_completed_event(make_usage()),
        ]
    )

    chunks = list(
        client.stream(
            system_prompt="You are helpful.",
            user_message="Say hello.",
            config=LlmConfig(model_name="test-model", temperature=0.2),
        )
    )

    *deltas, end = chunks
    assert delta_texts(deltas) == ["Hello ", "world."]
    assert isinstance(end, LlmStreamEnd)
    assert end.incomplete_reason is None
    assert end.usage == TokenUsage(
        input_tokens=11,
        output_tokens=22,
        total_tokens=33,
        cached_tokens=44,
        cache_write_tokens=55,
        reasoning_tokens=66,
    )


def test_stream_sends_the_same_request_as_complete_plus_stream_true() -> None:
    """Pins the request shape, like test_complete_returns_response_text does.

    Same keys as complete() with omit for the unset ones, plus stream=True.
    Use a reasoning-model config (temperature=None): that is the combination
    the omit path exists for.
    """
    client, _stream, responses = make_stream_client(
        [
            FakeStreamEvent(type="response.output_text.delta", delta="Hello."),
            make_completed_event(),
        ]
    )

    list(
        client.stream(
            system_prompt="You are helpful.",
            user_message="Say hello.",
            config=LlmConfig(model_name="reasoning-model", temperature=None),
        )
    )

    assert responses.calls == [
        {
            "model": "reasoning-model",
            "instructions": "You are helpful.",
            "input": "Say hello.",
            "temperature": omit,
            "max_output_tokens": omit,
            "stream": True,
        }
    ]


def test_stream_makes_no_request_before_the_first_event_is_pulled() -> None:
    """The generator promise from the port docstring.

    responses.calls stays empty right after stream() is called and fills on the
    first next(). That is what lets a caller turn a transport failure into a
    502 before it commits to a response.
    """
    client, _stream, responses = make_stream_client(
        [
            FakeStreamEvent(type="response.output_text.delta", delta="Hello."),
            make_completed_event(),
        ]
    )

    events = start_stream(client)

    assert responses.calls == []

    first = next(events)

    assert first == LlmTextDelta("Hello.")
    assert len(responses.calls) == 1


def test_stream_reports_an_incomplete_run_on_the_end_event() -> None:
    """incomplete_details -> LlmStreamEnd.incomplete_reason, and no raise.

    A truncated answer is a result, not a failure: the deltas already yielded
    are real text. Use "max_output_tokens" and assert the deltas survive.
    """
    client, _stream, _responses = make_stream_client(
        [
            FakeStreamEvent(type="response.output_text.delta", delta="Half an ans"),
            make_incomplete_event("max_output_tokens", make_usage()),
        ]
    )

    chunks = list(start_stream(client))

    *deltas, end = chunks
    assert delta_texts(deltas) == ["Half an ans"]
    assert isinstance(end, LlmStreamEnd)
    assert end.incomplete_reason == "max_output_tokens"
    assert end.usage is not None
    assert end.usage.total_tokens == 33


def test_stream_rejects_a_completed_run_without_any_delta() -> None:
    """The empty-stream case: completed, no incomplete_reason, no delta.

    Nothing explains the silence, so it is LlmUnavailableError - the same
    reading complete() gives an empty output_text, down to the wording.
    """
    client, _stream, _responses = make_stream_client(
        [make_completed_event(make_usage())]
    )

    with pytest.raises(LlmUnavailableError, match="contains no content"):
        list(start_stream(client))


@pytest.mark.parametrize(
    "event",
    [
        FakeStreamEvent(
            type="response.failed",
            response=FakeResponse(
                output_text="",
                error=FakeError(message="upstream exploded", code="server_error"),
            ),
        ),
        FakeStreamEvent(type="error", message="upstream exploded", code="server_error"),
    ],
    ids=["response.failed", "error"],
)
def test_stream_raises_on_a_provider_error_event(event: FakeStreamEvent) -> None:
    """Both error shapes, one test.

    "response.failed" carries its message on event.response.error, the
    top-level "error" event on the event itself. Assert the provider's message
    reaches the raised error, otherwise the two branches read identically in a
    log and the branch that fired cannot be told apart.
    """
    client, _stream, _responses = make_stream_client(
        [FakeStreamEvent(type="response.output_text.delta", delta="Half an ans"), event]
    )

    with pytest.raises(LlmUnavailableError) as exc_info:
        list(start_stream(client))

    assert "upstream exploded" in str(exc_info.value)
    assert "server_error" in str(exc_info.value)


def test_stream_raises_when_no_terminal_event_arrives() -> None:
    """The provider hung up mid-answer: deltas, then the iteration just ends.

    A different case from the empty stream, and the message has to say so.
    This is the one that must not quietly become LlmStreamEnd(None, None).
    """
    client, _stream, _responses = make_stream_client(
        [FakeStreamEvent(type="response.output_text.delta", delta="Half an ans")]
    )

    with pytest.raises(LlmUnavailableError, match="without a terminal event"):
        list(start_stream(client))


def test_stream_maps_a_transport_failure_during_iteration() -> None:
    """The except-ladder after the first yield, via FakeStream.raise_at.

    create() returns once the headers are in, so an APIConnectionError arrives
    while events are pulled. Pull one delta first, then assert the raise:
    that pins the ladder around the loop, which the create() tests cannot
    reach.
    """
    upstream_error = APIConnectionError(
        request=httpx2.Request("POST", "https://api.openai.com/v1/responses")
    )
    client, stream, _responses = make_stream_client(
        [
            FakeStreamEvent(type="response.output_text.delta", delta="Hello "),
            make_completed_event(make_usage()),
        ],
        error=upstream_error,
        raise_at=1,
    )

    events = start_stream(client)

    assert next(events) == LlmTextDelta("Hello ")

    with pytest.raises(LlmUnavailableError) as exc_info:
        next(events)

    assert exc_info.value.__cause__ is upstream_error
    assert stream.closed


def test_stream_closes_the_connection_when_the_caller_stops_early(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The reason for `with stream_obj:`, and the third exit's cost line.

    Pull one delta, then close the generator. GeneratorExit lands on the
    suspended yield, the with-block runs __exit__, FakeStream.closed flips.
    Without the with, the generator stays suspended forever holding a
    connection.

    The log line matters here for the same reason it does on the raising paths,
    and more so: from Commit 3 on, an SSE client hanging up is the common way a
    run ends early, so this exit must not be the one without a trace.
    GeneratorExit is not an Exception, so `except LlmError` never sees it.
    """
    client, stream, _responses = make_stream_client(
        [
            FakeStreamEvent(type="response.output_text.delta", delta="Hello "),
            FakeStreamEvent(type="response.output_text.delta", delta="world."),
            make_completed_event(make_usage()),
        ]
    )

    events = start_stream(client)

    with caplog.at_level(logging.INFO, logger="harness.infrastructure.llm.client"):
        assert next(events) == LlmTextDelta("Hello ")
        # Snapshotted, not asserted in place: `assert not stream.closed` narrows
        # the attribute to False for the rest of the function, mypy cannot see
        # that close() flips it, and it then reads everything below the next
        # assert as unreachable and stops checking it.
        closed_before = stream.closed

        events.close()

    assert not closed_before
    assert stream.closed
    runs = stream_run_records(caplog)
    assert len(runs) == 1
    assert runs[0].outcome == "abandoned"  # type: ignore[attr-defined]
    # No terminal event ever arrived, so the delta count is the only thing this
    # run has to show for what the provider already billed.
    assert runs[0].deltas == 1  # type: ignore[attr-defined]
    assert not hasattr(runs[0], "total_tokens")


def test_stream_closes_the_connection_on_a_provider_error() -> None:
    """Same guarantee on the raising path: closed is True after the raises."""
    client, stream, _responses = make_stream_client(
        [
            FakeStreamEvent(type="response.output_text.delta", delta="Half an ans"),
            FakeStreamEvent(
                type="response.failed",
                response=FakeResponse(
                    output_text="",
                    error=FakeError(message="upstream exploded", code="server_error"),
                ),
            ),
        ]
    )

    with pytest.raises(LlmUnavailableError):
        list(start_stream(client))

    assert stream.closed


def test_stream_ignores_an_event_type_it_does_not_know() -> None:
    # The elif chain has no else on purpose: the API ships new event types
    # continuously, and one of them between two deltas must not end the run.
    # A later rewrite into a match with a `case _: raise` would pass every
    # other test in this file.
    client, _stream, _responses = make_stream_client(
        [
            FakeStreamEvent(type="response.output_text.delta", delta="Hello "),
            FakeStreamEvent(type="response.output_item.added"),
            FakeStreamEvent(type="response.output_text.delta", delta="world."),
            make_completed_event(make_usage()),
        ]
    )

    chunks = list(start_stream(client))

    *deltas, end = chunks
    assert delta_texts(deltas) == ["Hello ", "world."]
    assert isinstance(end, LlmStreamEnd)


# The request itself can fail, and then nothing was ever streamed. Same ladder
# and the same four cases as the complete() block above, but asserted on the
# first next(): a generator body does not run until then, which is exactly the
# promise the port docstring makes to a caller that wants a 502 out of this.
@pytest.mark.parametrize(
    ("error_type", "status_code"),
    [
        (AuthenticationError, 401),
        (PermissionDeniedError, 403),
        (BadRequestError, 400),
        (NotFoundError, 404),
    ],
)
def test_stream_maps_rejected_requests_to_configuration_error(
    error_type: type[APIStatusError], status_code: int
) -> None:
    upstream_error = make_status_error(error_type, status_code)
    client, _stream, _responses = make_stream_client([], create_error=upstream_error)

    events = start_stream(client)

    with pytest.raises(LlmConfigurationError) as exc_info:
        next(events)

    assert exc_info.value.__cause__ is upstream_error


def test_stream_maps_rate_limit_to_unavailable_error() -> None:
    upstream_error = make_status_error(RateLimitError, 429)
    client, _stream, _responses = make_stream_client([], create_error=upstream_error)

    events = start_stream(client)

    with pytest.raises(LlmUnavailableError) as exc_info:
        next(events)

    assert exc_info.value.__cause__ is upstream_error


@pytest.mark.parametrize("error_type", [APIConnectionError, APITimeoutError])
def test_stream_maps_connection_error_to_unavailable_error(
    error_type: type[APIConnectionError],
) -> None:
    upstream_error = error_type(
        request=httpx2.Request("POST", "https://api.openai.com/v1/responses")
    )
    client, _stream, _responses = make_stream_client([], create_error=upstream_error)

    events = start_stream(client)

    with pytest.raises(LlmUnavailableError) as exc_info:
        next(events)

    assert exc_info.value.__cause__ is upstream_error


def test_stream_maps_unknown_openai_error_to_unavailable_error() -> None:
    upstream_error = OpenAIError("something else went wrong")
    client, _stream, _responses = make_stream_client([], create_error=upstream_error)

    events = start_stream(client)

    with pytest.raises(LlmUnavailableError) as exc_info:
        next(events)

    assert exc_info.value.__cause__ is upstream_error


def test_stream_logs_the_cost_of_a_run_that_failed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The empty-stream case is the sharpest version of the problem: the run was
    # billed and the terminal event even carried the numbers, so raising without
    # the cost line would throw away usage that was already in hand.
    client, _stream, _responses = make_stream_client(
        [make_completed_event(make_usage())]
    )

    with (
        caplog.at_level(logging.INFO, logger="harness.infrastructure.llm.client"),
        pytest.raises(LlmUnavailableError),
    ):
        list(start_stream(client))

    runs = stream_run_records(caplog)
    assert len(runs) == 1
    assert runs[0].outcome == "failed"  # type: ignore[attr-defined]
    assert runs[0].deltas == 0  # type: ignore[attr-defined]
    assert runs[0].total_tokens == 33  # type: ignore[attr-defined]
    assert runs[0].cache_write_tokens == 55  # type: ignore[attr-defined]


def test_stream_logs_a_run_that_never_reached_a_terminal_event(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # No terminal event means no usage, so this line carries no numbers. It is
    # still the only record that the run happened, and a run that produced
    # deltas was billed for them.
    client, _stream, _responses = make_stream_client(
        [FakeStreamEvent(type="response.output_text.delta", delta="Half an ans")]
    )

    with (
        caplog.at_level(logging.INFO, logger="harness.infrastructure.llm.client"),
        pytest.raises(LlmUnavailableError),
    ):
        list(start_stream(client))

    runs = stream_run_records(caplog)
    assert len(runs) == 1
    assert runs[0].outcome == "failed"  # type: ignore[attr-defined]
    assert runs[0].deltas == 1  # type: ignore[attr-defined]
    assert not hasattr(runs[0], "total_tokens")


def test_stream_logs_a_run_that_returned(caplog: pytest.LogCaptureFixture) -> None:
    client, _stream, _responses = make_stream_client(
        [
            FakeStreamEvent(type="response.output_text.delta", delta="Hello."),
            make_completed_event(make_usage()),
        ]
    )

    with caplog.at_level(logging.INFO, logger="harness.infrastructure.llm.client"):
        list(start_stream(client))

    runs = stream_run_records(caplog)
    assert len(runs) == 1
    assert runs[0].outcome == "completed"  # type: ignore[attr-defined]
    assert runs[0].incomplete_reason is None  # type: ignore[attr-defined]
    assert runs[0].deltas == 1  # type: ignore[attr-defined]
    assert runs[0].total_tokens == 33  # type: ignore[attr-defined]
