import logging
from collections.abc import Callable
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
    LlmStructuredToolCompletion,
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
class FakeResponse:
    output_text: str
    usage: FakeUsage | None = None
    incomplete_details: FakeIncompleteDetails | None = None


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

    def create(self, **kwargs: object) -> Any:
        self.calls.append(kwargs)
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
