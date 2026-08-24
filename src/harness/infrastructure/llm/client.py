import logging
from collections import Counter
from collections.abc import Sequence
from typing import Any, Literal, cast

from openai import (
    APIConnectionError,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    OpenAI,
    OpenAIError,
    PermissionDeniedError,
    RateLimitError,
    omit,
)
from openai.types.responses import FunctionToolParam
from openai.types.responses.response_usage import ResponseUsage as OpenAiResponseUsage
from pydantic import BaseModel, ValidationError

from harness.core.config import LlmConfig
from harness.core.interfaces import (
    LlmCompletion,
    LlmConfigurationError,
    LlmError,
    LlmResponseFormatError,
    LlmStructuredCompletion,
    LlmStructuredToolCompletion,
    LlmToolCompletion,
    LlmToolError,
    LlmToolStopReason,
    LlmUnavailableError,
    TokenUsage,
    ToolRound,
    ToolSpec,
)

logger = logging.getLogger(__name__)


def _to_token_usage(
    usage: OpenAiResponseUsage | None, model_name: str
) -> TokenUsage | None:
    if usage is None:
        logger.warning(
            "LLM call completed without usage data",
            extra={"model_name": model_name},
        )
        return None

    return TokenUsage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        total_tokens=usage.total_tokens,
        cached_tokens=usage.input_tokens_details.cached_tokens,
        cache_write_tokens=usage.input_tokens_details.cache_write_tokens,
        reasoning_tokens=usage.output_tokens_details.reasoning_tokens,
    )


LlmToolOutcome = LlmToolStopReason | Literal["failed"]
"""How a tool run ended, for the log only.

A stop reason describes a run that returned something. "failed" covers the runs
that raised instead, and those are billed exactly the same, so the log needs a
word for them that the port does not.
"""


def _log_tool_run(
    model_name: str, rounds: Sequence[ToolRound], outcome: LlmToolOutcome
) -> None:
    """Log one line per finished tool run.

    complete() logs every call it makes; without this the tool path only ever
    logged its warnings, so a successful multi-round run left no trace and its
    cost could not be reconstructed from the logs.
    """
    billed = [r.usage for r in rounds if r.usage is not None]
    logger.info(
        "LLM tool run completed",
        extra={
            "model_name": model_name,
            "outcome": outcome,
            "rounds": len(rounds),
            "tool_calls": sum(len(r.tool_names) for r in rounds),
            "input_tokens": sum(u.input_tokens for u in billed),
            "output_tokens": sum(u.output_tokens for u in billed),
            "total_tokens": sum(u.total_tokens for u in billed),
            "cached_tokens": sum(u.cached_tokens for u in billed),
            "cache_write_tokens": sum(u.cache_write_tokens for u in billed),
            "reasoning_tokens": sum(u.reasoning_tokens for u in billed),
        },
    )


def _to_tool_param(spec: ToolSpec[Any]) -> FunctionToolParam:
    """Describe one tool the way the Responses API wants it.

    Precondition on spec.params, unchecked here: strict mode wants
    additionalProperties: false and every property in required, so the model
    needs model_config = ConfigDict(extra="forbid") and no field defaults. Use
    `str | None` where a value may be absent, since strict makes every field
    required and a default therefore never fires. A model that breaks this
    still builds a schema, and the provider rejects it as a BadRequestError,
    which arrives as LlmConfigurationError far from the tool that caused it.

    The SDK has to_strict_json_schema for exactly this, but it lives in
    openai.lib._tools and is private, and model_json_schema() on a model that
    holds the precondition produces the same document.
    """
    param: FunctionToolParam = {
        "type": "function",
        "name": spec.name,
        "parameters": spec.params.model_json_schema(),
        "strict": True,
    }
    description = spec.description or spec.params.__doc__
    if description is not None:
        # The key stays out when there is nothing to say. An empty string would
        # tell the model the tool has an (empty) description.
        param["description"] = description
    return param


class OpenAiLlmClient:
    """LlmClient adapter for the OpenAI Responses API."""

    def __init__(self, client: OpenAI) -> None:
        self._client = client

    def complete(
        self, system_prompt: str, user_message: str, config: LlmConfig
    ) -> LlmCompletion:
        try:
            response = self._client.responses.create(
                model=config.model_name,
                instructions=system_prompt,
                input=user_message,
                temperature=config.temperature
                if config.temperature is not None
                else omit,
                max_output_tokens=config.max_output_tokens
                if config.max_output_tokens is not None
                else omit,
            )
        except (
            AuthenticationError,
            PermissionDeniedError,
            BadRequestError,
            NotFoundError,
        ) as exc:
            raise LlmConfigurationError(
                f"Request for model {config.model_name} was rejected."
            ) from exc
        except (RateLimitError, APIConnectionError) as exc:
            raise LlmUnavailableError(
                f"Model {config.model_name} is currently unavailable."
            ) from exc
        except OpenAIError as exc:
            raise LlmUnavailableError(
                f"Model {config.model_name} did not answer."
            ) from exc

        details = response.incomplete_details
        incomplete_reason = details.reason if details is not None else None

        text = response.output_text
        if incomplete_reason is not None:
            # The provider stopped early, so an empty text is explained and not
            # a sign that the model is unavailable.
            logger.warning(
                "LLM response is incomplete",
                extra={
                    "model_name": config.model_name,
                    "incomplete_reason": incomplete_reason,
                },
            )
        elif not text.strip():
            raise LlmUnavailableError(
                f"The response for the {config.model_name} model contains no content."
            )

        usage = response.usage
        token_usage = _to_token_usage(usage, config.model_name)
        if token_usage is not None:
            logger.info(
                "LLM call completed",
                extra={
                    "model_name": config.model_name,
                    "input_tokens": token_usage.input_tokens,
                    "output_tokens": token_usage.output_tokens,
                    "total_tokens": token_usage.total_tokens,
                    "cached_tokens": token_usage.cached_tokens,
                    "cache_write_tokens": token_usage.cache_write_tokens,
                    "reasoning_tokens": token_usage.reasoning_tokens,
                },
            )

        return LlmCompletion(text, token_usage, incomplete_reason)

    def complete_structured[T: BaseModel](
        self, system_prompt: str, user_message: str, config: LlmConfig, schema: type[T]
    ) -> LlmStructuredCompletion[T]:
        try:
            response = self._client.responses.parse(
                model=config.model_name,
                instructions=system_prompt,
                input=user_message,
                text_format=schema,
                temperature=config.temperature
                if config.temperature is not None
                else omit,
                max_output_tokens=config.max_output_tokens
                if config.max_output_tokens is not None
                else omit,
            )
        except (
            AuthenticationError,
            PermissionDeniedError,
            BadRequestError,
            NotFoundError,
        ) as exc:
            raise LlmConfigurationError(
                f"Request for model {config.model_name} was rejected."
            ) from exc
        except (RateLimitError, APIConnectionError) as exc:
            raise LlmUnavailableError(
                f"Model {config.model_name} is currently unavailable."
            ) from exc
        except ValidationError as exc:
            # Known gap: the SDK validates inside parse(), so the response object
            # and its usage are lost here. The call was billed but stays
            # unrecorded. Closing this would mean create() plus our own
            # model_validate_json, and with it the schema generation we
            # deliberately left to the SDK.
            raise LlmResponseFormatError(
                f"Response from {config.model_name} did not satisfy {schema.__name__}."
            ) from exc
        except OpenAIError as exc:
            raise LlmUnavailableError(
                f"Model {config.model_name} did not answer."
            ) from exc

        parsed = response.output_parsed
        if parsed is None:
            details = response.incomplete_details
            reason = (
                details.reason
                if details is not None and details.reason is not None
                else "refusal"
            )
            raise LlmResponseFormatError(
                f"Response from {config.model_name} could not be parsed ({reason})."
            )

        token_usage = _to_token_usage(response.usage, config.model_name)

        return LlmStructuredCompletion(parsed=parsed, usage=token_usage)

    def complete_with_tools(
        self,
        system_prompt: str,
        user_message: str,
        config: LlmConfig,
        tools: Sequence[ToolSpec[Any]],
        max_rounds: int = 5,
    ) -> LlmToolCompletion:
        result = self._run_tool_loop(
            system_prompt=system_prompt,
            user_message=user_message,
            config=config,
            tools=tools,
            max_rounds=max_rounds,
            schema=None,
        )
        # result.parsed is always None on this path, so it is dropped rather
        # than handed to callers who never asked for a schema.
        return LlmToolCompletion(result.text, result.rounds, result.stop_reason)

    def complete_with_tools_structured[T: BaseModel](
        self,
        system_prompt: str,
        user_message: str,
        config: LlmConfig,
        tools: Sequence[ToolSpec[Any]],
        schema: type[T],
        max_rounds: int = 5,
    ) -> LlmStructuredToolCompletion[T]:
        result = self._run_tool_loop(
            system_prompt=system_prompt,
            user_message=user_message,
            config=config,
            tools=tools,
            max_rounds=max_rounds,
            schema=schema,
        )
        # The loop is not generic, so it can only promise BaseModel. What it put
        # in `parsed` came out of responses.parse(text_format=schema) and is
        # therefore a T; the cast narrows the type parameter and nothing else,
        # on the same object.
        return cast(LlmStructuredToolCompletion[T], result)

    def _run_tool_loop(
        self,
        system_prompt: str,
        user_message: str,
        config: LlmConfig,
        tools: Sequence[ToolSpec[Any]],
        max_rounds: int,
        schema: type[BaseModel] | None,
    ) -> LlmStructuredToolCompletion[BaseModel]:
        """Drive the tool loop, with or without a schema on the final answer.

        One loop for both public methods. The two differ in exactly two places,
        the request call and how the final round is read, and everything else
        that matters here is easy to get subtly wrong twice: feeding the calls
        back so the next request is not rejected, stopping before a round whose
        output nobody would read, and getting the cost line out before an
        exception leaves the method.

        Not generic on purpose. A `type[T] | None` parameter cannot be solved
        when the caller passes None, so the plain path would need a type
        argument it has no use for. The structured path narrows with a cast
        instead, which is one documented line in one place.
        """
        if max_rounds < 1:
            raise ValueError(f"max_rounds must be at least 1, got {max_rounds}.")

        by_name = {s.name: s for s in tools}
        if len(by_name) != len(tools):
            counts = Counter(s.name for s in tools)
            duplicates = sorted(name for name, count in counts.items() if count > 1)
            raise ValueError(f"Duplicate tool names: {', '.join(duplicates)}.")
        tool_params = [_to_tool_param(s) for s in tools]
        items: list[Any] = [{"role": "user", "content": user_message}]
        rounds: list[ToolRound] = []
        last_response_text = ""

        # Any raise below leaves a run that already spent rounds. Those calls
        # were billed, so the cost line has to go out before the exception
        # does, or a failed run is the one run whose price nobody can see.
        try:
            for round_index in range(max_rounds):
                parsed_output: BaseModel | None = None
                try:
                    if schema is None:
                        response = self._client.responses.create(
                            model=config.model_name,
                            instructions=system_prompt,
                            input=items,
                            tools=tool_params,
                            temperature=config.temperature
                            if config.temperature is not None
                            else omit,
                            max_output_tokens=config.max_output_tokens
                            if config.max_output_tokens is not None
                            else omit,
                        )
                    else:
                        # text_format and tools travel in the same request. The
                        # schema applies to every round, but only a round that
                        # calls no tool produces output_parsed: while the model
                        # is still calling tools its output items are the calls,
                        # not an answer.
                        parsed_response = self._client.responses.parse(
                            model=config.model_name,
                            instructions=system_prompt,
                            input=items,
                            tools=tool_params,
                            text_format=schema,
                            temperature=config.temperature
                            if config.temperature is not None
                            else omit,
                            max_output_tokens=config.max_output_tokens
                            if config.max_output_tokens is not None
                            else omit,
                        )
                        response = parsed_response
                        parsed_output = parsed_response.output_parsed
                except ValidationError as exc:
                    # Only reachable on the parse() path: the SDK validates
                    # inside the call, so the response object and its usage are
                    # lost. Same known gap as complete_structured, and the round
                    # is still billed.
                    raise LlmResponseFormatError(
                        f"Response from {config.model_name} did not satisfy "
                        f"{schema.__name__ if schema is not None else '?'}."
                    ) from exc
                except (
                    AuthenticationError,
                    PermissionDeniedError,
                    BadRequestError,
                    NotFoundError,
                ) as exc:
                    raise LlmConfigurationError(
                        f"Request for model {config.model_name} was rejected."
                    ) from exc
                except (RateLimitError, APIConnectionError) as exc:
                    raise LlmUnavailableError(
                        f"Model {config.model_name} is currently unavailable."
                    ) from exc
                except OpenAIError as exc:
                    raise LlmUnavailableError(
                        f"Model {config.model_name} did not answer."
                    ) from exc

                usage = _to_token_usage(response.usage, config.model_name)
                calls = [i for i in response.output if i.type == "function_call"]
                rounds.append(ToolRound(tuple(c.name for c in calls), usage))
                last_response_text = response.output_text

                details = response.incomplete_details
                if details is not None:
                    # The provider cut this round off, so any call it contains may be
                    # truncated. We stop instead of running a handler on half an
                    # argument object.
                    logger.warning(
                        "LLM tool response is incomplete",
                        extra={
                            "model_name": config.model_name,
                            "incomplete_reason": details.reason,
                            "round": round_index,
                        },
                    )
                    _log_tool_run(config.model_name, rounds, "incomplete_details")
                    return LlmStructuredToolCompletion(
                        response.output_text, None, tuple(rounds), "incomplete_details"
                    )

                if not calls:
                    text = response.output_text
                    if schema is not None:
                        if parsed_output is None:
                            # incomplete_details was handled above, so the round
                            # ran to the end and still produced no object. What
                            # is left is a refusal or an empty output, and both
                            # are a broken answer rather than a broken provider.
                            raise LlmResponseFormatError(
                                f"Response from {config.model_name} could not be "
                                f"parsed into {schema.__name__}."
                            )
                    elif not text.strip():
                        # Nothing stopped the model early and it asked for no tool,
                        # so an empty answer has no explanation left. Under a schema
                        # this check is moot: output_text then holds the JSON, which
                        # is non-empty whenever parsed_output is set.
                        raise LlmUnavailableError(
                            f"The response for the {config.model_name} model "
                            f"contains no content."
                        )

                    _log_tool_run(config.model_name, rounds, "completed")
                    return LlmStructuredToolCompletion(
                        text, parsed_output, tuple(rounds), "completed"
                    )

                if round_index == max_rounds - 1:
                    # No round left to feed the results into, so we stop before doing
                    # work whose output nobody would read.
                    break

                # The calls must go back, otherwise the output is rejected with a 400.
                items += response.output
                for call in calls:
                    spec = by_name.get(call.name)
                    if spec is None:
                        raise LlmResponseFormatError(
                            f"Model {config.model_name} called the unknown tool "
                            f"{call.name!r}."
                        )
                    try:
                        args = spec.params.model_validate_json(call.arguments)
                    except ValidationError as exc:
                        raise LlmResponseFormatError(
                            f"Arguments for tool {call.name!r} did not satisfy "
                            f"{spec.params.__name__}."
                        ) from exc
                    try:
                        output = spec.handler(args)
                    except Exception as exc:
                        # A handler is our code, so this is not a provider failure.
                        # Wrapping it keeps the caller's single except intact and
                        # names the tool, which a raw exception from three frames
                        # down does not.
                        raise LlmToolError(
                            f"Tool {call.name!r} failed in round {round_index}."
                        ) from exc

                    items.append(
                        {
                            "type": "function_call_output",
                            "call_id": call.call_id,
                            "output": output,
                        }
                    )

        except LlmError:
            _log_tool_run(config.model_name, rounds, "failed")
            raise

        _log_tool_run(config.model_name, rounds, "max_rounds")
        # No parsed answer here even under a schema: the loop ran out of rounds
        # before the model stopped calling tools, so there was never a final
        # answer to parse. The partial text still travels, as it does without a
        # schema.
        return LlmStructuredToolCompletion(
            last_response_text, None, tuple(rounds), "max_rounds"
        )
