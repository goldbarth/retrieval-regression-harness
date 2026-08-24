from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast, get_args

from pydantic import BaseModel

from harness.core.config import LlmConfig

LlmIncompleteReason = Literal["max_output_tokens", "content_filter"]
LLM_INCOMPLETE_REASONS = cast(
    tuple[LlmIncompleteReason, ...],
    get_args(LlmIncompleteReason),
)

LlmToolStopReason = Literal["completed", "incomplete_details", "max_rounds"]
LLM_TOOL_STOP_REASONS = cast(
    tuple[LlmToolStopReason, ...],
    get_args(LlmToolStopReason),
)


class LlmError(Exception):
    """Base llm error. Raised when an exception occurred,
    but no differentiation is required."""


class LlmConfigurationError(LlmError):
    """Raised when the internal configuration is not set properly."""


class LlmUnavailableError(LlmError):
    """Raised when the language model cannot be reached or does not answer."""


class LlmResponseFormatError(LlmError):
    """Raised when the model's response cannot be parsed or is invalid."""


class LlmToolError(LlmError):
    """Raised when a tool handler fails during a tool-augmented run.

    The fault is ours, not the provider's: the model asked for a tool we
    offered and our handler could not deliver it. It stays inside the LlmError
    family so one except still covers a whole run, and it names the tool so the
    log points at the handler rather than at the model.
    """


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cached_tokens: int
    cache_write_tokens: int
    reasoning_tokens: int


@dataclass(frozen=True)
class LlmStructuredCompletion[T: BaseModel]:
    parsed: T
    usage: TokenUsage | None


@dataclass(frozen=True)
class LlmCompletion:
    text: str
    usage: TokenUsage | None = None
    incomplete_reason: LlmIncompleteReason | None = None
    """Set when the provider stopped early. The text is then a partial answer."""


class TextCompleter(Protocol):
    """Port for plain text completion. Implemented by adapters in infrastructure."""

    def complete(
        self, system_prompt: str, user_message: str, config: LlmConfig
    ) -> LlmCompletion:
        """Return the model's answer or raise LlmError."""
        ...


class StructuredCompleter(Protocol):
    """Port for schema-bound completion. What the judge needs, and no more."""

    def complete_structured[T: BaseModel](
        self, system_prompt: str, user_message: str, config: LlmConfig, schema: type[T]
    ) -> LlmStructuredCompletion[T]:
        """Return the model's answer parsed into the schema or raise LlmError.

        Args:
            system_prompt: Instructions that define the model's behavior and role.
            user_message: The input text to be processed by the model.
            config: Configuration settings for the language model.
            schema: Pydantic model type that defines the expected response structure.

        Returns:
            LlmStructuredCompletion containing the parsed response and token usage.

        Raises:
            LlmError: When an error occurs during completion or response parsing.
        """
        ...


class ToolCompleter(Protocol):
    """Port for tool-augmented completion. The model may call the given tools."""

    def complete_with_tools(
        self,
        system_prompt: str,
        user_message: str,
        config: LlmConfig,
        tools: Sequence[ToolSpec[Any]],
        max_rounds: int = 5,
    ) -> LlmToolCompletion:
        """Run the model until it answers without calling a tool.

        Raises:
            LlmError: On transport failure, an unknown tool name, an empty
                answer, or a tool handler that raised.
            ValueError: If max_rounds is below 1, or if two tools share a name.
                Both are mistakes in the calling code, checked before the first
                request. The call therefore always reaches the provider at
                least once, so "max_rounds" never means "no attempt".
        """
        ...


class StructuredToolCompleter(Protocol):
    """Port for a tool-augmented run whose final answer is bound to a schema."""

    def complete_with_tools_structured[T: BaseModel](
        self,
        system_prompt: str,
        user_message: str,
        config: LlmConfig,
        tools: Sequence[ToolSpec[Any]],
        schema: type[T],
        max_rounds: int = 5,
    ) -> LlmStructuredToolCompletion[T]:
        """Run the model until it answers, then parse that answer into `schema`.

        A role of its own rather than a `schema: type[T] | None` on
        complete_with_tools. An optional schema would make the return type
        depend on the value of an argument, which neither the type checker nor
        a test double can express, and every caller of the plain method would
        start paying for a parsed field it never asked for.

        Args:
            system_prompt: Instructions that define the model's behavior and role.
            user_message: The input text to be processed by the model.
            config: Configuration settings for the language model.
            tools: The tools the model may call during the run.
            schema: Pydantic model type the final answer is parsed into.
            max_rounds: Upper bound on provider calls, tool rounds included.

        Raises:
            LlmError: As complete_with_tools, plus LlmResponseFormatError when
                the model answered but the answer did not satisfy the schema.
            ValueError: If max_rounds is below 1, or if two tools share a name.
        """
        ...


class LlmClient(
    TextCompleter,
    StructuredCompleter,
    ToolCompleter,
    StructuredToolCompleter,
    Protocol,
):
    """The full adapter surface, for wiring only.

    Consumers depend on the single role they use, so a new method on one role
    cannot break a test double that never touches it. Protocol has to stay in
    the bases here: without it this becomes a plain ABC and adapters would have
    to inherit from it to satisfy it.
    """


@dataclass(frozen=True)
class ToolSpec[T: BaseModel]:
    name: str
    params: type[T]
    handler: Callable[[T], str]
    description: str | None = None


@dataclass(frozen=True)
class ToolRound:
    tool_names: tuple[str, ...]
    usage: TokenUsage | None


@dataclass(frozen=True)
class LlmToolCompletion:
    text: str
    rounds: tuple[ToolRound, ...]
    stop_reason: LlmToolStopReason


@dataclass(frozen=True)
class LlmStructuredToolCompletion[T: BaseModel]:
    """A tool-augmented run whose final answer was meant to be schema-bound.

    Carries `text` as well as `parsed`, so it is a superset of
    LlmToolCompletion rather than a replacement. An exhausted or truncated run
    has no schema-compliant answer but still produced text, and raising instead
    would throw that text away together with stop_reason, which exists
    precisely so a caller can tell "completed" from "max_rounds".
    """

    text: str
    """Raw output of the last round, always set. Under a schema this is the
    JSON the model emitted, so on "completed" it is the serialized form of
    `parsed`; on "max_rounds" it is whatever partial text the model had
    produced by then."""

    parsed: T | None
    """Not None exactly when stop_reason is "completed".

    An invariant, not a type guarantee: the adapter establishes it and
    test_llm_client pins it. A caller that needs the parsed answer therefore
    checks this field rather than stop_reason, because only this check narrows
    the type."""

    rounds: tuple[ToolRound, ...]
    stop_reason: LlmToolStopReason


@dataclass(frozen=True)
class SectionHit:
    doc_id: str
    section: str
    text: str


class SectionSearch(Protocol):
    def find(self, query: str, top_k: int) -> Sequence[SectionHit]:
        """Search for document sections matching the query.

        Args:
            query: Search terms to match against document sections.
            top_k: Maximum number of results to return.

        Returns:
            Sequence of matching section hits, ranked by relevance.
        """
        ...
