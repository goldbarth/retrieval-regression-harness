from dataclasses import dataclass

from harness.core.interfaces import (
    LlmConfigurationError,
    LlmError,
    LlmResponseFormatError,
    LlmToolError,
    LlmUnavailableError,
)


@dataclass(frozen=True)
class LlmErrorResponse:
    """What one LlmError becomes on its way out: to the caller, and to the log.

    log_message travels with the other two on purpose. One handler now answers
    every LlmError, so the log wording is the third thing that used to sit next
    to the class and would otherwise be the only one left behind. Together the
    row answers the question one actually has while something is broken: what
    did the caller see, and what did we write down about it.
    """

    status_code: int
    detail: str
    log_message: str


# The one place this wording exists. Both exits read from here: the exception
# handler in main.py and the SSE error frame in analyze.py, which is the
# streaming twin of that handler. What a caller learns about our internals
# therefore no longer depends on whether the run failed before or after the
# first byte.
LLM_ERROR_RESPONSES: dict[type[LlmError], LlmErrorResponse] = {
    # The base class has a row of its own rather than a fallback branch: with
    # the MRO lookup below, an error class added later resolves to its nearest
    # ancestor, and this row is the last one every LlmError shares. 500,
    # because an error we never classified is our gap, not the provider's.
    LlmError: LlmErrorResponse(
        status_code=500,
        detail="internal server error",
        log_message="LLM call failed for a reason we do not classify",
    ),
    LlmConfigurationError: LlmErrorResponse(
        status_code=500,
        detail="internal server error",
        log_message="LLM call failed because of our own configuration",
    ),
    # Our handler failed, not the provider, so this is a 500 like any other bug
    # of ours and must not be reported as an upstream problem.
    LlmToolError: LlmErrorResponse(
        status_code=500,
        detail="internal server error",
        log_message="LLM tool run failed inside one of our own handlers",
    ),
    # The provider answered, but with something we cannot use: a schema the
    # response missed, a tool we never offered, arguments that do not parse.
    # Same 502 as an absent answer, because the fault sits upstream either way
    # and the caller can do nothing differently.
    LlmResponseFormatError: LlmErrorResponse(
        status_code=502,
        detail="upstream model answer was unusable",
        log_message="LLM call failed because the response could not be used",
    ),
    LlmUnavailableError: LlmErrorResponse(
        status_code=502,
        detail="upstream model unavailable",
        log_message="LLM call failed because the provider did not answer",
    ),
}


def llm_error_response(exc: LlmError) -> LlmErrorResponse:
    """Find the row for this error along its own class hierarchy.

    Keyed by class and walked over type(exc).__mro__, rather than isinstance
    against an ordered table: order in the table would then decide which row a
    subclass gets, and a row moved for readability could silently change an
    answer. The MRO puts the most specific class first by construction, so a
    subclass added below one of the four inherits its answer without an edit
    here, and gets its own the moment it is given a row.
    """
    for cls in type(exc).__mro__:
        if (entry := LLM_ERROR_RESPONSES.get(cls)) is not None:
            return entry

    # Unreachable while the parameter is an LlmError: its MRO always contains
    # LlmError, which has a row. Kept because the type system cannot say that,
    # and because a caller passing something else deserves an answer rather
    # than an implicit None.
    return LLM_ERROR_RESPONSES[LlmError]
