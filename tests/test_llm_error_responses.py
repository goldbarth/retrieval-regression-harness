from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from harness.api.dependencies import get_llm_client
from harness.api.errors import LLM_ERROR_RESPONSES, LlmErrorResponse
from harness.core.config import LlmConfig
from harness.core.interfaces import LlmCompletion, LlmError
from harness.main import app


class RaisingLlmClient:
    def __init__(self, error_type: type[LlmError]) -> None:
        self._error_type = error_type

    def complete(
        self, system_prompt: str, user_message: str, config: LlmConfig
    ) -> LlmCompletion:
        # Deliberately not any of the fixed wordings: a fake that says what the
        # handler says could not tell an answer from a leaked provider message.
        raise self._error_type("provider detail we are not supposed to repeat")


@pytest.fixture
def error_client(request: pytest.FixtureRequest) -> Iterator[TestClient]:
    """Reach the LLM exception handler through the real /analyze route.

    /analyze calls the LLM port, so the tests need no throwaway routes. The
    override replaces only the LLM client dependency with a fake adapter that
    raises the requested domain error.

    FastAPI resolves dependency_overrides per request. Therefore this fixture
    must yield so the override stays active while the test sends its request.
    The overrides live on the global app instance and must be cleared afterwards
    to avoid leaking fake dependencies into later tests.
    """
    error_type: type[LlmError] = request.param
    app.dependency_overrides[get_llm_client] = lambda: RaisingLlmClient(error_type)

    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


@pytest.mark.parametrize(
    ("error_client", "expected"),
    [
        pytest.param(error_type, response, id=error_type.__name__)
        for error_type, response in LLM_ERROR_RESPONSES.items()
    ],
    indirect=["error_client"],
)
def test_llm_error_answers_its_table_row(
    error_client: TestClient, expected: LlmErrorResponse
) -> None:
    """Every row of the table, end to end through the handler.

    Parametrized over LLM_ERROR_RESPONSES rather than spelling the wording out
    once per error class: a row added there is covered without an edit here,
    and no assertion can keep claiming a wording the table no longer has. Which
    error deserves a 500 and which a 502 is argued at the table itself.

    The bare LlmError is a row like the others. It has no handler of its own -
    that is the point: before the single handler on the base class it fell
    through as an uncontrolled 500 with a traceback in the body.
    """
    response = error_client.post("/analyze", json={"text": "I love Donuts :)"})

    assert response.status_code == expected.status_code
    assert response.json() == {"detail": expected.detail}
