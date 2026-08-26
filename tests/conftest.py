from collections.abc import Generator, Iterator

import pytest
from fastapi.testclient import TestClient

from harness.api.dependencies import get_llm_client, get_llm_config
from harness.core.config import LlmConfig
from harness.core.interfaces import (
    LlmCompletion,
    LlmStreamEnd,
    LlmStreamEvent,
    LlmTextDelta,
    TokenUsage,
)
from harness.main import app


class FakeLlmClient:
    def complete(
        self, system_prompt: str, user_message: str, config: LlmConfig
    ) -> LlmCompletion:
        return LlmCompletion(f"Analyzed: {user_message}")

    def stream(
        self, system_prompt: str, user_message: str, config: LlmConfig
    ) -> Generator[LlmStreamEvent]:
        """Answer the same sentence as complete(), one word per delta.

        Word-wise rather than one block: the point of the endpoint test is that
        frames arrive separately, and a single-delta fake could not tell a
        streamed response from a buffered one.
        """
        yield LlmTextDelta("Analyzed:")
        yield LlmTextDelta(f" {user_message}")

        yield LlmStreamEnd(
            TokenUsage(
                input_tokens=9,
                output_tokens=10,
                total_tokens=19,
                cached_tokens=0,
                cache_write_tokens=0,
                reasoning_tokens=0,
            ),
            None,
        )


@pytest.fixture
def client() -> Iterator[TestClient]:
    """Run API tests without touching the real OpenAI client.

    FastAPI resolves dependency_overrides per request, not when TestClient is
    constructed. Therefore this fixture must yield so the overrides stay active
    for the whole test and are cleaned up afterwards.

    The overrides live on the global app instance. Clearing them in finally
    prevents fake dependencies from leaking into later tests.
    """
    app.dependency_overrides[get_llm_client] = lambda: FakeLlmClient()
    app.dependency_overrides[get_llm_config] = lambda: LlmConfig(
        model_name="test-model",
        temperature=0.2,
    )

    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()
