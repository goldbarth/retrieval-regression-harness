from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

from harness.api.dependencies import get_llm_client, get_llm_config, get_tools
from harness.core.config import LlmConfig
from harness.core.gold import ExpectedSource
from harness.core.interfaces import (
    LLM_TOOL_STOP_REASONS,
    LlmStructuredToolCompletion,
    LlmToolStopReason,
    SectionHit,
    StructuredToolCompleter,
    ToolRound,
    ToolSpec,
)
from harness.core.rag import RagAnswer
from harness.core.tools import build_section_search_tool
from harness.main import app


@dataclass
class RecordedToolCall:
    system_prompt: str
    user_message: str
    config: LlmConfig
    tools: tuple[str, ...]
    schema: type[BaseModel]


class RecordingStructuredToolCompleter:
    """Only implements StructuredToolCompleter. That it suffices is the point
    of the role: the endpoint asks for one method, so a double that provides
    one method is a complete stand-in."""

    def __init__(self, completion: LlmStructuredToolCompletion[RagAnswer]) -> None:
        self.completion = completion
        self.calls: list[RecordedToolCall] = []

    def complete_with_tools_structured[T: BaseModel](
        self,
        system_prompt: str,
        user_message: str,
        config: LlmConfig,
        tools: Sequence[ToolSpec[Any]],
        schema: type[T],
        max_rounds: int = 5,
    ) -> LlmStructuredToolCompletion[T]:
        self.calls.append(
            RecordedToolCall(
                system_prompt=system_prompt,
                user_message=user_message,
                config=config,
                tools=tuple(tool.name for tool in tools),
                schema=schema,
            )
        )
        # The double is built for RagAnswer and the endpoint only ever asks for
        # RagAnswer, so the stored completion is the right one. The signature
        # still has to stay generic to satisfy the protocol.
        return self.completion  # type: ignore[return-value]


def test_the_double_satisfies_the_structured_tool_role() -> None:
    # Static check, not a runtime one: if StructuredToolCompleter grows a
    # method, this assignment stops type-checking and the double gets fixed on
    # purpose rather than by a failing request test.
    completer: StructuredToolCompleter = RecordingStructuredToolCompleter(
        LlmStructuredToolCompletion("", None, (), "completed")
    )

    assert completer is not None


class FakeSearch:
    def find(self, query: str, top_k: int) -> list[SectionHit]:
        return [SectionHit("doc", "section", "body")]


RagClientFactory = Callable[[RecordingStructuredToolCompleter, LlmConfig], TestClient]


@pytest.fixture
def rag_client() -> Iterator[RagClientFactory]:
    """Build a TestClient with every /rag/analyze dependency overridden.

    Every test in this file must go through here, including the ones that
    expect a 422. FastAPI resolves the dependencies for a request even when the
    body fails validation, so an un-overridden test still runs the real
    get_llm_client, reads Settings and needs an API key. That passes locally
    where a .env sits in the working directory and fails in CI with a
    ValidationError that has nothing to do with the status code under test.
    """

    def build(
        llm: RecordingStructuredToolCompleter, llm_config: LlmConfig
    ) -> TestClient:
        tools: list[ToolSpec[Any]] = [build_section_search_tool(FakeSearch())]
        app.dependency_overrides[get_llm_client] = lambda: llm
        app.dependency_overrides[get_llm_config] = lambda: llm_config
        app.dependency_overrides[get_tools] = lambda: tools
        return TestClient(app)

    try:
        yield build
    finally:
        app.dependency_overrides.clear()


def _answered(text: str, *sections: tuple[str, str]) -> RagAnswer:
    return RagAnswer(
        answer_status="answered",
        answer=text,
        citations=[ExpectedSource(doc_id=d, section=s) for d, s in sections],
    )


def _completer(
    text: str, stop_reason: LlmToolStopReason = "completed"
) -> RecordingStructuredToolCompleter:
    """Build a double that honors the port's invariant.

    parsed is set exactly when stop_reason is "completed". Handing back a
    parsed answer together with "max_rounds" would let a test pass on a
    combination the adapter never produces.
    """
    parsed = _answered(text, ("doc", "section")) if stop_reason == "completed" else None
    return RecordingStructuredToolCompleter(
        LlmStructuredToolCompletion(text, parsed, (), stop_reason)
    )


def test_rag_analyze_returns_the_model_answer(rag_client: RagClientFactory) -> None:
    llm = _completer("tool backed answer")
    client = rag_client(llm, LlmConfig(model_name="rag-test-model"))

    response = client.post("/rag/analyze", json={"text": "  who? "})

    assert response.status_code == 200
    assert response.json() == {
        "result": "tool backed answer",
        "num_chars": len("tool backed answer"),
        "stop_reason": "completed",
        "citations": [{"doc_id": "doc", "section": "section"}],
        "answer_status": "answered",
    }


def test_rag_analyze_returns_the_citations_as_a_field(
    rag_client: RagClientFactory,
) -> None:
    """The point of commit 4. The smoke run on 2026-08-19 produced three
    citation formats across four answers, none of them parsable; the field
    replaces that convention, so it has to arrive structured and unchanged."""
    llm = RecordingStructuredToolCompleter(
        LlmStructuredToolCompletion(
            "grounded",
            _answered(
                "grounded",
                ("tutorial/response-model", "response-model-priority"),
                (
                    "advanced/custom-response",
                    "document-in-openapi-and-override-response",
                ),
            ),
            (),
            "completed",
        )
    )
    client = rag_client(llm, LlmConfig(model_name="rag-test-model"))

    response = client.post("/rag/analyze", json={"text": "who?"})

    assert response.status_code == 200
    assert response.json()["citations"] == [
        {"doc_id": "tutorial/response-model", "section": "response-model-priority"},
        {
            "doc_id": "advanced/custom-response",
            "section": "document-in-openapi-and-override-response",
        },
    ]


def test_rag_analyze_reports_an_ungrounded_answer_as_such(
    rag_client: RagClientFactory,
) -> None:
    """An empty citation list alone is ambiguous, so answer_status carries the
    difference: retrieval found nothing usable, and the model said so instead
    of inventing a source."""
    llm = RecordingStructuredToolCompleter(
        LlmStructuredToolCompletion(
            "nothing relevant",
            RagAnswer(
                answer_status="no_relevant_sections",
                answer="nothing relevant",
                citations=[],
            ),
            (),
            "completed",
        )
    )
    client = rag_client(llm, LlmConfig(model_name="rag-test-model"))

    response = client.post("/rag/analyze", json={"text": "who?"})

    assert response.status_code == 200
    assert response.json()["answer_status"] == "no_relevant_sections"
    assert response.json()["citations"] == []
    assert response.json()["stop_reason"] == "completed"


def test_rag_analyze_passes_request_config_tools_and_schema_to_the_llm(
    rag_client: RagClientFactory,
) -> None:
    llm_config = LlmConfig(model_name="rag-test-model", temperature=0.3)
    llm = RecordingStructuredToolCompleter(
        LlmStructuredToolCompletion(
            "answer",
            _answered("answer", ("doc", "section")),
            (ToolRound(("search_sections",), None),),
            "completed",
        )
    )
    client = rag_client(llm, llm_config)

    response = client.post("/rag/analyze", json={"text": "  who? "})

    assert response.status_code == 200
    assert len(llm.calls) == 1
    call = llm.calls[0]
    assert call.system_prompt
    assert call.user_message == "who?"
    assert call.config == llm_config
    assert call.tools == ("search_sections",)
    assert call.schema is RagAnswer


def test_rag_analyze_rejects_whitespace_only_text_returns_422(
    rag_client: RagClientFactory,
) -> None:
    llm = _completer("never reached")
    client = rag_client(llm, LlmConfig(model_name="rag-test-model"))

    response = client.post("/rag/analyze", json={"text": "   "})

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "text"]
    assert llm.calls == []


def test_rag_analyze_returns_the_partial_text_when_the_run_hit_max_rounds(
    rag_client: RagClientFactory,
) -> None:
    # max_rounds is not an error: the caller gets what the model produced, and
    # stop_reason is the only thing that says the answer is not a full one.
    # There is no parsed answer to take citations from, so the list is empty
    # and answer_status stays null, which is what separates this case from a
    # deliberate "no_relevant_sections".
    llm = _completer("partial", stop_reason="max_rounds")
    client = rag_client(llm, LlmConfig(model_name="rag-test-model"))

    response = client.post("/rag/analyze", json={"text": "who?"})

    assert response.status_code == 200
    assert response.json()["result"] == "partial"
    assert response.json()["stop_reason"] == "max_rounds"
    assert response.json()["citations"] == []
    assert response.json()["answer_status"] is None


@pytest.mark.parametrize("stop_reason", LLM_TOOL_STOP_REASONS)
def test_rag_analyze_reports_every_stop_reason(
    stop_reason: LlmToolStopReason, rag_client: RagClientFactory
) -> None:
    # Parametrized over the alias itself: a new stop reason cannot be added to
    # the port without this test demanding that the router passes it on.
    llm = _completer("answer", stop_reason=stop_reason)
    client = rag_client(llm, LlmConfig(model_name="rag-test-model"))

    response = client.post("/rag/analyze", json={"text": "who?"})

    assert response.status_code == 200
    assert response.json()["stop_reason"] == stop_reason
