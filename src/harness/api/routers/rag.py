from typing import Annotated, Any

from fastapi import APIRouter, Depends

from harness.api.dependencies import get_llm_client, get_llm_config, get_tools
from harness.core.config import LlmConfig
from harness.core.interfaces import StructuredToolCompleter, ToolSpec
from harness.core.prompts import RAG_SYSTEM_PROMPT
from harness.core.rag import RagAnswer
from harness.schemas.requests import TextRequest
from harness.schemas.responses import RagResponse

router = APIRouter(tags=["rag"])


@router.post("/rag/analyze")
def rag_analyze(
    request: TextRequest,
    # StructuredToolCompleter, not ToolCompleter: the endpoint now needs the
    # answer parsed, and naming the narrower role keeps a test double honest
    # about which method it has to provide.
    llm: Annotated[StructuredToolCompleter, Depends(get_llm_client)],
    llm_config: Annotated[LlmConfig, Depends(get_llm_config)],
    tools: Annotated[list[ToolSpec[Any]], Depends(get_tools)],
) -> RagResponse:
    result = llm.complete_with_tools_structured(
        system_prompt=RAG_SYSTEM_PROMPT,
        user_message=request.text,
        config=llm_config,
        tools=tools,
        schema=RagAnswer,
    )

    # stop_reason travels to the caller: "max_rounds" and "incomplete_details"
    # both carry a partial answer, and a 200 alone cannot say which one it is.
    answer = result.parsed
    if answer is None:
        # A run that never reached a final answer. The partial text still goes
        # out, exactly as before this endpoint became schema-bound, and the
        # empty citation list is the absence of an answer rather than an
        # ungrounded one.
        return RagResponse(
            result=result.text,
            num_chars=len(result.text),
            stop_reason=result.stop_reason,
        )

    # Citations pass through unchanged. Whether each one was actually returned
    # by search_sections is a different question, and answering it needs a
    # handler that records what it handed back. That belongs to the phase 3
    # runner, which is where the comparison against a gold question happens.
    return RagResponse(
        result=answer.answer,
        num_chars=len(answer.answer),
        stop_reason=result.stop_reason,
        citations=answer.citations,
        answer_status=answer.answer_status,
    )
