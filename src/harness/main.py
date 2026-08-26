import logging

from fastapi import FastAPI, Request
from starlette.responses import JSONResponse

from harness.api.errors import llm_error_response
from harness.api.routers import analyze_router, health_router, rag_router
from harness.core.interfaces import LlmError

logger = logging.getLogger(__name__)

app = FastAPI()

app.include_router(analyze_router)

app.include_router(rag_router)

app.include_router(health_router)


@app.exception_handler(LlmError)
def handle_llm_error(request: Request, exc: LlmError) -> JSONResponse:
    """Answer every LlmError from the one table in api/errors.py.

    Registered on the base class, not on the four subclasses: Starlette looks a
    handler up along type(exc).__mro__, so this covers all of them, and a bare
    LlmError as well - that one used to fall through as an uncontrolled 500.
    A handler registered on a subclass would still win, should one ever need
    something this cannot express.

    logger.exception stays here rather than moving into the table. The table
    says what to write; the handler is the place that knows there is a live
    traceback to attach to it.
    """
    response = llm_error_response(exc)
    logger.exception(response.log_message)
    return JSONResponse(
        status_code=response.status_code, content={"detail": response.detail}
    )
