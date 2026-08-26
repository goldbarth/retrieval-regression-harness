from pydantic import BaseModel

SSE_MEDIA_TYPE = "text/event-stream"

SSE_DELTA = "delta"
SSE_END = "end"
SSE_ERROR = "error"


def format_sse_frame(event_name: str, payload: BaseModel) -> str:
    """Serialize one server sent event frame.

    Wire format: `event: <name>\\ndata: <json>\\n\\n`. The blank line is what
    ends a frame; without it a browser's EventSource buffers everything until
    the connection closes, which is exactly the behaviour streaming exists to
    avoid.

    JSON in the data line, not raw text: a token can be a newline, and a raw
    newline inside `data:` would cut the frame in two and hand the client half
    an event.
    """
    json_str = payload.model_dump_json()
    return f"event: {event_name}\ndata: {json_str}\n\n"
