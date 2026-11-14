"""Tool that ends the voice session when the user clearly wants to stop.

Unlike the opt-in `disconnect_client` tool (which closes the WebSocket —
observed killing replies mid-sentence), this tool only sends a lightweight
`{"type":"stop_session"}` control message. The va_client firmware treats it
exactly like a local "stop" word: silence playback, gate the mic, suppress the
follow-up listening window, back to wake word. The socket stays up.
"""
import logging
from typing import Any, Callable, Awaitable, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from pipecat.services.llm_service import FunctionCallParams

logger = logging.getLogger(__name__)


def get_end_conversation_tool_definition() -> Dict[str, Any]:
    """Get the tool definition for the OpenAI Realtime API."""
    return {
        "type": "function",
        "name": "end_conversation",
        "description": (
            "End the voice session when the user clearly wants to stop talking to "
            "you or leave: they say just 'stop', 'never mind', 'goodbye', 'that's "
            "all' or 'be quiet' with no other request. Do NOT use it when 'stop' is "
            "part of a device command ('stop the music', 'stop the lights', 'stop "
            "the alarm') — that is a Home Assistant action, not an end of the "
            "conversation. When in doubt, do not call it and ask instead."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        }
    }


def create_end_conversation_handler(
    broadcast: Callable[[dict], Awaitable[None]],
    on_ended: Optional[Callable[[str], None]] = None,
) -> Callable[["FunctionCallParams"], Awaitable[None]]:
    """
    Create the tool handler for Pipecat's OpenAI Realtime Service.

    Args:
        broadcast: async callable that delivers a JSON dict to the connected
            device(s) (WebSocketHandler.broadcast_json).
        on_ended: optional sync callable(reason) that schedules the
            conversation rollover (context trim + carryover note) — called
            after the stop_session message is sent (WebSocketHandler.
            trigger_rollover).
    """
    async def end_conversation_handler(params: "FunctionCallParams") -> None:
        """Send stop_session to the device(s); the firmware ends the session."""
        logger.info(f"🛑 end_conversation tool called ({params.arguments}) → stop_session to device")
        try:
            await broadcast({"type": "stop_session"})
        except Exception as e:
            logger.warning(f"⚠️ could not send stop_session to device: {e!r}")
            await params.result_callback("Error: could not reach the device to end the session.")
            return
        # The device is ending the session (mic off, back to wake word) — this
        # is a conversation boundary, so trim the server-side conversation and
        # inject the carryover context note (agreed design: every conversation
        # ends with the last 12 items carried into the next one). Scheduled as
        # a task: the rollover waits out the in-flight response itself, and the
        # tool result should return promptly.
        if on_ended is not None:
            try:
                on_ended("end_conversation tool (spoken stop/goodbye)")
            except Exception as e:
                logger.warning(f"⚠️ could not schedule conversation rollover: {e!r}")
        await params.result_callback("Session ended. The device is back in wake-word mode.")

    return end_conversation_handler
