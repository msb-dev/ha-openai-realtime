"""Conversation rollover — bounded memory across conversation boundaries.

Design (agreed 2026-11-14): the full history is kept DURING a conversation
(wake → "stop" / silence timeout). When a conversation ends — for ANY reason —
the next conversation starts with the system prompt (session-level
`instructions`, never touched) plus the last N items of context, and nothing
more.

Two distinct mechanisms are needed because the OpenAI Realtime conversation
lives server-side and there is no "edit history" operation:

1. **In-place rollover** (same OpenAI session continues — "stop" via the
   end_conversation tool, silence timeout via the device's {"type":"flush"}):
   the server-side conversation is trimmed by deleting the oldest items with
   `conversation.item.delete`, and a small "context note" user item is
   injected so the model knows what happened:

       "(Context note: older conversation history has been trimmed. The
       messages below are carried over from earlier exchanges with this
       user; the most recent previous conversation ended at {time}. …)"

   The note carries an ABSOLUTE timestamp (not "just now"), so it stays
   truthful no matter how much later the next conversation happens; the
   model can pair it with the GetDateTime tool to reason about elapsed time.
   At each rollover any previous note/replay items are deleted and a fresh
   note injected, so at most one is ever present.

   This is instant (no reconnect), which matters: a rollover triggered right
   after "stop" must not eat a fast follow-up wake.

2. **Replay-on-new-session** (the OpenAI session genuinely changes — the
   55-min proactive refresh, an error-triggered reconnect, a new device
   WebSocket connection): the fresh session starts with an empty server-side
   conversation, so the carried context is replayed as ONE packed user item
   (the same trick pipecat's own history-initialization uses, because
   `conversation.item.create` of many items makes the model forget it can
   speak audio):

       "(Context note: … ended at {time}. …)
        <JSON dump of the carried messages>"

   The source of truth for the replay is the client-side context mirror,
   which — with input transcription enabled — tracks user turns
   (transcripts), assistant turns (output transcripts) and tool calls, and
   therefore mirrors the server conversation.

Neither path touches the session-level instructions: those are re-sent by
pipecat's `_update_settings()` on every (re)connect from the application
config, so durable facts (home location etc.) live there and can never be
trimmed away.

The client-side context is deliberately NOT trimmed in path 1: the server
conversation is what the model (and the per-turn token bill) sees, and the
cache/restore path already trims client-side history to the carryover limit
when a new connection restores it.
"""
import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

CONTEXT_NOTE_PREFIX = "(Context note:"


def _humanize_seconds(seconds: float) -> str:
    if seconds < 90:
        return "moments"
    minutes = seconds / 60
    if minutes < 90:
        return f"{int(round(minutes))} minutes"
    hours = seconds / 3600
    if hours < 36:
        return f"{int(round(hours))} hours"
    return f"{int(round(seconds / 86400))} days"


def _format_clock(ts: float) -> str:
    # Local time; the container runs with TZ set (Europe/London), matching the
    # GetDateTime tool the model can call to reason about elapsed time.
    return time.strftime("%H:%M on %A %d %B", time.localtime(ts))


def context_note_text(previous_end_ts: float, now_ts: Optional[float] = None) -> str:
    now_ts = now_ts if now_ts is not None else time.time()
    return (
        f"{CONTEXT_NOTE_PREFIX} older conversation history has been trimmed. "
        f"The messages below are carried over from earlier exchanges with this "
        f"user; the most recent previous conversation ended at "
        f"{_format_clock(previous_end_ts)} — {_humanize_seconds(max(0.0, now_ts - previous_end_ts))} "
        f"before this note was written. Treat them as background context, not a "
        f"live exchange; respond naturally to whatever the user says next."
    )


@dataclass
class TrackedItem:
    """One server-side conversation item, in conversation order."""

    item_id: str
    type: str
    role: Optional[str]
    call_id: Optional[str] = None
    is_note: bool = False


class ConversationItemTracker:
    """Ordered list of server-side conversation items, fed by
    `conversation.item.added` events. The OpenAI Realtime conversation is
    authoritative and server-side; this mirror is what makes item deletion
    possible (the delete API is id-based, and pipecat doesn't track ids)."""

    def __init__(self):
        self._items: List[TrackedItem] = []
        self._seen: set = set()

    def add(self, item_id: Optional[str], item: Any) -> None:
        if not item_id or item_id in self._seen:
            return
        self._seen.add(item_id)
        itype = getattr(item, "type", None) or "message"
        role = getattr(item, "role", None)
        call_id = getattr(item, "call_id", None)
        self._items.append(TrackedItem(
            item_id=item_id,
            type=itype,
            role=role,
            call_id=call_id,
            is_note=str(item_id).startswith("carryover_"),
        ))

    @property
    def items(self) -> List[TrackedItem]:
        return list(self._items)

    def drop_ids(self, ids: List[str]) -> None:
        dropped = set(ids)
        self._items = [t for t in self._items if t.item_id not in dropped]

    def reset(self) -> None:
        self._items = []
        self._seen = set()


class ConversationRollover:
    """Trims the server-side conversation at conversation boundaries (path 1).

    One instance per device pipeline (per OpenAI session) — the tracker is
    session-scoped by construction.
    """

    def __init__(self, service, tracker: ConversationItemTracker, max_carryover: int = 12):
        self._service = service
        self._tracker = tracker
        self._max_carryover = max(1, int(max_carryover))
        self._rolling = False

    async def rollover(self, reason: str) -> bool:
        """Trim the server-side conversation + inject the context note.

        Returns False if skipped (response in flight, nothing to do, or an
        error). Safe to call speculatively at any conversation boundary.
        """
        if self._rolling:
            return False
        self._rolling = True
        try:
            return await self._rollover_inner(reason)
        except Exception as e:
            logger.warning(f"⚠️ rollover failed ({reason[:60]!r}): {e!r}")
            return False
        finally:
            self._rolling = False

    async def _rollover_inner(self, reason: str) -> bool:
        # Wait out an in-flight response: deleting items while the model is
        # generating (or while a stop-cancel is settling) races the server.
        for _ in range(10):
            if getattr(self._service, "_current_assistant_response", None) is None:
                break
            await asyncio.sleep(0.5)
        else:
            logger.info(f"⏭️ rollover skipped — response still active after 5s ({reason[:60]})")
            return False

        items = self._tracker.items
        if not items:
            return False  # empty session — nothing to trim or annotate

        # Keep the newest N items, then make sure no kept function_call_output
        # is orphaned from its function_call (deleting the call while keeping
        # its result can make later turns error out).
        keep: List[TrackedItem] = items[-self._max_carryover:]
        keep_ids = {t.item_id for t in keep}
        by_id = {t.item_id: t for t in items}
        for t in list(keep):
            if t.type == "function_call_output" and t.call_id:
                call = by_id.get(t.call_id)
                if call and call.item_id not in keep_ids:
                    keep.insert(0, call)
                    keep_ids.add(call.item_id)

        # Any previous note/replay item inside the keep-set is deleted too —
        # a fresh note is injected below, so at most one note ever remains.
        drop = [t for t in items if t.item_id not in keep_ids or t.is_note]
        dropped = 0
        for t in drop:
            try:
                from pipecat.services.openai.realtime import events as rt_events
                await self._service.send_client_event(
                    rt_events.ConversationItemDeleteEvent(item_id=t.item_id)
                )
                dropped += 1
            except Exception as e:
                logger.debug(f"rollover: delete {t.item_id} failed: {e!r}")
        self._tracker.drop_ids([t.item_id for t in drop])

        note_id = f"carryover_note_{int(time.time() * 1000)}"
        try:
            from pipecat.services.openai.realtime import events as rt_events
            await self._service.send_client_event(rt_events.ConversationItemCreateEvent(
                item={
                    "id": note_id,
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": context_note_text(time.time())}],
                }
            ))
        except Exception as e:
            logger.warning(f"⚠️ rollover: could not inject context note: {e!r}")

        logger.info(
            f"🔄 conversation rollover ({reason[:50]}): {dropped} old item(s) removed, "
            f"{len(keep)} carried over, context note injected"
        )
        return True


def build_packed_carryover_message(
    messages: List[Dict[str, Any]], previous_end_ts: float
) -> Optional[Dict[str, Any]]:
    """Build the single packed user item used to replay carried context into a
    FRESH OpenAI session (path 2).

    Mirrors pipecat's own history-initialization strategy
    (`OpenAIRealtimeLLMContext.get_messages_for_initializing_history`): many
    individual `conversation.item.create`s make the model forget it can speak
    audio, so the whole carryover is packed into one user message as JSON.
    """
    # Strip prior notes defensively (restore paths may re-prepare the same list).
    kept = []
    for m in messages:
        content = m.get("content") if isinstance(m, dict) else None
        if isinstance(content, str) and content.startswith(CONTEXT_NOTE_PREFIX):
            continue
        if isinstance(content, list) and any(
            isinstance(c, dict) and str(c.get("text", "")).startswith(CONTEXT_NOTE_PREFIX)
            for c in content
        ):
            continue
        kept.append(m)
    if not kept:
        return None
    text = (
        f"{context_note_text(previous_end_ts)}\n\n"
        "Carried-over conversation history (JSON):\n"
        f"{json.dumps(kept, indent=2, default=str)}"
    )
    return {
        "id": f"carryover_replay_{int(time.time() * 1000)}",
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": text}],
    }


async def replay_carryover_after_reconnect(
    service, messages: List[Dict[str, Any]], previous_end_ts: float, tracker: Optional[ConversationItemTracker]
) -> bool:
    """Replay carried context into a freshly (re)connected OpenAI session.

    pipecat's own one-time history setup cannot be relied on here: under
    semantic VAD with automatic responses, `_create_response()` (the only
    thing that replays context) is not called on user turns, and the in-place
    `reset_conversation()` leaves `_context` set so the "first context" branch
    never re-runs. So we send the packed carryover explicitly, once the new
    session is ready.
    """
    if not messages:
        return False
    # Wait for session.ready (session.updated) — max ~10 s.
    for _ in range(40):
        if getattr(service, "_api_session_ready", False):
            break
        await asyncio.sleep(0.25)
    else:
        logger.warning("⚠️ carryover replay skipped — session not ready after 10s")
        return False

    packed = build_packed_carryover_message(messages, previous_end_ts)
    if packed is None:
        return False
    try:
        from pipecat.services.openai.realtime import events as rt_events
        await service.send_client_event(rt_events.ConversationItemCreateEvent(item=packed))
    except Exception as e:
        logger.warning(f"⚠️ carryover replay failed: {e!r}")
        return False
    if tracker is not None:
        tracker.add(packed["id"], packed)
    count = len(messages)
    logger.info(f"📤 Carried context replayed into fresh session ({count} messages)")
    return True
