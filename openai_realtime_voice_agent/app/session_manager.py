"""Session management with context caching for OpenAI Realtime API."""
import logging
import time
from typing import Optional, Dict
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.frames.frames import Frame, StartFrame, LLMMessagesUpdateFrame

logger = logging.getLogger(__name__)


class ContextCacheEntry:
    """Entry in the context cache for a specific client."""

    def __init__(self, context: LLMContext, timestamp: float):
        self.context = context
        self.timestamp = timestamp
        # When the conversation this context belongs to last ENDED (boundary:
        # "stop", silence timeout, WS drop). Used to timestamp the carryover
        # context note. Preserved across re-caches of a continuously-growing
        # context; reset to "now" whenever the context shrank to a carryover
        # set (i.e. a rollover/restore happened).
        self.ended_at: Optional[float] = None


class SessionManager:
    """Manages OpenAI Realtime sessions with context caching per client device.

    For each new WebSocket connection, a new session is created, but the context
    from previous sessions for the same client is preserved if the last connection
    closed within the reuse timeout period.
    """

    def __init__(self, reuse_timeout: float = 300.0, max_restored_messages: int = 0):
        """Initialize session manager.

        Args:
            reuse_timeout: UNUSED since 2026-11-14. The cache no longer expires:
                every new conversation starts with the last-12 carried items
                plus a timestamped context note (see conversation_rollover.py),
                so there is no "resume vs fresh" ambiguity to time out. The
                parameter is kept for wiring compatibility.
            max_restored_messages: Cap on how many of the most-recent cached
                messages are restored into a new session (0 = unlimited). This
                is the conversation-boundary carryover: within a conversation
                the history is unbounded (OpenAI's own `truncation: auto`
                degrades gracefully at the model's input limit), while across
                conversations the carried context — and thus the per-turn token
                cost — is bounded by this cap. A leading system message, if
                present, is always kept.
        """
        self.reuse_timeout = reuse_timeout
        self.max_restored_messages = max(0, int(max_restored_messages))
        # Dictionary mapping client_id to ContextCacheEntry
        self.context_caches: Dict[str, ContextCacheEntry] = {}
        # Dictionary mapping client_id to current service
        self.current_services: Dict[str, OpenAIRealtimeLLMService] = {}
        # Dictionary mapping client_id to context aggregator pair
        self.context_aggregators: Dict[str, LLMContextAggregatorPair] = {}

    def get_cached_context(self, client_id: str) -> Optional[LLMContext]:
        """Get cached context for a specific client (never expires).

        The carryover design (2026-11-14): a snapshot is always valid — every
        conversation, however long after the last one, continues with the last
        N messages plus a timestamped context note. Staleness is handled by
        the note telling the model WHEN the previous conversation ended, not
        by forgetting.
        """
        cache_entry = self.context_caches.get(client_id)
        if cache_entry is not None:
            age = time.time() - (cache_entry.ended_at or cache_entry.timestamp)
            logger.info(
                f"♻️ Using carried context for client {client_id} — previous conversation "
                f"ended {age:.0f}s ago"
            )
        return cache_entry
    
    def cache_context_from_service(self, client_id: str, service: OpenAIRealtimeLLMService):
        """Extract and cache context from a service before it's closed.
        
        Args:
            client_id: Unique identifier for the client device
            service: The OpenAI Realtime service to extract context from
        """
        # First try to get context from the context aggregator (more reliable)
        context = None
        if client_id in self.context_aggregators:
            aggregator_pair = self.context_aggregators[client_id]
            # The context is shared between user and assistant aggregators
            user_aggregator = aggregator_pair.user()
            if hasattr(user_aggregator, '_context') and user_aggregator._context:
                context = user_aggregator._context
                logger.debug(f"🔍 Found context in aggregator for client {client_id}")
        
        # Fallback: try to get context from service
        if not context and service and hasattr(service, '_context') and service._context:
            context = service._context
            logger.debug(f"🔍 Found context in service for client {client_id}")
        
        # Cache the context if we found one
        if context:
            messages = context.get_messages() if hasattr(context, 'get_messages') else []
            message_count = len(messages) if messages else 0
            existing = self.context_caches.get(client_id)
            now = time.time()
            if existing is not None:
                prev_len = len(existing.context.get_messages() or [])
                if message_count > prev_len:
                    # The conversation continued and grew since the last cache —
                    # this cache point is the (new) conversation end.
                    ended_at = now
                else:
                    # Same-or-smaller context (e.g. re-caching an already
                    # trimmed carryover): preserve the original boundary time
                    # so the context note stays truthful.
                    ended_at = existing.ended_at if existing.ended_at is not None else now
            else:
                ended_at = now
            entry = ContextCacheEntry(context=context, timestamp=now)
            entry.ended_at = ended_at
            self.context_caches[client_id] = entry
            logger.info(f"💾 Cached context from previous session for client {client_id} ({message_count} messages)")
        else:
            if not service:
                logger.warning(f"⚠️ No service provided to cache context for client {client_id}")
            elif client_id not in self.context_aggregators:
                logger.warning(f"⚠️ No context aggregator found for client {client_id}")
            elif not hasattr(service, '_context'):
                logger.warning(f"⚠️ Service has no '_context' attribute for client {client_id}")
            elif not service._context:
                logger.warning(f"⚠️ Service context is None for client {client_id}")
            else:
                logger.debug(f"No context to cache from service for client {client_id}")
    
    def create_context_for_new_session(self, client_id: str) -> LLMContext:
        """Create a new context for a new session, reusing cached context if available.
        
        Args:
            client_id: Unique identifier for the client device
            
        Returns:
            LLMContext for the new session (cached or new)
        """
        # Log available cache keys for debugging
        if self.context_caches:
            logger.debug(f"🔍 Available cached contexts: {list(self.context_caches.keys())}")
        else:
            logger.debug("🔍 No cached contexts available")
        
        cached_context = self.get_cached_context(client_id)
        if cached_context:
            # Create a new context instance with the same messages
            # Use the constructor to properly copy messages and tools
            cached_messages = cached_context.get_messages()
            # Carryover: trim to the most-recent N messages (a leading system
            # message, if present, is always kept) and prepend a timestamped
            # context note so the model knows the carried messages are from an
            # EARLIER conversation, not a live one. The note carries an
            # absolute wall-clock time (see conversation_rollover.py) so it
            # stays truthful however much later the next conversation is.
            restored_messages = self._prepare_carryover(
                cached_messages if cached_messages else [],
                cached_context.ended_at or cached_context.timestamp,
            )
            new_context = LLMContext(
                messages=restored_messages,
                tools=cached_context.tools if hasattr(cached_context, 'tools') else None,
                tool_choice=cached_context.tool_choice if hasattr(cached_context, 'tool_choice') else None
            )
            logger.info(f"✅ Created new context for client {client_id} with {len(new_context.get_messages())} messages from cache")
            return new_context
        else:
            logger.info(f"🆕 Creating new empty context for client {client_id}")
            return LLMContext()

    def _prepare_carryover(self, messages: list, ended_at: float) -> list:
        """Trim a message list to the carryover set and prepend the context note.

        Used both when restoring into a new session (WS disconnect path) and
        conceptually mirrored by the in-place rollover (see
        conversation_rollover.py, which does the same trimming server-side via
        conversation.item.delete + an injected note item).
        """
        from app.conversation_rollover import CONTEXT_NOTE_PREFIX, context_note_text

        def is_note(m):
            content = m.get("content") if isinstance(m, dict) else None
            if isinstance(content, str):
                return content.startswith(CONTEXT_NOTE_PREFIX)
            if isinstance(content, list):
                return any(
                    isinstance(c, dict) and str(c.get("text", "")).startswith(CONTEXT_NOTE_PREFIX)
                    for c in content
                )
            return False

        kept = [m for m in messages if not is_note(m)]
        head: list = []
        body = kept
        if kept and isinstance(kept[0], dict) and kept[0].get("role") == "system":
            head, body = [kept[0]], kept[1:]
        if self.max_restored_messages > 0 and len(body) > self.max_restored_messages:
            body = body[-self.max_restored_messages:]
            logger.info(
                f"✂️ Trimmed carried context: {len(messages)} → {len(head) + len(body) + 1} messages "
                f"(cap {self.max_restored_messages})"
            )
        if not body and not head:
            return []
        note = {
            "role": "user",
            "content": context_note_text(ended_at),
        }
        return head + [note] + body
    
    def get_current_service(self, client_id: str) -> Optional[OpenAIRealtimeLLMService]:
        """Get current OpenAI service for a specific client.
        
        Args:
            client_id: Unique identifier for the client device
            
        Returns:
            Current OpenAIRealtimeLLMService if exists, None otherwise
        """
        return self.current_services.get(client_id)
    
    def set_current_service(self, client_id: str, service: OpenAIRealtimeLLMService):
        """Set the current active service for a client.
        
        Args:
            client_id: Unique identifier for the client device
            service: The currently active OpenAI Realtime service
        """
        self.current_services[client_id] = service
    
    def set_context_aggregator(self, client_id: str, aggregator_pair: LLMContextAggregatorPair):
        """Set the context aggregator pair for a client.
        
        Args:
            client_id: Unique identifier for the client device
            aggregator_pair: The LLMContextAggregatorPair instance for this client
        """
        self.context_aggregators[client_id] = aggregator_pair
    
    def remove_context_aggregator(self, client_id: str):
        """Remove the context aggregator pair for a client.
        
        Args:
            client_id: Unique identifier for the client device
        """
        if client_id in self.context_aggregators:
            del self.context_aggregators[client_id]
    
    def cleanup_before_new_session(self, client_id: str):
        """Cleanup before creating a new session for a client.
        
        This should be called before creating a new session to cache
        the context from the current service.
        
        Args:
            client_id: Unique identifier for the client device
        """
        # Cache context from service/aggregator
        if client_id in self.current_services:
            self.cache_context_from_service(client_id, self.current_services[client_id])
            del self.current_services[client_id]
        
        # Remove context aggregator (will be recreated for new session)
        self.remove_context_aggregator(client_id)
    
    def create_context_aggregator(self, client_id: str) -> LLMContextAggregatorPair:
        """Create a context aggregator pair for a new session.
        
        Args:
            client_id: Unique identifier for the client device
            
        Returns:
            LLMContextAggregatorPair with cached or new context
        """
        context = self.create_context_for_new_session(client_id)
        aggregator_pair = LLMContextAggregatorPair(context)
        self.set_context_aggregator(client_id, aggregator_pair)
        return aggregator_pair
    
    def create_context_initializer(self, client_id: str, context_aggregator: LLMContextAggregatorPair) -> Optional['ContextInitializer']:
        """Create a context initializer if cached messages exist.
        
        Args:
            client_id: Unique identifier for the client device
            context_aggregator: The context aggregator pair
            
        Returns:
            ContextInitializer if cached messages exist, None otherwise
        """
        context = context_aggregator.user().context
        if len(context.get_messages()) > 0:
            return ContextInitializer(
                context_aggregator=context_aggregator,
                cached_context=context,
                client_id=client_id
            )
        return None
    
    def handle_client_disconnect(self, client_id: str, service: Optional[OpenAIRealtimeLLMService] = None):
        """Handle client disconnection by caching context.
        
        Args:
            client_id: Unique identifier for the client device
            service: Optional service instance to cache context from
        """
        logger.info(f"🔌 Client {client_id} disconnected - caching context")
        
        # Get service to cache from
        service_to_cache = None
        if client_id in self.current_services:
            service_to_cache = self.current_services[client_id]
        elif service:
            service_to_cache = service
        
        if service_to_cache:
            try:
                self.cache_context_from_service(client_id, service_to_cache)
                if client_id in self.current_services:
                    del self.current_services[client_id]
                logger.info(f"💾 Cached context for disconnected client {client_id}")
            except Exception as e:
                logger.warning(f"⚠️ Error caching context for disconnected client {client_id}: {e}")
        else:
            logger.debug(f"No service found to cache context for client {client_id}")


class ContextInitializer(FrameProcessor):
    """Processor that sends cached context after StartFrame has passed through the pipeline."""
    
    def __init__(self, context_aggregator, cached_context, client_id, **kwargs):
        super().__init__(**kwargs)
        self.context_aggregator = context_aggregator
        self.cached_context = cached_context
        self.client_id = client_id
        self.context_sent = False
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames and send cached context after StartFrame."""
        if isinstance(frame, StartFrame):
            await super().process_frame(frame, direction)
            await self.push_frame(frame, direction)
            
            # Send cached context after StartFrame has passed through the pipeline
            # Use LLMMessagesUpdateFrame with run_llm=False to set context without triggering a response
            if self.cached_context and not self.context_sent:
                context = self.cached_context
                messages = context.get_messages()
                if len(messages) > 0:
                    # Update messages without triggering LLM response
                    # The bot will wait for the user to speak first
                    update_frame = LLMMessagesUpdateFrame(messages=messages, run_llm=False)
                    await self.context_aggregator.user().push_frame(update_frame)
                    logger.info(f"📤 Sent cached context ({len(messages)} messages) to OpenAI for client {self.client_id} (waiting for user)")
                    self.context_sent = True
            return
        
        await self.push_frame(frame, direction)

