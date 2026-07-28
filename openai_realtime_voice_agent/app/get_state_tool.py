"""get_entity_state function-tool.

A DIRECT Home Assistant state read that bypasses the MCP `GetLiveContext` tool.
GetLiveContext (from the official mcp_server integration) does not surface sensor
*values* to the Realtime model — control tools and person lookups work, but
"how full is the battery?" / "what's the PV output?" return nothing. This tool
reads `GET /api/states` from HA core using the add-on's supervisor token and
selects the best-matching entity via :func:`select_states`, returning the
value(s) as a short, spoken-friendly string.

`select_states` is a *pure* function (no I/O) so the matching logic is unit
tested — see ``tests/test_get_state_tool.py``. Matching rules:

* dead states (``unavailable``/``unknown``/``none``/empty) are skipped;
* a hit on the entity's name/entity_id outweighs a hit on its device_class, so
  a specifically-named sensor wins over a generic one of the same class;
* a query word that names an attribute returns that attribute instead of the
  bare state;
* a bare device_class query with several candidates ("Akku") asks back which one
  instead of guessing;
* at most two entities are spoken.
"""
import logging
from typing import Dict, Any, List, Callable, Awaitable, TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from pipecat.services.llm_service import FunctionCallParams

logger = logging.getLogger(__name__)

# German question words -> HA device_class, so "Akku"/"Ladung" find battery
# sensors, "Temperatur" finds temperature sensors, etc. even when the entity's
# friendly-name is in another language (e.g. an English "... SoC Battery").
_SYNONYMS = {
    "akku": "battery", "batterie": "battery", "ladestand": "battery",
    "ladung": "battery", "soc": "battery",
    "temperatur": "temperature", "temp": "temperature",
    "feuchtigkeit": "humidity", "luftfeuchtigkeit": "humidity", "feuchte": "humidity",
    "leistung": "power", "strom": "power", "watt": "power",
    "energie": "energy", "verbrauch": "energy", "kwh": "energy",
    "co2": "carbon_dioxide", "kohlendioxid": "carbon_dioxide",
}
# Domains worth reading for a "what's the value/state" question.
_READ_DOMAINS = (
    "sensor", "binary_sensor", "number", "person", "device_tracker",
    "weather", "climate", "cover", "lock", "switch", "light", "fan",
    "vacuum", "humidifier", "media_player", "input_boolean", "input_number",
)
# States that carry no answer — never speak these as a value.
_DEAD_STATES = {"", "unknown", "unavailable", "none"}
# Attribute keys that are metadata, never a value the user asked for by name.
_META_ATTR_KEYS = {
    "friendly_name", "device_class", "unit_of_measurement", "icon",
    "entity_picture", "supported_features", "attribution", "state_class",
    "assumed_state", "restored",
}


def _norm(s: str) -> str:
    s = (s or "").lower()
    for a, b in (("ä", "ae"), ("ö", "oe"), ("ü", "ue"), ("ß", "ss")):
        s = s.replace(a, b)
    return s


def _tokens(s: str) -> List[str]:
    return [t for t in _norm(s).split() if len(t) > 1]


def _join_names(names: List[str]) -> str:
    if len(names) <= 1:
        return "".join(names)
    return f"{', '.join(names[:-1])} oder {names[-1]}"


def select_states(states: List[Dict[str, Any]], name: str, limit: int = 2) -> Dict[str, str]:
    """Pick the entity/entities that best answer a value question about *name*.

    Returns a dict ``{"kind": "answer"|"clarify"|"none", "text": "<spoken>"}``.
    Pure function: no network, no side effects.
    """
    q_tokens = _tokens(name)
    dc_implied = {_SYNONYMS[t] for t in q_tokens if t in _SYNONYMS}
    norm_name = _norm(name).strip()

    scored: List[Dict[str, Any]] = []
    for s in states:
        eid = s.get("entity_id", "")
        if eid.split(".", 1)[0] not in _READ_DOMAINS:
            continue
        raw_state = s.get("state")
        if str(raw_state or "").strip().lower() in _DEAD_STATES:
            continue

        attrs = s.get("attributes", {}) or {}
        fn = attrs.get("friendly_name") or eid
        name_words = set(_norm(fn).split()) | set(
            _norm(eid.replace(".", " ").replace("_", " ")).split()
        )
        whole = [t for t in q_tokens if t in name_words]
        sub = [
            t for t in q_tokens
            if t not in name_words and len(t) >= 3 and any(t in w for w in name_words)
        ]
        dc = attrs.get("device_class") or ""
        dc_hit = 1 if dc and dc in dc_implied else 0
        exact = 1 if norm_name and _norm(fn) == norm_name else 0

        score = 100 * exact + 10 * len(whole) + 3 * len(sub) + 2 * dc_hit
        if score == 0:
            continue

        # If a query word names an attribute (e.g. "Restzeit"), speak that
        # attribute's value instead of the bare state.
        value, unit = raw_state, attrs.get("unit_of_measurement")
        for k, v in attrs.items():
            if k in _META_ATTR_KEYS:
                continue
            if any(t in set(_norm(str(k)).split()) for t in q_tokens):
                value, unit = v, None
                break

        scored.append({
            "score": score,
            "nmc": len(whole) + len(sub),  # name-match count
            "spec": len(fn),               # shorter name == more specific
            "fn": fn,
            "value": value,
            "unit": unit,
        })

    if not scored:
        return {"kind": "none", "text": f"Ich habe keinen Wert für „{name}“ gefunden."}

    scored.sort(key=lambda c: (-c["score"], -c["nmc"], c["spec"]))
    best = scored[0]

    if best["nmc"] == 0:
        # Only a device_class matched — ambiguous when several qualify ("Akku").
        if len(scored) >= 2:
            names = [c["fn"] for c in scored[:3]]
            return {
                "kind": "clarify",
                "text": f"Ich habe mehrere Werte für „{name}“ — meinst du {_join_names(names)}?",
            }
        chosen = scored[:1]
    else:
        # Real name match found. Return only the *equally-best* matches, never
        # pad up to `limit` with a weaker entity that merely shares one generic
        # token (e.g. a named "... SoC" must not append a random battery at 0 %).
        best_score = best["score"]
        chosen = [c for c in scored if c["score"] == best_score][:limit]

    parts = []
    for c in chosen:
        unit_s = f" {c['unit']}" if c["unit"] else ""
        parts.append(f"{c['fn']}: {c['value']}{unit_s}")
    return {"kind": "answer", "text": "; ".join(parts)}


def get_state_tool_definition() -> Dict[str, Any]:
    """OpenAI Realtime function-tool definition for reading a state/value."""
    return {
        "type": "function",
        "name": "get_entity_state",
        "description": (
            "Read the CURRENT value or state of a Home Assistant sensor or entity "
            "by name. Use this for ANY question about a value or status — battery "
            "or SoC, PV / solar production, power, energy, temperature, humidity, "
            "CO2, air quality, a person's location/room, washing-machine time "
            "left, car range/fuel, whether a window is open, etc. Prefer this tool "
            "for reading values; do NOT use it to control devices (use the Hass* "
            "tools for that). If it asks which entity you mean, relay that question."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": (
                        "What to look up, in natural language and in the user's "
                        "language, e.g. a battery/SoC, solar production, the "
                        "temperature of a room, a person's location, or an "
                        "appliance's status."
                    ),
                }
            },
            "required": ["name"],
        },
    }


def create_get_state_tool_handler(
    supervisor_token: str, base_url: str = "http://supervisor/core/api"
) -> Callable[["FunctionCallParams"], Awaitable[None]]:
    """Create a get_entity_state handler for pipecat's OpenAIRealtimeLLMService."""

    async def get_state_tool_handler(params: "FunctionCallParams") -> None:
        name = ((params.arguments or {}).get("name") or "").strip()
        logger.info(f"📊 get_entity_state called: {name!r}")

        if not name:
            await params.result_callback("Ich habe nicht verstanden, welcher Wert gemeint ist.")
            return
        if not supervisor_token:
            await params.result_callback("Ich kann die Werte gerade nicht auslesen.")
            return

        try:
            headers = {"Authorization": f"Bearer {supervisor_token}"}
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(f"{base_url}/states", headers=headers)
                resp.raise_for_status()
                states = resp.json()

            result = select_states(states, name)
            logger.info(f"📊 get_entity_state [{result['kind']}]: {result['text'][:200]}")
            await params.result_callback(result["text"])
        except Exception as e:
            logger.error(f"❌ get_entity_state failed: {e}", exc_info=True)
            await params.result_callback("Das Auslesen des Werts hat gerade nicht geklappt.")

    return get_state_tool_handler
