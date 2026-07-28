"""get_entity_state function-tool.

A DIRECT Home Assistant state read that bypasses the MCP `GetLiveContext` tool.
GetLiveContext (from the official mcp_server integration) does not surface sensor
*values* to the Realtime model — control tools and person lookups work, but
"how full is the battery?" / "what's the PV output?" return nothing. This tool
reads HA state and selects the best-matching entity via :func:`select_states`,
returning the value(s) as a short, spoken-friendly string.

`select_states` is a *pure* function (no I/O) so the matching logic is unit
tested — see ``tests/test_get_state_tool.py``. It matches a natural-language
query against each entity's friendly-name, entity_id, **Assist aliases**, its
**HA area** and its device_class. Ranking is *coverage-first*: the entity that
accounts for the most distinct query words wins, so "Temperatur Büro" prefers a
sensor that matches both the value type AND the room over one that only matches
the type. Query words are normalised, German filler words are dropped, and
compounds are decomposed ("Solarleistung" -> "solar" + "leistung") so the model
does not have to guess the exact entity name.

The handler enriches the live `/api/states` read with each entity's effective
area and its Assist aliases, fetched together in one Home Assistant WebSocket
session (area/device/entity registry lists). It is cached with a TTL and fails
*soft*: if the lookup breaks, matching falls back to name/device_class only —
never worse than before, and the pure stopword/compound/coverage logic still
applies.
"""
import asyncio
import json
import logging
import time
from typing import Dict, Any, List, Callable, Awaitable, Tuple, TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from pipecat.services.llm_service import FunctionCallParams

logger = logging.getLogger(__name__)

# German question words -> HA device_class, so "Akku"/"Ladung" find battery
# sensors, "Temperatur" finds temperature sensors, etc. even when the entity's
# friendly-name is in another language (e.g. an English "... SoC Battery").
_SYNONYMS = {
    "akku": "battery", "batterie": "battery", "ladestand": "battery",
    "ladung": "battery", "soc": "battery", "akkustand": "battery",
    "batteriestand": "battery",
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
# Generic filler words to drop from a query so they can't hijack the match
# (e.g. "aktuelle" matching "Aktuelle Seite"). Kept deliberately small.
_STOPWORDS = {
    "aktuelle", "aktueller", "aktuelles", "aktuell", "der", "die", "das",
    "den", "dem", "des", "ein", "eine", "einen", "einem", "im", "in", "am",
    "an", "auf", "von", "vom", "zum", "zur", "wie", "viel", "wieviel", "ist",
    "sind", "war", "grad", "gerade", "momentan", "jetzt", "wert", "werte",
    "stand", "status", "mein", "meine", "meiner", "unser", "unsere", "wo",
    "welche", "welcher", "welches", "gibt", "es", "und", "mir", "sag",
    "zeig", "nochmal", "bitte", "hier", "da", "denn",
}
# Compound tails: split "<prefix><tail>" into prefix + tail so a spoken compound
# still matches a two-word entity name / a device_class synonym.
_COMPOUND_TAILS = (
    "leistung", "temperatur", "temperaturen", "luftfeuchtigkeit",
    "feuchtigkeit", "feuchte", "verbrauch", "spannung", "zaehler", "stand",
    "ladung", "level", "zeit",
)

# Coverage/strength weights per matched query term (best signal wins).
_W_NAME = 10   # word hit on friendly-name / entity_id / alias
_W_ATTR = 8    # word hit on an attribute key ("Restzeit")
_W_AREA = 6    # word hit on the entity's HA area
_W_SUB = 3     # substring hit on a name word (German compounds)
_W_DC = 2      # device_class implied by a synonym


def _norm(s: str) -> str:
    s = (s or "").lower()
    for a, b in (("ä", "ae"), ("ö", "oe"), ("ü", "ue"), ("ß", "ss")):
        s = s.replace(a, b)
    return s


def _query_terms(name: str) -> Tuple[set, set]:
    """Return (name_terms, dc_classes) for a query.

    Drops filler words, decomposes German compounds, and derives implied
    device_classes from synonyms. name_terms are matched against entity
    names/aliases/areas; dc_classes against device_class.
    """
    raw = [t for t in _norm(name).split() if len(t) > 1]
    toks: List[str] = []
    for t in raw:
        toks.append(t)
        for tail in _COMPOUND_TAILS:
            if t != tail and t.endswith(tail) and len(t) - len(tail) >= 3:
                toks.append(t[: -len(tail)])
                toks.append(tail)
                break
    name_terms, dc = set(), set()
    for t in toks:
        if len(t) <= 1 or t in _STOPWORDS:
            continue
        name_terms.add(t)
        if t in _SYNONYMS:
            dc.add(_SYNONYMS[t])
    return name_terms, dc


def _join_names(names: List[str]) -> str:
    if len(names) <= 1:
        return "".join(names)
    return f"{', '.join(names[:-1])} oder {names[-1]}"


def select_states(entities: List[Dict[str, Any]], name: str, limit: int = 2) -> Dict[str, str]:
    """Pick the entity/entities that best answer a value question about *name*.

    Each entity is ``{entity_id, state, attributes, area, aliases}``. Returns
    ``{"kind": "answer"|"clarify"|"none", "text": "<spoken>"}``. Pure function.
    """
    name_terms, dc_classes = _query_terms(name)
    norm_name = _norm(name).strip()
    if not name_terms:
        return {"kind": "none", "text": f"Ich habe keinen Wert für „{name}“ gefunden."}

    scored: List[Dict[str, Any]] = []
    for e in entities:
        eid = e.get("entity_id", "")
        if eid.split(".", 1)[0] not in _READ_DOMAINS:
            continue
        raw_state = e.get("state")
        if str(raw_state or "").strip().lower() in _DEAD_STATES:
            continue

        attrs = e.get("attributes", {}) or {}
        fn = attrs.get("friendly_name") or eid
        aliases = [a for a in (e.get("aliases") or []) if a]
        fn_words = set(_norm(fn).split()) | set(
            _norm(eid.replace(".", " ").replace("_", " ")).split()
        )
        alias_sets = [(al, set(_norm(al).split())) for al in aliases]
        name_words = set(fn_words)
        for _al, _aw in alias_sets:
            name_words |= _aw
        area_words = set(_norm(e.get("area") or "").split())

        strength = 0
        strong = False           # matched by something other than device_class
        value, unit = raw_state, attrs.get("unit_of_measurement")
        attr_value = None
        covered_terms = set()

        for t in name_terms:
            w = 0
            if t in name_words:
                w = _W_NAME
            elif t in area_words:
                w = _W_AREA
            elif len(t) >= 3 and any(t in ww for ww in name_words):
                w = _W_SUB
            else:
                for k, v in attrs.items():
                    if k in _META_ATTR_KEYS:
                        continue
                    if t in set(_norm(str(k)).split()):
                        w, attr_value = _W_ATTR, v
                        break
            if w:
                covered_terms.add(t)
                strength += w
                strong = True

        # device_class is an additive tie-break toward the *right* class (so
        # "Temperatur" prefers the temperature entity over its battery sibling),
        # and it covers a synonym term not otherwise matched (bare "Akku").
        edc = attrs.get("device_class") or ""
        if edc and edc in dc_classes:
            strength += _W_DC
            for t in name_terms:
                if t not in covered_terms and _SYNONYMS.get(t) == edc:
                    covered_terms.add(t)
        covered = len(covered_terms)

        if norm_name and (norm_name == _norm(fn) or any(norm_name == _norm(a) for a in aliases)):
            strength += 100
            strong = True
            covered = max(covered, 1)

        if covered == 0:
            continue
        if attr_value is not None:
            value, unit = attr_value, None

        # Spoken label: normally the friendly-name, but if the friendly-name
        # contributed nothing to the match (e.g. it's a useless "Öffnung" /
        # "Leistung") and an alias did, speak the best-matching alias instead.
        # Judge only against the *displayed* friendly-name, not the entity_id.
        fn_only = set(_norm(fn).split())

        def _hits(words):
            return sum(1 for t in name_terms
                       if t in words or (len(t) >= 3 and any(t in w for w in words)))
        label = fn
        if _hits(fn_only) == 0:
            best_al, best_n = None, 0
            for al, aw in alias_sets:
                n = _hits(aw)
                if n > best_n or (n == best_n and n > 0 and len(al) > len(best_al or "")):
                    best_al, best_n = al, n
            if best_al and best_n > 0:
                label = best_al

        scored.append({
            "covered": covered, "strength": strength, "strong": strong,
            "spec": len(fn), "fn": fn, "label": label, "value": value, "unit": unit,
        })

    if not scored:
        return {"kind": "none", "text": f"Ich habe keinen Wert für „{name}“ gefunden."}

    scored.sort(key=lambda c: (-c["covered"], -c["strength"], c["spec"]))
    best = scored[0]
    top = [c for c in scored
           if c["covered"] == best["covered"] and c["strength"] == best["strength"]]

    if not best["strong"]:
        # Only device_class matched — ambiguous when several qualify ("Akku").
        if len(top) >= 2:
            names = [c["label"] for c in scored[:3]]
            return {"kind": "clarify",
                    "text": f"Ich habe mehrere Werte für „{name}“ — meinst du {_join_names(names)}?"}
        chosen = scored[:1]
    else:
        chosen = top[:limit]

    parts = []
    for c in chosen:
        unit_s = f" {c['unit']}" if c["unit"] else ""
        parts.append(f"{c['label']}: {c['value']}{unit_s}")
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
            "left, car range/fuel, whether a window is open, etc. You may include "
            "the room in the query. Prefer this tool for reading values; do NOT "
            "use it to control devices (use the Hass* tools for that). If it asks "
            "which entity you mean, relay that question."
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
                        "appliance's status. Naming the room helps."
                    ),
                }
            },
            "required": ["name"],
        },
    }


# --- live enrichment (effective area + Assist aliases), cached, fail-soft ---

_REGISTRY_TTL = 600.0
_registry_cache: Dict[str, Any] = {"ts": 0.0, "area": {}, "aliases": {}}


async def _fetch_registry(token: str) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """(area_map, alias_map) from one HA WebSocket session.

    Reads area/device/entity registries and resolves each entity's *effective*
    area (its own area_id, else its device's) plus its Assist aliases.
    """
    import websockets  # provided transitively by pipecat

    async def _run() -> Tuple[Dict[str, str], Dict[str, List[str]]]:
        # max_size=None: the entity registry list on a large install exceeds the
        # library's default 1 MiB frame cap (observed 1009 "message too big").
        async with websockets.connect(
            "ws://supervisor/core/api/websocket", max_size=None
        ) as ws:
            await ws.recv()  # auth_required
            await ws.send(json.dumps({"type": "auth", "access_token": token}))
            if json.loads(await ws.recv()).get("type") != "auth_ok":
                raise RuntimeError("HA WS auth failed")

            async def call(cmd_id: int, typ: str, extra: dict = None):
                msg = {"id": cmd_id, "type": typ}
                if extra:
                    msg.update(extra)
                await ws.send(json.dumps(msg))
                while True:
                    m = json.loads(await ws.recv())
                    if m.get("id") == cmd_id and m.get("type") == "result":
                        return m.get("result")

            areas = await call(1, "config/area_registry/list") or []
            devices = await call(2, "config/device_registry/list") or []
            # The list command omits aliases; get_entries returns full entries
            # (aliases + area_id + device_id) in one bulk call.
            listing = await call(3, "config/entity_registry/list") or []
            eids = [e["entity_id"] for e in listing if e.get("entity_id")]
            full = await call(4, "config/entity_registry/get_entries",
                              {"entity_ids": eids}) or {}

        area_by_id = {a["area_id"]: a.get("name", "") for a in areas}
        dev_area = {d["id"]: d.get("area_id") for d in devices}
        area_map, alias_map = {}, {}
        for eid, ent in full.items():
            if not ent:
                continue
            aid = ent.get("area_id") or dev_area.get(ent.get("device_id"))
            if aid and area_by_id.get(aid):
                area_map[eid] = area_by_id[aid]
            al = [a for a in (ent.get("aliases") or []) if a]
            if al:
                alias_map[eid] = al
        return area_map, alias_map

    return await asyncio.wait_for(_run(), timeout=8)


async def _enrich(token: str) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """Return (area_map, alias_map), refreshing the TTL cache. Fail-soft."""
    now = time.monotonic()
    if now - _registry_cache["ts"] < _REGISTRY_TTL and (
        _registry_cache["area"] or _registry_cache["aliases"]
    ):
        return _registry_cache["area"], _registry_cache["aliases"]
    try:
        area_map, alias_map = await _fetch_registry(token)
        _registry_cache["area"], _registry_cache["aliases"] = area_map, alias_map
        logger.info(f"get_entity_state: registry loaded ({len(area_map)} with area, "
                    f"{len(alias_map)} with aliases)")
    except Exception as e:
        logger.warning(f"get_entity_state: registry lookup failed ({e}); "
                       "matching without areas/aliases")
    _registry_cache["ts"] = now  # back off retries to the TTL either way
    return _registry_cache["area"], _registry_cache["aliases"]


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
            area_map, alias_map = await _enrich(supervisor_token)

            for s in states:
                eid = s.get("entity_id", "")
                s["area"] = area_map.get(eid)
                s["aliases"] = alias_map.get(eid, [])

            result = select_states(states, name)
            logger.info(f"📊 get_entity_state [{result['kind']}]: {result['text'][:200]}")
            await params.result_callback(result["text"])
        except Exception as e:
            logger.error(f"❌ get_entity_state failed: {e}", exc_info=True)
            await params.result_callback("Das Auslesen des Werts hat gerade nicht geklappt.")

    return get_state_tool_handler
