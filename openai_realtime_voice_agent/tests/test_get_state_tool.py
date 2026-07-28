"""Tests for the pure state-selection logic of get_entity_state.

The scoring/selection is a pure function over a list of enriched HA state dicts
(state + attributes + effective ``area`` + Assist ``aliases``) so it can be
tested without hitting the HA API. Each test names the behaviour it pins.
"""
from app.get_state_tool import select_states


def st(entity_id, state, friendly_name=None, device_class=None, unit=None,
       area=None, aliases=None, **attrs):
    a = dict(attrs)
    if friendly_name is not None:
        a["friendly_name"] = friendly_name
    if device_class is not None:
        a["device_class"] = device_class
    if unit is not None:
        a["unit_of_measurement"] = unit
    return {"entity_id": entity_id, "state": state, "attributes": a,
            "area": area, "aliases": aliases or []}


# --- baseline behaviour ---------------------------------------------------

def test_returns_none_kind_when_nothing_matches():
    r = select_states([st("sensor.foo", "1", "Foo")], "Quatschbegriff")
    assert r["kind"] == "none"


def test_answer_includes_value_and_unit():
    states = [st("sensor.wz_temp", "21.5", "Wohnzimmer Temperatur",
                 device_class="temperature", unit="°C")]
    r = select_states(states, "Temperatur Wohnzimmer")
    assert r["kind"] == "answer"
    assert "21.5" in r["text"] and "°C" in r["text"]


def test_skips_unavailable_and_unknown_states():
    states = [
        st("sensor.solar_soc", "unavailable", "Solar SoC Battery",
           device_class="battery", unit="%"),
        st("sensor.phone", "unknown", "Handy Akku", device_class="battery", unit="%"),
    ]
    assert select_states(states, "Solar SoC")["kind"] == "none"


def test_name_match_beats_device_class_only_match():
    states = [
        st("sensor.solar_soc", "70", "Solar SoC Battery", device_class="battery", unit="%"),
        st("sensor.phone", "45", "Handy Akku", device_class="battery", unit="%"),
    ]
    r = select_states(states, "Solar SoC")
    assert r["kind"] == "answer"
    assert r["text"].startswith("Solar SoC Battery")
    assert "Handy" not in r["text"]


def test_ambiguous_device_class_only_asks_for_clarification():
    states = [
        st("sensor.solar_soc", "70", "Solar SoC Battery", device_class="battery", unit="%"),
        st("sensor.phone", "45", "Smartphone", device_class="battery", unit="%"),
    ]
    r = select_states(states, "Akku")
    assert r["kind"] == "clarify"
    assert "Solar SoC Battery" in r["text"] and "Smartphone" in r["text"]


def test_single_device_class_match_answers_without_clarifying():
    states = [st("sensor.solar_soc", "70", "Solar SoC Battery",
                 device_class="battery", unit="%")]
    r = select_states(states, "Akku")
    assert r["kind"] == "answer" and "70" in r["text"]


def test_caps_answer_to_two_entities():
    states = [st(f"sensor.t{i}", str(i), f"Temperatur Raum {i}",
                 device_class="temperature", unit="°C") for i in range(5)]
    r = select_states(states, "Temperatur")
    assert r["kind"] == "answer"
    assert r["text"].count(";") <= 1


def test_query_token_matching_an_attribute_key_returns_that_attribute():
    states = [st("sensor.waschmaschine", "läuft", "Waschmaschine", restzeit=42)]
    r = select_states(states, "Waschmaschine Restzeit")
    assert r["kind"] == "answer"
    assert "42" in r["text"] and "läuft" not in r["text"]


def test_returns_only_top_scored_match_not_weaker_padding():
    states = [
        st("sensor.solar_soc", "62.4", "Solar SoC Battery", device_class="battery", unit="%"),
        st("sensor.solar_soh", "100", "Solar SoH Battery", device_class="battery", unit="%"),
        st("sensor.car_soc", "0.0", "Ladestand", device_class="battery", unit="%"),
    ]
    r = select_states(states, "Solar SoC")
    assert r["kind"] == "answer"
    assert r["text"] == "Solar SoC Battery: 62.4 %"


def test_ignores_non_read_domains():
    states = [st("automation.foo", "on", "Foo Automation"),
              st("scene.foo", "on", "Foo Scene")]
    assert select_states(states, "Foo")["kind"] == "none"


# --- new: stopwords, compounds, area, aliases, confidence gate ------------

def test_filler_words_do_not_hijack_the_match():
    # "aktuelle" must not match "Aktuelle Seite" and drown the real answer.
    states = [
        st("sensor.tab_page", "https://x", "Tablet Aktuelle Seite"),
        st("sensor.warn", "0", "Wetter Aktuelle Warnstufe"),
        st("sensor.solar", "800", "Wechselrichter Solarleistung", device_class="power", unit="W"),
    ]
    r = select_states(states, "aktuelle Solarleistung")
    assert r["kind"] == "answer"
    assert r["text"].startswith("Wechselrichter Solarleistung")
    assert "Aktuelle Seite" not in r["text"] and "Warnstufe" not in r["text"]


def test_german_compound_is_decomposed():
    # "Solarleistung" as one spoken word finds a "Solar ... Leistung" sensor.
    states = [
        st("sensor.pv", "800", "PV Solar Leistung", device_class="power", unit="W"),
        st("sensor.house", "300", "Hauslast", device_class="power", unit="W"),
    ]
    r = select_states(states, "Solarleistung")
    assert r["kind"] == "answer"
    assert r["text"].startswith("PV Solar Leistung")


def test_area_lets_a_room_word_find_an_unnamed_sensor():
    # The Büro sensor is named generically but sits in area "Büro"; the query
    # names the room, so it must beat a temperature sensor in another room.
    states = [
        st("sensor.aqara3", "21.5", "Aqara 3", device_class="temperature",
           unit="°C", area="Büro"),
        st("sensor.wz", "22.0", "Wohnzimmer Temperatur", device_class="temperature",
           unit="°C", area="Wohnzimmer"),
    ]
    r = select_states(states, "Temperatur Büro")
    assert r["kind"] == "answer"
    assert r["text"].startswith("Aqara 3")
    assert "Wohnzimmer" not in r["text"]


def test_assist_alias_matches_strongly_and_avoids_clarify():
    # "Akkustand" is an alias of the house battery; it must resolve directly
    # even though other battery-class sensors exist.
    states = [
        st("sensor.house_batt", "54", "Solar SoC Battery", device_class="battery",
           unit="%", aliases=["Akkustand", "Batteriestand"]),
        st("sensor.phone", "45", "Handy", device_class="battery", unit="%"),
    ]
    r = select_states(states, "Akkustand")
    assert r["kind"] == "answer"
    assert r["text"].startswith("Solar SoC Battery")


def test_device_class_breaks_ties_toward_the_right_class():
    # One device exposes battery/humidity/temperature entities that all share
    # the name+room. "Temperatur" must pick the temperature one, not its
    # battery or humidity sibling.
    states = [
        st("sensor.ts_batt", "49", "Temperatursensor Schlafzimmer Battery",
           device_class="battery", unit="%"),
        st("sensor.ts_hum", "49.3", "Temperatursensor Schlafzimmer Humidity",
           device_class="humidity", unit="%"),
        st("sensor.ts_temp", "21.0", "Temperatursensor Schlafzimmer Temperature",
           device_class="temperature", unit="°C"),
    ]
    r = select_states(states, "Temperatur Schlafzimmer")
    assert r["kind"] == "answer"
    assert r["text"].startswith("Temperatursensor Schlafzimmer Temperature")
    assert "Battery" not in r["text"] and "Humidity" not in r["text"]


def test_query_of_only_stopwords_returns_none():
    states = [st("sensor.wz_temp", "21.5", "Wohnzimmer Temperatur",
                 device_class="temperature", unit="°C")]
    assert select_states(states, "wie viel ist es")["kind"] == "none"
