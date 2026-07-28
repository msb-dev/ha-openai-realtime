"""Tests for the pure state-selection logic of get_entity_state.

The scoring/selection is a pure function over a list of HA state dicts so it can
be tested without hitting the HA API. Each test names the behaviour it pins.
"""
from app.get_state_tool import select_states


def st(entity_id, state, friendly_name=None, device_class=None, unit=None, **attrs):
    a = dict(attrs)
    if friendly_name is not None:
        a["friendly_name"] = friendly_name
    if device_class is not None:
        a["device_class"] = device_class
    if unit is not None:
        a["unit_of_measurement"] = unit
    return {"entity_id": entity_id, "state": state, "attributes": a}


def test_returns_none_kind_when_nothing_matches():
    states = [st("sensor.foo", "1", "Foo")]
    r = select_states(states, "Quatschbegriff")
    assert r["kind"] == "none"


def test_answer_includes_value_and_unit():
    states = [st("sensor.wz_temp", "21.5", "Wohnzimmer Temperatur",
                 device_class="temperature", unit="°C")]
    r = select_states(states, "Temperatur Wohnzimmer")
    assert r["kind"] == "answer"
    assert "21.5" in r["text"]
    assert "°C" in r["text"]


def test_skips_unavailable_and_unknown_states():
    states = [
        st("sensor.solar_soc", "unavailable", "Solar SoC Battery",
           device_class="battery", unit="%"),
        st("sensor.phone", "unknown", "Handy Akku", device_class="battery", unit="%"),
    ]
    r = select_states(states, "Solar SoC")
    assert r["kind"] == "none"


def test_name_match_beats_device_class_only_match():
    states = [
        st("sensor.solar_soc", "70", "Solar SoC Battery",
           device_class="battery", unit="%"),
        st("sensor.phone", "45", "Handy Akku", device_class="battery", unit="%"),
    ]
    r = select_states(states, "Solar SoC")
    assert r["kind"] == "answer"
    # The named solar battery must be the (first) answer, not the phone battery.
    assert r["text"].startswith("Solar SoC Battery")
    assert "Handy" not in r["text"]


def test_ambiguous_device_class_only_asks_for_clarification():
    # "Akku" matches no friendly-name word; both are battery device_class only.
    states = [
        st("sensor.solar_soc", "70", "Solar SoC Battery",
           device_class="battery", unit="%"),
        st("sensor.phone", "45", "Smartphone", device_class="battery", unit="%"),
    ]
    r = select_states(states, "Akku")
    assert r["kind"] == "clarify"
    assert "Solar SoC Battery" in r["text"]
    assert "Smartphone" in r["text"]


def test_single_device_class_match_answers_without_clarifying():
    states = [
        st("sensor.solar_soc", "70", "Solar SoC Battery",
           device_class="battery", unit="%"),
    ]
    r = select_states(states, "Akku")
    assert r["kind"] == "answer"
    assert "70" in r["text"]


def test_caps_answer_to_two_entities():
    states = [
        st(f"sensor.t{i}", str(i), f"Temperatur Raum {i}",
           device_class="temperature", unit="°C")
        for i in range(5)
    ]
    r = select_states(states, "Temperatur")
    assert r["kind"] == "answer"
    # at most two "name: value" segments (joined by "; ")
    assert r["text"].count(";") <= 1


def test_query_token_matching_an_attribute_key_returns_that_attribute():
    states = [
        st("sensor.waschmaschine", "läuft", "Waschmaschine", restzeit=42),
    ]
    r = select_states(states, "Waschmaschine Restzeit")
    assert r["kind"] == "answer"
    assert "42" in r["text"]
    assert "läuft" not in r["text"]


def test_returns_only_top_scored_match_not_weaker_padding():
    # A distinctive two-token match must not be padded with a weaker entity that
    # only matched one generic token: a car "soc" (entity_id contains "soc") and
    # a second solar metric both otherwise qualify as battery device_class.
    states = [
        st("sensor.solar_soc", "62.4", "Solar SoC Battery",
           device_class="battery", unit="%"),
        st("sensor.solar_soh", "100", "Solar SoH Battery",
           device_class="battery", unit="%"),
        st("sensor.car_soc", "0.0", "Ladestand", device_class="battery", unit="%"),
    ]
    r = select_states(states, "Solar SoC")
    assert r["kind"] == "answer"
    assert r["text"] == "Solar SoC Battery: 62.4 %"


def test_ignores_non_read_domains():
    states = [
        st("automation.foo", "on", "Foo Automation"),
        st("scene.foo", "on", "Foo Scene"),
    ]
    r = select_states(states, "Foo")
    assert r["kind"] == "none"
