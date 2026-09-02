"""Unit tests for the Open-Meteo helper without making network requests."""

from __future__ import annotations

import httpx

from conftest import load_module

weather_api = load_module("examples/weather_api.py", "weather_api_under_test")


def test_best_match_expands_us_state_abbreviation() -> None:
    results = [
        {"name": "Paris", "admin1": "Île-de-France", "country": "France"},
        {"name": "Paris", "admin1": "Texas", "country": "United States"},
    ]

    match = weather_api._best_match(results, "TX")

    assert match["admin1"] == "Texas"


def test_current_conditions_rejects_unknown_unit() -> None:
    try:
        weather_api._current_conditions(1.0, 2.0, "kelvin")
    except weather_api.WeatherError as exc:
        assert "unsupported unit" in str(exc)
    else:
        raise AssertionError("Expected WeatherError for unsupported unit")


def test_get_json_wraps_network_errors(monkeypatch) -> None:
    request = httpx.Request("GET", "https://example.test")

    def fake_get(*args, **kwargs):
        raise httpx.ConnectError("network down", request=request)

    monkeypatch.setattr(httpx, "get", fake_get)

    try:
        weather_api._get_json("https://example.test", {})
    except weather_api.WeatherError as exc:
        assert "could not reach" in str(exc)
        assert "network down" in str(exc)
    else:
        raise AssertionError("Expected WeatherError for network failure")


def test_get_json_wraps_http_status_errors(monkeypatch) -> None:
    request = httpx.Request("GET", "https://example.test")
    response = httpx.Response(503, request=request)

    def fake_get(*args, **kwargs):
        return response

    monkeypatch.setattr(httpx, "get", fake_get)

    try:
        weather_api._get_json("https://example.test", {})
    except weather_api.WeatherError as exc:
        assert "HTTP 503" in str(exc)
    else:
        raise AssertionError("Expected WeatherError for HTTP failure")


def test_get_weather_returns_readable_error_when_geocoding_fails(monkeypatch) -> None:
    def fake_geocode(location: str):
        raise weather_api.WeatherError(f"no place found matching {location!r}")

    monkeypatch.setattr(weather_api, "_geocode", fake_geocode)

    result = weather_api.get_weather("Not A Real Place")

    assert result.startswith("Error:")
    assert "no place found" in result
