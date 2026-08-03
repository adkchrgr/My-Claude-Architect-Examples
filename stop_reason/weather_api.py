"""A real weather lookup for the `get_weather` tool.

Uses Open-Meteo, which needs no API key, so the example stays runnable with
nothing but `ANTHROPIC_API_KEY` set. Two calls per lookup:

  1. geocoding  — turn "Paris" into a latitude/longitude
  2. forecast   — read the current conditions at those coordinates

HTTP goes through `httpx`, which `anthropic` already depends on, so there's no
extra install. Failures are returned as plain strings rather than raised: the
tool result is handed straight back to Claude, and a readable error lets it
explain what went wrong instead of crashing the loop.
"""

from __future__ import annotations

import httpx

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
TIMEOUT_SECONDS = 10

# Open-Meteo reports conditions as WMO weather codes. Only the descriptions we
# need — anything unmapped falls back to the raw code.
WMO_CODES: dict[int, str] = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "foggy",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    61: "light rain",
    63: "moderate rain",
    65: "heavy rain",
    66: "light freezing rain",
    67: "heavy freezing rain",
    71: "light snow",
    73: "moderate snow",
    75: "heavy snow",
    77: "snow grains",
    80: "light rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    85: "light snow showers",
    86: "heavy snow showers",
    95: "a thunderstorm",
    96: "a thunderstorm with light hail",
    99: "a thunderstorm with heavy hail",
}


# Claude writes US cities as "Paris, TX", but Open-Meteo reports the state
# spelled out ("Texas"), so a two-letter hint needs translating before it can be
# matched. Non-US hints ("France", "Ontario") are already spelled out.
US_STATES: dict[str, str] = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas",
    "ca": "california", "co": "colorado", "ct": "connecticut", "de": "delaware",
    "fl": "florida", "ga": "georgia", "hi": "hawaii", "id": "idaho",
    "il": "illinois", "in": "indiana", "ia": "iowa", "ks": "kansas",
    "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland",
    "ma": "massachusetts", "mi": "michigan", "mn": "minnesota", "ms": "mississippi",
    "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada",
    "nh": "new hampshire", "nj": "new jersey", "nm": "new mexico", "ny": "new york",
    "nc": "north carolina", "nd": "north dakota", "oh": "ohio", "ok": "oklahoma",
    "or": "oregon", "pa": "pennsylvania", "ri": "rhode island", "sc": "south carolina",
    "sd": "south dakota", "tn": "tennessee", "tx": "texas", "ut": "utah",
    "vt": "vermont", "va": "virginia", "wa": "washington", "wv": "west virginia",
    "wi": "wisconsin", "wy": "wyoming", "dc": "district of columbia",
}


class WeatherError(Exception):
    """Raised when the weather endpoints can't answer the question."""


def get_weather(location: str, unit: str = "celsius") -> str:
    """Return a one-line current-conditions summary for `location`.

    This is the function the `get_weather` tool_use block dispatches to. It
    returns a string either way — a summary on success, an `Error: ...` message
    on failure — because that string becomes the `tool_result` content.
    """
    try:
        place = _geocode(location)
        current = _current_conditions(place["latitude"], place["longitude"], unit)
    except WeatherError as exc:
        return f"Error: {exc}"

    code = current["weather_code"]
    condition = WMO_CODES.get(code, f"WMO code {code}")
    degrees = "°C" if unit == "celsius" else "°F"
    return (
        f"The weather in {place['label']} is {condition}, "
        f"{current['temperature']}{degrees} "
        f"(humidity {current['humidity']}%, wind {current['wind_speed']} km/h). "
        f"Observed at {current['time']} local time."
    )


def _geocode(location: str) -> dict:
    """Resolve a city name to coordinates plus a human-readable label.

    Claude tends to pass "San Francisco, CA" — a form the geocoder rejects, since
    it matches on the city name alone. So we search on the part before the comma
    and use the rest ("CA", "France") to pick among the matches.
    """
    city, _, hint = location.partition(",")
    params = {"name": city.strip(), "count": 10, "language": "en", "format": "json"}
    results = _get_json(GEOCODE_URL, params).get("results") or []
    if not results:
        raise WeatherError(f"no place found matching {location!r}")

    hit = _best_match(results, hint.strip())
    # Include region/country so "Springfield" isn't ambiguous in the answer.
    label = ", ".join(
        part for part in (hit.get("name"), hit.get("admin1"), hit.get("country")) if part
    )
    return {
        "label": label or location,
        "latitude": hit["latitude"],
        "longitude": hit["longitude"],
    }


def _best_match(results: list[dict], hint: str) -> dict:
    """Pick the geocoding result that matches a region hint, else the first one.

    Open-Meteo orders results by population, so result 0 is the sensible default
    when there's no hint (or nothing matches it).
    """
    if not hint:
        return results[0]

    needle = hint.casefold()
    needle = US_STATES.get(needle, needle)  # "tx" -> "texas"
    for hit in results:
        candidates = (
            hit.get("admin1"),        # "Texas", "Île-de-France Region"
            hit.get("country"),       # "United States"
            hit.get("country_code"),  # "US"
        )
        # `startswith` catches Open-Meteo's suffixed names, e.g. a "new york"
        # needle against the admin1 "New York State".
        if any(field and field.casefold().startswith(needle) for field in candidates):
            return hit
    return results[0]


def _current_conditions(latitude: float, longitude: float, unit: str) -> dict:
    """Fetch current conditions at a coordinate pair."""
    if unit not in ("celsius", "fahrenheit"):
        raise WeatherError(f"unsupported unit {unit!r} — use 'celsius' or 'fahrenheit'")

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
        "temperature_unit": unit,
        "wind_speed_unit": "kmh",
        "timezone": "auto",
    }
    current = _get_json(FORECAST_URL, params).get("current")
    if not current:
        raise WeatherError("the forecast response had no `current` block")

    return {
        "temperature": current.get("temperature_2m"),
        "humidity": current.get("relative_humidity_2m"),
        "weather_code": current.get("weather_code"),
        "wind_speed": current.get("wind_speed_10m"),
        "time": current.get("time"),
    }


def _get_json(url: str, params: dict) -> dict:
    """GET a URL with query params and decode the JSON body."""
    try:
        response = httpx.get(url, params=params, timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        raise WeatherError(f"{url} returned HTTP {exc.response.status_code}") from exc
    except httpx.RequestError as exc:
        raise WeatherError(f"could not reach {url}: {exc}") from exc
    except ValueError as exc:  # body wasn't JSON
        raise WeatherError(f"{url} returned a body that wasn't JSON") from exc


if __name__ == "__main__":
    # Quick smoke test without burning any tokens: python weather_api.py
    print(get_weather("Paris"))
    print(get_weather("San Francisco, CA", unit="fahrenheit"))
