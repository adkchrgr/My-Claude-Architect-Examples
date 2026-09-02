# `stop_reason` — how the tool-use loop is driven

A small, runnable example showing how the Claude Messages API uses
`stop_reason` to tell you what to do next. It uses a `get_weather` tool so both
of the reasons that matter show up in one run:

| `stop_reason` | What it means | What the code does |
| --- | --- | --- |
| `tool_use` | Claude paused to call a tool | run the tool, append the result, loop |
| `end_turn` | Claude is finished (the **end result**) | print the final answer and stop |

## Files

- `weather_example.py` — the manual tool-use loop. Watch how each response's
  `stop_reason` decides whether to keep looping or finish.
- `weather_api.py` — the actual tool implementation. Calls the Open-Meteo
  endpoints for live conditions; no API key required.
- `lib/sdk_parser.py` — formatting helpers used to log the `stop_reason`,
  content blocks, and token usage on every turn (imported as `lib`).
- `requirements.txt` — the one dependency (`anthropic`).

## Run it

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...   # or run: ant auth login
python weather_example.py
```

## What you'll see

The run makes two API calls (two turns):

1. **Turn 1** → `stop_reason: tool_use`. Claude asks to call `get_weather`.
   The code appends Claude's `tool_use` turn, runs the tool, and appends the
   `tool_result` back to the history.
2. **Turn 2** → `stop_reason: end_turn`. With the weather in hand, Claude
   writes the final answer. This is the **end result** and the loop exits.

Because the API is stateless, the full `messages` list is resent on every turn
and grows as results are appended — the logs make that growth explicit.

The `tool_result` is live weather, so the numbers change between runs. That's the
point: the value Claude reads on turn 2 came from outside the model.

## The weather endpoint

`weather_api.get_weather()` makes two [Open-Meteo](https://open-meteo.com) calls
— both keyless and free for non-commercial use:

1. `geocoding-api.open-meteo.com/v1/search` — city name → latitude/longitude.
   `"Paris, TX"` is split on the comma so the state can disambiguate the match,
   since the geocoder only searches on the city name.
2. `api.open-meteo.com/v1/forecast` — current temperature, humidity, wind, and a
   WMO weather code that gets mapped to a description.

HTTP goes through `httpx`, which `anthropic` already installs, so there's still
just one dependency. Errors (unknown city, endpoint unreachable, bad unit) are
returned as `Error: ...` **strings**, not raised — that text becomes the
`tool_result`, so Claude can explain the failure and the loop keeps its shape.
Try it without spending tokens:

```bash
python weather_api.py
```

## The three appends that make the loop work

1. Append Claude's response **verbatim** (`{"role": "assistant", "content":
   response.content}`) — the `tool_use` blocks must be preserved so the
   `tool_result` blocks can reference them by `id`.
2. Run each tool and collect a `tool_result` block per `tool_use` (matching
   `tool_use_id`).
3. Append all results in **one** `{"role": "user", ...}` message, then loop.

Model used: `claude-opus-4-8`.
