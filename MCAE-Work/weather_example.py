"""Demonstrate how `stop_reason` drives a tool-use loop with the Claude API.

The Messages API is stateless and turn-based. On every turn Claude returns a
`stop_reason` that tells you what to do next:

  * `stop_reason == "tool_use"`   -> Claude paused to call a tool. You run the
                                     tool, append its result to the conversation,
                                     and send everything back for another turn.
  * `stop_reason == "end_turn"`   -> Claude is done. This is the end result.

This script uses a `get_weather` tool to make both cases show up: the first
response stops with `tool_use`, and after we hand back the weather the second
response stops with `end_turn`. Every step is logged through
`lib/sdk_parser` so you can watch the loop and see how each message is appended
to the running history.

The tool calls a real endpoint (Open-Meteo, via `weather_api.py`), so the
`tool_result` you see in the logs is live data — no API key needed for it.

Run it:

    pip install anthropic
    export ANTHROPIC_API_KEY=sk-ant-...   # or: ant auth login
    python weather_example.py
"""

from __future__ import annotations

import anthropic

import weather_api
from lib import sdk_parser

log = sdk_parser.get_logger()

MODEL = "claude-opus-4-8"
MAX_TOKENS = 1024

# --- Tool definition ---------------------------------------------------------
# A single weather tool. The schema is what Claude sees when deciding whether
# (and how) to call it.
WEATHER_TOOL = {
    "name": "get_weather",
    "description": (
        "Get the current, real-time weather for a given city. Returns "
        "conditions, temperature, humidity, and wind speed."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City name, e.g. 'Paris' or 'San Francisco, CA'.",
            },
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "Temperature unit to report in.",
            },
        },
        "required": ["location"],
    },
}


def run_tool(name: str, tool_input: dict) -> str:
    """Dispatch a tool_use block to the matching Python function.

    `get_weather` hits the Open-Meteo endpoints (see `weather_api.py`). It
    always returns a string — an error message included — because whatever it
    returns becomes the `tool_result` content Claude reads on the next turn.
    """
    if name == "get_weather":
        try:
            return weather_api.get_weather(**tool_input)
        except TypeError as exc:
            # Claude sent arguments that don't match the schema. Tell it so it
            # can retry rather than killing the loop.
            return f"Error: bad arguments for get_weather: {exc}"
    return f"Error: unknown tool {name!r}"


def main() -> None:
    client = anthropic.Anthropic()

    # The conversation history. The API is stateless, so we resend this whole
    # list on every turn — and grow it as we go.
    messages: list[dict] = [
        {"role": "user", "content": "What's the weather in Paris right now?"}
    ]

    turn = 0
    while True:
        turn += 1
        log.info("=" * 70)
        log.info("TURN %d — sending %d message(s) to %s", turn, len(messages), MODEL)

        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            tools=[WEATHER_TOOL],
            messages=messages,
        )

        # Log the whole response: stop_reason, content blocks, and token usage.
        log.info("RESPONSE:\n%s", sdk_parser.format_response(response))

        # --- The end result -------------------------------------------------
        # `end_turn` means Claude is finished. Print the final text and stop.
        if response.stop_reason == "end_turn":
            log.info("Reached the end result (stop_reason=end_turn). Done.")
            final_text = "".join(
                block.text for block in response.content if block.type == "text"
            )
            log.info("FINAL ANSWER: %s", final_text)
            break

        # --- The tool_use branch -------------------------------------------
        # `tool_use` means Claude wants one or more tools run before it can
        # continue. Anything else (max_tokens, refusal, ...) we surface and stop.
        if response.stop_reason != "tool_use":
            log.info("Stopping on unexpected stop_reason=%s", response.stop_reason)
            break

        # 1. Append Claude's turn *verbatim* — the tool_use blocks must be
        #    preserved so the follow-up tool_result blocks line up by id.
        assistant_message = {"role": "assistant", "content": response.content}
        log.info("%s", sdk_parser.format_tool_result_message(assistant_message))
        messages.append(assistant_message)

        # 2. Run every tool Claude asked for and collect the results. All the
        #    results for one turn go back in a *single* user message.
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            log.info("Running tool %s(%s)", block.name, block.input)
            result_text = run_tool(block.name, block.input)
            log.info("Tool returned: %s", result_text)
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,  # must match the tool_use block's id
                    "content": result_text,
                }
            )

        # 3. Append the tool results as a user turn, then loop — the next
        #    `messages.create` call gives Claude the weather it asked for.
        user_message = {"role": "user", "content": tool_results}
        log.info("%s", sdk_parser.format_tool_result_message(user_message))
        messages.append(user_message)

    log.info("=" * 70)
    log.info("Conversation ended after %d turn(s), %d message(s) total.", turn, len(messages))


if __name__ == "__main__":
    main()
