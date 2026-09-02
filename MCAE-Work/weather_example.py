"""Demonstrate how `stop_reason` drives a bounded tool-use loop with Claude.

The Messages API is stateless and turn-based. Each response includes a
`stop_reason` that tells the application whether Claude is finished or paused
to request one or more tools. This example preserves every assistant tool-use
block, returns matching tool-result blocks, and caps the total number of model
turns so a malformed interaction cannot loop forever.
"""

from __future__ import annotations

from typing import Any

import anthropic

import weather_api
from lib import sdk_parser

log = sdk_parser.get_logger()

MODEL = "claude-opus-4-8"
MAX_TOKENS = 1024
MAX_TURNS = 8

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


def run_tool(name: str, tool_input: dict[str, Any]) -> tuple[str, bool]:
    """Dispatch a tool call and return `(content, is_error)`."""
    if name != "get_weather":
        return f"Error: unknown tool {name!r}", True

    try:
        return weather_api.get_weather(**tool_input), False
    except (TypeError, ValueError) as exc:
        return f"Error: bad arguments for get_weather: {exc}", True
    except Exception as exc:
        return f"Error: get_weather failed unexpectedly: {exc}", True


def main() -> None:
    client = anthropic.Anthropic()
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "What's the weather in Paris right now?"}
    ]

    for turn in range(1, MAX_TURNS + 1):
        log.info("=" * 70)
        log.info("TURN %d — sending %d message(s) to %s", turn, len(messages), MODEL)

        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            tools=[WEATHER_TOOL],
            messages=messages,
        )
        log.info("RESPONSE:\n%s", sdk_parser.format_response(response))

        if response.stop_reason == "end_turn":
            final_text = "".join(
                block.text for block in response.content if block.type == "text"
            ).strip()
            log.info("FINAL ANSWER: %s", final_text or "(no text)")
            log.info(
                "Conversation ended after %d turn(s), %d message(s) total.",
                turn,
                len(messages),
            )
            return

        if response.stop_reason != "tool_use":
            log.warning("Stopping on unexpected stop_reason=%s", response.stop_reason)
            return

        assistant_message = {"role": "assistant", "content": response.content}
        messages.append(assistant_message)

        tool_results: list[dict[str, Any]] = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            log.info("Running tool %s(%s)", block.name, block.input)
            result_text, is_error = run_tool(block.name, block.input)
            log.info("Tool returned: %s", result_text)

            result_block: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result_text,
            }
            if is_error:
                result_block["is_error"] = True
            tool_results.append(result_block)

        if not tool_results:
            log.error("stop_reason=tool_use but no tool_use blocks were returned")
            return

        user_message = {"role": "user", "content": tool_results}
        log.info("%s", sdk_parser.format_tool_result_message(user_message))
        messages.append(user_message)

    log.warning("Reached MAX_TURNS (%d) without end_turn; stopping.", MAX_TURNS)


if __name__ == "__main__":
    main()
