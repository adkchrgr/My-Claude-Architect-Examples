"""Demonstrate model-driven tool selection with a bounded tool-use loop."""

from __future__ import annotations

import asyncio
import json
import os
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

from anthropic import AsyncAnthropic
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 1024
MAX_TURNS = 8


def tool_magic_eyeball(question: str) -> str:
    """Return a toy yes/no fortune response."""
    _ = question
    return random.choice(["Yes", "No", "Ask again later"])


def tool_roll_dice(sides: int = 6) -> str:
    """Roll an n-sided die with basic input validation."""
    if sides < 2 or sides > 1000:
        raise ValueError("sides must be between 2 and 1000")
    return str(random.randint(1, sides))


TOOLS = [
    {
        "name": "magic_eyeball",
        "description": "Use for yes-or-no fortune-telling questions.",
        "input_schema": {
            "type": "object",
            "properties": {"question": {"type": "string"}},
            "required": ["question"],
        },
    },
    {
        "name": "roll_dice",
        "description": "Use when the user wants to roll a die or asks for a random number for luck.",
        "input_schema": {
            "type": "object",
            "properties": {
                "sides": {
                    "type": "integer",
                    "minimum": 2,
                    "maximum": 1000,
                    "description": "Number of sides on the die. Defaults to 6.",
                }
            },
            "required": [],
        },
    },
]

TOOL_FUNCTIONS: dict[str, Callable[..., str]] = {
    "magic_eyeball": tool_magic_eyeball,
    "roll_dice": tool_roll_dice,
}

SYSTEM_PROMPT = (
    "You are a whimsical fortune-telling assistant. Use the tools available "
    "to you as needed to answer the user."
)


async def create(client: AsyncAnthropic, messages: list[dict[str, Any]]) -> object:
    return await client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        tools=TOOLS,
        messages=messages,
    )


def run_tool(name: str, tool_input: dict[str, Any]) -> tuple[str, bool]:
    """Execute a model-selected tool and convert failures into tool-result data."""
    handler = TOOL_FUNCTIONS.get(name)
    if handler is None:
        return f"Unknown tool: {name}", True

    try:
        return handler(**tool_input), False
    except (TypeError, ValueError) as exc:
        return f"Invalid arguments for {name}: {exc}", True
    except Exception as exc:  # keep demo loop alive while surfacing the failure
        return f"Tool {name} failed: {exc}", True


async def main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    async with AsyncAnthropic(api_key=api_key) as client:
        user_message = (
            "Hey Claude, will I be a billionaire living on Mars in 2026? "
            "Also roll me a 20-sided die for luck."
        )
        messages: list[dict[str, Any]] = [{"role": "user", "content": user_message}]

        for turn in range(1, MAX_TURNS + 1):
            response = await create(client, messages)
            print(f"round {turn}: stop_reason={response.stop_reason}")
            print(json.dumps(response.model_dump(), indent=2))

            if response.stop_reason == "end_turn":
                return

            if response.stop_reason != "tool_use":
                print(f"Stopping on unexpected stop_reason={response.stop_reason!r}")
                return

            messages.append({"role": "assistant", "content": response.content})
            tool_results: list[dict[str, Any]] = []

            for block in response.content:
                if block.type != "tool_use":
                    continue

                result, is_error = run_tool(block.name, block.input)
                result_block: dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                }
                if is_error:
                    result_block["is_error"] = True
                tool_results.append(result_block)

            if not tool_results:
                print("Model reported tool_use but emitted no tool calls; stopping.")
                return

            messages.append({"role": "user", "content": tool_results})

        print(f"Reached MAX_TURNS ({MAX_TURNS}) without end_turn; stopping.")


if __name__ == "__main__":
    asyncio.run(main())
