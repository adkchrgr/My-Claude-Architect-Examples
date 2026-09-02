"""Demonstrate how `stop_reason` changes control flow in a tool-use exchange."""

from __future__ import annotations

import asyncio
import json
import os
import random
from pathlib import Path
from typing import Any

from anthropic import AsyncAnthropic, DefaultAioHttpClient
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 1024

TOOLS = [
    {
        "name": "magic_eyeball",
        "description": "Use for yes-or-no fortune-telling questions.",
        "input_schema": {
            "type": "object",
            "properties": {"question": {"type": "string"}},
            "required": ["question"],
        },
    }
]


def tool_magic_eyeball(question: str) -> str:
    _ = question
    return random.choice(["Yes", "No", "Ask again later"])


async def create(client: AsyncAnthropic, messages: list[dict[str, Any]]) -> object:
    return await client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        tools=TOOLS,
        messages=messages,
    )


async def main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    async with AsyncAnthropic(
        api_key=api_key,
        http_client=DefaultAioHttpClient(),
    ) as client:
        user_message = "Hey Claude, will I be a billionaire living on Mars in 2026?"
        messages: list[dict[str, Any]] = [{"role": "user", "content": user_message}]

        response = await create(client, messages)
        print("original:")
        print(json.dumps(response.model_dump(), indent=2))

        if response.stop_reason != "tool_use":
            print(f"No tool call requested; stop_reason={response.stop_reason!r}")
            return

        tool_uses = [block for block in response.content if block.type == "tool_use"]
        if not tool_uses:
            print("stop_reason was tool_use but no tool_use block was present")
            return

        tool_results: list[dict[str, Any]] = []
        for tool_use in tool_uses:
            if tool_use.name != "magic_eyeball":
                result = f"Unknown tool: {tool_use.name}"
                is_error = True
            else:
                try:
                    result = tool_magic_eyeball(**tool_use.input)
                    is_error = False
                except (TypeError, ValueError) as exc:
                    result = f"Invalid arguments for magic_eyeball: {exc}"
                    is_error = True

            result_block: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": tool_use.id,
                "content": result,
            }
            if is_error:
                result_block["is_error"] = True
            tool_results.append(result_block)

        messages.extend(
            [
                {"role": "assistant", "content": response.content},
                {"role": "user", "content": tool_results},
            ]
        )

        follow_up = await create(client, messages)
        print("follow up:")
        print(json.dumps(follow_up.model_dump(), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
