"""Demonstrate code-driven routing around an LLM classification step."""

from __future__ import annotations

import asyncio
import json
import os
import random
from pathlib import Path

from anthropic import AsyncAnthropic
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 1024
VALID_CATEGORIES = {"FORTUNE", "OTHER"}

CLASSIFY_PROMPT = """Classify the user's message into exactly one category. Respond with only the category name, nothing else.

Categories:
- FORTUNE: a yes/no fortune-telling question
- OTHER: anything else

User message: {question}"""


def tool_magic_eyeball(_: str) -> str:
    """Return a deliberately toy fortune response for the routing demo."""
    return random.choice(["Yes", "No", "Ask again later"])


async def create(client: AsyncAnthropic, messages: list[dict]) -> object:
    return await client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=messages,
    )


def extract_text(response: object) -> str:
    """Collect text blocks from a Messages API response."""
    return "".join(
        block.text for block in response.content if getattr(block, "type", None) == "text"
    ).strip()


async def main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    async with AsyncAnthropic(api_key=api_key) as client:
        user_message = "Hey Claude, will I be a billionaire living on Mars in 2026?"

        classify_messages = [
            {
                "role": "user",
                "content": CLASSIFY_PROMPT.format(question=user_message),
            }
        ]
        classify_response = await create(client, classify_messages)
        category = extract_text(classify_response).upper()

        if category not in VALID_CATEGORIES:
            print(f"classification was unexpected ({category!r}); defaulting to OTHER")
            category = "OTHER"
        else:
            print(f"classification: {category}")

        # The branch itself is deterministic application code. The model only
        # supplies the classification value consumed by the decision tree.
        if category == "FORTUNE":
            fortune = tool_magic_eyeball(user_message)
            print(f"magic eyeball says: {fortune}")
            messages = [
                {
                    "role": "user",
                    "content": (
                        f"{user_message}\n\n"
                        f"(The magic eyeball says: {fortune}. Give the user a fun, "
                        "in-character answer based on this.)"
                    ),
                }
            ]
        else:
            messages = [{"role": "user", "content": user_message}]

        response = await create(client, messages)
        print("final answer:")
        print(json.dumps(response.model_dump(), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
