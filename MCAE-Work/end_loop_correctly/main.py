"""Demonstrate a bounded agentic loop with explicit tool dispatch safeguards."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from anthropic import AsyncAnthropic, DefaultAioHttpClient
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 1024
MAX_ITERATIONS = 10


def tool_check_inventory(item: str) -> str:
    inventory = {"widget-A": 12, "widget-B": 0, "gadget-X": 3}
    quantity = inventory.get(item, 0)
    if quantity > 0:
        return f"{quantity} units of '{item}' available"
    return f"'{item}' is out of stock"


def tool_place_order(item: str, quantity: int) -> str:
    if quantity < 1:
        raise ValueError("quantity must be at least 1")
    order_id = abs(hash(item)) % 10000
    return f"Order confirmed: {quantity}x '{item}'. Order ID: ORD-{order_id:04d}"


def tool_send_notification(recipient: str, message: str) -> str:
    if not recipient.strip():
        raise ValueError("recipient cannot be empty")
    if not message.strip():
        raise ValueError("message cannot be empty")
    return f"Notification sent to '{recipient}': {message}"


TOOL_HANDLERS: dict[str, Callable[..., str]] = {
    "check_inventory": tool_check_inventory,
    "place_order": tool_place_order,
    "send_notification": tool_send_notification,
}

TOOLS = [
    {
        "name": "check_inventory",
        "description": "Check how many units of an item are currently in stock.",
        "input_schema": {
            "type": "object",
            "properties": {"item": {"type": "string", "description": "Item SKU or name"}},
            "required": ["item"],
        },
    },
    {
        "name": "place_order",
        "description": "Place a purchase order for a given item and quantity.",
        "input_schema": {
            "type": "object",
            "properties": {
                "item": {"type": "string"},
                "quantity": {"type": "integer", "minimum": 1},
            },
            "required": ["item", "quantity"],
        },
    },
    {
        "name": "send_notification",
        "description": "Send a notification message to a recipient such as 'warehouse' or 'manager'.",
        "input_schema": {
            "type": "object",
            "properties": {
                "recipient": {"type": "string"},
                "message": {"type": "string"},
            },
            "required": ["recipient", "message"],
        },
    },
]

SYSTEM_PROMPT = """You are an inventory management assistant with tools to check
stock levels, place orders, and send notifications.

When handling a request:
1. Always verify inventory availability before ordering.
2. Only place an order if stock is confirmed available.
3. Notify the relevant team after completing actions.

Use your tools step-by-step — the order in which you call them matters."""


async def create(client: AsyncAnthropic, messages: list[dict[str, Any]]) -> object:
    return await client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        tools=TOOLS,
        messages=messages,
    )


def run_tool(name: str, tool_input: dict[str, Any]) -> tuple[str, bool]:
    handler = TOOL_HANDLERS.get(name)
    if handler is None:
        return f"Unknown tool: {name}", True

    try:
        return handler(**tool_input), False
    except (TypeError, ValueError) as exc:
        return f"Invalid arguments for {name}: {exc}", True
    except Exception as exc:
        return f"Tool {name} failed: {exc}", True


async def main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    async with AsyncAnthropic(
        api_key=api_key,
        http_client=DefaultAioHttpClient(),
    ) as client:
        user_message = (
            "I need 5 units of widget-A. Check if they're available, "
            "place the order if so, then notify the warehouse."
        )
        print(f"User: {user_message}\n")
        messages: list[dict[str, Any]] = [{"role": "user", "content": user_message}]

        for step in range(1, MAX_ITERATIONS + 1):
            response = await create(client, messages)
            print(f"[Step {step}] stop_reason={response.stop_reason}")

            if response.stop_reason == "end_turn":
                final_text = "".join(
                    block.text
                    for block in response.content
                    if getattr(block, "type", None) == "text"
                ).strip()
                print(f"\nAssistant: {final_text or '(no text)'}")
                return

            if response.stop_reason != "tool_use":
                print(f"Stopping on unexpected stop_reason={response.stop_reason!r}")
                return

            tool_results: list[dict[str, Any]] = []
            for block in response.content:
                if block.type != "tool_use":
                    continue

                print(f"  → Model calls: {block.name}({json.dumps(block.input)})")
                result, is_error = run_tool(block.name, block.input)
                print(f"    Result: {result}")

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

            messages.extend(
                [
                    {"role": "assistant", "content": response.content},
                    {"role": "user", "content": tool_results},
                ]
            )

        print(f"\nReached MAX_ITERATIONS ({MAX_ITERATIONS}) without end_turn — stopping.")


if __name__ == "__main__":
    asyncio.run(main())
