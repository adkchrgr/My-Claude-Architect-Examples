"""Basic orchestrator-workers example using Claude for decomposition and aggregation."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from anthropic import AsyncAnthropic
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

COORDINATOR_MODEL = "claude-haiku-4-5-20251001"
WORKER_MODEL = {
    "simple": "claude-haiku-4-5-20251001",
    "complex": "claude-haiku-4-5-20251001",
}
WORKER_SYSTEM = {
    "simple": "You are a fast worker. Answer in 2-3 sentences. No preamble.",
    "complex": (
        "You are a senior analyst. Give a rigorous, structured answer that names "
        "the relevant trade-offs and ends with a clear bottom line."
    ),
}
WORKER_MAX_TOKENS = {"simple": 300, "complex": 1024}
WORKER_MAX_ATTEMPTS = 3
WORKER_RETRY_BASE_DELAY = 1.0
MAX_COORDINATOR_TURNS = 6

TOOLS = [
    {
        "name": "dispatch_worker",
        "description": (
            "Delegate ONE atomic sub-task to a worker. Call this once per sub-task; "
            "emit all independent calls in the same turn. Set complexity to simple "
            "for short factual work and complex for analysis, trade-offs, or judgment."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "subtask": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Self-contained instruction for the worker",
                },
                "complexity": {"type": "string", "enum": ["simple", "complex"]},
            },
            "required": ["subtask", "complexity"],
        },
    }
]

COORDINATOR_SYSTEM = (
    "You are a coordinator agent. When you receive a request:\n"
    "1. Decompose it into the smallest useful set of independent sub-tasks.\n"
    "2. Judge each sub-task's complexity and call dispatch_worker for it. Emit all "
    "independent dispatches in the same turn.\n"
    "3. Once workers report back, aggregate their answers into one coherent response. "
    "Resolve conflicts, drop redundancy, and make an explicit recommendation when "
    "one was requested. If a worker fails, either re-dispatch or clearly note the gap."
)


def validate_dispatch(tool_input: dict[str, Any]) -> tuple[str, str]:
    subtask = tool_input.get("subtask")
    complexity = tool_input.get("complexity")
    if not isinstance(subtask, str) or not subtask.strip():
        raise ValueError("subtask must be a non-empty string")
    if complexity not in WORKER_MODEL:
        raise ValueError(f"unsupported complexity {complexity!r}")
    return subtask.strip(), complexity


async def run_worker(client: AsyncAnthropic, subtask: str, complexity: str) -> str:
    """Run one worker with per-worker retries and exponential backoff."""
    last_error: Exception | None = None

    for attempt in range(1, WORKER_MAX_ATTEMPTS + 1):
        try:
            response = await client.messages.create(
                model=WORKER_MODEL[complexity],
                max_tokens=WORKER_MAX_TOKENS[complexity],
                system=WORKER_SYSTEM[complexity],
                messages=[{"role": "user", "content": subtask}],
            )
            text = "".join(
                block.text
                for block in response.content
                if getattr(block, "type", None) == "text"
            ).strip()
            return text or "(worker returned no text)"
        except Exception as exc:  # demo retries transport/API failures at this boundary
            last_error = exc
            if attempt < WORKER_MAX_ATTEMPTS:
                delay = WORKER_RETRY_BASE_DELAY * (2 ** (attempt - 1))
                print(f"    worker attempt {attempt} failed ({exc!r}); retrying in {delay:.0f}s")
                await asyncio.sleep(delay)

    raise RuntimeError("worker failed after all retry attempts") from last_error


async def create_coordinator(client: AsyncAnthropic, messages: list[dict[str, Any]]) -> object:
    return await client.messages.create(
        model=COORDINATOR_MODEL,
        max_tokens=2048,
        system=COORDINATOR_SYSTEM,
        tools=TOOLS,
        messages=messages,
    )


async def main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    async with AsyncAnthropic(api_key=api_key) as client:
        user_message = (
            "We're choosing a database for a new service that needs high write "
            "throughput, occasional analytical queries, and strong durability. "
            "Compare PostgreSQL, MongoDB, and Cassandra for this workload and recommend one."
        )
        messages: list[dict[str, Any]] = [{"role": "user", "content": user_message}]
        print(f"User: {user_message}\n")

        for coordinator_turn in range(1, MAX_COORDINATOR_TURNS + 1):
            response = await create_coordinator(client, messages)

            if response.stop_reason == "end_turn":
                final_text = "".join(
                    block.text
                    for block in response.content
                    if getattr(block, "type", None) == "text"
                ).strip()
                print("\n[AGGREGATION] coordinator synthesized the worker outputs:\n")
                print(final_text or "(no text)")
                return

            if response.stop_reason != "tool_use":
                print(f"Stopping on unexpected stop_reason={response.stop_reason!r}")
                return

            messages.append({"role": "assistant", "content": response.content})
            dispatches = [block for block in response.content if block.type == "tool_use"]
            if not dispatches:
                print("Coordinator reported tool_use but emitted no tool calls; stopping.")
                return

            print(f"[DECOMPOSITION] coordinator emitted {len(dispatches)} sub-task(s):")
            parsed_dispatches: list[tuple[object, str, str]] = []
            immediate_errors: list[dict[str, Any]] = []

            for block in dispatches:
                if block.name != "dispatch_worker":
                    immediate_errors.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": f"Unknown coordinator tool: {block.name}",
                            "is_error": True,
                        }
                    )
                    continue

                try:
                    subtask, complexity = validate_dispatch(block.input)
                except ValueError as exc:
                    immediate_errors.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": f"Invalid dispatch: {exc}",
                            "is_error": True,
                        }
                    )
                    continue

                print(f"  - ({complexity}) {subtask}")
                parsed_dispatches.append((block, subtask, complexity))

            print("\n[ROUTING] running valid workers in parallel...")
            outputs = await asyncio.gather(
                *(run_worker(client, subtask, complexity) for _, subtask, complexity in parsed_dispatches),
                return_exceptions=True,
            )

            tool_results = list(immediate_errors)
            for (block, subtask, _), output in zip(parsed_dispatches, outputs):
                if isinstance(output, Exception):
                    print(f"  FAILED: {subtask[:70]} -- {output!r}")
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": f"Worker failed after {WORKER_MAX_ATTEMPTS} attempts: {output!r}",
                            "is_error": True,
                        }
                    )
                else:
                    print(f"  done: {subtask[:70]}")
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": output,
                        }
                    )

            messages.append({"role": "user", "content": tool_results})

        print(f"Reached MAX_COORDINATOR_TURNS ({MAX_COORDINATOR_TURNS}) without end_turn; stopping.")


if __name__ == "__main__":
    asyncio.run(main())
