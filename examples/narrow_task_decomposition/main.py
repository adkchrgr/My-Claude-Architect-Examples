"""Coordinator/worker example with bounded turns and validated dispatches."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from anthropic import AsyncAnthropic
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

COORDINATOR_MODEL = "claude-haiku-4-5-20251001"
WORKER_MODEL = {
    "simple": "claude-haiku-4-5-20251001",
    "complex": "claude-haiku-4-5-20251001",
}
WORKER_SYSTEM = {
    "simple": "You are a fast worker. Answer in 2-3 sentences. No preamble.",
    "complex": (
        "You are a senior analyst. Give a rigorous, structured answer that names the "
        "relevant trade-offs and ends with a clear bottom line."
    ),
}
WORKER_MAX_TOKENS = {"simple": 300, "complex": 1024}
WORKER_MAX_ATTEMPTS = 3
WORKER_RETRY_BASE_DELAY = 1.0
MAX_COORDINATOR_TURNS = 6
VALID_COMPLEXITIES = frozenset(WORKER_MODEL)

TOOLS = [
    {
        "name": "dispatch_worker",
        "description": (
            "Delegate one atomic sub-task to a worker. Emit all independent calls in "
            "the same turn. Use simple for short factual work and complex for analysis."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "subtask": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Self-contained instruction for the worker",
                },
                "complexity": {
                    "type": "string",
                    "enum": ["simple", "complex"],
                },
            },
            "required": ["subtask", "complexity"],
            "additionalProperties": False,
        },
    }
]

COORDINATOR_SYSTEM = (
    "You are a coordinator agent. First draft an obvious decomposition, then audit "
    "it for implicit requirements, missing perspectives, failure modes, edge cases, "
    "and dimensions that should be evaluated consistently. Dispatch one atomic "
    "sub-task per tool call, fan out independent work in the same turn, then "
    "synthesize the worker results into one coherent answer. If a worker returns an "
    "error, either retry that sub-task or explicitly note the gap."
)


def get_api_key() -> str:
    """Return the configured API key or fail with a clear setup message."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return api_key


def validate_dispatch(tool_input: Any) -> tuple[str, str]:
    """Validate one coordinator dispatch before handing it to a worker."""
    if not isinstance(tool_input, dict):
        raise ValueError("dispatch input must be an object")

    subtask = tool_input.get("subtask")
    complexity = tool_input.get("complexity")

    if not isinstance(subtask, str) or not subtask.strip():
        raise ValueError("subtask must be a non-empty string")
    if complexity not in VALID_COMPLEXITIES:
        raise ValueError("complexity must be 'simple' or 'complex'")

    return subtask.strip(), complexity


def extract_text(response: Any) -> str:
    """Join text blocks from an Anthropic response."""
    return "".join(block.text for block in response.content if hasattr(block, "text"))


async def run_worker(client: AsyncAnthropic, subtask: str, complexity: str) -> str:
    """Run one worker with bounded exponential-backoff retries."""
    last_error: Exception | None = None

    for attempt in range(1, WORKER_MAX_ATTEMPTS + 1):
        try:
            response = await client.messages.create(
                model=WORKER_MODEL[complexity],
                max_tokens=WORKER_MAX_TOKENS[complexity],
                system=WORKER_SYSTEM[complexity],
                messages=[{"role": "user", "content": subtask}],
            )
            return extract_text(response)
        except Exception as exc:  # noqa: BLE001 - worker failures become coordinator data
            last_error = exc
            if attempt < WORKER_MAX_ATTEMPTS:
                delay = WORKER_RETRY_BASE_DELAY * (2 ** (attempt - 1))
                print(
                    f"    worker attempt {attempt} failed ({exc!r}); "
                    f"retrying in {delay:.0f}s"
                )
                await asyncio.sleep(delay)

    raise RuntimeError("worker failed after all retry attempts") from last_error


async def create_coordinator_response(client: AsyncAnthropic, messages: list[dict[str, Any]]):
    """Create one coordinator turn."""
    return await client.messages.create(
        model=COORDINATOR_MODEL,
        max_tokens=2048,
        system=COORDINATOR_SYSTEM,
        tools=TOOLS,
        messages=messages,
    )


async def run_example(client: AsyncAnthropic, user_message: str) -> str:
    """Run the bounded coordinator loop and return the final synthesized text."""
    messages: list[dict[str, Any]] = [{"role": "user", "content": user_message}]

    for turn in range(1, MAX_COORDINATOR_TURNS + 1):
        response = await create_coordinator_response(client, messages)

        if response.stop_reason == "end_turn":
            return extract_text(response) or "(no text)"

        if response.stop_reason != "tool_use":
            raise RuntimeError(f"unexpected coordinator stop_reason: {response.stop_reason!r}")

        messages.append({"role": "assistant", "content": response.content})
        dispatches = [block for block in response.content if block.type == "tool_use"]
        if not dispatches:
            raise RuntimeError("coordinator returned tool_use without any tool blocks")

        print(f"[TURN {turn}] coordinator dispatched {len(dispatches)} sub-task(s):")

        validated: list[tuple[Any, str | None, str | None, str | None]] = []
        for block in dispatches:
            if block.name != "dispatch_worker":
                validated.append((block, None, None, f"unknown tool: {block.name}"))
                continue
            try:
                subtask, complexity = validate_dispatch(block.input)
            except ValueError as exc:
                validated.append((block, None, None, str(exc)))
                continue

            print(f"  - ({complexity}) {subtask}")
            validated.append((block, subtask, complexity, None))

        worker_tasks = [
            run_worker(client, subtask, complexity)
            for _, subtask, complexity, error in validated
            if error is None and subtask is not None and complexity is not None
        ]
        worker_outputs = await asyncio.gather(*worker_tasks, return_exceptions=True)
        output_iter = iter(worker_outputs)

        tool_results: list[dict[str, Any]] = []
        for block, subtask, _, validation_error in validated:
            if validation_error is not None:
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": f"Invalid dispatch: {validation_error}",
                        "is_error": True,
                    }
                )
                continue

            output = next(output_iter)
            if isinstance(output, Exception):
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": f"Worker failed: {output}",
                        "is_error": True,
                    }
                )
                print(f"  FAILED: {(subtask or '')[:70]} -- {output!r}")
            else:
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": output,
                    }
                )
                print(f"  done: {(subtask or '')[:70]}")

        messages.append({"role": "user", "content": tool_results})

    raise RuntimeError(f"coordinator exceeded {MAX_COORDINATOR_TURNS} turns")


async def main() -> None:
    user_message = (
        "We're choosing a database for a new service that needs high write throughput, "
        "occasional analytical queries, and strong durability. Compare PostgreSQL, "
        "MongoDB, and Cassandra for this workload and recommend one."
    )
    print(f"User: {user_message}\n")

    async with AsyncAnthropic(api_key=get_api_key()) as client:
        final_text = await run_example(client, user_message)

    print("\n[AGGREGATION] coordinator synthesized the worker outputs:\n")
    print(final_text)


if __name__ == "__main__":
    asyncio.run(main())
