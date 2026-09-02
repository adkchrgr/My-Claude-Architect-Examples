"""Formatting helpers for logging Claude Messages API responses.

This module turns the objects returned by the Anthropic Python SDK into
readable log lines so the stop_reason example can narrate exactly what the
model did on each turn. It intentionally does no API work of its own — pass it
a `Message` response (or the content list you are about to append) and it hands
back a formatted string.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Iterable

# Every value the Messages API can put in `response.stop_reason`, paired with a
# one-line explanation. The two the example cares about are `tool_use` (Claude
# paused to call a tool — you must run it and continue) and `end_turn` (Claude
# finished — this is the "end result").
STOP_REASON_MEANINGS: dict[str, str] = {
    "tool_use": "Claude wants to call a tool — run it, append the result, and loop.",
    "end_turn": "Claude finished naturally — this is the final answer (end result).",
    "max_tokens": "Hit the max_tokens cap — output is truncated; raise the limit.",
    "stop_sequence": "Hit a custom stop sequence.",
    "pause_turn": "A long server-tool turn paused — re-send to resume.",
    "refusal": "Claude declined for safety reasons — check stop_details.",
}


def get_logger(name: str = "stop_reason") -> logging.Logger:
    """Return a console logger with a compact, aligned format.

    Safe to call repeatedly — it won't stack duplicate handlers.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def format_stop_reason(stop_reason: str | None) -> str:
    """Render a stop_reason with its meaning, e.g. `tool_use — Claude wants...`."""
    if stop_reason is None:
        return "None — (still streaming or no reason reported)"
    meaning = STOP_REASON_MEANINGS.get(stop_reason, "(unrecognized stop reason)")
    return f"{stop_reason} — {meaning}"


def format_content_blocks(content: Iterable[Any], indent: str = "    ") -> str:
    """Pretty-print the content blocks of a message (text / thinking / tool_use).

    Accepts both SDK block objects (which expose `.type`) and plain dicts, so it
    works on a live response and on the message dicts you build to send back.
    """
    lines: list[str] = []
    for block in content:
        block_type = _get(block, "type")
        if block_type == "text":
            lines.append(f"{indent}[text] {_get(block, 'text')}")
        elif block_type == "thinking":
            lines.append(f"{indent}[thinking] {_get(block, 'thinking')}")
        elif block_type == "tool_use":
            tool_input = json.dumps(_get(block, "input"), ensure_ascii=False)
            lines.append(
                f"{indent}[tool_use] name={_get(block, 'name')} "
                f"id={_get(block, 'id')} input={tool_input}"
            )
        elif block_type == "tool_result":
            lines.append(
                f"{indent}[tool_result] tool_use_id={_get(block, 'tool_use_id')} "
                f"content={_stringify(_get(block, 'content'))}"
            )
        else:
            lines.append(f"{indent}[{block_type}] {_stringify(block)}")
    return "\n".join(lines) if lines else f"{indent}(no content blocks)"


def format_usage(usage: Any) -> str:
    """Render token usage from a response, if present."""
    if usage is None:
        return "usage: (none)"
    inp = _get(usage, "input_tokens")
    out = _get(usage, "output_tokens")
    return f"usage: input={inp} tokens, output={out} tokens"


def format_response(response: Any) -> str:
    """Format a full Message response for logging: stop_reason, blocks, usage."""
    return (
        f"stop_reason: {format_stop_reason(_get(response, 'stop_reason'))}\n"
        f"  content:\n{format_content_blocks(_get(response, 'content') or [])}\n"
        f"  {format_usage(_get(response, 'usage'))}"
    )


def format_tool_result_message(message: dict[str, Any]) -> str:
    """Format a message you are about to append to the conversation history."""
    role = message.get("role", "?")
    content = message.get("content")
    if isinstance(content, str):
        return f"appending {role} message: {content!r}"
    return f"appending {role} message with content:\n{format_content_blocks(content or [])}"


def _get(obj: Any, key: str) -> Any:
    """Read an attribute from an SDK object or a key from a dict."""
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _stringify(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)
