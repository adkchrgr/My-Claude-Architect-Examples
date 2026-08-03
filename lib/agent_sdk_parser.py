"""Formatting helpers for the messages `claude_agent_sdk.query()` yields.

`query()` streams dataclass instances (`SystemMessage`, `AssistantMessage`,
`UserMessage`, `ResultMessage`, ...) whose default `repr()` is a wall of
nested fields — useful for debugging the SDK itself, but not for reading what
the agent actually did. This module turns each message into a short, labeled
line for a human to skim: raw tool output is unescaped instead of
double-JSON-encoded, long content is truncated, empty blocks and per-token
bookkeeping events are dropped, and IDs are shortened. Pass it a message as
it comes out of the `async for` loop; it does no API work of its own.

`format_message` returns `""` for messages that carry no human-relevant
signal (e.g. `thinking_tokens` deltas) — callers should skip printing empty
results rather than emitting a blank line.

`RunLogger` wraps `format_message` to also persist a copy of the run to
`<project>/logs/<timestamp>.log`, one new file per process.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ResultMessage,
    ServerToolResultBlock,
    ServerToolUseBlock,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

_MAX_CONTENT_CHARS = 1500
_INLINE_JSON_LIMIT = 120


def format_message(message: Any) -> str:
    """Render one message from `query()`/`ClaudeSDKClient` as a readable line.

    Dispatches on type, so subclasses of `SystemMessage` (e.g.
    `TaskStartedMessage`, `HookEventMessage`) fall through to the generic
    system-message branch automatically.
    """
    if isinstance(message, AssistantMessage):
        return _format_assistant(message)
    if isinstance(message, UserMessage):
        return _format_user(message)
    if isinstance(message, ResultMessage):
        return _format_result(message)
    if isinstance(message, SystemMessage):
        return _format_system(message)
    return _format_generic(message)


def _format_system(message: SystemMessage) -> str:
    if message.subtype == "thinking_tokens":
        return ""  # per-delta token counts, not useful to a human reader
    if message.subtype == "init":
        data = message.data or {}
        return (
            f"[system:init] model={data.get('model')} cwd={data.get('cwd')} "
            f"tools={len(data.get('tools', []))}"
        )
    return f"[system:{message.subtype}] {_stringify(message.data)}"


def _format_generic(message: Any) -> str:
    name = type(message).__name__
    if dataclasses.is_dataclass(message):
        return f"[{name}] {_stringify(dataclasses.asdict(message))}"
    return f"[{name}] {message!r}"


def _format_assistant(message: AssistantMessage) -> str:
    body = []
    for block in message.content:
        formatted = _format_block(block)
        if formatted:
            body.append(f"  {formatted}")
    if message.stop_reason:
        body.append(f"  stop_reason: {message.stop_reason}")
    if message.error:
        body.append(f"  error: {message.error}")
    if not body:
        return ""  # e.g. a turn whose only content was an empty thinking block
    return "\n".join([f"[assistant:{message.model}]", *body])


def _format_user(message: UserMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return f"[user] {content}"
    body = []
    for block in content:
        formatted = _format_block(block)
        if formatted:
            body.append(f"  {formatted}")
    if not body:
        return ""
    return "\n".join(["[user]", *body])


def _format_result(message: ResultMessage) -> str:
    status = "error" if message.is_error else "success"
    lines = [
        f"[result:{message.subtype}] {status} "
        f"({message.num_turns} turn(s), {message.duration_ms}ms, "
        f"${message.total_cost_usd or 0:.4f})"
    ]
    if message.result:
        lines.append(f"  {message.result}")
    if message.errors:
        lines.append(f"  errors: {_stringify(message.errors)}")
    return "\n".join(lines)


def _format_block(block: Any) -> str | None:
    if isinstance(block, TextBlock):
        return f"[text] {block.text}" if block.text.strip() else None
    if isinstance(block, ThinkingBlock):
        return f"[thinking] {block.thinking}" if block.thinking.strip() else None
    if isinstance(block, ToolUseBlock):
        return f"[tool_use:{_short_id(block.id)}] {block.name}({_stringify(block.input)})"
    if isinstance(block, ToolResultBlock):
        marker = " [ERROR]" if block.is_error else ""
        return f"[tool_result:{_short_id(block.tool_use_id)}]{marker}{_render_content(block.content)}"
    if isinstance(block, ServerToolUseBlock):
        return f"[server_tool_use:{_short_id(block.id)}] {block.name}({_stringify(block.input)})"
    if isinstance(block, ServerToolResultBlock):
        return f"[server_tool_result:{_short_id(block.tool_use_id)}]{_render_content(block.content)}"
    return f"[{type(block).__name__}] {block!r}"


def _short_id(id_: str | None) -> str:
    return id_[-8:] if id_ else "?"


def _render_content(content: Any) -> str:
    """Render tool-result content, indenting multi-line text under the marker."""
    text = _stringify(content)
    if "\n" not in text:
        return f" {text}"
    indented = "\n".join(f"      {line}" for line in text.splitlines())
    return "\n" + indented


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return _truncate(value)
    if isinstance(value, list) and value and all(
        isinstance(item, dict) and item.get("type") == "text" for item in value
    ):
        return _stringify("\n".join(item.get("text", "") for item in value))
    try:
        compact = json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return _truncate(str(value))
    if len(compact) <= _INLINE_JSON_LIMIT:
        return compact
    try:
        return _truncate(json.dumps(value, ensure_ascii=False, indent=2, default=str))
    except (TypeError, ValueError):
        return _truncate(compact)


def _truncate(text: str, limit: int = _MAX_CONTENT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [{len(text) - limit} more characters truncated]"


class RunLogger:
    """Writes formatted `query()` messages to a per-run log file.

    The log file lives at `<project_dir>/logs/<timestamp>.log`, where
    `project_dir` defaults to the directory of the `__main__` script (i.e.
    the entry-point script that's actually running, such as
    `hello_world/main.py`) rather than the current working directory or
    `lib/`'s own location — so logs land next to the script regardless of
    where it was launched from. Each `RunLogger()` instance starts a new
    file, named after the moment it was created, so every run gets its own
    log.
    """

    def __init__(self, project_dir: Path | str | None = None) -> None:
        base = Path(project_dir) if project_dir is not None else _default_project_dir()
        log_dir = base / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.path = log_dir / f"{_run_timestamp()}.log"
        self._file = self.path.open("a", encoding="utf-8")

    def log(self, message: Any) -> str:
        """Format `message`, append it to this run's log file, and return it."""
        line = format_message(message)
        if line:
            self._file.write(line + "\n")
            self._file.flush()
        return line

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _default_project_dir() -> Path:
    main_module = sys.modules.get("__main__")
    main_file = getattr(main_module, "__file__", None)
    if main_file:
        return Path(main_file).resolve().parent
    return Path.cwd()


def _run_timestamp() -> str:
    # Colons aren't safe in filenames on every filesystem, so use hyphens
    # while keeping the ISO 8601 date/time ordering (sorts chronologically).
    return datetime.now().astimezone().strftime("%Y-%m-%dT%H-%M-%S%z")
