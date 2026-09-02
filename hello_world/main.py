"""Run a small Claude Agent SDK debugging task and log each streamed message."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Shared helper libraries live in lib/ at the repo root, not per-project.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from claude_agent_sdk import ClaudeAgentOptions, query

from lib import agent_sdk_parser


async def main() -> None:
    working_directory = Path(__file__).resolve().parent

    with agent_sdk_parser.RunLogger() as logger:
        print(f"Logging to {logger.path}")
        async for message in query(
            prompt="Find and fix the bug in hello_world.rb",
            options=ClaudeAgentOptions(
                allowed_tools=["Read", "Edit", "Bash"],
                cwd=str(working_directory),
            ),
        ):
            line = logger.log(message)
            if line:
                print(line)


if __name__ == "__main__":
    asyncio.run(main())
