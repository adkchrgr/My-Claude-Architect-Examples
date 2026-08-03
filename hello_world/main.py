import asyncio
import sys
from pathlib import Path

# Shared helper libraries live in lib/ at the repo root, not per-project.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from claude_agent_sdk import query, ClaudeAgentOptions

from lib import agent_sdk_parser


async def main():
    with agent_sdk_parser.RunLogger() as logger:
        print(f"Logging to {logger.path}")
        async for message in query(
            prompt="Find and fix the bug in hello_world.rb",
            options=ClaudeAgentOptions(
                allowed_tools=["Read", "Edit", "Bash"],
                cwd=str(Path(__file__).resolve().parent),
            ),
        ):
            line = logger.log(message)
            if line:
                print(line)


asyncio.run(main())