# `lib/`

Shared Python helpers used by more than one project folder in this repo.
Each project folder (`hello_world/`, `stop_reason/`, ...) has its own
`venv` and dependencies, but code in here is plain Python with no
per-project state, so it's kept in one place instead of copy-pasted.

## What's here

- `agent_sdk_parser.py` — turns the message objects `claude_agent_sdk.query()`
  yields (`SystemMessage`, `AssistantMessage`, `UserMessage`, `ResultMessage`,
  and their content blocks) into short, readable log lines instead of raw
  dataclass reprs. Used by `hello_world/main.py`.

## Importing it from a project folder

Every project's entry-point script is one directory below the repo root, so
add these two lines **before** the import, at the top of the script:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import agent_sdk_parser
```

`parents[1]` is "one level up from this file" — i.e. the repo root — so it
works regardless of which project's venv is running the script or what the
current working directory is. See `hello_world/main.py` for a working
example.

No install step needed: the project's own venv doesn't need `lib/`'s
dependencies pre-installed as a package — it just needs whatever `lib/`
imports (e.g. `claude-agent-sdk`) already in that venv's
`requirements.txt`, same as any other import.

## Adding a new module

Drop a new `.py` file in here if the code will be reused by more than one
project. If it's only useful to one project, keep it local to that project
instead (e.g. `stop_reason/lib/sdk_parser.py` is scoped to the `stop_reason`
example for now and hasn't been promoted here).
