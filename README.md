# Claude Agent Patterns

![CI](https://github.com/adkchrgr/My-Claude-Architect-Examples/actions/workflows/ci.yml/badge.svg)

Runnable Python experiments exploring how Claude-based applications behave beyond a single API request.

I'm building these while working through Claude architecture material. The goal is not simply to complete exercises, but to understand the control flow, failure modes, and engineering tradeoffs involved in building reliable agentic systems.

## Repository Layout

```text
.
├── examples/        # runnable Claude/API architecture examples
├── tests/           # deterministic unit tests
├── pyproject.toml   # pytest + Ruff configuration
├── requirements.txt
├── requirements-dev.txt
└── .env.example     # local configuration template; never contains a real key
```

## What This Repository Covers

| Pattern | Example | Key Concept |
| --- | --- | --- |
| Basic API usage | `examples/anthropic_sdk_test/` | Messages API and model discovery |
| Tool use | `examples/stop_reason/` | `tool_use`, `tool_result`, and `end_turn` |
| External tools | `examples/weather_example.py` | Combining Claude with a real HTTP API |
| Loop control | `examples/end_loop_correctly/` | Stop conditions and max-iteration safeguards |
| Code-driven routing | `examples/decision_making/code-driven.py` | Application controls the branch |
| Model-driven routing | `examples/decision_making/model-driven.py` | Model chooses tools dynamically |
| Coordinator/workers | `examples/coordinator-agent_basic/` | Async worker fan-out and aggregation |
| Task decomposition | `examples/narrow_task_decomposition/` | Validated, bounded decomposition and dispatch |

## Design Principles I'm Exploring

### Explicit control flow
Agentic systems still need deterministic boundaries. I inspect `stop_reason`, limit loops, validate tool calls, and keep application control separate from model decisions where appropriate.

### Failure as data
A failed tool or worker should not necessarily destroy an entire workflow. Where possible, failures are returned to the coordinator with enough context to retry, degrade gracefully, or disclose the missing result.

### Cost-aware routing
Not every task needs the same model. Several examples deliberately separate simple and complex work so model selection can eventually be driven by cost, latency, and reasoning requirements.

### Observable behavior
I prefer examples that expose message history, tool calls, token usage, and stop conditions rather than hiding the agent loop behind an abstraction.

## Setup

```bash
git clone https://github.com/adkchrgr/My-Claude-Architect-Examples.git
cd My-Claude-Architect-Examples
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set your own `ANTHROPIC_API_KEY`. The real `.env` file is excluded from version control.

## Example

```bash
cd examples
python weather_example.py
```

The weather example demonstrates a full tool-use loop using live data from Open-Meteo: Claude requests the tool, Python executes it, the result is appended to the message history, and Claude completes the answer on a later turn.

## Tests and CI

The automated tests focus on deterministic application behavior and external boundaries without making live Anthropic API calls. This keeps CI fast, repeatable, and free of API-key requirements.

Current coverage includes:

- Open-Meteo network and HTTP failure handling
- location disambiguation behavior
- weather input validation
- successful and failed tool dispatch
- malformed or invalid tool arguments
- task-decomposition dispatch validation

Run the same checks locally with:

```bash
pip install -r requirements-dev.txt
ruff check .
pytest -v
```

GitHub Actions runs the lint and test suite automatically on every push and pull request.

## Production Considerations

These are learning implementations rather than production services.

Areas I'm progressively adding include:

- broader automated test coverage
- structured logging
- consistent exception handling
- bounded retries and exponential backoff
- tool-input validation
- explicit loop limits
- configurable model selection
- tracing and evaluation

## Why I'm Building This

My professional background is technical support engineering, where I regularly work from customer symptoms through logs and evidence to a reproducible case.

I'm interested in applying the same discipline to AI systems: observable behavior, testable hypotheses, explicit failure handling, and tooling that turns one-off investigation work into reusable systems.
