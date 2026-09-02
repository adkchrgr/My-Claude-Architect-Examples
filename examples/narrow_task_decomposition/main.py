import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from anthropic import AsyncAnthropic

load_dotenv(Path(__file__).parent.parent / ".env")

# Coordinator agent (orchestrator-workers pattern).
#
# The coordinator model does three things, and every one of those decisions
# lives in the model's output -- not in our control flow:
#
#   1. DECOMPOSITION            -- draft sub-tasks, then AUDIT the draft for the
#                                  angles a first pass misses, then add sub-tasks
#                                  for the gaps before delegating
#   2. COMPLEXITY + ROUTING     -- tag each sub-task simple/complex; Python routes
#                                  it to a cheap fast worker or a deliberate one
#   3. AGGREGATION              -- synthesize the workers' answers into one reply
#
# The only thing that lives in code is the routing table below.
#
# Why the audit step: "split the request into sub-tasks" only ever splits what
# the request literally says. The coordinator never pins down vague requirements
# ("high write throughput" -- how many writes/sec?), never scores each option on
# a shared rubric, and never reaches dimensions the user didn't name (operational
# burden, cost at scale, migration risk). The draft -> audit -> add loop, plus a
# domain checklist of commonly-missed angles, is what turns "split the request"
# into "work out the whole problem."

COORDINATOR_MODEL = "claude-haiku-4-5-20251001"

# The "complex" lane would point at a stronger model (e.g. Sonnet) in production;
# kept on Haiku here to keep costs low.
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

# Per-spoke resilience: each worker call is retried on failure with exponential
# backoff before the hub gives up on that spoke.
WORKER_MAX_ATTEMPTS = 3
WORKER_RETRY_BASE_DELAY = 1.0  # seconds; doubled each attempt

# The coordinator's single tool. It decomposes the request in its own reasoning,
# then calls this once per sub-task -- fanning out multiple calls in one turn for
# independent sub-tasks. `complexity` is the model's routing decision.
tools = [
  {
    "name": "dispatch_worker",
    "description": (
      "Delegate ONE atomic sub-task to a worker. Call this once per sub-task you "
      "have broken the request into; emit all calls for independent sub-tasks in "
      "the same turn to fan out. Set complexity to 'simple' for lookups or short "
      "factual answers, or 'complex' for anything needing analysis, trade-offs, "
      "or judgment."
    ),
    "input_schema": {
      "type": "object",
      "properties": {
        "subtask": {"type": "string", "description": "Self-contained instruction for the worker"},
        "complexity": {"type": "string", "enum": ["simple", "complex"]},
      },
      "required": ["subtask", "complexity"],
    },
  }
]

COORDINATOR_SYSTEM = (
  "You are a coordinator agent. When you receive a request:\n"
  "\n"
  "1. DRAFT a decomposition: list the sub-tasks that obviously follow from the "
  "request as stated.\n"
  "\n"
  "2. AUDIT the draft before delegating. Ask yourself:\n"
  "   - What implicit requirements or constraints has the user assumed but not "
  "spelled out? Add a sub-task to pin each one down.\n"
  "   - What perspectives or stakeholders does this decision affect that the "
  "draft doesn't cover?\n"
  "   - What risks, failure modes, or edge cases deserve their own investigation?\n"
  "   - What am I treating as one lump that should be split -- e.g. a comparison "
  "where each option needs the same dimensions evaluated?\n"
  "   - What would a domain expert check that a first pass would miss?\n"
  "   Add a sub-task for every gap you find.\n"
  "\n"
  "   For infrastructure- or tool-selection decisions specifically, make sure the "
  "plan covers: the concrete numbers behind each vague requirement; every "
  "candidate scored on the SAME dimensions; operational burden (backups, scaling, "
  "upgrades, on-call); total cost at the target scale; team familiarity; "
  "migration effort and lock-in risk; and any place where requirements conflict.\n"
  "\n"
  "3. Judge each sub-task's complexity and call dispatch_worker for it. Emit all "
  "dispatch_worker calls for independent sub-tasks in the same turn.\n"
  "\n"
  "4. Once every worker has reported back, aggregate their answers into a single "
  "coherent response for the user -- resolve conflicts, drop redundancy, and make "
  "an explicit recommendation when one was asked for. If a worker returned an "
  "error, either re-dispatch that sub-task or proceed and note the gap explicitly."
)


async def run_worker(client, subtask: str, complexity: str) -> str:
  """Run one spoke, retrying that spoke alone on failure. Raises if every
  attempt fails -- the caller decides how to degrade."""
  last_error = None
  for attempt in range(1, WORKER_MAX_ATTEMPTS + 1):
    try:
      response = await client.messages.create(
        model=WORKER_MODEL[complexity],
        max_tokens=WORKER_MAX_TOKENS[complexity],
        system=WORKER_SYSTEM[complexity],
        messages=[{"role": "user", "content": subtask}],
      )
      return "".join(b.text for b in response.content if hasattr(b, "text"))
    except Exception as e:  # noqa: BLE001 -- retry any transient failure
      last_error = e
      if attempt < WORKER_MAX_ATTEMPTS:
        delay = WORKER_RETRY_BASE_DELAY * (2 ** (attempt - 1))
        print(f"    worker attempt {attempt} failed ({e!r}); retrying in {delay:.0f}s")
        await asyncio.sleep(delay)
  raise last_error


async def create(client, messages):
  return await client.messages.create(
    model=COORDINATOR_MODEL,
    max_tokens=2048,
    system=COORDINATOR_SYSTEM,
    tools=tools,
    messages=messages,
  )


async def main() -> None:
  async with AsyncAnthropic(
      api_key=os.environ.get("ANTHROPIC_API_KEY"),
  ) as client:
    user_message = (
      "We're choosing a database for a new service that needs high write "
      "throughput, occasional analytical queries, and strong durability. "
      "Compare PostgreSQL, MongoDB, and Cassandra for this workload and "
      "recommend one."
    )
    messages = [{"role": "user", "content": user_message}]
    print(f"User: {user_message}\n")

    response = await create(client, messages)

    while response.stop_reason == "tool_use":
      messages.append({"role": "assistant", "content": response.content})

      dispatches = [b for b in response.content if b.type == "tool_use"]

      print(f"[DECOMPOSITION] coordinator split the request into {len(dispatches)} sub-task(s):")
      for b in dispatches:
        print(f"  - ({b.input['complexity']}) {b.input['subtask']}")

      print("\n[ROUTING] running workers in parallel...")
      # return_exceptions=True so one failed spoke doesn't sink the whole batch;
      # the hub still gets every other worker's result and reports the failure
      # back to the coordinator as a tool error.
      worker_outputs = await asyncio.gather(*(
        run_worker(client, b.input["subtask"], b.input["complexity"])
        for b in dispatches
      ), return_exceptions=True)

      tool_results = []
      for b, output in zip(dispatches, worker_outputs):
        if isinstance(output, Exception):
          print(f"  FAILED: {b.input['subtask'][:70]} -- {output!r}")
          tool_results.append({
            "type": "tool_result",
            "tool_use_id": b.id,
            "content": f"Worker failed after {WORKER_MAX_ATTEMPTS} attempts: {output!r}",
            "is_error": True,
          })
        else:
          print(f"  done: {b.input['subtask'][:70]}")
          tool_results.append({
            "type": "tool_result",
            "tool_use_id": b.id,
            "content": output,
          })

      messages.append({"role": "user", "content": tool_results})
      response = await create(client, messages)

    print("\n[AGGREGATION] coordinator synthesized the worker outputs:\n")
    final_text = "".join(b.text for b in response.content if hasattr(b, "text"))
    print(final_text or "(no text)")


asyncio.run(main())
