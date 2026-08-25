import os
import asyncio
import random
from pathlib import Path
from dotenv import load_dotenv
import json
from anthropic import AsyncAnthropic

load_dotenv(Path(__file__).parent.parent / ".env")

# Two unrelated tools. Nothing in the code below tells Claude which one (if either)
# to use for a given message -- that choice is made by the model at inference time,
# based on the user's question and each tool's "description". That's the "model-driven"
# part of this demo: the decision of *whether* to call a tool, *which* tool, and *how
# many times* lives entirely in the model's output, not in our control flow.

def tool_magic_eyeball(question):
  return random.choice(["Yes", "No", "Ask again later"])

def tool_roll_dice(sides=6):
  return str(random.randint(1, sides))

tools = [
  {
    "name": "magic_eyeball",
    "description": "When the user asks a yes or no fortune telling question call this function",
    "input_schema": {
      "type": "object",
      "properties": {
        "question": {"type": "string"}
      },
      "required": ["question"]
    }
  },
  {
    "name": "roll_dice",
    "description": "When the user wants to roll a die or asks for a random number for luck, call this function",
    "input_schema": {
      "type": "object",
      "properties": {
        "sides": {"type": "integer", "description": "Number of sides on the die, defaults to 6"}
      },
      "required": []
    }
  }
]

# Maps a tool name Claude chooses back to the Python function that implements it.
tool_functions = {
  "magic_eyeball": tool_magic_eyeball,
  "roll_dice": tool_roll_dice,
}

# Claude doesn't need this to figure out which tool to call -- the tool descriptions
# above are already sufficient for that decision. This is here purely to show WHERE
# a system prompt would go if you wanted to add steering on top of that decision
# (tone, constraints, tie-breaking rules between overlapping tools, etc.).
system_prompt = "You are a whimsical fortune-telling assistant. Use the tools available to you as needed to answer the user."

model = "claude-haiku-4-5-20251001"

async def create(client, messages):
  return await client.messages.create(
      model=model,
      max_tokens=1024,
      system=system_prompt,
      tools=tools,
      messages=messages,
  )

async def main() -> None:
  async with AsyncAnthropic(
      api_key=os.environ.get("ANTHROPIC_API_KEY"),
  ) as client:
    user_message = "Hey Claude, will I be a billionaire living on Mars in 2026? Also roll me a 20-sided die for luck."
    messages = [{"role": "user", "content": user_message}]

    response = await create(client, messages)
    print("round 1:")
    print(json.dumps(response.model_dump(), indent=2))

    # Loop for as long as Claude decides it wants to call tools. We don't assume a
    # fixed number of rounds or a fixed set of tools -- we just keep responding to
    # whatever stop_reason and tool_use blocks the model produces.
    while response.stop_reason == "tool_use":
      messages.append({"role": "assistant", "content": response.content})

      tool_results = []
      for block in response.content:
        if block.type != "tool_use":
          continue
        func = tool_functions[block.name]
        result = func(**block.input)
        tool_results.append({
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": result,
        })

      messages.append({"role": "user", "content": tool_results})

      response = await create(client, messages)
      print("follow up:")
      print(json.dumps(response.model_dump(), indent=2))

asyncio.run(main())
