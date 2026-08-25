import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv
import json
from anthropic import AsyncAnthropic

load_dotenv(Path(__file__).parent.parent / ".env")

def tool_magic_eyeball(question):
  import random
  return random.choice(["Yes", "No", "Ask again later"])

model = "claude-haiku-4-5-20251001"

CLASSIFY_PROMPT = """Classify the user's message into exactly one category. Respond with only the category name, nothing else.

Categories:
- FORTUNE: a yes/no fortune-telling question
- OTHER: anything else

User message: {question}"""

async def create(client, messages):
  return await client.messages.create(
      model=model,
      max_tokens=1024,
      messages=messages,
  )

async def main() -> None:
  async with AsyncAnthropic(
      api_key=os.environ.get("ANTHROPIC_API_KEY"),
  ) as client:
    user_message = "Hey Claude, will I be a billionaire living on Mars in 2026?"

    classify_messages = [{"role": "user", "content": CLASSIFY_PROMPT.format(question=user_message)}]
    classify_response = await create(client, classify_messages)
    category = classify_response.content[0].text.strip()
    print(f"classification: {category}")

    # Hardcoded decision tree: the branch taken is decided in code by
    # inspecting the LLM's plain-text output, not by the model choosing a tool.
    if category == "FORTUNE":
      fortune = tool_magic_eyeball(user_message)
      print(f"magic eyeball says: {fortune}")
      messages = [
          {"role": "user", "content": f"{user_message}\n\n(The magic eyeball says: {fortune}. Give the user a fun, in-character answer based on this.)"},
      ]
    else:
      messages = [{"role": "user", "content": user_message}]

    response = await create(client, messages)
    print('final answer:')
    print(json.dumps(response.model_dump(), indent=2))

asyncio.run(main())
