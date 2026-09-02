"""Minimal Anthropic SDK smoke tests.

This module intentionally keeps the examples small: one function sends a basic
message and another lists models visible to the configured account. Nothing
runs on import, which makes the module safe to reuse from tests or other code.
"""

from __future__ import annotations

import os
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 1024


def get_client() -> Anthropic:
    """Create an Anthropic client and fail clearly when credentials are missing."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Add it to MCAE-Work/.env or export it."
        )
    return Anthropic(api_key=api_key)


def send_hello(client: Anthropic) -> None:
    """Send a minimal Messages API request and print the returned content."""
    message = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": "Hello, Claude"}],
    )
    print(message.content)


def list_models(client: Anthropic) -> None:
    """Print model IDs available to the configured Anthropic account."""
    page = client.models.list()
    for model in page.data:
        print(model.id)


def main() -> None:
    client = get_client()
    send_hello(client)


if __name__ == "__main__":
    main()
