"""Shared model selection and explicit OpenRouter transport configuration."""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from openrouter import OpenRouter


load_dotenv(Path(__file__).with_name(".env"))
DEFAULT_MODEL = "dots-studio/dots-3-note-preview:free"
REQUEST_TIMEOUT_MS = 60_000


def create_model(*, cache=None, require_parameters=False) -> ChatOpenRouter:
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key or api_key == "paste_your_key_here":
        raise ValueError("Set OPENROUTER_API_KEY in the project-root .env file.")
    model_name = os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL).strip()
    if not model_name:
        raise ValueError("OPENROUTER_MODEL must not be empty.")

    # max_retries=0 in integration 0.2.9 leaves the SDK default active.
    # Explicit None disables retries in openrouter SDK 0.11.46.
    sdk = OpenRouter(
        api_key=api_key,
        timeout_ms=REQUEST_TIMEOUT_MS,
        retry_config=None,
    )
    return ChatOpenRouter(
        model=model_name,
        client=sdk,
        temperature=0,
        max_tokens=4096,
        max_retries=0,
        cache=cache,
        openrouter_provider=(
            {"require_parameters": True} if require_parameters else None
        ),
    )
