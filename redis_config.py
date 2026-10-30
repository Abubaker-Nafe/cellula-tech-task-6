"""Use one Redis endpoint for local commands and the deployed application."""

import os
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv
from redis import Redis


load_dotenv(Path(__file__).with_name(".env"))
DEFAULT_REDIS_URL = "redis://127.0.0.1:6379/0"


def get_redis_url() -> str:
    url = os.getenv("REDIS_URL", DEFAULT_REDIS_URL).strip()
    try:
        parsed = urlsplit(url)
        valid = parsed.scheme in {"redis", "rediss"} and bool(parsed.hostname)
        # Accessing port also validates a supplied port without displaying the URL.
        parsed.port
    except ValueError:
        valid = False
    if not valid:
        raise ValueError(
            "Set REDIS_URL to a valid redis:// or rediss:// connection URL "
            "in Streamlit Secrets or the project-root .env file."
        )
    return url


def create_redis_client(*, decode_responses=False) -> Redis:
    return Redis.from_url(
        get_redis_url(),
        decode_responses=decode_responses,
        socket_connect_timeout=5,
        socket_timeout=5,
    )
