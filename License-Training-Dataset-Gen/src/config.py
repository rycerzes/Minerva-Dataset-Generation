"""Centralized configuration for the Minerva Dataset Pipeline.

All LLM-related config is loaded from the `.env` file.
No defaults are provided — missing values raise errors at startup.
"""

from __future__ import annotations

import time
import threading
from pathlib import Path
from typing import Optional


class ConfigurationError(Exception):
    """Raised when a required environment variable is missing or invalid."""

    pass


def _parse_env_file(env_path: Path) -> dict[str, str]:
    """Parse a `.env` file into a dict, skipping comments and blank lines."""
    env_vars: dict[str, str] = {}

    if not env_path.exists():
        raise ConfigurationError(
            f".env file not found at {env_path}. "
            "Copy .env.example to .env and fill in all required values."
        )

    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                env_vars[key.strip()] = value.strip()

    return env_vars


def _require(env_vars: dict[str, str], key: str) -> str:
    """Return the value for *key* or raise if missing/empty."""
    value = env_vars.get(key, "").strip()
    if not value:
        raise ConfigurationError(
            f"Required environment variable '{key}' is not set in .env. "
            "See .env.example for the full list of required variables."
        )
    return value


def _require_float(env_vars: dict[str, str], key: str) -> float:
    raw = _require(env_vars, key)
    try:
        return float(raw)
    except ValueError:
        raise ConfigurationError(
            f"Environment variable '{key}' must be a number, got: {raw!r}"
        )


def _require_int(env_vars: dict[str, str], key: str) -> int:
    raw = _require(env_vars, key)
    try:
        return int(raw)
    except ValueError:
        raise ConfigurationError(
            f"Environment variable '{key}' must be an integer, got: {raw!r}"
        )


class LLMConfig:
    """Immutable LLM configuration loaded from `.env`.

    Required `.env` variables:
        LITELLM_API_BASE_URL  — LLM provider base URL
        LITELLM_API_KEY       — API key / token
        LITELLM_MODEL         — Model identifier (e.g. anthropic/claude-3-haiku)
        LITELLM_TEMPERATURE   — Sampling temperature (float)
        LITELLM_MAX_TOKENS    — Max tokens per response (int)
        LITELLM_RPM           — Rate limit in requests per minute (int)
    """

    def __init__(self, env_path: Optional[Path] = None):
        if env_path is None:
            env_path = Path(__file__).parent.parent / ".env"

        env_vars = _parse_env_file(env_path)

        self.api_base_url: str = _require(env_vars, "LITELLM_API_BASE_URL")
        self.api_key: str = _require(env_vars, "LITELLM_API_KEY")
        self.model: str = _require(env_vars, "LITELLM_MODEL")
        self.temperature: float = _require_float(env_vars, "LITELLM_TEMPERATURE")
        self.max_tokens: int = _require_int(env_vars, "LITELLM_MAX_TOKENS")
        self.rpm: int = _require_int(env_vars, "LITELLM_RPM")


class RateLimiter:
    """Thread-safe token-bucket rate limiter.

    Enforces at most `rpm` requests per 60-second window.
    Blocks the calling thread when the limit is exceeded.
    """

    def __init__(self, rpm: int):
        self.rpm = rpm
        self.interval = 60.0 / rpm  # minimum seconds between requests
        self._lock = threading.Lock()
        self._last_call: float = 0.0

    def acquire(self) -> None:
        """Block until a request slot is available."""
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call
            if elapsed < self.interval:
                sleep_time = self.interval - elapsed
                time.sleep(sleep_time)
            self._last_call = time.monotonic()
