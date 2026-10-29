"""Configuration management for NOVA Trading Execution Adapter.

Loads environment variables, validates connection parameters, and ensures
that access tokens are strictly protected against accidental exposure or logging.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

DEFAULT_API_BASE_URL: str = "http://192.168.65.157:3002"
DEFAULT_HTTP_TIMEOUT_SECONDS: float = 5.0
DEFAULT_STAGE_CACHE_TTL_SECONDS: float = 2.0
DEFAULT_STOCKS_CACHE_TTL_SECONDS: float = 60.0
DEFAULT_PERMITTED_STAGES: tuple[str, ...] = (
    "development",
    "running",
    "live",
    "competition",
    "simulation",
)


class ConfigValidationError(ValueError):
    """Raised when configuration validation fails."""
    pass


@dataclass(frozen=True)
class ExecutionConfig:
    """Immutable execution configuration parameters."""

    bot_access_token: str
    api_base_url: str = DEFAULT_API_BASE_URL
    http_timeout_seconds: float = DEFAULT_HTTP_TIMEOUT_SECONDS
    stage_cache_ttl_seconds: float = DEFAULT_STAGE_CACHE_TTL_SECONDS
    stocks_cache_ttl_seconds: float = DEFAULT_STOCKS_CACHE_TTL_SECONDS
    permitted_stages: tuple[str, ...] = DEFAULT_PERMITTED_STAGES
    allow_development_stage: bool = True
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        """Validate configuration invariants upon instantiation."""
        self.validate()

    def validate(self) -> None:
        """Validate all configuration attributes.

        Raises:
            ConfigValidationError: If any attribute is invalid or missing.
        """
        if not self.bot_access_token or not self.bot_access_token.strip():
            raise ConfigValidationError(
                "BOT_ACCESS_TOKEN must be provided and non-empty. "
                "Ensure BOT_ACCESS_TOKEN is set in your local .env file."
            )

        if not self.api_base_url or not self.api_base_url.strip():
            raise ConfigValidationError("api_base_url cannot be empty.")

        parsed = urlparse(self.api_base_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ConfigValidationError(
                f"Invalid api_base_url '{self.api_base_url}'. Must be a valid HTTP or HTTPS URL."
            )

        if self.http_timeout_seconds <= 0:
            raise ConfigValidationError(
                f"http_timeout_seconds must be positive, got {self.http_timeout_seconds}."
            )

        if self.stage_cache_ttl_seconds < 0:
            raise ConfigValidationError(
                f"stage_cache_ttl_seconds must be non-negative, got {self.stage_cache_ttl_seconds}."
            )

        if self.stocks_cache_ttl_seconds < 0:
            raise ConfigValidationError(
                f"stocks_cache_ttl_seconds must be non-negative, got {self.stocks_cache_ttl_seconds}."
            )

    def safe_repr(self) -> str:
        """Return a representation with sensitive tokens safely masked."""
        token_masked = "***REDACTED***" if self.bot_access_token else "EMPTY"
        return (
            f"ExecutionConfig(api_base_url='{self.api_base_url}', "
            f"bot_access_token='{token_masked}', "
            f"http_timeout_seconds={self.http_timeout_seconds}, "
            f"stage_cache_ttl_seconds={self.stage_cache_ttl_seconds}, "
            f"stocks_cache_ttl_seconds={self.stocks_cache_ttl_seconds}, "
            f"permitted_stages={self.permitted_stages}, "
            f"allow_development_stage={self.allow_development_stage})"
        )

    def __repr__(self) -> str:
        """Protect against token exposure in interactive debugging or logs."""
        return self.safe_repr()

    def __str__(self) -> str:
        """Protect against token exposure in print() or string conversion."""
        return self.safe_repr()

    @classmethod
    def load_from_env(cls, env_path: Optional[Path] = None) -> ExecutionConfig:
        """Load configuration from environment variables and local .env files.

        Search precedence for .env files:
        1. Explicitly provided env_path
        2. Root project .env (./.env)
        3. Kafka module .env (./kafka/.env)

        Args:
            env_path: Optional explicit Path to .env file.

        Returns:
            Validated ExecutionConfig instance.

        Raises:
            ConfigValidationError: If required settings are missing.
        """
        env_vars: dict[str, str] = {}

        candidate_paths: list[Path] = []
        if env_path is not None:
            candidate_paths.append(Path(env_path))
        else:
            candidate_paths.append(Path(".env"))
            candidate_paths.append(Path("kafka/.env"))

        for p in candidate_paths:
            if p.is_file():
                try:
                    from dotenv import dotenv_values
                    loaded = dotenv_values(str(p))
                    for k, v in loaded.items():
                        if k and v is not None and k not in env_vars:
                            env_vars[k] = v
                except ImportError:
                    # Minimal fallback parser if dotenv is not yet installed
                    try:
                        with open(p, "r", encoding="utf-8") as f:
                            for line in f:
                                line = line.strip()
                                if line and not line.startswith("#") and "=" in line:
                                    k, _, v = line.partition("=")
                                    k = k.strip()
                                    v = v.strip().strip("'\"")
                                    if k not in env_vars:
                                        env_vars[k] = v
                    except Exception:
                        pass

        # Environment variables override .env file contents
        for key in (
            "BOT_ACCESS_TOKEN",
            "TRADING_API_BASE_URL",
            "API_BASE_URL",
            "HTTP_TIMEOUT_SECONDS",
            "STAGE_CACHE_TTL_SECONDS",
            "STOCKS_CACHE_TTL_SECONDS",
            "ALLOW_DEVELOPMENT_STAGE",
            "PERMITTED_STAGES",
            "LOG_LEVEL",
        ):
            if key in os.environ and os.environ[key]:
                env_vars[key] = os.environ[key]

        raw_token = env_vars.get("BOT_ACCESS_TOKEN", "").strip()
        if not raw_token:
            raise ConfigValidationError(
                "BOT_ACCESS_TOKEN is missing or empty. Please define BOT_ACCESS_TOKEN "
                "in your environment or local .env file."
            )

        api_url = env_vars.get("TRADING_API_BASE_URL") or env_vars.get("API_BASE_URL") or DEFAULT_API_BASE_URL

        def _parse_float(val: Optional[str], default: float, name: str) -> float:
            if val is None or not val.strip():
                return default
            try:
                parsed_val = float(val.strip())
                return parsed_val
            except ValueError as exc:
                raise ConfigValidationError(f"{name} must be a valid float, got '{val}'") from exc

        http_timeout = _parse_float(env_vars.get("HTTP_TIMEOUT_SECONDS"), DEFAULT_HTTP_TIMEOUT_SECONDS, "HTTP_TIMEOUT_SECONDS")
        stage_ttl = _parse_float(env_vars.get("STAGE_CACHE_TTL_SECONDS"), DEFAULT_STAGE_CACHE_TTL_SECONDS, "STAGE_CACHE_TTL_SECONDS")
        stocks_ttl = _parse_float(env_vars.get("STOCKS_CACHE_TTL_SECONDS"), DEFAULT_STOCKS_CACHE_TTL_SECONDS, "STOCKS_CACHE_TTL_SECONDS")
        log_level = env_vars.get("LOG_LEVEL", "INFO").upper()

        allow_dev_raw = env_vars.get("ALLOW_DEVELOPMENT_STAGE", "true").strip().lower()
        allow_dev = allow_dev_raw in ("true", "1", "yes")

        permitted_stages_raw = env_vars.get("PERMITTED_STAGES")
        if permitted_stages_raw:
            stages_list = tuple(s.strip() for s in permitted_stages_raw.split(",") if s.strip())
            permitted_stages = stages_list if stages_list else DEFAULT_PERMITTED_STAGES
        else:
            permitted_stages = DEFAULT_PERMITTED_STAGES

        return cls(
            bot_access_token=raw_token,
            api_base_url=api_url.strip(),
            http_timeout_seconds=http_timeout,
            stage_cache_ttl_seconds=stage_ttl,
            stocks_cache_ttl_seconds=stocks_ttl,
            permitted_stages=permitted_stages,
            allow_development_stage=allow_dev,
            log_level=log_level,
        )
