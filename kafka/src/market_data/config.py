"""Kafka Ingestion Configuration for MANTHAN 2026.

Handles loading, validating, and formatting Kafka client configurations.
Guarantees default consumer group ID 'group-mn2651' and rejects invalid settings.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
    _DOTENV_AVAILABLE = True
except ImportError:  # pragma: no cover
    _DOTENV_AVAILABLE = False


class ConfigValidationError(ValueError):
    """Raised when configuration values are missing, invalid, or malformed."""
    pass


SUPPORTED_SECURITY_PROTOCOLS = {
    "PLAINTEXT",
    "SSL",
    "SASL_PLAINTEXT",
    "SASL_SSL",
}

SUPPORTED_AUTO_OFFSET_RESET = {
    "latest",
    "earliest",
    "none",
}

# Documented competition constants
DEFAULT_BOOTSTRAP_SERVERS = "<SERVER_IP>:9094"
DEFAULT_SECURITY_PROTOCOL = "PLAINTEXT"
DEFAULT_GROUP_ID = "group-mn2651"
DEFAULT_TOPIC_TICKS = "market.ticks"
DEFAULT_TOPIC_NEWS = "market.news"
DEFAULT_AUTO_OFFSET_RESET = "latest"


@dataclass(frozen=True)
class KafkaConfig:
    """Immutable Kafka configuration representation with built-in validation."""

    bootstrap_servers: str = DEFAULT_BOOTSTRAP_SERVERS
    security_protocol: str = DEFAULT_SECURITY_PROTOCOL
    group_id: str = DEFAULT_GROUP_ID
    topic_ticks: str = DEFAULT_TOPIC_TICKS
    topic_news: str = DEFAULT_TOPIC_NEWS
    auto_offset_reset: str = DEFAULT_AUTO_OFFSET_RESET
    enable_auto_commit: bool = True
    auto_commit_interval_ms: int = 5000
    session_timeout_ms: int = 45000
    max_poll_interval_ms: int = 300000
    log_level: str = "INFO"

    def is_placeholder_broker(self) -> bool:
        """Check if bootstrap_servers contains placeholder IP indicators."""
        clean = self.bootstrap_servers.strip()
        return "<SERVER_IP>" in clean or "SERVER_IP" in clean or clean.startswith("<")

    def validate(self) -> None:
        """Validate configuration settings.

        Raises:
            ConfigValidationError: If any critical configuration setting is invalid.
        """
        errors: list[str] = []

        if not self.bootstrap_servers or not self.bootstrap_servers.strip():
            errors.append("KAFKA_BOOTSTRAP_SERVERS cannot be empty.")

        if not self.group_id or not self.group_id.strip():
            errors.append("KAFKA_GROUP_ID cannot be empty. Must match assigned 'group-${username}' (e.g. group-mn2651).")

        protocol_upper = self.security_protocol.upper()
        if protocol_upper not in SUPPORTED_SECURITY_PROTOCOLS:
            errors.append(
                f"KAFKA_SECURITY_PROTOCOL '{self.security_protocol}' is unsupported. "
                f"Expected one of: {', '.join(sorted(SUPPORTED_SECURITY_PROTOCOLS))}."
            )

        offset_lower = self.auto_offset_reset.lower()
        if offset_lower not in SUPPORTED_AUTO_OFFSET_RESET:
            errors.append(
                f"KAFKA_AUTO_OFFSET_RESET '{self.auto_offset_reset}' is invalid. "
                f"Expected one of: {', '.join(sorted(SUPPORTED_AUTO_OFFSET_RESET))}."
            )

        if not self.topic_ticks or not self.topic_ticks.strip():
            errors.append("KAFKA_TOPIC_TICKS cannot be empty.")

        if not self.topic_news or not self.topic_news.strip():
            errors.append("KAFKA_TOPIC_NEWS cannot be empty.")

        if self.auto_commit_interval_ms <= 0:
            errors.append("KAFKA_AUTO_COMMIT_INTERVAL_MS must be greater than 0.")

        if self.session_timeout_ms <= 0:
            errors.append("KAFKA_SESSION_TIMEOUT_MS must be greater than 0.")

        if self.max_poll_interval_ms <= 0:
            errors.append("KAFKA_MAX_POLL_INTERVAL_MS must be greater than 0.")

        if self.session_timeout_ms >= self.max_poll_interval_ms:
            errors.append(
                f"KAFKA_SESSION_TIMEOUT_MS ({self.session_timeout_ms}) must be strictly less than "
                f"KAFKA_MAX_POLL_INTERVAL_MS ({self.max_poll_interval_ms}) to prevent consumer timeouts."
            )

        if errors:
            raise ConfigValidationError("; ".join(errors))

    def to_confluent_kafka_dict(self) -> dict[str, Any]:
        """Convert to confluent-kafka compatible configuration dictionary."""
        self.validate()
        return {
            "bootstrap.servers": self.bootstrap_servers,
            "group.id": self.group_id,
            "security.protocol": self.security_protocol.upper(),
            "auto.offset.reset": self.auto_offset_reset.lower(),
            "enable.auto.commit": self.enable_auto_commit,
            "auto.commit.interval.ms": self.auto_commit_interval_ms,
            "session.timeout.ms": self.session_timeout_ms,
            "max.poll.interval.ms": self.max_poll_interval_ms,
        }

    def safe_repr(self) -> str:
        """Return safe string representation masking potential credentials."""
        return (
            f"KafkaConfig(bootstrap_servers='{self.bootstrap_servers}', "
            f"security_protocol='{self.security_protocol}', "
            f"group_id='{self.group_id}', "
            f"topic_ticks='{self.topic_ticks}', "
            f"topic_news='{self.topic_news}', "
            f"auto_offset_reset='{self.auto_offset_reset}', "
            f"enable_auto_commit={self.enable_auto_commit})"
        )

    def __repr__(self) -> str:
        return self.safe_repr()

    @classmethod
    def load_from_env(
        cls,
        env_path: Path | str | None = None,
        load_env_file: bool = True,
    ) -> KafkaConfig:
        """Load configuration from environment variables, optionally loading a .env file."""
        if load_env_file:
            if env_path is not None:
                path_obj = Path(env_path)
                if path_obj.is_file() and _DOTENV_AVAILABLE:
                    load_dotenv(dotenv_path=path_obj, override=False)
            else:
                # Check default locations: current dir .env or kafka/.env
                if _DOTENV_AVAILABLE:
                    kafka_env = Path("kafka/.env")
                    root_env = Path(".env")
                    if kafka_env.is_file():
                        load_dotenv(dotenv_path=kafka_env, override=False)
                    elif root_env.is_file():
                        load_dotenv(dotenv_path=root_env, override=False)

        bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", DEFAULT_BOOTSTRAP_SERVERS)
        security_protocol = os.getenv("KAFKA_SECURITY_PROTOCOL", DEFAULT_SECURITY_PROTOCOL)
        group_id = os.getenv("KAFKA_GROUP_ID", DEFAULT_GROUP_ID)
        topic_ticks = os.getenv("KAFKA_TOPIC_TICKS", DEFAULT_TOPIC_TICKS)
        topic_news = os.getenv("KAFKA_TOPIC_NEWS", DEFAULT_TOPIC_NEWS)
        auto_offset_reset = os.getenv("KAFKA_AUTO_OFFSET_RESET", DEFAULT_AUTO_OFFSET_RESET)
        log_level = os.getenv("LOG_LEVEL", "INFO")

        # Parse boolean enable_auto_commit if explicitly configured
        raw_auto_commit = os.getenv("KAFKA_ENABLE_AUTO_COMMIT")
        if raw_auto_commit is not None:
            enable_auto_commit = raw_auto_commit.strip().lower() in ("true", "1", "yes")
        else:
            enable_auto_commit = True

        # Parse integer timing settings with strict validation
        def _parse_int_env(key: str, default: int) -> int:
            val = os.getenv(key)
            if val is None or not val.strip():
                return default
            try:
                return int(val.strip())
            except ValueError as exc:
                raise ConfigValidationError(f"{key} must be a valid integer, received {val!r}") from exc

        auto_commit_interval_ms = _parse_int_env("KAFKA_AUTO_COMMIT_INTERVAL_MS", 5000)
        session_timeout_ms = _parse_int_env("KAFKA_SESSION_TIMEOUT_MS", 45000)
        max_poll_interval_ms = _parse_int_env("KAFKA_MAX_POLL_INTERVAL_MS", 300000)


        config = cls(
            bootstrap_servers=bootstrap_servers.strip(),
            security_protocol=security_protocol.strip(),
            group_id=group_id.strip(),
            topic_ticks=topic_ticks.strip(),
            topic_news=topic_news.strip(),
            auto_offset_reset=auto_offset_reset.strip(),
            enable_auto_commit=enable_auto_commit,
            auto_commit_interval_ms=auto_commit_interval_ms,
            session_timeout_ms=session_timeout_ms,
            max_poll_interval_ms=max_poll_interval_ms,
            log_level=log_level.strip().upper(),
        )
        return config

