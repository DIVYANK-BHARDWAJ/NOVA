"""Unit tests for execution configuration and token protection."""

import os
from pathlib import Path
import pytest

from execution.config import ConfigValidationError, ExecutionConfig


def test_missing_access_token_raises():
    """Missing or empty BOT_ACCESS_TOKEN must raise ConfigValidationError."""
    with pytest.raises(ConfigValidationError, match="BOT_ACCESS_TOKEN must be provided"):
        ExecutionConfig(bot_access_token="")

    with pytest.raises(ConfigValidationError, match="BOT_ACCESS_TOKEN must be provided"):
        ExecutionConfig(bot_access_token="   ")


def test_valid_config():
    """Valid config instantiates successfully."""
    cfg = ExecutionConfig(
        bot_access_token="test-secret-token-12345",
        api_base_url="http://192.168.65.157:3002",
        http_timeout_seconds=4.0,
        stage_cache_ttl_seconds=5.0,
        stocks_cache_ttl_seconds=30.0,
    )
    assert cfg.api_base_url == "http://192.168.65.157:3002"
    assert cfg.http_timeout_seconds == 4.0
    assert cfg.stage_cache_ttl_seconds == 5.0
    assert cfg.stocks_cache_ttl_seconds == 30.0


def test_credential_redaction_in_repr_and_str():
    """Token must NEVER appear in repr() or str() output."""
    secret = "super-secret-production-token-998877"
    cfg = ExecutionConfig(bot_access_token=secret)

    repr_str = repr(cfg)
    str_str = str(cfg)
    safe_repr = cfg.safe_repr()

    assert secret not in repr_str
    assert secret not in str_str
    assert secret not in safe_repr
    assert "***REDACTED***" in repr_str
    assert "***REDACTED***" in str_str


def test_invalid_base_url():
    """Invalid URL scheme or format must raise ConfigValidationError."""
    with pytest.raises(ConfigValidationError, match="Invalid api_base_url"):
        ExecutionConfig(bot_access_token="valid-token", api_base_url="ftp://invalid-server")

    with pytest.raises(ConfigValidationError, match="Invalid api_base_url"):
        ExecutionConfig(bot_access_token="valid-token", api_base_url="not-a-url")

    with pytest.raises(ConfigValidationError, match="api_base_url cannot be empty"):
        ExecutionConfig(bot_access_token="valid-token", api_base_url="")


def test_invalid_timeouts_and_ttls():
    """Non-positive timeout and negative TTLs must raise ConfigValidationError."""
    with pytest.raises(ConfigValidationError, match="http_timeout_seconds must be positive"):
        ExecutionConfig(bot_access_token="token", http_timeout_seconds=0)

    with pytest.raises(ConfigValidationError, match="http_timeout_seconds must be positive"):
        ExecutionConfig(bot_access_token="token", http_timeout_seconds=-1.0)

    with pytest.raises(ConfigValidationError, match="stage_cache_ttl_seconds must be non-negative"):
        ExecutionConfig(bot_access_token="token", stage_cache_ttl_seconds=-5.0)

    with pytest.raises(ConfigValidationError, match="stocks_cache_ttl_seconds must be non-negative"):
        ExecutionConfig(bot_access_token="token", stocks_cache_ttl_seconds=-5.0)


def test_load_from_env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """load_from_env correctly loads from specified .env file."""
    monkeypatch.delenv("BOT_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("TRADING_API_BASE_URL", raising=False)
    monkeypatch.delenv("API_BASE_URL", raising=False)
    env_file = tmp_path / "custom.env"
    env_file.write_text(
        "BOT_ACCESS_TOKEN=file-token-abc-123\n"
        "TRADING_API_BASE_URL=http://localhost:3002\n"
        "HTTP_TIMEOUT_SECONDS=7.5\n"
    )

    cfg = ExecutionConfig.load_from_env(env_file)
    assert cfg.bot_access_token == "file-token-abc-123"
    assert cfg.api_base_url == "http://localhost:3002"
    assert cfg.http_timeout_seconds == 7.5


def test_load_from_env_missing_token_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """load_from_env raises ConfigValidationError if token is missing."""
    empty_env = tmp_path / "empty.env"
    empty_env.write_text("TRADING_API_BASE_URL=http://localhost:3002\n")

    monkeypatch.delenv("BOT_ACCESS_TOKEN", raising=False)
    with pytest.raises(ConfigValidationError, match="BOT_ACCESS_TOKEN is missing or empty"):
        ExecutionConfig.load_from_env(empty_env)


def test_load_from_env_invalid_timeout_raises(tmp_path: Path):
    """load_from_env raises ConfigValidationError if timeout is not a valid float."""
    bad_env = tmp_path / "bad.env"
    bad_env.write_text("BOT_ACCESS_TOKEN=some-token\nHTTP_TIMEOUT_SECONDS=not-a-number\n")

    with pytest.raises(ConfigValidationError, match="HTTP_TIMEOUT_SECONDS must be a valid float"):
        ExecutionConfig.load_from_env(bad_env)


def test_load_from_env_var_overrides(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Environment variables take precedence over file contents."""
    env_file = tmp_path / "test.env"
    env_file.write_text("BOT_ACCESS_TOKEN=file-token\nHTTP_TIMEOUT_SECONDS=2.0\n")

    monkeypatch.setenv("BOT_ACCESS_TOKEN", "override-env-token")
    monkeypatch.setenv("API_BASE_URL", "http://env-host:4000")
    monkeypatch.setenv("HTTP_TIMEOUT_SECONDS", "8.0")
    monkeypatch.setenv("STAGE_CACHE_TTL_SECONDS", "15.0")
    monkeypatch.setenv("STOCKS_CACHE_TTL_SECONDS", "45.0")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    cfg = ExecutionConfig.load_from_env(env_file)
    assert cfg.bot_access_token == "override-env-token"
    assert cfg.api_base_url == "http://env-host:4000"
    assert cfg.http_timeout_seconds == 8.0
    assert cfg.stage_cache_ttl_seconds == 15.0
    assert cfg.stocks_cache_ttl_seconds == 45.0
    assert cfg.log_level == "DEBUG"


def test_load_from_env_development_stage_and_permitted_stages(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """load_from_env parses ALLOW_DEVELOPMENT_STAGE and PERMITTED_STAGES correctly."""
    env_file = tmp_path / "stages.env"
    env_file.write_text(
        "BOT_ACCESS_TOKEN=file-token\n"
        "ALLOW_DEVELOPMENT_STAGE=false\n"
        "PERMITTED_STAGES=running,competition,round_1\n"
    )

    monkeypatch.delenv("BOT_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("ALLOW_DEVELOPMENT_STAGE", raising=False)
    monkeypatch.delenv("PERMITTED_STAGES", raising=False)

    cfg = ExecutionConfig.load_from_env(env_file)
    assert cfg.allow_development_stage is False
    assert cfg.permitted_stages == ("running", "competition", "round_1")
