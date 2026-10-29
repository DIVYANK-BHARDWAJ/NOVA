"""Tests for Kafka configuration loading, validation, and defaults."""

import os
from unittest.mock import patch
import pytest

from market_data.config import (
    DEFAULT_AUTO_OFFSET_RESET,
    DEFAULT_BOOTSTRAP_SERVERS,
    DEFAULT_GROUP_ID,
    DEFAULT_SECURITY_PROTOCOL,
    DEFAULT_TOPIC_NEWS,
    DEFAULT_TOPIC_TICKS,
    ConfigValidationError,
    KafkaConfig,
)


def test_default_config_values():
    """Verify default values match MANTHAN 2026 documented competition requirements."""
    config = KafkaConfig()
    assert config.group_id == "group-mn2651"
    assert config.group_id == DEFAULT_GROUP_ID
    assert config.bootstrap_servers == "<SERVER_IP>:9094"
    assert config.bootstrap_servers == DEFAULT_BOOTSTRAP_SERVERS
    assert config.security_protocol == "PLAINTEXT"
    assert config.security_protocol == DEFAULT_SECURITY_PROTOCOL
    assert config.topic_ticks == "market.ticks"
    assert config.topic_ticks == DEFAULT_TOPIC_TICKS
    assert config.topic_news == "market.news"
    assert config.topic_news == DEFAULT_TOPIC_NEWS
    assert config.auto_offset_reset == "latest"
    assert config.auto_offset_reset == DEFAULT_AUTO_OFFSET_RESET


def test_load_from_env_defaults():
    """Verify loading from clean environment uses the required defaults."""
    with patch.dict(os.environ, {}, clear=True):
        config = KafkaConfig.load_from_env(load_env_file=False)
        assert config.group_id == "group-mn2651"
        assert config.bootstrap_servers == "<SERVER_IP>:9094"
        assert config.security_protocol == "PLAINTEXT"
        assert config.topic_ticks == "market.ticks"
        assert config.topic_news == "market.news"
        assert config.auto_offset_reset == "latest"


def test_load_from_env_custom():
    """Verify configured environment variables override defaults."""
    env_vars = {
        "KAFKA_BOOTSTRAP_SERVERS": "10.0.1.100:9094",
        "KAFKA_SECURITY_PROTOCOL": "PLAINTEXT",
        "KAFKA_GROUP_ID": "group-mn2651-custom",
        "KAFKA_TOPIC_TICKS": "custom.ticks",
        "KAFKA_TOPIC_NEWS": "custom.news",
        "KAFKA_AUTO_OFFSET_RESET": "earliest",
        "LOG_LEVEL": "DEBUG",
    }
    with patch.dict(os.environ, env_vars, clear=True):
        config = KafkaConfig.load_from_env(load_env_file=False)
        assert config.bootstrap_servers == "10.0.1.100:9094"
        assert config.group_id == "group-mn2651-custom"
        assert config.topic_ticks == "custom.ticks"
        assert config.topic_news == "custom.news"
        assert config.auto_offset_reset == "earliest"
        assert config.log_level == "DEBUG"


def test_placeholder_detection():
    """Verify placeholder detection identifies documented placeholder IP."""
    cfg_placeholder = KafkaConfig(bootstrap_servers="<SERVER_IP>:9094")
    assert cfg_placeholder.is_placeholder_broker() is True

    cfg_custom_placeholder = KafkaConfig(bootstrap_servers="SERVER_IP:9094")
    assert cfg_custom_placeholder.is_placeholder_broker() is True

    cfg_real = KafkaConfig(bootstrap_servers="192.168.1.10:9094")
    assert cfg_real.is_placeholder_broker() is False


@pytest.mark.parametrize(
    "invalid_kwargs,error_substr",
    [
        ({"bootstrap_servers": ""}, "KAFKA_BOOTSTRAP_SERVERS cannot be empty"),
        ({"group_id": "   "}, "KAFKA_GROUP_ID cannot be empty"),
        ({"security_protocol": "INVALID_PROTO"}, "unsupported"),
        ({"auto_offset_reset": "invalid_offset"}, "invalid"),
        ({"topic_ticks": ""}, "KAFKA_TOPIC_TICKS cannot be empty"),
        ({"topic_news": ""}, "KAFKA_TOPIC_NEWS cannot be empty"),
    ],
)
def test_validation_errors(invalid_kwargs, error_substr):
    """Verify invalid configuration options raise ConfigValidationError with actionable message."""
    cfg = KafkaConfig(**invalid_kwargs)
    with pytest.raises(ConfigValidationError) as exc_info:
        cfg.validate()
    assert error_substr in str(exc_info.value)


def test_to_confluent_kafka_dict():
    """Verify dictionary conversion matches librdkafka / confluent-kafka property names."""
    cfg = KafkaConfig(
        bootstrap_servers="10.10.10.1:9094",
        group_id="group-mn2651",
        security_protocol="PLAINTEXT",
        auto_offset_reset="latest",
    )
    kafka_dict = cfg.to_confluent_kafka_dict()
    assert kafka_dict["bootstrap.servers"] == "10.10.10.1:9094"
    assert kafka_dict["group.id"] == "group-mn2651"
    assert kafka_dict["security.protocol"] == "PLAINTEXT"
    assert kafka_dict["auto.offset.reset"] == "latest"
    assert kafka_dict["enable.auto.commit"] is True


def test_safe_repr_does_not_reveal_secrets():
    """Verify string representation is safe."""
    cfg = KafkaConfig()
    repr_str = repr(cfg)
    assert "group-mn2651" in repr_str
    assert "KafkaConfig" in repr_str


def test_env_group_id_reaches_confluent_kafka_client():
    """Verify explicitly configured KAFKA_GROUP_ID reaches confluent-kafka dictionary."""
    custom_group = "group-custom-test"
    with patch.dict(os.environ, {"KAFKA_GROUP_ID": custom_group}, clear=True):
        cfg = KafkaConfig.load_from_env(load_env_file=False)
        kafka_dict = cfg.to_confluent_kafka_dict()
        assert kafka_dict["group.id"] == custom_group


def test_broker_address_from_config():
    """Verify broker address is loaded from KAFKA_BOOTSTRAP_SERVERS environment variable."""
    custom_broker = "192.168.65.157:9094"
    with patch.dict(os.environ, {"KAFKA_BOOTSTRAP_SERVERS": custom_broker}, clear=True):
        cfg = KafkaConfig.load_from_env(load_env_file=False)
        assert cfg.bootstrap_servers == custom_broker
        kafka_dict = cfg.to_confluent_kafka_dict()
        assert kafka_dict["bootstrap.servers"] == custom_broker
        assert cfg.topic_ticks == "market.ticks"
        assert cfg.topic_news == "market.news"


def test_load_from_env_auto_commit_and_timeouts():
    """Verify loading KAFKA_ENABLE_AUTO_COMMIT and timeout values from environment."""
    custom_env = {
        "KAFKA_ENABLE_AUTO_COMMIT": "false",
        "KAFKA_AUTO_COMMIT_INTERVAL_MS": "2500",
        "KAFKA_SESSION_TIMEOUT_MS": "30000",
        "KAFKA_MAX_POLL_INTERVAL_MS": "150000",
    }
    with patch.dict(os.environ, custom_env, clear=True):
        cfg = KafkaConfig.load_from_env(load_env_file=False)
        assert cfg.enable_auto_commit is False
        assert cfg.auto_commit_interval_ms == 2500
        assert cfg.session_timeout_ms == 30000
        assert cfg.max_poll_interval_ms == 150000

        kafka_dict = cfg.to_confluent_kafka_dict()
        assert kafka_dict["enable.auto.commit"] is False
        assert kafka_dict["auto.commit.interval.ms"] == 2500
        assert kafka_dict["session.timeout.ms"] == 30000
        assert kafka_dict["max.poll.interval.ms"] == 150000


def test_timeout_and_interval_validations():
    """Verify invalid timeouts and session >= max_poll raise ConfigValidationError."""
    # 1. session_timeout >= max_poll_interval
    cfg_bad_ratio = KafkaConfig(session_timeout_ms=50000, max_poll_interval_ms=40000)
    with pytest.raises(ConfigValidationError) as exc:
        cfg_bad_ratio.validate()
    assert "must be strictly less than" in str(exc.value)

    # 2. Non-positive interval
    cfg_bad_interval = KafkaConfig(auto_commit_interval_ms=0)
    with pytest.raises(ConfigValidationError) as exc:
        cfg_bad_interval.validate()
    assert "KAFKA_AUTO_COMMIT_INTERVAL_MS must be greater than 0" in str(exc.value)


def test_load_from_env_invalid_integer_raises_validation_error():
    """Verify non-integer environment variable values raise ConfigValidationError."""
    with patch.dict(os.environ, {"KAFKA_AUTO_COMMIT_INTERVAL_MS": "not_an_int"}, clear=True):
        with pytest.raises(ConfigValidationError) as exc:
            KafkaConfig.load_from_env(load_env_file=False)
        assert "must be a valid integer" in str(exc.value)


