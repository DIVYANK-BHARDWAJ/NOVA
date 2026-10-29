"""Tests for main.py CLI entrypoint and startup validation."""

from unittest.mock import MagicMock, patch
import pytest

from market_data.main import build_arg_parser, configure_logging, main


def test_build_arg_parser():
    """Verify CLI arguments are properly configured."""
    parser = build_arg_parser()
    args = parser.parse_args(["--validate-config", "--max-messages", "10"])
    assert args.validate_config is True
    assert args.max_messages == 10


def test_main_validate_config_with_placeholder():
    """Verify --validate-config checks settings and exits 0 even if broker is placeholder."""
    test_args = ["main.py", "--validate-config"]
    with patch("sys.argv", test_args):
        ret = main()
        assert ret == 0


def test_main_rejects_placeholder_when_attempting_run():
    """Verify main aborts with exit code 1 when attempting live connection to placeholder."""
    test_args = ["main.py"]
    with patch("sys.argv", test_args), patch.dict("os.environ", {"KAFKA_BOOTSTRAP_SERVERS": "<SERVER_IP>:9094"}):
        ret = main()
        assert ret == 1


def test_main_rejects_invalid_configuration():
    """Verify main aborts with exit code 1 when configuration validation fails."""
    test_args = ["main.py", "--validate-config"]
    with patch("sys.argv", test_args):
        with patch.dict("os.environ", {"KAFKA_GROUP_ID": ""}):
            ret = main()
            assert ret == 1


def test_main_successful_run_with_mock_consumer():
    """Verify main initializes dispatcher and runs consumer cleanly."""
    test_args = ["main.py", "--max-messages", "5"]
    env = {
        "KAFKA_BOOTSTRAP_SERVERS": "127.0.0.1:9094",
        "KAFKA_GROUP_ID": "group-mn2651",
    }
    with patch("sys.argv", test_args), patch.dict("os.environ", env):
        with patch("market_data.main.MarketDataConsumer") as MockConsumerClass:
            mock_consumer_instance = MagicMock()
            MockConsumerClass.return_value = mock_consumer_instance

            ret = main()
            assert ret == 0
            MockConsumerClass.assert_called_once()
            mock_consumer_instance.run.assert_called_once_with(max_messages=5)
