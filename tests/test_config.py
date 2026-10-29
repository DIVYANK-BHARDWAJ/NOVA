"""Unit tests for NOVA MAX Research Configuration (Engineer B — Phase 1 & 1.1 Remediation).

Uses Python 3.11 standard-library unittest only.
"""

from pathlib import Path
import sys
import unittest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nova.quant.config import load_research_config, validate_config_dict
from nova.quant.contracts import ContractValidationError, NumericalValidationError

CONFIG_FILE = Path(__file__).resolve().parent.parent / "config" / "research_config.toml"


class TestResearchConfig(unittest.TestCase):
    """Test suite for research configuration immutability, validation, and fingerprinting."""

    def test_load_default_research_config(self) -> None:
        """Verify that the official repository config/research_config.toml loads and passes validation."""
        self.assertTrue(CONFIG_FILE.exists(), f"Configuration file missing at {CONFIG_FILE}")
        config = load_research_config(CONFIG_FILE)

        self.assertEqual(config.config_version, "1.0.0")
        self.assertEqual(config.environment, "research")
        self.assertEqual(config.timezone, "UTC")
        self.assertTrue(config.strict_causality_mode)
        self.assertTrue(config.dynamic_universe_discovery)
        self.assertEqual(config.default_benchmark, "NO_TRADE")
        self.assertAlmostEqual(config.risk_free_rate_annual, 0.065)
        self.assertEqual(config.fee_schedule_status, "UNVERIFIED_PENDING_PLATFORM_CONTRACT")

        # Core strategies must be enabled
        self.assertTrue(config.is_strategy_enabled("momentum"))
        self.assertTrue(config.is_strategy_enabled("relative_strength"))
        self.assertTrue(config.is_strategy_enabled("regime"))

        # Optional strategies must be disabled by default
        self.assertFalse(config.is_strategy_enabled("breakout"))
        self.assertFalse(config.is_strategy_enabled("volatility_shock"))
        self.assertFalse(config.is_strategy_enabled("news_reaction"))
        self.assertFalse(config.is_strategy_enabled("mean_reversion"))
        self.assertFalse(config.is_strategy_enabled("pairs_trading"))
        self.assertFalse(config.is_strategy_enabled("ml_ranking"))
        self.assertFalse(config.is_strategy_enabled("portfolio_capital_allocation"))
        self.assertFalse(config.is_strategy_enabled("adaptive_selection"))

    def test_immutability_of_strategy_settings(self) -> None:
        """Remediation C: Callers must not be able to mutate nested strategy settings."""
        config = load_research_config(CONFIG_FILE)

        # 1. Attempting to modify top-level strategy_settings must raise TypeError
        with self.assertRaises(TypeError):
            config.strategy_settings["momentum"] = {}  # type: ignore

        # 2. Attempting to modify nested dictionary must raise TypeError
        with self.assertRaises(TypeError):
            config.strategy_settings["momentum"]["enabled"] = False  # type: ignore

        # 3. Attempting to add a new strategy must raise TypeError
        with self.assertRaises(TypeError):
            config.strategy_settings["unauthorized_new_strategy"] = {"enabled": True}  # type: ignore

    def test_deterministic_fingerprint(self) -> None:
        """The SHA-256 configuration fingerprint must be deterministic across identical loads."""
        c1 = load_research_config(CONFIG_FILE)
        c2 = load_research_config(CONFIG_FILE)
        self.assertEqual(len(c1.fingerprint), 64)
        self.assertEqual(c1.fingerprint, c2.fingerprint)

    def test_fingerprint_sensitivity_on_fee_schedule_status(self) -> None:
        """Remediation C: Changing fee_schedule_status must alter the fingerprint."""
        c1 = load_research_config(CONFIG_FILE)

        # Base valid dict
        valid_dict = {
            "metadata": {"config_version": "1.0.0", "environment": "research", "description": "d", "author": "Engineer B"},
            "temporal": {"timezone": "UTC", "max_clock_skew_seconds": 5, "strict_causality_mode": True},
            "universe": {"dynamic_universe_discovery": True, "max_active_universe_size": 500, "min_warmup_bars": 50},
            "benchmark": {"default_benchmark": "NO_TRADE", "risk_free_rate_annual": 0.065},
            "execution_friction_provisional": {
                "fee_schedule_status": "UNVERIFIED_PENDING_PLATFORM_CONTRACT",
                "provisional_slippage_bps": 5.0,
                "provisional_brokerage_bps": 2.0,
            },
            "strategies": {
                "momentum": {"enabled": True},
                "relative_strength": {"enabled": True},
                "regime": {"enabled": True},
            },
        }
        c_base = validate_config_dict(valid_dict)

        # Change fee_schedule_status
        valid_dict["execution_friction_provisional"]["fee_schedule_status"] = "VERIFIED_OFFICIAL_SCHEDULE_V1"
        c_mod = validate_config_dict(valid_dict)

        self.assertNotEqual(c_base.fingerprint, c_mod.fingerprint)

    def test_fingerprint_sensitivity_on_strategy_parameters(self) -> None:
        """Remediation C: Changing any strategy parameter must alter the fingerprint."""
        valid_dict = {
            "metadata": {"config_version": "1.0.0", "environment": "research", "description": "d", "author": "Engineer B"},
            "temporal": {"timezone": "UTC", "max_clock_skew_seconds": 5, "strict_causality_mode": True},
            "universe": {"dynamic_universe_discovery": True, "max_active_universe_size": 500, "min_warmup_bars": 50},
            "benchmark": {"default_benchmark": "NO_TRADE", "risk_free_rate_annual": 0.065},
            "execution_friction_provisional": {
                "fee_schedule_status": "UNVERIFIED",
                "provisional_slippage_bps": 5.0,
                "provisional_brokerage_bps": 2.0,
            },
            "strategies": {
                "momentum": {"enabled": True, "fast_window": 12},
                "relative_strength": {"enabled": True},
                "regime": {"enabled": True},
            },
        }
        c1 = validate_config_dict(valid_dict)

        valid_dict["strategies"]["momentum"]["fast_window"] = 15
        c2 = validate_config_dict(valid_dict)

        self.assertNotEqual(c1.fingerprint, c2.fingerprint)

    def test_boolean_supplied_for_numbers_rejected(self) -> None:
        """Remediation C: Booleans must not silently pass as numeric settings."""
        base_dict = {
            "metadata": {"config_version": "1.0.0", "environment": "research", "description": "d", "author": "a"},
            "temporal": {"timezone": "UTC", "max_clock_skew_seconds": 5, "strict_causality_mode": True},
            "universe": {"dynamic_universe_discovery": True, "max_active_universe_size": 500, "min_warmup_bars": 50},
            "benchmark": {"default_benchmark": "NO_TRADE", "risk_free_rate_annual": 0.065},
            "execution_friction_provisional": {"fee_schedule_status": "U", "provisional_slippage_bps": 5.0, "provisional_brokerage_bps": 2.0},
            "strategies": {"momentum": {"enabled": True}, "relative_strength": {"enabled": True}, "regime": {"enabled": True}},
        }

        # 1. Boolean for max_clock_skew_seconds
        d1 = dict(base_dict)
        d1["temporal"] = dict(base_dict["temporal"])
        d1["temporal"]["max_clock_skew_seconds"] = True
        with self.assertRaises(NumericalValidationError):
            validate_config_dict(d1)

        # 2. Boolean for risk_free_rate_annual
        d2 = dict(base_dict)
        d2["benchmark"] = dict(base_dict["benchmark"])
        d2["benchmark"]["risk_free_rate_annual"] = True
        with self.assertRaises(NumericalValidationError):
            validate_config_dict(d2)

        # 3. Boolean for min_warmup_bars
        d3 = dict(base_dict)
        d3["universe"] = dict(base_dict["universe"])
        d3["universe"]["min_warmup_bars"] = True
        with self.assertRaises(NumericalValidationError):
            validate_config_dict(d3)

    def test_unknown_sections_and_keys_rejected(self) -> None:
        """Remediation C: Unknown configuration sections and keys must be strictly rejected."""
        valid_dict = {
            "metadata": {"config_version": "1.0.0", "environment": "research", "description": "d", "author": "a"},
            "temporal": {"timezone": "UTC", "max_clock_skew_seconds": 5, "strict_causality_mode": True},
            "universe": {"dynamic_universe_discovery": True, "max_active_universe_size": 500, "min_warmup_bars": 50},
            "benchmark": {"default_benchmark": "NO_TRADE", "risk_free_rate_annual": 0.065},
            "execution_friction_provisional": {"fee_schedule_status": "U", "provisional_slippage_bps": 5.0, "provisional_brokerage_bps": 2.0},
            "strategies": {"momentum": {"enabled": True}, "relative_strength": {"enabled": True}, "regime": {"enabled": True}},
        }

        # Unknown section
        d_unknown_sec = dict(valid_dict)
        d_unknown_sec["unauthorized_secret_section"] = {"api_key": "secret"}
        with self.assertRaises(ContractValidationError):
            validate_config_dict(d_unknown_sec)

        # Unknown key in metadata
        d_unknown_key = dict(valid_dict)
        d_unknown_key["metadata"] = dict(valid_dict["metadata"])
        d_unknown_key["metadata"]["extra_invalid_key"] = "bad"
        with self.assertRaises(ContractValidationError):
            validate_config_dict(d_unknown_key)

        # Unknown strategy name
        d_unknown_strat = dict(valid_dict)
        d_unknown_strat["strategies"] = dict(valid_dict["strategies"])
        d_unknown_strat["strategies"]["magic_crypto_strat"] = {"enabled": True}
        with self.assertRaises(ContractValidationError):
            validate_config_dict(d_unknown_strat)

    def test_missing_file_raises_filenotfound(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_research_config("non_existent_config.toml")


if __name__ == "__main__":
    unittest.main()
