"""Unit tests for NOVA MAX Research Configuration (Engineer B — Phase 1).

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
    """Test suite for research configuration parsing, validation, and fingerprinting."""

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

    def test_deterministic_fingerprint(self) -> None:
        """The SHA-256 configuration fingerprint must be deterministic across identical loads."""
        c1 = load_research_config(CONFIG_FILE)
        c2 = load_research_config(CONFIG_FILE)
        self.assertEqual(len(c1.fingerprint), 64)
        self.assertEqual(c1.fingerprint, c2.fingerprint)

    def test_fingerprint_sensitivity(self) -> None:
        """Modifying any parameter must alter the configuration fingerprint."""
        c1 = load_research_config(CONFIG_FILE)

        # Create modified raw dict
        mod_dict = {
            "metadata": {
                "config_version": "1.0.0",
                "environment": "research",
                "description": "test",
                "author": "Engineer B",
            },
            "temporal": {
                "timezone": "UTC",
                "max_clock_skew_seconds": 10,  # Changed from 5 to 10
                "strict_causality_mode": True,
            },
            "universe": {
                "dynamic_universe_discovery": True,
                "max_active_universe_size": 500,
                "min_warmup_bars": 50,
            },
            "benchmark": {
                "default_benchmark": "NO_TRADE",
                "risk_free_rate_annual": 0.065,
            },
            "execution_friction_provisional": {
                "fee_schedule_status": "UNVERIFIED",
                "provisional_slippage_bps": 5.0,
                "provisional_brokerage_bps": 2.0,
            },
            "strategies": {
                "momentum": {"enabled": True},
                "relative_strength": {"enabled": True},
                "regime": {"enabled": True},
            },
        }
        c2 = validate_config_dict(mod_dict)
        self.assertNotEqual(c1.fingerprint, c2.fingerprint)

    def test_missing_required_section_rejected(self) -> None:
        """Omitting a required configuration section must raise ContractValidationError."""
        bad_dict = {
            "metadata": {"config_version": "1.0.0", "environment": "research", "description": "d", "author": "a"},
            # Missing "temporal"
            "universe": {"dynamic_universe_discovery": True, "max_active_universe_size": 100, "min_warmup_bars": 10},
            "benchmark": {"default_benchmark": "NO_TRADE", "risk_free_rate_annual": 0.05},
            "execution_friction_provisional": {"fee_schedule_status": "U", "provisional_slippage_bps": 5.0, "provisional_brokerage_bps": 2.0},
            "strategies": {},
        }
        with self.assertRaises(ContractValidationError):
            validate_config_dict(bad_dict)

    def test_invalid_types_rejected(self) -> None:
        """Negative values or wrong types must raise validation exceptions."""
        bad_temporal = {
            "metadata": {"config_version": "1.0.0", "environment": "research", "description": "d", "author": "a"},
            "temporal": {
                "timezone": "UTC",
                "max_clock_skew_seconds": -5,  # Negative skew rejected
                "strict_causality_mode": True,
            },
            "universe": {"dynamic_universe_discovery": True, "max_active_universe_size": 100, "min_warmup_bars": 10},
            "benchmark": {"default_benchmark": "NO_TRADE", "risk_free_rate_annual": 0.05},
            "execution_friction_provisional": {"fee_schedule_status": "U", "provisional_slippage_bps": 5.0, "provisional_brokerage_bps": 2.0},
            "strategies": {"momentum": {}, "relative_strength": {}, "regime": {}},
        }
        with self.assertRaises(NumericalValidationError):
            validate_config_dict(bad_temporal)

    def test_missing_file_raises_filenotfound(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_research_config("non_existent_config.toml")


if __name__ == "__main__":
    unittest.main()
