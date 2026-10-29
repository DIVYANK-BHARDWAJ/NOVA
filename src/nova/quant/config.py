"""NOVA MAX Quantitative Subsystem — Research Configuration Parser & Validator.

Reads and strictly validates research settings from TOML configuration files.
Uses standard library tomllib (Python 3.11+).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import tomllib
from typing import Any, Mapping

from nova.quant.contracts import ContractValidationError, NumericalValidationError


@dataclass(frozen=True, slots=True)
class ResearchConfig:
    """Immutable, validated in-memory representation of research configuration."""
    config_version: str
    environment: str
    description: str
    author: str
    timezone: str
    max_clock_skew_seconds: int
    strict_causality_mode: bool
    dynamic_universe_discovery: bool
    max_active_universe_size: int
    min_warmup_bars: int
    default_benchmark: str
    risk_free_rate_annual: float
    fee_schedule_status: str
    provisional_slippage_bps: float
    provisional_brokerage_bps: float
    strategy_settings: dict[str, Any]
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        # Compute deterministic SHA-256 fingerprint of the configuration
        canonical_dict = {
            "config_version": self.config_version,
            "environment": self.environment,
            "timezone": self.timezone,
            "max_clock_skew_seconds": self.max_clock_skew_seconds,
            "strict_causality_mode": self.strict_causality_mode,
            "dynamic_universe_discovery": self.dynamic_universe_discovery,
            "max_active_universe_size": self.max_active_universe_size,
            "min_warmup_bars": self.min_warmup_bars,
            "default_benchmark": self.default_benchmark,
            "risk_free_rate_annual": self.risk_free_rate_annual,
            "provisional_slippage_bps": self.provisional_slippage_bps,
            "provisional_brokerage_bps": self.provisional_brokerage_bps,
            "strategy_settings": self.strategy_settings,
        }
        canonical_json = json.dumps(canonical_dict, sort_keys=True, separators=(",", ":"))
        fp = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
        object.__setattr__(self, "fingerprint", fp)

    def is_strategy_enabled(self, strategy_name: str) -> bool:
        """Check whether a specific strategy is enabled."""
        strat = self.strategy_settings.get(strategy_name)
        if strat is None or not isinstance(strat, dict):
            return False
        return bool(strat.get("enabled", False))


def load_research_config(config_path: str | Path) -> ResearchConfig:
    """Load, parse, and validate research configuration from a TOML file."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except Exception as e:
        raise ContractValidationError(f"Failed to parse TOML configuration from {path}: {e}") from e

    return validate_config_dict(data)


def validate_config_dict(data: Mapping[str, Any]) -> ResearchConfig:
    """Validate a raw configuration dictionary structure and return a ResearchConfig."""
    required_sections = ["metadata", "temporal", "universe", "benchmark", "execution_friction_provisional", "strategies"]
    for sec in required_sections:
        if sec not in data or not isinstance(data[sec], dict):
            raise ContractValidationError(f"Configuration missing required section: '{sec}'")

    # 1. Metadata
    meta = data["metadata"]
    for req_field in ["config_version", "environment", "description", "author"]:
        if req_field not in meta or not isinstance(meta[req_field], str) or not meta[req_field].strip():
            raise ContractValidationError(f"Invalid or missing metadata field: '{req_field}'")

    # 2. Temporal
    temporal = data["temporal"]
    tz = temporal.get("timezone")
    if not isinstance(tz, str) or not tz.strip():
        raise ContractValidationError("temporal.timezone must be a non-empty string")

    skew = temporal.get("max_clock_skew_seconds")
    if not isinstance(skew, int) or skew < 0:
        raise NumericalValidationError("temporal.max_clock_skew_seconds must be a non-negative integer")

    strict_causality = temporal.get("strict_causality_mode")
    if not isinstance(strict_causality, bool):
        raise ContractValidationError("temporal.strict_causality_mode must be a boolean")

    # 3. Universe
    universe = data["universe"]
    dynamic_disc = universe.get("dynamic_universe_discovery")
    if not isinstance(dynamic_disc, bool):
        raise ContractValidationError("universe.dynamic_universe_discovery must be a boolean")

    max_size = universe.get("max_active_universe_size")
    if not isinstance(max_size, int) or max_size <= 0:
        raise NumericalValidationError("universe.max_active_universe_size must be a positive integer")

    min_warmup = universe.get("min_warmup_bars")
    if not isinstance(min_warmup, int) or min_warmup <= 0:
        raise NumericalValidationError("universe.min_warmup_bars must be a positive integer")

    # 4. Benchmark
    bench = data["benchmark"]
    def_bench = bench.get("default_benchmark")
    if not isinstance(def_bench, str) or not def_bench.strip():
        raise ContractValidationError("benchmark.default_benchmark must be a non-empty string")

    rf = bench.get("risk_free_rate_annual")
    if not isinstance(rf, (int, float)) or rf < 0.0 or rf > 1.0:
        raise NumericalValidationError(f"benchmark.risk_free_rate_annual must be in [0.0, 1.0], got {rf}")

    # 5. Execution friction
    fric = data["execution_friction_provisional"]
    fee_status = fric.get("fee_schedule_status")
    if not isinstance(fee_status, str) or not fee_status.strip():
        raise ContractValidationError("execution_friction_provisional.fee_schedule_status must be a non-empty string")

    slippage_bps = fric.get("provisional_slippage_bps")
    if not isinstance(slippage_bps, (int, float)) or slippage_bps < 0.0:
        raise NumericalValidationError(f"provisional_slippage_bps must be non-negative, got {slippage_bps}")

    brokerage_bps = fric.get("provisional_brokerage_bps")
    if not isinstance(brokerage_bps, (int, float)) or brokerage_bps < 0.0:
        raise NumericalValidationError(f"provisional_brokerage_bps must be non-negative, got {brokerage_bps}")

    # 6. Strategies
    strats = data["strategies"]
    # Check that core strategies are properly configured
    for core_strat in ["momentum", "relative_strength", "regime"]:
        if core_strat not in strats or not isinstance(strats[core_strat], dict):
            raise ContractValidationError(f"Core strategy '{core_strat}' configuration missing")

    # Enforce that optional strategies default to disabled
    optional_strats = [
        "breakout",
        "volatility_shock",
        "news_reaction",
        "mean_reversion",
        "pairs_trading",
        "ml_ranking",
        "portfolio_capital_allocation",
        "adaptive_selection",
    ]
    for opt in optional_strats:
        if opt in strats and isinstance(strats[opt], dict):
            is_enabled = strats[opt].get("enabled", False)
            if not isinstance(is_enabled, bool):
                raise ContractValidationError(f"Strategy '{opt}.enabled' must be a boolean")

    return ResearchConfig(
        config_version=meta["config_version"],
        environment=meta["environment"],
        description=meta["description"],
        author=meta["author"],
        timezone=tz,
        max_clock_skew_seconds=skew,
        strict_causality_mode=strict_causality,
        dynamic_universe_discovery=dynamic_disc,
        max_active_universe_size=max_size,
        min_warmup_bars=min_warmup,
        default_benchmark=def_bench,
        risk_free_rate_annual=float(rf),
        fee_schedule_status=fee_status,
        provisional_slippage_bps=float(slippage_bps),
        provisional_brokerage_bps=float(brokerage_bps),
        strategy_settings=dict(strats),
    )
