"""NOVA MAX Quantitative Subsystem — Research Configuration Parser & Validator.

Reads and strictly validates research settings from TOML configuration files.
Enforces configuration immutability (via MappingProxyType), strict non-boolean numeric typing,
deterministic SHA-256 fingerprinting, and strict rejection of unknown configuration keys.
Uses standard library tomllib (Python 3.11+).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import tomllib
from types import MappingProxyType
from typing import Any, Mapping

from nova.quant.contracts import ContractValidationError, NumericalValidationError

# ============================================================================
# Known Schema Keys for Strict Unknown-Key Detection
# ============================================================================

KNOWN_SECTIONS = {
    "metadata",
    "temporal",
    "universe",
    "benchmark",
    "execution_friction_provisional",
    "strategies",
}

KNOWN_METADATA_KEYS = {"config_version", "environment", "description", "author"}
KNOWN_TEMPORAL_KEYS = {"timezone", "max_clock_skew_seconds", "strict_causality_mode"}
KNOWN_UNIVERSE_KEYS = {"dynamic_universe_discovery", "max_active_universe_size", "min_warmup_bars"}
KNOWN_BENCHMARK_KEYS = {"default_benchmark", "risk_free_rate_annual"}
KNOWN_FRICTION_KEYS = {"fee_schedule_status", "provisional_slippage_bps", "provisional_brokerage_bps"}

KNOWN_STRATEGIES = {
    "momentum",
    "ema_crossover",
    "relative_strength",
    "regime",
    "breakout",
    "volatility_shock",
    "news_reaction",
    "mean_reversion",
    "pairs_trading",
    "ml_ranking",
    "portfolio_capital_allocation",
    "adaptive_selection",
}


def _freeze_mapping(obj: Any) -> Any:
    """Recursively freeze dictionaries to MappingProxyType and sequences to tuples."""
    if isinstance(obj, dict):
        return MappingProxyType({k: _freeze_mapping(v) for k, v in obj.items()})
    elif isinstance(obj, (list, tuple)):
        return tuple(_freeze_mapping(v) for v in obj)
    return obj


def _to_serializable(obj: Any) -> Any:
    """Recursively convert MappingProxyType and nested structures into canonical serializable dicts."""
    if isinstance(obj, (dict, MappingProxyType)):
        return {k: _to_serializable(v) for k, v in sorted(obj.items())}
    elif isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return obj


def _is_int(val: Any) -> bool:
    """Returns True if val is an int and strictly NOT a boolean."""
    return isinstance(val, int) and not isinstance(val, bool)


def _is_finite_num(val: Any) -> bool:
    """Returns True if val is a finite number and strictly NOT a boolean."""
    return isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(val)


@dataclass(frozen=True, slots=True)
class ResearchConfig:
    """Immutable, validated in-memory representation of research configuration.

    Nested strategy_settings is an immutable MappingProxyType to prevent post-validation mutation.
    """
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
    strategy_settings: Mapping[str, Any]
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        # Compute deterministic SHA-256 fingerprint representing all behaviorally relevant settings
        canonical_dict = {
            "author": self.author,
            "config_version": self.config_version,
            "default_benchmark": self.default_benchmark,
            "description": self.description,
            "dynamic_universe_discovery": self.dynamic_universe_discovery,
            "environment": self.environment,
            "fee_schedule_status": self.fee_schedule_status,
            "max_active_universe_size": self.max_active_universe_size,
            "max_clock_skew_seconds": self.max_clock_skew_seconds,
            "min_warmup_bars": self.min_warmup_bars,
            "provisional_brokerage_bps": self.provisional_brokerage_bps,
            "provisional_slippage_bps": self.provisional_slippage_bps,
            "risk_free_rate_annual": self.risk_free_rate_annual,
            "strategy_settings": _to_serializable(self.strategy_settings),
            "strict_causality_mode": self.strict_causality_mode,
            "timezone": self.timezone,
        }
        canonical_json = json.dumps(canonical_dict, sort_keys=True, separators=(",", ":"))
        fp = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
        object.__setattr__(self, "fingerprint", fp)

    def is_strategy_enabled(self, strategy_name: str) -> bool:
        """Check whether a specific strategy is marked enabled in configuration.

        NOTE: Being enabled in configuration means the strategy is eligible for evaluation;
        it does not mean it has been validated or authorized to trade.
        """
        strat = self.strategy_settings.get(strategy_name)
        if strat is None or not isinstance(strat, (dict, MappingProxyType)):
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
    """Validate a raw configuration dictionary structure and return an immutable ResearchConfig."""
    # 0. Reject unknown top-level sections
    extra_sections = set(data.keys()) - KNOWN_SECTIONS
    if extra_sections:
        raise ContractValidationError(f"Unknown configuration section(s) detected: {sorted(extra_sections)}")

    for sec in KNOWN_SECTIONS:
        if sec not in data or not isinstance(data[sec], dict):
            raise ContractValidationError(f"Configuration missing required section: '{sec}'")

    # 1. Metadata
    meta = data["metadata"]
    extra_meta = set(meta.keys()) - KNOWN_METADATA_KEYS
    if extra_meta:
        raise ContractValidationError(f"Unknown key(s) in 'metadata': {sorted(extra_meta)}")
    for req_field in ["config_version", "environment", "description", "author"]:
        if req_field not in meta or not isinstance(meta[req_field], str) or not meta[req_field].strip():
            raise ContractValidationError(f"Invalid or missing metadata field: '{req_field}'")

    # 2. Temporal
    temporal = data["temporal"]
    extra_temp = set(temporal.keys()) - KNOWN_TEMPORAL_KEYS
    if extra_temp:
        raise ContractValidationError(f"Unknown key(s) in 'temporal': {sorted(extra_temp)}")
    tz = temporal.get("timezone")
    if not isinstance(tz, str) or not tz.strip():
        raise ContractValidationError("temporal.timezone must be a non-empty string")

    skew = temporal.get("max_clock_skew_seconds")
    if not _is_int(skew) or skew < 0:
        raise NumericalValidationError("temporal.max_clock_skew_seconds must be a non-negative integer (non-boolean)")

    strict_causality = temporal.get("strict_causality_mode")
    if not isinstance(strict_causality, bool):
        raise ContractValidationError("temporal.strict_causality_mode must be a boolean")

    # 3. Universe
    universe = data["universe"]
    extra_univ = set(universe.keys()) - KNOWN_UNIVERSE_KEYS
    if extra_univ:
        raise ContractValidationError(f"Unknown key(s) in 'universe': {sorted(extra_univ)}")
    dynamic_disc = universe.get("dynamic_universe_discovery")
    if not isinstance(dynamic_disc, bool):
        raise ContractValidationError("universe.dynamic_universe_discovery must be a boolean")

    max_size = universe.get("max_active_universe_size")
    if not _is_int(max_size) or max_size <= 0:
        raise NumericalValidationError("universe.max_active_universe_size must be a positive integer (non-boolean)")

    min_warmup = universe.get("min_warmup_bars")
    if not _is_int(min_warmup) or min_warmup <= 0:
        raise NumericalValidationError("universe.min_warmup_bars must be a positive integer (non-boolean)")

    # 4. Benchmark
    bench = data["benchmark"]
    extra_bench = set(bench.keys()) - KNOWN_BENCHMARK_KEYS
    if extra_bench:
        raise ContractValidationError(f"Unknown key(s) in 'benchmark': {sorted(extra_bench)}")
    def_bench = bench.get("default_benchmark")
    if not isinstance(def_bench, str) or not def_bench.strip():
        raise ContractValidationError("benchmark.default_benchmark must be a non-empty string")

    rf = bench.get("risk_free_rate_annual")
    if not _is_finite_num(rf) or rf < 0.0 or rf > 1.0:
        raise NumericalValidationError(
            f"benchmark.risk_free_rate_annual must be a finite number in [0.0, 1.0] (non-boolean), got {rf}"
        )

    # 5. Execution friction
    fric = data["execution_friction_provisional"]
    extra_fric = set(fric.keys()) - KNOWN_FRICTION_KEYS
    if extra_fric:
        raise ContractValidationError(f"Unknown key(s) in 'execution_friction_provisional': {sorted(extra_fric)}")
    fee_status = fric.get("fee_schedule_status")
    if not isinstance(fee_status, str) or not fee_status.strip():
        raise ContractValidationError("execution_friction_provisional.fee_schedule_status must be a non-empty string")

    slippage_bps = fric.get("provisional_slippage_bps")
    if not _is_finite_num(slippage_bps) or slippage_bps < 0.0:
        raise NumericalValidationError(
            f"provisional_slippage_bps must be a non-negative finite number (non-boolean), got {slippage_bps}"
        )

    brokerage_bps = fric.get("provisional_brokerage_bps")
    if not _is_finite_num(brokerage_bps) or brokerage_bps < 0.0:
        raise NumericalValidationError(
            f"provisional_brokerage_bps must be a non-negative finite number (non-boolean), got {brokerage_bps}"
        )

    # 6. Strategies
    strats = data["strategies"]
    extra_strats = set(strats.keys()) - KNOWN_STRATEGIES
    if extra_strats:
        raise ContractValidationError(f"Unknown strategy key(s) detected: {sorted(extra_strats)}")

    # Check that core strategies are properly configured
    for core_strat in ["momentum", "relative_strength", "regime"]:
        if core_strat not in strats or not isinstance(strats[core_strat], dict):
            raise ContractValidationError(f"Core strategy '{core_strat}' configuration missing")

    # Validate individual strategy parameters
    for strat_name, strat_dict in strats.items():
        if not isinstance(strat_dict, dict):
            raise ContractValidationError(f"Strategy section '{strat_name}' must be a dictionary")
        is_enabled = strat_dict.get("enabled", False)
        if not isinstance(is_enabled, bool):
            raise ContractValidationError(f"Strategy '{strat_name}.enabled' must be a boolean")

        # Validate numeric parameters inside strategy dictionary
        for p_name, p_val in strat_dict.items():
            if p_name in ("fast_window", "slow_window", "signal_window", "ranking_window_bars", "adx_window", "volatility_window", "channel_bars", "recovery_window", "lookback_bars"):
                if not _is_int(p_val) or p_val <= 0:
                    raise NumericalValidationError(f"Strategy '{strat_name}.{p_name}' must be a positive integer, got {p_val}")
            elif p_name in ("top_quantile", "bottom_quantile", "min_sentiment_confidence", "min_correlation"):
                if not _is_finite_num(p_val) or not (0.0 < p_val <= 1.0):
                    raise NumericalValidationError(f"Strategy '{strat_name}.{p_name}' must be in (0.0, 1.0], got {p_val}")
            elif p_name in ("adx_trending_threshold", "atr_multiplier", "shock_z_score", "reversion_entry_z"):
                if not _is_finite_num(p_val) or p_val <= 0.0:
                    raise NumericalValidationError(f"Strategy '{strat_name}.{p_name}' must be a positive number, got {p_val}")

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
            if strats[opt].get("enabled") is True:
                raise ContractValidationError(
                    f"Optional strategy '{opt}' cannot be enabled by default in base configuration"
                )

    # Freeze strategy settings into an immutable MappingProxyType
    frozen_strats = _freeze_mapping(dict(strats))

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
        strategy_settings=frozen_strats,
    )
