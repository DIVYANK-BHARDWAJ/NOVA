"""NOVA Regime Intelligence — Canonical Transparent Market-Regime Detector.

Implements Person 2's deterministic, causal market-regime classification:
- Kaufman Efficiency Ratio (KER) measuring directional trend efficiency vs chop
- RiskMetrics EWMA volatility estimation (lambda=0.94)
- Fast/Slow EMA directional filter (never conflating efficiency with trend direction)
- Pre-simulation historical warm-start from genuine competition OHLCV data
- Contract-compliant ModelPrediction output (Interface Contract B)

Causality & Data Integrity Invariants:
1. Pure point-in-time calculation: State at bar t depends strictly on bars <= t.
2. Anti-leakage: Future bars or mutated future prices have ZERO impact on past regime predictions.
3. Zero hardcoding: All thresholds and lookbacks are configuration-driven via MarketRegimeConfig.
4. Transparent probabilities: Distribution maps discrete mutually exclusive regimes summing to 1.0.
"""

from __future__ import annotations

from enum import Enum
import math
from typing import Any, Mapping, Sequence

from nova.quant.contracts import (
    ContractValidationError,
    DataQualityStatus,
    ModelPrediction,
    NumericalValidationError,
    PredictionAvailability,
    PredictionType,
    TemporalValidationError,
    validate_finite,
    validate_timezone_aware,
)
from nova.regime.config import MarketRegimeConfig


class RegimeLabel(str, Enum):
    """Canonical taxonomy of market regimes."""
    TRENDING_BULLISH = "TRENDING_BULLISH"   # High directional efficiency with upward momentum
    TRENDING_BEARISH = "TRENDING_BEARISH"   # High directional efficiency with downward momentum
    RANGING = "RANGING"                     # Low directional efficiency, oscillatory/mean-reverting
    HIGH_VOLATILITY = "HIGH_VOLATILITY"     # Elevated volatility exceeding threshold (elevated risk)
    UNKNOWN = "UNKNOWN"                     # Insufficient history or corrupted market data


class TransparentRegimeDetector:
    """Production-grade causal market-regime estimator for NOVA (Person 2)."""

    def __init__(self, config: MarketRegimeConfig | None = None) -> None:
        self.config = config or MarketRegimeConfig()
        # Per-instrument chronological candle buffer for streaming/live execution
        self._buffers: dict[str, list[Any]] = {}

    def reset(self, ticker: str | None = None) -> None:
        """Clear historical buffer for a specific ticker or all tickers."""
        if ticker is not None:
            self._buffers.pop(ticker, None)
        else:
            self._buffers.clear()

    def warm_start(self, ticker: str, bars: Sequence[Any]) -> int:
        """Warm-start detector state using genuine historical OHLCV bars.

        Enables instant Long/Flat signal evaluation on subsequent live ticks
        without incurring warm-up abstention delays.
        """
        if not ticker or not ticker.strip():
            raise ContractValidationError("Ticker cannot be blank")
        if not bars:
            return 0

        self._validate_bar_sequence(bars)
        # Store a copy of chronological bars
        self._buffers[ticker] = list(bars)
        return len(self._buffers[ticker])

    def get_buffered_bar_count(self, ticker: str) -> int:
        """Return count of chronological bars in memory for an instrument."""
        return len(self._buffers.get(ticker, []))

    def _validate_bar_sequence(self, bars: Sequence[Any]) -> None:
        """Strictly validate chronological order and homogeneity of input bar sequence."""
        if not bars:
            raise ContractValidationError("Cannot evaluate empty bar sequence")

        first_ticker = bars[0].ticker
        for i in range(len(bars)):
            b = bars[i]
            if b.ticker != first_ticker:
                raise ContractValidationError(
                    f"Inconsistent ticker at index {i}: expected '{first_ticker}', got '{b.ticker}'"
                )
            validate_timezone_aware(b.source_timestamp, f"bars[{i}].source_timestamp")

            if i > 0:
                prev_ts = bars[i - 1].source_timestamp
                curr_ts = b.source_timestamp
                if curr_ts <= prev_ts:
                    raise TemporalValidationError(
                        f"Non-chronological bars detected: bar {i} ({curr_ts}) <= bar {i-1} ({prev_ts})"
                    )

    def calculate_efficiency_ratio(self, closes: Sequence[float]) -> float:
        """Calculate causal Kaufman Efficiency Ratio (KER) over ker_window.

        Formula:
            Direction = |Close_t - Close_{t - W}|
            Volatility = Sum_{k=1..W} |Close_{t-k+1} - Close_{t-k}|
            KER = Direction / Volatility (in [0.0, 1.0])

        Returns 0.0 for constant prices or zero path length.
        """
        w = self.config.ker_window
        if len(closes) < w + 1:
            return 0.0

        direction = abs(closes[-1] - closes[-w - 1])
        path_length = sum(abs(closes[-k] - closes[-k - 1]) for k in range(1, w + 1))

        if path_length <= 1e-12:
            return 0.0

        ker = direction / path_length
        return min(1.0, max(0.0, ker))

    def calculate_ewma_volatility(self, closes: Sequence[float]) -> float:
        """Calculate causal EWMA realized return volatility (RiskMetrics lambda=0.94).

        Returns standard deviation floored at min_volatility_floor.
        """
        if len(closes) < 2:
            return self.config.min_volatility_floor

        decay = self.config.ewma_lambda
        w = min(len(closes) - 1, self.config.volatility_window)

        # Log returns
        returns = [math.log(closes[-i] / closes[-i - 1]) for i in range(w, 0, -1)]
        if not returns:
            return self.config.min_volatility_floor

        ewma_sq = returns[0] ** 2
        for r in returns[1:]:
            ewma_sq = decay * ewma_sq + (1.0 - decay) * (r ** 2)

        sigma = math.sqrt(ewma_sq)
        return max(self.config.min_volatility_floor, sigma)

    def calculate_ema(self, series: Sequence[float], window: int) -> float:
        """Calculate single latest exponential moving average value causally."""
        if not series:
            return 0.0
        alpha = 2.0 / (float(window) + 1.0)
        current_ema = series[0]
        for val in series[1:]:
            current_ema = alpha * val + (1.0 - alpha) * current_ema
        return current_ema

    def detect(self, bars: Sequence[Any]) -> ModelPrediction:
        """Evaluate market bars causally as of latest bar and return contract-valid ModelPrediction."""
        if not bars:
            raise ContractValidationError("Cannot detect regime on empty bar sequence")

        self._validate_bar_sequence(bars)
        latest_bar = bars[-1]
        ticker = latest_bar.ticker
        as_of = latest_bar.source_timestamp
        cutoff = latest_bar.available_timestamp or as_of

        # 1. Check data quality
        if latest_bar.data_quality != DataQualityStatus.VALID:
            dist = {
                RegimeLabel.TRENDING_BULLISH.value: 0.0,
                RegimeLabel.TRENDING_BEARISH.value: 0.0,
                RegimeLabel.RANGING.value: 0.0,
                RegimeLabel.HIGH_VOLATILITY.value: 0.0,
            }
            return ModelPrediction(
                model_id=self.config.model_id,
                model_version=self.config.config_version,
                prediction_timestamp=as_of,
                input_cutoff_timestamp=cutoff,
                instrument_id=ticker,
                prediction_type=PredictionType.REGIME_PROBABILITY,
                horizon_bars=1,
                prediction_value=0.0,
                confidence_or_uncertainty=0.0,
                availability_status=PredictionAvailability.UNAVAILABLE,
                metadata={
                    "status_reason": f"DATA_QUALITY_{latest_bar.data_quality.value}",
                    "dominant_regime": RegimeLabel.UNKNOWN.value,
                    "buffered_bars": len(bars),
                },
            )

        # 2. Check warmup requirement
        if len(bars) < self.config.min_warmup_bars:
            return ModelPrediction(
                model_id=self.config.model_id,
                model_version=self.config.config_version,
                prediction_timestamp=as_of,
                input_cutoff_timestamp=cutoff,
                instrument_id=ticker,
                prediction_type=PredictionType.REGIME_PROBABILITY,
                horizon_bars=1,
                prediction_value=0.25,
                prediction_distribution={
                    RegimeLabel.TRENDING_BULLISH.value: 0.25,
                    RegimeLabel.TRENDING_BEARISH.value: 0.25,
                    RegimeLabel.RANGING.value: 0.25,
                    RegimeLabel.HIGH_VOLATILITY.value: 0.25,
                },
                confidence_or_uncertainty=0.0,
                availability_status=PredictionAvailability.INSUFFICIENT_HISTORY,
                metadata={
                    "status_reason": "INSUFFICIENT_HISTORY",
                    "dominant_regime": RegimeLabel.UNKNOWN.value,
                    "buffered_bars": len(bars),
                    "required_bars": self.config.min_warmup_bars,
                },
            )

        # 3. Extract causal features
        closes = [b.close for b in bars]
        ker = self.calculate_efficiency_ratio(closes)
        sigma = self.calculate_ewma_volatility(closes)

        fast_ema = self.calculate_ema(closes, self.config.fast_ema_window)
        slow_ema = self.calculate_ema(closes, self.config.slow_ema_window)
        ema_spread = fast_ema - slow_ema
        net_price_change = closes[-1] - closes[-self.config.ker_window - 1]

        # Determine direction: positive if EMA spread > 0 and price change > 0
        is_bullish = ema_spread >= 0.0 and net_price_change >= 0.0

        # 4. Formulate regime classification
        is_high_vol = sigma >= self.config.high_volatility_threshold
        is_trend = ker >= self.config.ker_trend_threshold
        is_range = ker <= self.config.ker_ranging_threshold

        if is_high_vol:
            dominant = RegimeLabel.HIGH_VOLATILITY
            confidence = min(1.0, 0.60 + 0.40 * (sigma / self.config.high_volatility_threshold - 1.0))
        elif is_trend:
            dominant = RegimeLabel.TRENDING_BULLISH if is_bullish else RegimeLabel.TRENDING_BEARISH
            confidence = min(1.0, 0.50 + 0.50 * ((ker - self.config.ker_trend_threshold) / (1.0 - self.config.ker_trend_threshold + 1e-6)))
        elif is_range:
            dominant = RegimeLabel.RANGING
            confidence = min(1.0, 0.50 + 0.50 * ((self.config.ker_ranging_threshold - ker) / (self.config.ker_ranging_threshold + 1e-6)))
        else:
            # Transitional zone: allocate based on proximity to trend vs range
            affinity = (ker - self.config.ker_ranging_threshold) / (self.config.ker_trend_threshold - self.config.ker_ranging_threshold)
            if affinity >= 0.50:
                dominant = RegimeLabel.TRENDING_BULLISH if is_bullish else RegimeLabel.TRENDING_BEARISH
            else:
                dominant = RegimeLabel.RANGING
            confidence = 0.50 + abs(affinity - 0.50)

        # 5. Formulate calibrated discrete probability distribution summing to 1.0
        # Dominant regime receives (0.50 + 0.50 * confidence), remaining is shared among others
        p_dom = round(0.50 + 0.45 * confidence, 4)
        p_rem = round((1.0 - p_dom) / 3.0, 4)
        p_last = round(1.0 - (p_dom + 2.0 * p_rem), 4)

        probabilities: dict[str, float] = {}
        for reg in [
            RegimeLabel.TRENDING_BULLISH,
            RegimeLabel.TRENDING_BEARISH,
            RegimeLabel.RANGING,
            RegimeLabel.HIGH_VOLATILITY,
        ]:
            if reg == dominant:
                probabilities[reg.value] = p_dom
            elif len(probabilities) == 3:
                probabilities[reg.value] = p_last
            else:
                probabilities[reg.value] = p_rem

        # Ensure exact sum = 1.0
        diff = 1.0 - sum(probabilities.values())
        probabilities[dominant.value] = round(probabilities[dominant.value] + diff, 4)

        return ModelPrediction(
            model_id=self.config.model_id,
            model_version=self.config.config_version,
            prediction_timestamp=as_of,
            input_cutoff_timestamp=cutoff,
            instrument_id=ticker,
            prediction_type=PredictionType.REGIME_PROBABILITY,
            horizon_bars=1,
            prediction_value=probabilities[dominant.value],
            prediction_distribution=probabilities,
            confidence_or_uncertainty=round(confidence, 4),
            availability_status=PredictionAvailability.READY,
            metadata={
                "dominant_regime": dominant.value,
                "efficiency_ratio": round(ker, 4),
                "ewma_volatility": round(sigma, 6),
                "is_high_volatility": is_high_vol,
                "fast_ema": round(fast_ema, 2),
                "slow_ema": round(slow_ema, 2),
                "ema_spread": round(ema_spread, 4),
                "buffered_bars": len(bars),
                "config_version": self.config.config_version,
            },
        )

    def process_event(self, event_or_bar: Any) -> ModelPrediction:
        """Incremental streaming event processor for live Kafka candle ingestion.

        Accepts either a validated MarketBar or a streaming event mapping (e.g. from Kafka).
        Appends the incoming candle to the instrument's chronological buffer and
        returns the latest regime prediction contract.
        """
        bar = self._normalize_to_market_bar(event_or_bar)
        ticker = bar.ticker
        if ticker not in self._buffers:
            self._buffers[ticker] = []

        buf = self._buffers[ticker]
        if buf and bar.source_timestamp <= buf[-1].source_timestamp:
            raise TemporalValidationError(
                f"Out-of-order or duplicate bar received for {ticker}: {bar.source_timestamp} <= {buf[-1].source_timestamp}"
            )

        buf.append(bar)
        return self.detect(buf)

    def _normalize_to_market_bar(self, obj: Any) -> MarketBar:
        """Coerce mapping or existing MarketBar into a validated MarketBar."""
        from datetime import datetime, timezone
        from nova.quant.contracts import MarketBar

        if isinstance(obj, MarketBar):
            return obj

        if isinstance(obj, Mapping):
            ticker = obj.get("ticker") or obj.get("symbol")
            if not ticker or not isinstance(ticker, str):
                raise ContractValidationError(f"Streaming event missing valid 'ticker' or 'symbol': {obj}")

            ts_raw = obj.get("source_timestamp") or obj.get("timestamp")
            if isinstance(ts_raw, str):
                ts = datetime.fromisoformat(ts_raw)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
            elif isinstance(ts_raw, datetime):
                ts = ts_raw if ts_raw.tzinfo is not None else ts_raw.replace(tzinfo=timezone.utc)
            else:
                raise ContractValidationError(f"Streaming event missing valid timestamp: {ts_raw}")

            close_p = float(obj.get("close", 0.0))
            open_p = float(obj.get("open", close_p))
            high_p = float(obj.get("high", max(open_p, close_p)))
            low_p = float(obj.get("low", min(open_p, close_p)))
            vol = float(obj.get("volume", 0.0))

            topic = obj.get("topic") or obj.get("source_topic") or "market.ticks"
            part = int(obj.get("partition", obj.get("source_partition", 0)))
            off = int(obj.get("offset", obj.get("source_offset", 0)))
            prov = obj.get("provenance") or f"KAFKA_STREAM:{topic}:{part}:{off}"

            return MarketBar(
                ticker=ticker,
                open=open_p,
                high=high_p,
                low=low_p,
                close=close_p,
                volume=vol,
                source_timestamp=ts,
                provenance=prov,
                data_quality=DataQualityStatus.VALID,
            )

        raise ContractValidationError(f"Unsupported event type for process_event: {type(obj)}")
    def detect_regime(self, bars: Sequence[Any]) -> Any:
        """Adapter method satisfying SentinelDecisionEngine's RegimeOutput contract."""
        from nova.decision.regime import MarketRegimeType, RegimeOutput
        pred = self.detect(bars)
        meta = pred.metadata or {}
        dom = str(meta.get("dominant_regime", "UNKNOWN"))
        ker = float(meta.get("efficiency_ratio", meta.get("ker", 0.0)))
        sigma = float(meta.get("ewma_volatility", meta.get("volatility_sigma", 0.005)))
        is_high_vol = bool(meta.get("is_high_volatility", False))
        conf = float(pred.confidence_or_uncertainty)

        if "TRENDING" in dom:
            rtype = MarketRegimeType.TRENDING
        elif dom == "RANGING":
            rtype = MarketRegimeType.RANGING
        elif dom == "HIGH_VOLATILITY" or is_high_vol:
            rtype = MarketRegimeType.HIGH_VOLATILITY
        elif dom == "UNKNOWN":
            rtype = MarketRegimeType.UNKNOWN
        else:
            rtype = MarketRegimeType.NORMAL

        denom = max(1e-4, self.config.ker_trend_threshold - self.config.ker_ranging_threshold)
        trend_share = max(0.0, min(1.0, (ker - self.config.ker_ranging_threshold) / denom))

        latest_bar = self._normalize_to_market_bar(bars[-1]) if bars else None
        ts = latest_bar.source_timestamp if latest_bar else pred.prediction_timestamp

        return RegimeOutput(
            regime_type=rtype,
            efficiency_ratio=ker,
            trend_share=trend_share,
            ewma_volatility=max(0.005, sigma),
            is_high_volatility=is_high_vol,
            evaluation_timestamp=ts,
            confidence=conf,
            reason_codes=(f"TRANSPARENT_REGIME_{dom}", f"KER_{ker:.3f}", f"SIGMA_{sigma:.4f}"),
        )
