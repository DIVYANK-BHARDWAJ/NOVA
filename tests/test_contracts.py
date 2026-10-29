"""Unit tests for NOVA MAX Quantitative Data Contracts (Engineer B — Phase 1).

Uses Python 3.11 standard-library unittest only.
All test fixtures are purely synthetic.
"""

from datetime import datetime, timezone, timedelta
import math
from pathlib import Path
import sys
import unittest

# Ensure src/ is on PYTHONPATH for tests
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nova.quant.contracts import (
    ContractValidationError,
    DataQualityStatus,
    EvaluationStatus,
    MarketBar,
    CandidateSignal,
    RegimeCategory,
    RegimeState,
    ResearchMetadata,
    ScoreSemantics,
    SignalDirection,
    StrategyEvaluation,
    TemporalValidationError,
    NumericalValidationError,
    TimestampSemantics,
)


class TestMarketBarContracts(unittest.TestCase):
    """Synthetic test suite for MarketBar validation and causal integrity."""

    def setUp(self) -> None:
        self.utc = timezone.utc
        self.t0 = datetime(2026, 3, 1, 9, 15, tzinfo=self.utc)
        self.t1 = datetime(2026, 3, 1, 9, 20, tzinfo=self.utc)

    def test_valid_market_bar(self) -> None:
        """Synthetic fixture: standard valid 5-minute candle."""
        bar = MarketBar(
            ticker="SYNTHETIC_ASSET_A",
            open=100.0,
            high=105.0,
            low=98.0,
            close=102.5,
            volume=5000.0,
            source_timestamp=self.t0,
            available_timestamp=self.t1,
            timestamp_semantics=TimestampSemantics.CLOSE_TIME,
            data_quality=DataQualityStatus.VALID,
            provenance="SYNTHETIC_FIXTURE_SRC",
        )
        self.assertEqual(bar.ticker, "SYNTHETIC_ASSET_A")
        self.assertEqual(bar.close, 102.5)
        self.assertEqual(bar.contract_version, "PROPOSED_V0")

    def test_blank_ticker_rejected(self) -> None:
        """Ticker cannot be blank or whitespace-only."""
        with self.assertRaises(ContractValidationError):
            MarketBar(
                ticker="   ",
                open=100.0,
                high=105.0,
                low=98.0,
                close=102.5,
                volume=100.0,
                source_timestamp=self.t0,
            )

    def test_negative_or_zero_price_rejected(self) -> None:
        """Prices must be strictly positive."""
        with self.assertRaises(NumericalValidationError):
            MarketBar(
                ticker="SYNTH_TICKER",
                open=0.0,
                high=105.0,
                low=0.0,
                close=102.0,
                volume=100.0,
                source_timestamp=self.t0,
            )

        with self.assertRaises(NumericalValidationError):
            MarketBar(
                ticker="SYNTH_TICKER",
                open=-10.0,
                high=105.0,
                low=-15.0,
                close=102.0,
                volume=100.0,
                source_timestamp=self.t0,
            )

    def test_negative_volume_rejected(self) -> None:
        """Volume cannot be negative."""
        with self.assertRaises(NumericalValidationError):
            MarketBar(
                ticker="SYNTH_TICKER",
                open=100.0,
                high=105.0,
                low=95.0,
                close=102.0,
                volume=-1.0,
                source_timestamp=self.t0,
            )

    def test_ohlc_relationship_invariants(self) -> None:
        """Low cannot exceed high; open/close must stay inside [low, high]."""
        # Low > High
        with self.assertRaises(ContractValidationError):
            MarketBar(
                ticker="SYNTH_TICKER",
                open=100.0,
                high=95.0,
                low=105.0,
                close=100.0,
                volume=100.0,
                source_timestamp=self.t0,
            )

        # Open > High
        with self.assertRaises(ContractValidationError):
            MarketBar(
                ticker="SYNTH_TICKER",
                open=110.0,
                high=105.0,
                low=95.0,
                close=100.0,
                volume=100.0,
                source_timestamp=self.t0,
            )

        # Close < Low
        with self.assertRaises(ContractValidationError):
            MarketBar(
                ticker="SYNTH_TICKER",
                open=100.0,
                high=105.0,
                low=95.0,
                close=90.0,
                volume=100.0,
                source_timestamp=self.t0,
            )

    def test_non_finite_numeric_rejected(self) -> None:
        """NaN or inf in prices or volume must be strictly rejected."""
        with self.assertRaises(NumericalValidationError):
            MarketBar(
                ticker="SYNTH_TICKER",
                open=float("nan"),
                high=105.0,
                low=95.0,
                close=100.0,
                volume=100.0,
                source_timestamp=self.t0,
            )

        with self.assertRaises(NumericalValidationError):
            MarketBar(
                ticker="SYNTH_TICKER",
                open=100.0,
                high=float("inf"),
                low=95.0,
                close=100.0,
                volume=100.0,
                source_timestamp=self.t0,
            )

    def test_naive_timestamp_rejected(self) -> None:
        """Naive datetime without timezone must raise TemporalValidationError."""
        naive_dt = datetime(2026, 3, 1, 9, 15)  # No tzinfo
        with self.assertRaises(TemporalValidationError):
            MarketBar(
                ticker="SYNTH_TICKER",
                open=100.0,
                high=105.0,
                low=95.0,
                close=100.0,
                volume=100.0,
                source_timestamp=naive_dt,
            )

    def test_causal_availability_rules(self) -> None:
        """A bar is NEVER causally available if semantics are UNKNOWN or available_timestamp is None."""
        # Case 1: UNKNOWN semantics cannot be assumed causally available
        bar_unknown = MarketBar(
            ticker="SYNTH_TICKER",
            open=100.0,
            high=105.0,
            low=95.0,
            close=100.0,
            volume=100.0,
            source_timestamp=self.t0,
            available_timestamp=self.t1,
            timestamp_semantics=TimestampSemantics.UNKNOWN,
        )
        self.assertFalse(bar_unknown.is_causally_available(as_of=self.t1 + timedelta(minutes=5)))

        # Case 2: Available timestamp missing
        bar_no_avail = MarketBar(
            ticker="SYNTH_TICKER",
            open=100.0,
            high=105.0,
            low=95.0,
            close=100.0,
            volume=100.0,
            source_timestamp=self.t0,
            available_timestamp=None,
            timestamp_semantics=TimestampSemantics.CLOSE_TIME,
        )
        self.assertFalse(bar_no_avail.is_causally_available(as_of=self.t1 + timedelta(minutes=5)))

        # Case 3: Available timestamp verified and <= as_of
        bar_valid = MarketBar(
            ticker="SYNTH_TICKER",
            open=100.0,
            high=105.0,
            low=95.0,
            close=100.0,
            volume=100.0,
            source_timestamp=self.t0,
            available_timestamp=self.t1,
            timestamp_semantics=TimestampSemantics.CLOSE_TIME,
        )
        # Not available prior to t1
        self.assertFalse(bar_valid.is_causally_available(as_of=self.t0))
        # Available at or after t1
        self.assertTrue(bar_valid.is_causally_available(as_of=self.t1))
        self.assertTrue(bar_valid.is_causally_available(as_of=self.t1 + timedelta(seconds=1)))


class TestCandidateSignalContracts(unittest.TestCase):
    """Synthetic test suite for CandidateSignal invariants and boundary rules."""

    def setUp(self) -> None:
        self.utc = timezone.utc
        self.now = datetime(2026, 3, 1, 9, 30, tzinfo=self.utc)

    def test_valid_candidate_signal(self) -> None:
        """Synthetic fixture: valid momentum signal with explicit units."""
        sig = CandidateSignal(
            strategy_id="MOM_ADAPTIVE_V1",
            strategy_version="1.0.0",
            ticker="SYNTH_ASSET_B",
            direction=SignalDirection.BULLISH,
            signal_score=0.75,
            score_semantics=ScoreSemantics.RAW_INTENSITY,
            evaluation_timestamp=self.now,
            prediction_horizon_bars=12,
            expected_gross_return=0.0150,      # +1.50% decimal return
            trading_cost_estimate=0.0020,      # 20 bps
            execution_friction_estimate=0.0010, # 10 bps
            uncertainty_penalty=0.0020,        # 20 bps
            expected_net_edge=0.0100,          # 1.50% - (0.20% + 0.10% + 0.20%) = 1.00%
            reason_codes=("TREND_ACCELERATION_CONFIRMED", "LOW_VOLATILITY_EXPANSION"),
        )
        self.assertEqual(sig.direction, SignalDirection.BULLISH)
        self.assertAlmostEqual(sig.compute_expected_net_edge(), 0.0100)

    def test_candidate_signal_cannot_execute_orders(self) -> None:
        """Strict Ownership Guard: CandidateSignal has NO execution capabilities."""
        sig = CandidateSignal(
            strategy_id="MOM_ADAPTIVE_V1",
            strategy_version="1.0.0",
            ticker="SYNTH_ASSET_B",
            direction=SignalDirection.BULLISH,
            signal_score=0.5,
            score_semantics=ScoreSemantics.RAW_INTENSITY,
            evaluation_timestamp=self.now,
        )
        # Ensure no order methods or execution fields exist
        self.assertFalse(hasattr(sig, "submit_order"))
        self.assertFalse(hasattr(sig, "execute"))
        self.assertFalse(hasattr(sig, "order_id"))
        self.assertFalse(hasattr(sig, "quantity"))
        self.assertFalse(hasattr(sig, "order_type"))
        self.assertFalse(hasattr(sig, "limit_price"))

    def test_no_fabricated_zero_for_missing_estimates(self) -> None:
        """If gross return or costs are unknown, compute_expected_net_edge must return None."""
        sig = CandidateSignal(
            strategy_id="MOM_ADAPTIVE_V1",
            strategy_version="1.0.0",
            ticker="SYNTH_ASSET_B",
            direction=SignalDirection.BULLISH,
            signal_score=0.5,
            score_semantics=ScoreSemantics.RAW_INTENSITY,
            evaluation_timestamp=self.now,
            expected_gross_return=None,
            trading_cost_estimate=None,
        )
        self.assertIsNone(sig.compute_expected_net_edge())

    def test_negative_costs_rejected(self) -> None:
        """Costs and penalties cannot be negative."""
        with self.assertRaises(NumericalValidationError):
            CandidateSignal(
                strategy_id="MOM_V1",
                strategy_version="1.0.0",
                ticker="SYNTH_ASSET_B",
                direction=SignalDirection.BULLISH,
                signal_score=0.5,
                score_semantics=ScoreSemantics.RAW_INTENSITY,
                evaluation_timestamp=self.now,
                trading_cost_estimate=-0.001,
            )

    def test_net_edge_inconsistency_rejected(self) -> None:
        """Explicitly passed net edge must match computed components."""
        with self.assertRaises(ContractValidationError):
            CandidateSignal(
                strategy_id="MOM_V1",
                strategy_version="1.0.0",
                ticker="SYNTH_ASSET_B",
                direction=SignalDirection.BULLISH,
                signal_score=0.5,
                score_semantics=ScoreSemantics.RAW_INTENSITY,
                evaluation_timestamp=self.now,
                expected_gross_return=0.02,
                trading_cost_estimate=0.005,
                expected_net_edge=0.05,  # Inconsistent: 0.02 - 0.005 != 0.05
            )


class TestStrategyEvaluationContracts(unittest.TestCase):
    """Synthetic test suite for StrategyEvaluation consistency rules."""

    def setUp(self) -> None:
        self.utc = timezone.utc
        self.now = datetime(2026, 3, 1, 9, 30, tzinfo=self.utc)
        self.valid_candidate = CandidateSignal(
            strategy_id="REGIME_RS_V1",
            strategy_version="1.0.0",
            ticker="SYNTH_STOCK_X",
            direction=SignalDirection.BULLISH,
            signal_score=0.8,
            score_semantics=ScoreSemantics.RANK_DECILE,
            evaluation_timestamp=self.now,
        )

    def test_valid_candidate_produced(self) -> None:
        """CANDIDATE_PRODUCED with matching candidate payload."""
        eval_res = StrategyEvaluation(
            status=EvaluationStatus.CANDIDATE_PRODUCED,
            strategy_id="REGIME_RS_V1",
            strategy_version="1.0.0",
            evaluation_timestamp=self.now,
            candidate=self.valid_candidate,
            reason_codes=("TOP_QUINTILE_BREAKOUT",),
        )
        self.assertEqual(eval_res.status, EvaluationStatus.CANDIDATE_PRODUCED)
        self.assertIsNotNone(eval_res.candidate)

    def test_candidate_produced_missing_candidate_rejected(self) -> None:
        """CANDIDATE_PRODUCED status requires candidate to be non-None."""
        with self.assertRaises(ContractValidationError):
            StrategyEvaluation(
                status=EvaluationStatus.CANDIDATE_PRODUCED,
                strategy_id="REGIME_RS_V1",
                strategy_version="1.0.0",
                evaluation_timestamp=self.now,
                candidate=None,
            )

    def test_strategy_id_mismatch_rejected(self) -> None:
        """Strategy ID in evaluation must match candidate's strategy ID."""
        with self.assertRaises(ContractValidationError):
            StrategyEvaluation(
                status=EvaluationStatus.CANDIDATE_PRODUCED,
                strategy_id="DIFFERENT_STRATEGY_ID",
                strategy_version="1.0.0",
                evaluation_timestamp=self.now,
                candidate=self.valid_candidate,
            )

    def test_abstention_with_candidate_payload_rejected(self) -> None:
        """Abstention status with non-None candidate payload is contradictory and must raise."""
        with self.assertRaises(ContractValidationError):
            StrategyEvaluation(
                status=EvaluationStatus.ABSTAIN_MARKET_CONDITIONS,
                strategy_id="REGIME_RS_V1",
                strategy_version="1.0.0",
                evaluation_timestamp=self.now,
                candidate=self.valid_candidate,  # Contradictory!
                reason_codes=("HIGH_VOLATILITY_REGIME",),
            )

    def test_valid_abstention(self) -> None:
        """Abstention with candidate=None is completely valid."""
        eval_res = StrategyEvaluation(
            status=EvaluationStatus.ABSTAIN_INSUFFICIENT_HISTORY,
            strategy_id="REGIME_RS_V1",
            strategy_version="1.0.0",
            evaluation_timestamp=self.now,
            candidate=None,
            reason_codes=("WARMUP_BARS_NOT_REACHED",),
        )
        self.assertEqual(eval_res.status, EvaluationStatus.ABSTAIN_INSUFFICIENT_HISTORY)
        self.assertIsNone(eval_res.candidate)


class TestRegimeStateAndMetadata(unittest.TestCase):
    """Synthetic test suite for RegimeState and ResearchMetadata."""

    def setUp(self) -> None:
        self.utc = timezone.utc
        self.now = datetime(2026, 3, 1, 9, 30, tzinfo=self.utc)

    def test_valid_regime_state(self) -> None:
        regime = RegimeState(
            regime_label=RegimeCategory.TRENDING_BULLISH,
            evaluation_timestamp=self.now,
            confidence_score=0.85,
            supporting_indicators=(("ADX_14", 32.5), ("ADVANCE_DECLINE_RATIO", 2.1)),
            reason_codes=("STRONG_TREND_PERSISTENCE",),
        )
        self.assertEqual(regime.regime_label, RegimeCategory.TRENDING_BULLISH)
        self.assertEqual(regime.confidence_score, 0.85)

    def test_invalid_confidence_score_rejected(self) -> None:
        with self.assertRaises(NumericalValidationError):
            RegimeState(
                regime_label=RegimeCategory.HIGH_VOLATILITY_CHOP,
                evaluation_timestamp=self.now,
                confidence_score=1.5,  # Out of range [0.0, 1.0]
            )

    def test_research_metadata_blank_field_rejected(self) -> None:
        with self.assertRaises(ContractValidationError):
            ResearchMetadata(
                strategy_id="MOM_V1",
                strategy_version="1.0.0",
                config_version="",  # Blank string rejected
                config_fingerprint="abc12345",
            )


if __name__ == "__main__":
    unittest.main()
