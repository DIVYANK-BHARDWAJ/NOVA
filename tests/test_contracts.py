"""Unit tests for NOVA MAX Quantitative Data Contracts (Engineer B — Phase 1 & 1.1 Remediation).

Uses Python 3.11 standard-library unittest only.
All test fixtures are purely synthetic.
"""

from datetime import datetime, timezone, timedelta
import math
from pathlib import Path
import sys
import unittest

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

    def test_boolean_price_rejected(self) -> None:
        """Booleans must not silently pass as prices."""
        with self.assertRaises(NumericalValidationError):
            MarketBar(
                ticker="SYNTH_TICKER",
                open=True,  # Boolean rejected
                high=105.0,
                low=95.0,
                close=100.0,
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
        naive_dt = datetime(2026, 3, 1, 9, 15)
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
        self.assertFalse(bar_valid.is_causally_available(as_of=self.t0))
        self.assertTrue(bar_valid.is_causally_available(as_of=self.t1))
        self.assertTrue(bar_valid.is_causally_available(as_of=self.t1 + timedelta(seconds=1)))


class TestCandidateSignalContracts(unittest.TestCase):
    """Synthetic test suite for CandidateSignal invariants and boundary rules."""

    def setUp(self) -> None:
        self.utc = timezone.utc
        self.now = datetime(2026, 3, 1, 9, 30, tzinfo=self.utc)

    def test_valid_candidate_signal_with_complete_net_edge(self) -> None:
        """Synthetic fixture: valid momentum signal with complete net edge components."""
        sig = CandidateSignal(
            strategy_id="MOM_ADAPTIVE_V1",
            strategy_version="1.0.0",
            ticker="SYNTH_ASSET_B",
            direction=SignalDirection.BULLISH,
            signal_score=0.75,
            score_semantics=ScoreSemantics.RAW_INTENSITY,
            evaluation_timestamp=self.now,
            prediction_horizon_bars=12,
            prediction_horizon_unit="BARS",
            expected_gross_return=0.0150,      # +1.50% decimal return
            trading_cost_estimate=0.0020,      # 20 bps
            execution_friction_estimate=0.0010, # 10 bps
            uncertainty_penalty=0.0020,        # 20 bps
            expected_net_edge=0.0100,          # 1.50% - (0.20% + 0.10% + 0.20%) = 1.00%
            reason_codes=("TREND_ACCELERATION_CONFIRMED", "LOW_VOLATILITY_EXPANSION"),
        )
        self.assertEqual(sig.direction, SignalDirection.BULLISH)
        net_edge = sig.compute_expected_net_edge()
        self.assertIsNotNone(net_edge)
        assert net_edge is not None
        self.assertAlmostEqual(net_edge, 0.0100)

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
        self.assertFalse(hasattr(sig, "submit_order"))
        self.assertFalse(hasattr(sig, "execute"))
        self.assertFalse(hasattr(sig, "order_id"))
        self.assertFalse(hasattr(sig, "quantity"))
        self.assertFalse(hasattr(sig, "order_type"))
        self.assertFalse(hasattr(sig, "limit_price"))

    def test_net_edge_returns_none_if_any_component_missing(self) -> None:
        """Remediation A: If ANY of gross return, cost, friction, or penalty is None, net edge must be None (no fabricated zeros)."""
        base_params = {
            "strategy_id": "MOM_V1",
            "strategy_version": "1.0.0",
            "ticker": "SYNTH_ASSET_B",
            "direction": SignalDirection.BULLISH,
            "signal_score": 0.5,
            "score_semantics": ScoreSemantics.RAW_INTENSITY,
            "evaluation_timestamp": self.now,
        }

        # 1. Missing gross return
        s1 = CandidateSignal(**base_params, expected_gross_return=None, trading_cost_estimate=0.002, execution_friction_estimate=0.001, uncertainty_penalty=0.001)
        self.assertIsNone(s1.compute_expected_net_edge())

        # 2. Missing trading cost
        s2 = CandidateSignal(**base_params, expected_gross_return=0.02, trading_cost_estimate=None, execution_friction_estimate=0.001, uncertainty_penalty=0.001)
        self.assertIsNone(s2.compute_expected_net_edge())

        # 3. Missing execution friction
        s3 = CandidateSignal(**base_params, expected_gross_return=0.02, trading_cost_estimate=0.002, execution_friction_estimate=None, uncertainty_penalty=0.001)
        self.assertIsNone(s3.compute_expected_net_edge())

        # 4. Missing uncertainty penalty
        s4 = CandidateSignal(**base_params, expected_gross_return=0.02, trading_cost_estimate=0.002, execution_friction_estimate=0.001, uncertainty_penalty=None)
        self.assertIsNone(s4.compute_expected_net_edge())

    def test_explicit_net_edge_with_incomplete_components_rejected(self) -> None:
        """Remediation A: Supplying explicit expected_net_edge when components are incomplete must raise ContractValidationError."""
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
                trading_cost_estimate=0.002,
                execution_friction_estimate=None,  # Incomplete!
                uncertainty_penalty=0.001,
                expected_net_edge=0.017,
            )

    def test_net_edge_mismatch_rejected(self) -> None:
        """Remediation A: Explicitly supplied expected_net_edge must match computed components."""
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
                trading_cost_estimate=0.002,
                execution_friction_estimate=0.001,
                uncertainty_penalty=0.001,
                expected_net_edge=0.050,  # Inconsistent: 0.02 - 0.004 = 0.016 != 0.050
            )

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

        with self.assertRaises(NumericalValidationError):
            CandidateSignal(
                strategy_id="MOM_V1",
                strategy_version="1.0.0",
                ticker="SYNTH_ASSET_B",
                direction=SignalDirection.BULLISH,
                signal_score=0.5,
                score_semantics=ScoreSemantics.RAW_INTENSITY,
                evaluation_timestamp=self.now,
                uncertainty_penalty=-0.001,
            )

    def test_score_semantics_raw_intensity_boundaries(self) -> None:
        """Remediation B: RAW_INTENSITY must be bounded in [-1.0, 1.0]."""
        base = {
            "strategy_id": "MOM_V1",
            "strategy_version": "1.0.0",
            "ticker": "SYNTH_TICKER",
            "direction": SignalDirection.BULLISH,
            "score_semantics": ScoreSemantics.RAW_INTENSITY,
            "evaluation_timestamp": self.now,
        }
        # Valid boundaries
        CandidateSignal(**base, signal_score=-1.0)
        CandidateSignal(**base, signal_score=0.0)
        CandidateSignal(**base, signal_score=1.0)

        # Invalid
        with self.assertRaises(NumericalValidationError):
            CandidateSignal(**base, signal_score=1.0001)

        with self.assertRaises(NumericalValidationError):
            CandidateSignal(**base, signal_score=-1.0001)

    def test_score_semantics_rank_decile_boundaries(self) -> None:
        """Remediation B: RANK_DECILE must be in [1.0, 10.0]."""
        base = {
            "strategy_id": "MOM_V1",
            "strategy_version": "1.0.0",
            "ticker": "SYNTH_TICKER",
            "direction": SignalDirection.BULLISH,
            "score_semantics": ScoreSemantics.RANK_DECILE,
            "evaluation_timestamp": self.now,
        }
        CandidateSignal(**base, signal_score=1.0)
        CandidateSignal(**base, signal_score=5.5)
        CandidateSignal(**base, signal_score=10.0)

        with self.assertRaises(NumericalValidationError):
            CandidateSignal(**base, signal_score=0.9)

        with self.assertRaises(NumericalValidationError):
            CandidateSignal(**base, signal_score=10.1)

    def test_score_semantics_rank_percentile_boundaries(self) -> None:
        """Remediation B: RANK_PERCENTILE must be in [0.0, 1.0]."""
        base = {
            "strategy_id": "MOM_V1",
            "strategy_version": "1.0.0",
            "ticker": "SYNTH_TICKER",
            "direction": SignalDirection.BULLISH,
            "score_semantics": ScoreSemantics.RANK_PERCENTILE,
            "evaluation_timestamp": self.now,
        }
        CandidateSignal(**base, signal_score=0.0)
        CandidateSignal(**base, signal_score=0.95)
        CandidateSignal(**base, signal_score=1.0)

        with self.assertRaises(NumericalValidationError):
            CandidateSignal(**base, signal_score=-0.01)

        with self.assertRaises(NumericalValidationError):
            CandidateSignal(**base, signal_score=1.01)

    def test_score_semantics_z_score_and_edge_bps(self) -> None:
        """Remediation B: Z_SCORE and EMPIRICAL_EDGE_BPS allow unbounded finite floats."""
        base = {
            "strategy_id": "MOM_V1",
            "strategy_version": "1.0.0",
            "ticker": "SYNTH_TICKER",
            "direction": SignalDirection.BULLISH,
            "evaluation_timestamp": self.now,
        }
        sig_z = CandidateSignal(**base, score_semantics=ScoreSemantics.Z_SCORE, signal_score=3.85)
        self.assertEqual(sig_z.signal_score, 3.85)

        sig_bps = CandidateSignal(**base, score_semantics=ScoreSemantics.EMPIRICAL_EDGE_BPS, signal_score=45.2)
        self.assertEqual(sig_bps.signal_score, 45.2)

    def test_prediction_horizon_validation(self) -> None:
        """Remediation B: Prediction horizon must be a positive integer, non-boolean, and unit non-empty."""
        base = {
            "strategy_id": "MOM_V1",
            "strategy_version": "1.0.0",
            "ticker": "SYNTH_TICKER",
            "direction": SignalDirection.BULLISH,
            "signal_score": 0.5,
            "score_semantics": ScoreSemantics.RAW_INTENSITY,
            "evaluation_timestamp": self.now,
        }
        with self.assertRaises(ContractValidationError):
            CandidateSignal(**base, prediction_horizon_bars=0)

        with self.assertRaises(ContractValidationError):
            CandidateSignal(**base, prediction_horizon_bars=True)  # Boolean rejected

        with self.assertRaises(ContractValidationError):
            CandidateSignal(**base, prediction_horizon_unit="   ")  # Blank unit rejected


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
            signal_score=8.0,
            score_semantics=ScoreSemantics.RANK_DECILE,
            evaluation_timestamp=self.now,
        )

    def test_valid_candidate_produced(self) -> None:
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
        with self.assertRaises(ContractValidationError):
            StrategyEvaluation(
                status=EvaluationStatus.CANDIDATE_PRODUCED,
                strategy_id="REGIME_RS_V1",
                strategy_version="1.0.0",
                evaluation_timestamp=self.now,
                candidate=None,
            )

    def test_strategy_id_mismatch_rejected(self) -> None:
        with self.assertRaises(ContractValidationError):
            StrategyEvaluation(
                status=EvaluationStatus.CANDIDATE_PRODUCED,
                strategy_id="DIFFERENT_STRATEGY_ID",
                strategy_version="1.0.0",
                evaluation_timestamp=self.now,
                candidate=self.valid_candidate,
            )

    def test_abstention_with_candidate_payload_rejected(self) -> None:
        with self.assertRaises(ContractValidationError):
            StrategyEvaluation(
                status=EvaluationStatus.ABSTAIN_MARKET_CONDITIONS,
                strategy_id="REGIME_RS_V1",
                strategy_version="1.0.0",
                evaluation_timestamp=self.now,
                candidate=self.valid_candidate,
                reason_codes=("HIGH_VOLATILITY_REGIME",),
            )

    def test_valid_abstention(self) -> None:
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
                confidence_score=1.5,
            )

    def test_research_metadata_blank_field_rejected(self) -> None:
        with self.assertRaises(ContractValidationError):
            ResearchMetadata(
                strategy_id="MOM_V1",
                strategy_version="1.0.0",
                config_version="",
                config_fingerprint="abc12345",
            )


if __name__ == "__main__":
    unittest.main()
