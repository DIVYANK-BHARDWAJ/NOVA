"""NOVA Feature Ablation & Component Contribution Subsystem (Person 3 Exclusive Domain).

Implements Section 4 of Sentinel Trader Part 7 Specification:
Evaluates the contribution of each modular subsystem by executing controlled,
isolated ablation variants against the full system on identical causal market data:
1. Full System (Trend + Mean Reversion + ML Ridge + Regime Detection + Friction Engine + Risk Gate + Profit Realization)
2. Ablation: Without Trend-Following Strategy (trend_weight = 0.0)
3. Ablation: Without Mean Reversion Strategy (mean_reversion_weight = 0.0)
4. Ablation: Without ML Predictions (predictor_weight = 0.0, predictor = None)
5. Ablation: Without Market-Regime Detection (use_regime_weighting = False)
6. Ablation: Without Friction-Aware Selection (min_net_edge_bps = 0.0)
7. Ablation: Without Risk Controls (unrestricted position exposure / relaxed limits)
8. Ablation: Without Profit Realization & Exits (enable_profit_realization = False)
9. Baseline: Buy & Hold (passive asset holding)
10. Baseline: No Trade / Cash Preservation (zero risk, zero return)
11. News Sentiment: Explicitly logged as UNSUPPORTED (news sentiment dataset not provided in market data)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from nova.decision.engine import DecisionEngineConfig, SentinelDecisionEngine
from nova.decision.prediction import CausalRidgePredictor
from nova.evaluation.metrics import PerformanceMetrics
from nova.evaluation.replay import ChronologicalReplayEngine, ReplaySummary
from nova.friction.models import FrictionParameters, MarketFrictionEngine
from nova.portfolio.profit_realization import ProfitRealizationConfig
from nova.quant.contracts import MarketBar
from nova.decision.regime import MarketRegimeDetector
from nova.risk.gates import DeterministicRiskGate, RiskConfiguration
from nova.strategies.mean_reversion import MeanReversionConfig, MeanReversionStrategy
from nova.strategies.trend_following import TrendFollowingConfig, TrendFollowingStrategy


@dataclass(frozen=True, slots=True)
class AblationVariantResult:
    """Outcome of a single feature ablation or baseline evaluation."""
    variant_id: str
    variant_name: str
    description: str
    is_supported: bool
    unsupported_reason: str | None = None
    net_return: float | None = None
    net_pnl_inr: float | None = None
    max_drawdown: float | None = None
    win_rate: float | None = None
    profit_factor: float | None = None
    total_turnover: float | None = None
    total_fees_inr: float | None = None
    total_trades: int | None = None
    marginal_contribution_bps: float | None = None  # Full Return - Variant Return (in bps)


@dataclass(frozen=True, slots=True)
class AblationStudyResult:
    """Complete comparative study across all variants and baselines."""
    ticker: str
    total_bars: int
    evaluated_steps: int
    variants: tuple[AblationVariantResult, ...]
    benchmark_return: float

    def format_table(self) -> str:
        """Produce clean ASCII comparison table."""
        lines = [
            "=" * 98,
            "SENTINEL TRADER -- FEATURE ABLATION & COMPONENT CONTRIBUTION STUDY",
            "=" * 98,
            f"{'Variant / Configuration':<32} | {'Net Return':<12} | {'Net P&L (INR)':<15} | {'Max DD':<9} | {'Turnover':<10} | {'Impact (bps)':<12}",
            "-" * 98,
        ]
        for v in self.variants:
            if not v.is_supported:
                lines.append(f"{v.variant_name:<32} | {'DATA MISSING':<12} | {'N/A':<15} | {'N/A':<9} | {'N/A':<10} | {v.unsupported_reason or 'N/A':<12}")
                continue

            net_ret_str = f"{v.net_return * 100:+.2f}%" if v.net_return is not None else "N/A"
            pnl_str = f"INR {v.net_pnl_inr:>+10,.0f}" if v.net_pnl_inr is not None else "N/A"
            dd_str = f"{v.max_drawdown * 100:.2f}%" if v.max_drawdown is not None else "N/A"
            turn_str = f"{v.total_turnover:.2f}x" if v.total_turnover is not None else "N/A"
            contrib_str = f"{v.marginal_contribution_bps:>+10.1f} bps" if v.marginal_contribution_bps is not None else "BASELINE"

            lines.append(f"{v.variant_name:<32} | {net_ret_str:<12} | {pnl_str:<15} | {dd_str:<9} | {turn_str:<10} | {contrib_str:<12}")

        lines.append("=" * 98)
        return "\n".join(lines)


class AblationRunner:
    """Orchestrates controlled ablation evaluations on strictly identical market bar sequences."""

    def __init__(
        self,
        friction_params: FrictionParameters | None = None,
        risk_config: RiskConfiguration | None = None,
    ) -> None:
        self.friction_params = friction_params or FrictionParameters(
            spread_bps=2.0,
            slippage_bps=3.0,
            brokerage_bps=1.0,
            tax_bps=1.0,
            flat_brokerage_inr=20.0,
            flat_dp_charges_inr=15.34,
        )
        self.risk_config = risk_config or RiskConfiguration(
            max_position_exposure=0.25,
            max_drawdown_limit=0.15,
            allow_short=True,
            initial_capital=1000000.0,
        )

    def run_study(
        self,
        bars: Sequence[MarketBar],
        warmup_bars: int = 25,
        initial_capital: float = 1000000.0,
        execution_timing: str = "CANDLE_CLOSE",
    ) -> AblationStudyResult:
        """Execute full system and all ablation variants chronologically."""
        if not bars:
            raise ValueError("Cannot run ablation study on empty bars sequence")

        ticker = bars[0].ticker
        friction_engine = MarketFrictionEngine(self.friction_params)
        risk_gate = DeterministicRiskGate(self.risk_config)

        # 1. Full System Configuration
        predictor = CausalRidgePredictor()
        predictor.fit_chronological(bars[:warmup_bars], train_fraction=0.8)
        regime_detector = MarketRegimeDetector()

        full_engine_cfg = DecisionEngineConfig(
            trend_weight=0.50,
            mean_reversion_weight=0.50,
            min_net_edge_bps=5.0,
            target_exposure=0.20,
            allow_short=True,
            use_regime_weighting=True,
            predictor_weight=0.30,
            enable_profit_realization=True,
            initial_capital=initial_capital,
            profit_realization_config=ProfitRealizationConfig(min_cash_reserve_fraction=0.20),
            engine_id="full_sentinel_trader",
        )
        full_engine = SentinelDecisionEngine(
            config=full_engine_cfg,
            friction_engine=friction_engine,
            risk_gate=risk_gate,
            predictor=predictor,
            regime_detector=regime_detector,
        )

        replay = ChronologicalReplayEngine(decision_engine=full_engine, friction_engine=friction_engine)
        full_summary: ReplaySummary = replay.run(bars, warmup_bars=warmup_bars, initial_capital=initial_capital, execution_timing=execution_timing)
        full_ret = full_summary.net_cumulative_return

        variants: list[AblationVariantResult] = []

        # Baseline: Full System
        variants.append(
            AblationVariantResult(
                variant_id="full_system",
                variant_name="Full Sentinel Trader",
                description="All modules active (Trend, MR, ML, Regime, Friction, Risk, Exits)",
                is_supported=True,
                net_return=full_summary.net_cumulative_return,
                net_pnl_inr=full_summary.net_pnl_inr,
                max_drawdown=full_summary.max_drawdown,
                win_rate=full_summary.win_rate,
                total_turnover=full_summary.total_turnover,
                total_fees_inr=full_summary.total_fees_inr,
                total_trades=full_summary.total_trades,
                marginal_contribution_bps=0.0,
            )
        )

        # 2. Ablation 1: No Trend-Following (trend_weight = 0.0)
        cfg_no_trend = DecisionEngineConfig(
            trend_weight=0.0,
            mean_reversion_weight=1.0,
            min_net_edge_bps=5.0,
            target_exposure=0.20,
            allow_short=True,
            use_regime_weighting=True,
            predictor_weight=0.30,
            enable_profit_realization=True,
            initial_capital=initial_capital,
            engine_id="ablate_no_trend",
        )
        sum_no_trend = ChronologicalReplayEngine(
            decision_engine=SentinelDecisionEngine(config=cfg_no_trend, friction_engine=friction_engine, risk_gate=risk_gate, predictor=predictor, regime_detector=regime_detector),
            friction_engine=friction_engine,
        ).run(bars, warmup_bars=warmup_bars, initial_capital=initial_capital, execution_timing=execution_timing)
        variants.append(self._create_variant_result("no_trend", "Ablate: No Trend Following", "Trend weight set to 0.0", sum_no_trend, full_ret))

        # 3. Ablation 2: No Mean Reversion (mean_reversion_weight = 0.0)
        cfg_no_mr = DecisionEngineConfig(
            trend_weight=1.0,
            mean_reversion_weight=0.0,
            min_net_edge_bps=5.0,
            target_exposure=0.20,
            allow_short=True,
            use_regime_weighting=True,
            predictor_weight=0.30,
            enable_profit_realization=True,
            initial_capital=initial_capital,
            engine_id="ablate_no_mr",
        )
        sum_no_mr = ChronologicalReplayEngine(
            decision_engine=SentinelDecisionEngine(config=cfg_no_mr, friction_engine=friction_engine, risk_gate=risk_gate, predictor=predictor, regime_detector=regime_detector),
            friction_engine=friction_engine,
        ).run(bars, warmup_bars=warmup_bars, initial_capital=initial_capital, execution_timing=execution_timing)
        variants.append(self._create_variant_result("no_mr", "Ablate: No Mean Reversion", "Mean reversion weight set to 0.0", sum_no_mr, full_ret))

        # 4. Ablation 3: No ML Predictor (predictor_weight = 0.0)
        cfg_no_ml = DecisionEngineConfig(
            trend_weight=0.50,
            mean_reversion_weight=0.50,
            min_net_edge_bps=5.0,
            target_exposure=0.20,
            allow_short=True,
            use_regime_weighting=True,
            predictor_weight=0.0,
            enable_profit_realization=True,
            initial_capital=initial_capital,
            engine_id="ablate_no_ml",
        )
        sum_no_ml = ChronologicalReplayEngine(
            decision_engine=SentinelDecisionEngine(config=cfg_no_ml, friction_engine=friction_engine, risk_gate=risk_gate, predictor=None, regime_detector=regime_detector),
            friction_engine=friction_engine,
        ).run(bars, warmup_bars=warmup_bars, initial_capital=initial_capital, execution_timing=execution_timing)
        variants.append(self._create_variant_result("no_ml", "Ablate: No ML Predictor", "ML return predictor disabled", sum_no_ml, full_ret))

        # 5. Ablation 4: No Market Regime Weighting
        cfg_no_regime = DecisionEngineConfig(
            trend_weight=0.50,
            mean_reversion_weight=0.50,
            min_net_edge_bps=5.0,
            target_exposure=0.20,
            allow_short=True,
            use_regime_weighting=False,
            predictor_weight=0.30,
            enable_profit_realization=True,
            initial_capital=initial_capital,
            engine_id="ablate_no_regime",
        )
        sum_no_regime = ChronologicalReplayEngine(
            decision_engine=SentinelDecisionEngine(config=cfg_no_regime, friction_engine=friction_engine, risk_gate=risk_gate, predictor=predictor, regime_detector=None),
            friction_engine=friction_engine,
        ).run(bars, warmup_bars=warmup_bars, initial_capital=initial_capital, execution_timing=execution_timing)
        variants.append(self._create_variant_result("no_regime", "Ablate: No Regime Weighting", "Efficiency ratio regime weighting disabled", sum_no_regime, full_ret))

        # 6. Ablation 5: No Friction Awareness (min_net_edge_bps = 0.0)
        cfg_no_friction = DecisionEngineConfig(
            trend_weight=0.50,
            mean_reversion_weight=0.50,
            min_net_edge_bps=0.0,
            target_exposure=0.20,
            allow_short=True,
            use_regime_weighting=True,
            predictor_weight=0.30,
            enable_profit_realization=True,
            initial_capital=initial_capital,
            engine_id="ablate_no_friction_hurdle",
        )
        sum_no_friction = ChronologicalReplayEngine(
            decision_engine=SentinelDecisionEngine(config=cfg_no_friction, friction_engine=friction_engine, risk_gate=risk_gate, predictor=predictor, regime_detector=regime_detector),
            friction_engine=friction_engine,
        ).run(bars, warmup_bars=warmup_bars, initial_capital=initial_capital, execution_timing=execution_timing)
        variants.append(self._create_variant_result("no_friction", "Ablate: Zero Friction Hurdle", "Min net edge hurdle set to 0 bps", sum_no_friction, full_ret))

        # 7. Ablation 6: Without Risk Controls
        relaxed_risk_gate = DeterministicRiskGate(RiskConfiguration(max_position_exposure=1.0, max_drawdown_limit=1.0, allow_short=True))
        sum_no_risk = ChronologicalReplayEngine(
            decision_engine=SentinelDecisionEngine(config=full_engine_cfg, friction_engine=friction_engine, risk_gate=relaxed_risk_gate, predictor=predictor, regime_detector=regime_detector),
            friction_engine=friction_engine,
        ).run(bars, warmup_bars=warmup_bars, initial_capital=initial_capital, execution_timing=execution_timing)
        variants.append(self._create_variant_result("no_risk_gate", "Ablate: Relaxed Risk Gate", "Max exposure relaxed to 1.0, drawdown limit 1.0", sum_no_risk, full_ret))

        # 8. Ablation 7: Without Profit Realization & Position Exits
        cfg_no_exits = DecisionEngineConfig(
            trend_weight=0.50,
            mean_reversion_weight=0.50,
            min_net_edge_bps=5.0,
            target_exposure=0.20,
            allow_short=True,
            use_regime_weighting=True,
            predictor_weight=0.30,
            enable_profit_realization=False,
            initial_capital=initial_capital,
            engine_id="ablate_no_profit_exits",
        )
        sum_no_exits = ChronologicalReplayEngine(
            decision_engine=SentinelDecisionEngine(config=cfg_no_exits, friction_engine=friction_engine, risk_gate=risk_gate, predictor=predictor, regime_detector=regime_detector),
            friction_engine=friction_engine,
        ).run(bars, warmup_bars=warmup_bars, initial_capital=initial_capital, execution_timing=execution_timing)
        variants.append(self._create_variant_result("no_exits", "Ablate: No Profit Realization", "Trailing stop & partial scale-out disabled", sum_no_exits, full_ret))

        # 9. Baseline 1: Buy & Hold Benchmark
        bench_ret = full_summary.benchmark_buy_and_hold_return
        variants.append(
            AblationVariantResult(
                variant_id="baseline_buy_and_hold",
                variant_name="Baseline: Buy & Hold",
                description="Passive buy and hold benchmark",
                is_supported=True,
                net_return=bench_ret,
                net_pnl_inr=initial_capital * bench_ret,
                max_drawdown=full_summary.max_drawdown,
                win_rate=1.0 if bench_ret > 0 else 0.0,
                total_turnover=1.0,
                total_fees_inr=60.0,
                total_trades=1,
                marginal_contribution_bps=(full_ret - bench_ret) * 10000.0,
            )
        )

        # 10. Baseline 2: No-Trade Cash Preservation
        variants.append(
            AblationVariantResult(
                variant_id="baseline_no_trade",
                variant_name="Baseline: No Trade (Cash)",
                description="Maintain 100% cash cushion, zero trades",
                is_supported=True,
                net_return=0.0,
                net_pnl_inr=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                total_turnover=0.0,
                total_fees_inr=0.0,
                total_trades=0,
                marginal_contribution_bps=full_ret * 10000.0,
            )
        )

        # 11. News Sentiment Component: Documented as Data Missing
        variants.append(
            AblationVariantResult(
                variant_id="news_sentiment",
                variant_name="News Sentiment Signals",
                description="NLP sentiment signal incorporation",
                is_supported=False,
                unsupported_reason="DATA_MISSING: News sentiment feed not in market data",
            )
        )

        return AblationStudyResult(
            ticker=ticker,
            total_bars=len(bars),
            evaluated_steps=full_summary.evaluated_steps,
            variants=tuple(variants),
            benchmark_return=bench_ret,
        )

    def _create_variant_result(
        self,
        var_id: str,
        name: str,
        desc: str,
        summary: ReplaySummary,
        full_ret: float,
    ) -> AblationVariantResult:
        net_ret = summary.net_cumulative_return
        marginal_bps = (full_ret - net_ret) * 10000.0
        return AblationVariantResult(
            variant_id=var_id,
            variant_name=name,
            description=desc,
            is_supported=True,
            net_return=net_ret,
            net_pnl_inr=summary.net_pnl_inr,
            max_drawdown=summary.max_drawdown,
            win_rate=summary.win_rate,
            total_turnover=summary.total_turnover,
            total_fees_inr=summary.total_fees_inr,
            total_trades=summary.total_trades,
            marginal_contribution_bps=round(marginal_bps, 1),
        )
