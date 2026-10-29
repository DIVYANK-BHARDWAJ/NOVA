"""Unit and regression tests for NOVA causal moving average feature calculations."""

from datetime import datetime, timezone, timedelta
import math
from pathlib import Path
import sys
import unittest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nova.features.moving_averages import (
    EMACrossoverFeatures,
    calculate_ema_series,
    compute_ema_crossover_features,
    validate_window,
)
from nova.quant.contracts import ContractValidationError, NumericalValidationError


class TestMovingAverageFeatures(unittest.TestCase):
    """Verifies deterministic, causal EMA calculations and invariant handling."""

    def test_hand_calculated_ema_series(self) -> None:
        """Verify against an independently hand-calculated 3-period EMA series.

        Given series = [10.0, 11.0, 12.0] and window = 3:
            alpha = 2 / (3 + 1) = 0.5
            EMA_0 = 10.0
            EMA_1 = 0.5 * 11.0 + 0.5 * 10.0 = 10.5
            EMA_2 = 0.5 * 12.0 + 0.5 * 10.5 = 11.25
        """
        series = [10.0, 11.0, 12.0]
        ema = calculate_ema_series(series, window=3)
        self.assertEqual(len(ema), 3)
        self.assertAlmostEqual(ema[0], 10.0, places=7)
        self.assertAlmostEqual(ema[1], 10.5, places=7)
        self.assertAlmostEqual(ema[2], 11.25, places=7)

    def test_constant_prices(self) -> None:
        """Constant prices must produce an identical constant EMA."""
        series = [50.0] * 10
        ema = calculate_ema_series(series, window=5)
        for val in ema:
            self.assertAlmostEqual(val, 50.0, places=7)

    def test_rising_and_falling_prices(self) -> None:
        """Rising prices yield strictly increasing EMAs; falling yield decreasing."""
        rising = [10.0 + i for i in range(10)]
        ema_rising = calculate_ema_series(rising, window=4)
        for i in range(1, len(ema_rising)):
            self.assertGreater(ema_rising[i], ema_rising[i - 1])

        falling = [100.0 - i * 2 for i in range(10)]
        ema_falling = calculate_ema_series(falling, window=4)
        for i in range(1, len(ema_falling)):
            self.assertLess(ema_falling[i], ema_falling[i - 1])

    def test_empty_series(self) -> None:
        """Empty input series returns an empty list without error."""
        self.assertEqual(calculate_ema_series([], window=5), [])

    def test_invalid_window_validation(self) -> None:
        """Windows < 2 or non-integers/booleans must be rejected."""
        with self.assertRaises(NumericalValidationError):
            validate_window(1)
        with self.assertRaises(NumericalValidationError):
            validate_window(0)
        with self.assertRaises(NumericalValidationError):
            validate_window(-5)
        with self.assertRaises(NumericalValidationError):
            validate_window(True)  # type: ignore
        with self.assertRaises(NumericalValidationError):
            validate_window("12")  # type: ignore

    def test_non_finite_and_negative_prices_rejected(self) -> None:
        """Prices with NaN, Inf, or <= 0.0 must raise NumericalValidationError."""
        with self.assertRaises(NumericalValidationError):
            calculate_ema_series([10.0, float("nan"), 12.0], window=3)
        with self.assertRaises(NumericalValidationError):
            calculate_ema_series([10.0, float("inf"), 12.0], window=3)
        with self.assertRaises(NumericalValidationError):
            calculate_ema_series([10.0, -1.0, 12.0], window=3)
        with self.assertRaises(NumericalValidationError):
            calculate_ema_series([10.0, 0.0, 12.0], window=3)
        with self.assertRaises(NumericalValidationError):
            calculate_ema_series([10.0, True, 12.0], window=3)  # type: ignore

    def test_fast_period_must_be_strictly_less_than_slow_period(self) -> None:
        """fast_window >= slow_window must raise ContractValidationError."""
        closes = [10.0] * 10
        ts = [datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=i) for i in range(10)]

        with self.assertRaises(ContractValidationError):
            compute_ema_crossover_features(closes, ts, fast_window=10, slow_window=10)
        with self.assertRaises(ContractValidationError):
            compute_ema_crossover_features(closes, ts, fast_window=15, slow_window=10)

    def test_warmup_flag_semantics(self) -> None:
        """Features must have is_warmed_up=False for index < slow_window - 1."""
        closes = [10.0 + i for i in range(20)]
        ts = [datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=i) for i in range(20)]
        features = compute_ema_crossover_features(closes, ts, fast_window=3, slow_window=6)

        self.assertEqual(len(features), 20)
        for i, feat in enumerate(features):
            if i < 5:  # slow_window - 1 = 5
                self.assertFalse(feat.is_warmed_up)
            else:
                self.assertTrue(feat.is_warmed_up)


if __name__ == "__main__":
    unittest.main()
