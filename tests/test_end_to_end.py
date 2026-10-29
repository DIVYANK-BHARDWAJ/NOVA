"""End-to-end integration tests for NOVA research execution pipeline and CLI."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
for d in (SRC_DIR, SCRIPTS_DIR):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

from nova.quant.contracts import ContractValidationError, NumericalValidationError
from nova.research.adapters import (
    generate_synthetic_ohlcv,
    load_market_bars_from_csv,
)
from run_research_strategy import main as cli_main
from start_nova_bot import main as bot_main


class TestEndToEndPipeline(unittest.TestCase):
    """Verifies complete workflow from ingestion through evaluation and CLI reporting."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_cli_demo_mode_execution(self) -> None:
        """CLI demo mode must execute successfully and generate a valid JSON report."""
        report_file = self.test_dir / "demo_report.json"
        exit_code = cli_main([
            "--demo",
            "--bars", "50",
            "--fast-window", "4",
            "--slow-window", "8",
            "--output", str(report_file),
        ])

        self.assertEqual(exit_code, 0)
        self.assertTrue(report_file.is_file())

        with report_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data["strategy_id"], "ema_crossover")
        self.assertEqual(data["strategy_version"], "1.0.0")
        self.assertTrue(data["is_synthetic_data"])
        self.assertIn("performance", data)
        self.assertIn("gross_cumulative_return", data["performance"])
        self.assertIn("max_drawdown", data["performance"])
        self.assertEqual(data["performance"]["friction_status"], "UNVERIFIED_PENDING_PLATFORM_CONTRACT")
        self.assertIsNone(data["performance"]["net_cumulative_return"])

    def test_cli_demo_mode_with_friction(self) -> None:
        """CLI demo mode with friction flags generates verified net return."""
        report_file = self.test_dir / "friction_report.json"
        exit_code = cli_main([
            "--demo",
            "--bars", "50",
            "--fast-window", "4",
            "--slow-window", "8",
            "--brokerage-bps", "2.0",
            "--slippage-bps", "5.0",
            "--output", str(report_file),
        ])

        self.assertEqual(exit_code, 0)
        with report_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data["performance"]["friction_status"], "VERIFIED_PROVISIONAL")
        self.assertIsNotNone(data["performance"]["net_cumulative_return"])

    def test_csv_loading_and_pipeline_execution(self) -> None:
        """Pipeline must load valid CSV file and produce evaluation successfully."""
        csv_file = self.test_dir / "valid_bars.csv"
        csv_file.write_text(
            "timestamp,open,high,low,close,volume\n"
            + "\n".join(
                f"2026-01-01T{10 + i // 60:02d}:{i % 60:02d}:00+00:00,100.0,105.0,95.0,102.0,1000.0"
                for i in range(40)
            ),
            encoding="utf-8",
        )

        report_file = self.test_dir / "csv_report.json"
        exit_code = cli_main([
            "--file", str(csv_file),
            "--ticker", "TEST_CSV",
            "--fast-window", "4",
            "--slow-window", "8",
            "--output", str(report_file),
        ])

        self.assertEqual(exit_code, 0)
        with report_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data["ticker"], "TEST_CSV")
        self.assertFalse(data["is_synthetic_data"])
        self.assertEqual(data["sample_counts"]["total_bars_input"], 40)

    def test_csv_missing_column_rejection(self) -> None:
        """CSV missing a required column (e.g. close) must be rejected."""
        csv_file = self.test_dir / "missing_col.csv"
        csv_file.write_text(
            "timestamp,open,high,low,volume\n"
            "2026-01-01T10:00:00+00:00,100.0,105.0,95.0,1000.0\n",
            encoding="utf-8",
        )
        with self.assertRaises(ContractValidationError):
            load_market_bars_from_csv(csv_file, ticker="TEST")

    def test_csv_malformed_numeric_rejection(self) -> None:
        """CSV with non-numeric value in price column must be rejected."""
        csv_file = self.test_dir / "malformed_num.csv"
        csv_file.write_text(
            "timestamp,open,high,low,close,volume\n"
            "2026-01-01T10:00:00+00:00,100.0,105.0,INVALID_PRICE,102.0,1000.0\n",
            encoding="utf-8",
        )
        with self.assertRaises(NumericalValidationError):
            load_market_bars_from_csv(csv_file, ticker="TEST")


    def test_start_nova_bot_simulation_runner(self) -> None:
        """start_nova_bot simulation mode must execute cleanly and return 0."""
        # Save stdout/argv, run bounded simulation
        exit_code = bot_main([
            "--simulated-feed",
            "--max-ticks", "10",
            "--tick-delay-seconds", "0.0",
        ])
        self.assertEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main()

