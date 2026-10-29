"""Test for synthetic benchmark execution and metric verification."""

try:
    from tests.benchmark_ingestion import run_benchmark
except (ImportError, ModuleNotFoundError):
    try:
        from kafka.tests.benchmark_ingestion import run_benchmark
    except (ImportError, ModuleNotFoundError):
        from benchmark_ingestion import run_benchmark


def test_synthetic_benchmark_execution():
    """Verify that synthetic benchmark runs cleanly and returns valid latency metrics."""
    results = run_benchmark(sample_size=200, warmup_size=50)

    assert results["sample_size"] == 200
    assert results["num_ticks"] == 160
    assert results["num_news"] == 40
    assert results["ticks_dispatched"] == 160
    assert results["news_dispatched"] == 40

    # Ensure latency measurements are positive and realistic
    assert results["min_us"] > 0
    assert results["median_us"] > 0
    assert results["p95_us"] >= results["median_us"]
    assert results["p99_us"] >= results["p95_us"]
    assert results["throughput_msg_sec"] > 0
