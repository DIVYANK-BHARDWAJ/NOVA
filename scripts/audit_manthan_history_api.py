"""NOVA Research — MANTHAN Historical Data Availability & XGBoost Feasibility Prober.

Performs read-only inspection of the official MANTHAN competition API:
1. Probe A: Dynamic universe discovery (GET /api/v1/stocks)
2. Probe B: Historical OHLCV extraction (GET /api/v1/stocks/{ticker}/history?days=...)
3. Probe C: Data quality, coverage, date ranges, nulls, volume, and simulated_date alignment
4. Probe D: XGBoost training data sufficiency calculation (warmup deduction, target creation, chronological splits)
5. Probe E & F: Indicator warm-start vs supervised ML training vs Kafka streaming

Strict Safety:
- ONLY sends read-only GET requests
- Never modifies orders or portfolio
- Never logs credentials, tokens, or JWTs
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any
import urllib.request
import urllib.error

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_env_config() -> tuple[str, str]:
    """Extract GATEWAY_URL and BOT_ACCESS_TOKEN safely from .env without printing secrets."""
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        raise FileNotFoundError(f".env file missing at {env_path}")

    gateway = "http://192.168.65.157:3002"
    token = ""
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("GATEWAY_URL="):
                gateway = line.split("=", 1)[1].strip().strip('"').strip("'")
            elif line.startswith("BOT_ACCESS_TOKEN="):
                token = line.split("=", 1)[1].strip().strip('"').strip("'")

    if not token:
        raise ValueError("BOT_ACCESS_TOKEN is not configured in .env")

    return gateway.rstrip("/"), token


def send_get_request(url: str, token: str, timeout: float = 8.0) -> tuple[int, dict[str, Any] | None, str | None]:
    """Perform read-only authenticated GET request using urllib."""
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "User-Agent": "NOVA-Forge-Audit/1.0",
        "Connection": "close",
    }
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.status
            body_bytes = resp.read()
            data = json.loads(body_bytes.decode("utf-8"))
            return status, data, None
    except urllib.error.HTTPError as http_err:
        err_msg = f"HTTP Error {http_err.code}: {http_err.reason}"
        try:
            body = http_err.read().decode("utf-8")
            err_msg += f" - Response: {body}"
        except Exception:
            pass
        return http_err.code, None, err_msg
    except Exception as exc:
        return 0, None, f"Network/Socket Error: {type(exc).__name__}: {exc}"


def main() -> None:
    print("=" * 80)
    print("NOVA — MANTHAN HISTORICAL DATA AVAILABILITY & XGBOOST FEASIBILITY PROBE")
    print("=" * 80)

    try:
        gateway, token = load_env_config()
        print(f"[*] Gateway URL: {gateway}")
        print("[*] Bot Access Token: [PRESENT - SANITIZED]")
    except Exception as e:
        print(f"[!] Configuration Error: {e}")
        sys.exit(1)

    raw_dir = REPO_ROOT / "data" / "raw" / "manthan_history"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------------
    # PROBE A: Dynamic Stock Universe Discovery
    # ------------------------------------------------------------------------
    stocks_url = f"{gateway}/api/v1/stocks"
    print(f"\n[1] Probing Universe Discovery: GET {stocks_url} ...")
    status, stocks_data, err = send_get_request(stocks_url, token)

    if status != 200 or not stocks_data:
        print(f"[!] Failed to fetch stock universe. Status: {status}, Error: {err}")
        sys.exit(1)

    print(f"[+] GET /api/v1/stocks succeeded with HTTP {status}")
    print(f"[+] Top-level response keys: {list(stocks_data.keys())}")

    stocks_list = stocks_data.get("stocks", [])
    sim_date_reported = stocks_data.get("current_date") or stocks_data.get("simulated_date")
    print(f"[+] Discovered {len(stocks_list)} instruments in universe")
    if sim_date_reported:
        print(f"[+] Platform reported simulation date: {sim_date_reported}")

    tickers: list[str] = []
    stock_details: list[dict[str, Any]] = []
    for item in stocks_list:
        if isinstance(item, dict):
            t = item.get("ticker") or item.get("symbol")
            if t:
                tickers.append(t)
                stock_details.append(item)
        elif isinstance(item, str):
            tickers.append(item)

    print(f"[+] Active Ticker Universe ({len(tickers)} symbols): {tickers}")

    # Save raw stocks universe payload
    stocks_raw_path = raw_dir / "universe_stocks.json"
    with open(stocks_raw_path, "w", encoding="utf-8") as f:
        json.dump(stocks_data, f, indent=2)
    stocks_sha = hashlib.sha256(stocks_raw_path.read_bytes()).hexdigest()
    print(f"[+] Saved raw universe response to {stocks_raw_path.name} (SHA-256: {stocks_sha[:16]}...)")

    # ------------------------------------------------------------------------
    # PROBE B: Historical OHLCV Data Retrieval
    # ------------------------------------------------------------------------
    print("\n[2] Probing Historical OHLCV Data for Discovered Universe...")

    history_stats: dict[str, Any] = {}
    total_raw_bars = 0
    total_valid_bars = 0

    for ticker in tickers:
        print(f"\n--- Inquiring History for [{ticker}] ---")
        # Try days=90
        hist_url_90 = f"{gateway}/api/v1/stocks/{ticker}/history?days=90"
        h_status, h_data, h_err = send_get_request(hist_url_90, token)

        if h_status != 200 or not h_data:
            print(f"[!] GET {hist_url_90} failed (Status: {h_status}, Error: {h_err})")
            history_stats[ticker] = {"status": h_status, "error": h_err, "records": 0}
            continue

        print(f"[+] GET {hist_url_90} succeeded (HTTP {h_status})")
        print(f"[+] Response top-level keys: {list(h_data.keys())}")

        # Save raw ticker history
        t_raw_path = raw_dir / f"{ticker}_history_90d.json"
        with open(t_raw_path, "w", encoding="utf-8") as f:
            json.dump(h_data, f, indent=2)
        t_sha = hashlib.sha256(t_raw_path.read_bytes()).hexdigest()

        # Find candle array
        bars_array = None
        for key_candidate in ["history", "candles", "data", "bars", "prices"]:
            if key_candidate in h_data and isinstance(h_data[key_candidate], list):
                bars_array = h_data[key_candidate]
                break

        if bars_array is None and isinstance(h_data, list):
            bars_array = h_data

        if bars_array is None:
            print(f"[!] Could not locate list of bars in response: {h_data}")
            history_stats[ticker] = {"status": 200, "records": 0, "error": "No bars list found in response"}
            continue

        num_bars = len(bars_array)
        total_raw_bars += num_bars
        print(f"[+] Returned {num_bars} raw historical bars for {ticker}")

        if num_bars > 0:
            sample_bar = bars_array[0]
            print(f"[+] Sample bar schema keys: {list(sample_bar.keys()) if isinstance(sample_bar, dict) else sample_bar}")
            print(f"[+] Earliest bar sample: {sample_bar}")
            latest_bar = bars_array[-1]
            print(f"[+] Latest bar sample: {latest_bar}")

            # Inspect quality
            valid_bars = 0
            dates_list = []
            for b in bars_array:
                if isinstance(b, dict):
                    # date/timestamp
                    d = b.get("date") or b.get("timestamp") or b.get("datetime")
                    c = b.get("close")
                    o = b.get("open")
                    h = b.get("high")
                    l = b.get("low")
                    v = b.get("volume", 0)

                    if d:
                        dates_list.append(str(d))
                    if all(x is not None and isinstance(x, (int, float)) and math.isfinite(x) and x > 0 for x in [o, h, l, c]):
                        valid_bars += 1

            total_valid_bars += valid_bars
            earliest_d = dates_list[0] if dates_list else "UNKNOWN"
            latest_d = dates_list[-1] if dates_list else "UNKNOWN"
            unique_dates = len(set(dates_list))

            history_stats[ticker] = {
                "status": 200,
                "raw_records": num_bars,
                "valid_records": valid_bars,
                "unique_dates": unique_dates,
                "earliest_date": earliest_d,
                "latest_date": latest_d,
                "sha256": t_sha,
                "sim_date": h_data.get("current_date") or h_data.get("simulated_date"),
            }
            print(f"[+] Date Coverage: {earliest_d} to {latest_d} (Unique dates: {unique_dates}/{num_bars})")

        # Also test with days=365 to see if deeper history is available
        hist_url_365 = f"{gateway}/api/v1/stocks/{ticker}/history?days=365"
        h365_status, h365_data, _ = send_get_request(hist_url_365, token)
        if h365_status == 200 and h365_data:
            bars_365 = h365_data.get("history") or h365_data.get("candles") or h365_data.get("data") or []
            print(f"[+] Probe with days=365 returned {len(bars_365)} bars for {ticker}")
            history_stats[ticker]["records_365d"] = len(bars_365)
            # Save 365d payload as well
            with open(raw_dir / f"{ticker}_history_365d.json", "w", encoding="utf-8") as f:
                json.dump(h365_data, f, indent=2)

    # ------------------------------------------------------------------------
    # PROBE C & D: Dataset Sufficiency & Usability for XGBoost
    # ------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("DATA QUALITY, COVERAGE & XGBOOST FEASIBILITY ASSESSMENT")
    print("=" * 80)

    print(f"Total Active Instruments:   {len(tickers)}")
    print(f"Total Raw Observations:     {total_raw_bars}")
    print(f"Total Valid Observations:   {total_valid_bars}")

    # Canonical Feature Pipeline Warmup Requirement:
    # FeaturePipeline requires 26 bars for slow EMA, 28 bars for ADX (2*14), 20 bars for rolling vol/mean
    # Maximum warmup requirement = 28 bars
    WARMUP_BARS = 28
    TARGET_HORIZONS = [1, 5]  # 1-day and 5-day forward return targets

    print(f"\nFeature Extraction Warmup Requirement: {WARMUP_BARS} bars")
    print(f"Target Horizons: {TARGET_HORIZONS} bars")

    for ticker, stat in history_stats.items():
        raw_n = stat.get("raw_records", 0)
        valid_n = stat.get("valid_records", 0)
        rec_365 = stat.get("records_365d", raw_n)
        usable_pool = max(raw_n, rec_365)

        print(f"\n--- Instrument Feasibility Breakdown: [{ticker}] ---")
        print(f"Available historical bars:          {usable_pool}")
        if usable_pool < WARMUP_BARS:
            print(f"[!] INSUFFICIENT FOR WARMUP: {usable_pool} < {WARMUP_BARS} bars required")
            continue

        bars_post_warmup = usable_pool - WARMUP_BARS
        print(f"Bars after feature warmup (t >= 28): {bars_post_warmup}")

        for H in TARGET_HORIZONS:
            # Last H bars cannot have target label without future data
            labelled_rows = max(0, bars_post_warmup - H)
            # Train / Val / Test split: 60% Train, 20% Val, 20% Test
            # With purging gap of H bars between splits
            n_train = int(labelled_rows * 0.60)
            n_val = int(labelled_rows * 0.20)
            n_test = labelled_rows - n_train - n_val
            # Account for embargo gap H bars between train->val and val->test
            n_val_purged = max(0, n_val - H)
            n_test_purged = max(0, n_test - H)

            print(f"  [Target Horizon H={H} bars]:")
            print(f"    Supervised labelled rows:        {labelled_rows}")
            print(f"    Train Split (60%):               {n_train} rows")
            print(f"    Validation Split (purged):       {n_val_purged} rows")
            print(f"    Untouched Test Split (purged):   {n_test_purged} rows")

    summary_file = raw_dir / "feasibility_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(history_stats, f, indent=2)
    print(f"\n[+] Full feasibility diagnostic metrics written to {summary_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()

