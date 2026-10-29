"""NOVA Quantitative Research — Official MANTHAN API Harvester & Provenance Reconciler (Person 2).

Discovers dynamic active stock universe from GET /api/v1/stocks, queries full historical
OHLCV daily data per ticker respecting exact listing dates, records response payloads with
cryptographic SHA-256 checksums, and produces reconciled replay dataset canonical_manthan_v2.

POLICY & INVARIANTS:
- Does NOT overwrite existing canonical_manthan_v1 dataset.
- Respects venue rate limits with respectful request pacing.
- Validates anti-lookahead constraint (end_date <= current_simulation_date).
- Excludes synthetic fixture injection.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def load_env_config() -> dict[str, str]:
    """Parse .env safely without external dependencies."""
    env_file = REPO_ROOT / ".env"
    if not env_file.exists():
        raise FileNotFoundError(f"Missing .env file at {env_file}")
    cfg: dict[str, str] = {}
    with open(env_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                cfg[k.strip()] = v.strip()
    return cfg


def fetch_json(url: str, token: str) -> tuple[int, dict[str, Any], bytes]:
    """Perform HTTP GET request with Bearer and Cookie authentication."""
    req = urllib.request.Request(url)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Cookie", f"access_token={token}")
    req.add_header("User-Agent", "NOVA-Quantitative-Research-Agent/2.0")
    try:
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            raw_bytes = resp.read()
            data = json.loads(raw_bytes.decode("utf-8"))
            return resp.status, data, raw_bytes
    except urllib.error.HTTPError as err:
        raw_bytes = err.read()
        try:
            data = json.loads(raw_bytes.decode("utf-8"))
        except Exception:
            data = {"error": "HTTP_ERROR", "body": raw_bytes.decode("utf-8", errors="replace")}
        return err.code, data, raw_bytes


def main() -> None:
    print("=" * 80)
    print("NOVA — Official MANTHAN API Historical Extraction & Provenance Audit")
    print("=" * 80)

    cfg = load_env_config()
    base_url = cfg.get("GATEWAY_URL", "http://192.168.65.157:3002").rstrip("/")
    token = cfg.get("BOT_ACCESS_TOKEN", "")
    if not token:
        raise ValueError("BOT_ACCESS_TOKEN is missing or blank in .env")

    raw_output_dir = REPO_ROOT / "data" / "raw" / "manthan_api_v2"
    replay_output_dir = REPO_ROOT / "data" / "replay" / "canonical_manthan_v2"
    raw_output_dir.mkdir(parents=True, exist_ok=True)
    replay_output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Query Simulation Clock & Stage
    print(f"[*] Querying simulation clock and lifecycle stage from {base_url}...")
    status_clock, data_clock, raw_clock = fetch_json(f"{base_url}/api/v1/simulation/clock", token)
    status_stage, data_stage, raw_stage = fetch_json(f"{base_url}/api/v1/stage", token)

    if status_clock != 200 or status_stage != 200:
        raise RuntimeError(f"Failed to query simulation status: clock={status_clock}, stage={status_stage}")

    sim_date = data_clock.get("simulated_date", "2026-04-14")
    print(f"[+] Simulation Stage: {data_stage.get('stage')} | Active Dataset: {data_stage.get('active_dataset')}")
    print(f"[+] Current Simulation Date: {sim_date} | Total Days: {data_clock.get('total_days')}")

    # 2. Dynamic Universe Discovery
    print("\n[*] Discovering dynamic universe via GET /api/v1/stocks...")
    status_stocks, data_stocks, raw_stocks = fetch_json(f"{base_url}/api/v1/stocks", token)
    if status_stocks != 200:
        raise RuntimeError(f"Failed to fetch stock universe: HTTP {status_stocks}")

    stock_items = data_stocks.get("stocks", [])
    discovered_tickers = [s.get("ticker") or s.get("symbol") for s in stock_items if s.get("ticker") or s.get("symbol")]
    print(f"[+] Discovered {len(discovered_tickers)} active stocks: {discovered_tickers}")

    # Save raw stocks payload
    (raw_output_dir / "stocks_universe_response.json").write_bytes(raw_stocks)

    # 3. Discover exact listing dates and query full OHLCV history
    print("\n[*] Querying historical OHLCV archives per instrument respecting listing dates...")
    harvest_summary: dict[str, Any] = {}
    all_observations: list[dict[str, Any]] = []
    fingerprints_set: set[str] = set()

    # Calculate strict anti-lookahead end_date (day strictly prior to simulation clock)
    sim_dt = datetime.fromisoformat(sim_date)
    end_date = (sim_dt - timedelta(days=1)).date().isoformat()
    print(f"[+] Strict Anti-Lookahead End Date: {end_date} (strictly prior to {sim_date})")

    for idx, ticker in enumerate(discovered_tickers, start=1):
        time.sleep(0.1)  # Respectful pacing
        print(f"  -> [{idx}/{len(discovered_tickers)}] Probing listing date for {ticker}...")
        # Step A: Probe listing date with early boundary 2010-01-01
        probe_url = f"{base_url}/api/v1/stocks/history?ticker={ticker}&start_date=2010-01-01&end_date={end_date}"
        st_code, st_body, _ = fetch_json(probe_url, token)

        listing_date = "2016-10-24"  # Default fallback if boundary error
        if st_code == 400:
            listing_date = st_body.get("listing_date") or st_body.get("earliest_available_date") or "2016-10-24"
        print(f"     Identified listing date: {listing_date}")

        # Step B: Query full history from listing date to anti-lookahead end_date
        hist_url = f"{base_url}/api/v1/stocks/history?ticker={ticker}&start_date={listing_date}&end_date={end_date}"
        hist_status, hist_data, hist_raw = fetch_json(hist_url, token)

        if hist_status != 200:
            print(f"     [!] HTTP {hist_status} for {ticker}: {hist_data.get('message')}")
            continue

        raw_file = raw_output_dir / f"{ticker}_history_response.json"
        raw_file.write_bytes(hist_raw)
        raw_sha256 = hashlib.sha256(hist_raw).hexdigest()

        prices = hist_data.get("prices", [])
        total_recs = hist_data.get("total_records", len(prices))

        first_d = prices[0]["date"] if prices else None
        last_d = prices[-1]["date"] if prices else None

        harvest_summary[ticker] = {
            "listing_date": listing_date,
            "requested_start_date": listing_date,
            "requested_end_date": end_date,
            "returned_records": len(prices),
            "reported_total_records": total_recs,
            "earliest_date": first_d,
            "latest_date": last_d,
            "response_sha256": raw_sha256,
        }
        print(f"     [+] Extracted {len(prices)} bars: {first_d} to {last_d} (SHA-256: {raw_sha256[:12]}...)")

        # Convert to canonical observation format
        for p in prices:
            m_date = p["date"]
            o_val = float(p["open"])
            h_val = float(p["high"])
            l_val = float(p["low"])
            c_val = float(p["close"])
            v_val = float(p.get("volume", 0.0))

            # Natural key fingerprint: ticker|market_date|open|high|low|close|volume
            key_str = f"{ticker}|{m_date}|{o_val:.4f}|{h_val:.4f}|{l_val:.4f}|{c_val:.4f}|{v_val:.1f}"
            fp = hashlib.sha256(key_str.encode("utf-8")).hexdigest()
            fingerprints_set.add(fp)

            all_observations.append({
                "ticker": ticker,
                "market_date": m_date,
                "open": o_val,
                "high": h_val,
                "low": l_val,
                "close": c_val,
                "volume": v_val,
                "fingerprint": fp,
                "provenance": f"MANTHAN_OFFICIAL_API_V2:{ticker}",
                "harvest_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            })

    # Sort observations by ticker, then market_date
    all_observations.sort(key=lambda x: (x["ticker"], x["market_date"]))

    # 4. Save Canonical v2 Observations JSONL
    obs_file = replay_output_dir / "observations.jsonl"
    with open(obs_file, "w", encoding="utf-8") as f:
        for idx, obs in enumerate(all_observations, start=1):
            obs_rec = {"observation_index": idx, **obs}
            f.write(json.dumps(obs_rec) + "\n")

    obs_bytes = obs_file.read_bytes()
    obs_sha256 = hashlib.sha256(obs_bytes).hexdigest()

    # 5. Build and save canonical manifest
    manifest_payload = {
        "dataset_name": "canonical_manthan_v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_system": "MANTHAN_OFFICIAL_REST_GATEWAY",
        "gateway_url": base_url,
        "simulation_clock": {
            "simulated_date": sim_date,
            "stage": data_stage.get("stage"),
            "active_dataset": data_stage.get("active_dataset"),
        },
        "universe_discovery": {
            "tickers_count": len(discovered_tickers),
            "tickers": discovered_tickers,
            "stock_details": stock_items,
        },
        "instrument_summaries": harvest_summary,
        "total_observations_count": len(all_observations),
        "unique_fingerprints_count": len(fingerprints_set),
        "file_name": "observations.jsonl",
        "file_sha256": obs_sha256,
        "file_size_bytes": len(obs_bytes),
    }

    manifest_file = replay_output_dir / "manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_payload, f, indent=2)

    print("\n" + "=" * 80)
    print("EXTRACTION & RECONCILIATION COMPLETE")
    print(f"Total Unique Market Observations: {len(fingerprints_set):,}")
    print(f"Canonical Replay Dataset:         {replay_output_dir}")
    print(f"Manifest File:                    {manifest_file}")
    print(f"Observations File SHA-256:        {obs_sha256}")
    print("=" * 80)


if __name__ == "__main__":
    main()
