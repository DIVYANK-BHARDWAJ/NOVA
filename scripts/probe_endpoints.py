import json
from pathlib import Path
import sys
import time
import urllib.request
import urllib.error

env_path = Path(".env")
token = ""
gateway = "http://192.168.65.157:3002"

if env_path.exists():
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("BOT_ACCESS_TOKEN="):
                token = line.split("=", 1)[1].strip().strip('"').strip("'")
            elif line.startswith("GATEWAY_URL="):
                gateway = line.split("=", 1)[1].strip().strip('"').strip("'")

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json",
}

# 1. Discover tickers
stocks_req = urllib.request.Request(f"{gateway}/api/v1/stocks", headers=headers)
with urllib.request.urlopen(stocks_req) as resp:
    stocks_data = json.loads(resp.read().decode("utf-8"))

tickers = [s["ticker"] for s in stocks_data.get("stocks", [])]
print(f"Discovered {len(tickers)} tickers: {tickers}")

raw_dir = Path("data/raw/manthan_history_full")
raw_dir.mkdir(parents=True, exist_ok=True)

summary = {}
for ticker in tickers:
    time.sleep(0.2)
    # Probe with days=3650 to get earliest_available_date from HTTP 400 or get full history
    probe_url = f"{gateway}/api/v1/stocks/history?ticker={ticker}&days=3650"
    earliest_date = "2016-10-24"
    try:
        req = urllib.request.Request(probe_url, headers=headers)
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            prices = data.get("prices", [])
            summary[ticker] = {"records": len(prices), "start": prices[0]["date"], "end": prices[-1]["date"]}
            print(f"[{ticker}] SUCCESS via days=3650: {len(prices)} bars from {prices[0]['date']} to {prices[-1]['date']}")
            with open(raw_dir / f"{ticker}_history_full.json", "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            continue
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8")
        try:
            err = json.loads(err_body)
            earliest_date = err.get("earliest_available_date", "2016-10-24")
        except Exception:
            print(f"[{ticker}] Failed parsing error: {err_body}")

    # Now query with start_date=earliest_date
    fetch_url = f"{gateway}/api/v1/stocks/history?ticker={ticker}&start_date={earliest_date}"
    try:
        req = urllib.request.Request(fetch_url, headers=headers)
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            prices = data.get("prices", [])
            summary[ticker] = {
                "records": len(prices),
                "start": prices[0]["date"] if prices else None,
                "end": prices[-1]["date"] if prices else None,
            }
            print(f"[{ticker}] SUCCESS via start_date={earliest_date}: {len(prices)} bars from {prices[0]['date']} to {prices[-1]['date']}")
            with open(raw_dir / f"{ticker}_history_full.json", "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8")
        print(f"[{ticker}] FAILED start_date={earliest_date}: {err_body}")
        # Fallback to days=2000 or days=1500
        for fallback_days in [2000, 1500, 1000, 500]:
            try:
                fb_url = f"{gateway}/api/v1/stocks/history?ticker={ticker}&days={fallback_days}"
                req = urllib.request.Request(fb_url, headers=headers)
                with urllib.request.urlopen(req) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    prices = data.get("prices", [])
                    summary[ticker] = {
                        "records": len(prices),
                        "start": prices[0]["date"] if prices else None,
                        "end": prices[-1]["date"] if prices else None,
                    }
                    print(f"[{ticker}] SUCCESS via days={fallback_days}: {len(prices)} bars from {prices[0]['date']} to {prices[-1]['date']}")
                    with open(raw_dir / f"{ticker}_history_full.json", "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2)
                    break
            except Exception:
                continue

print("\n" + "=" * 60)
print("COMPREHENSIVE GENUINE HISTORICAL DATASET SUMMARY:")
print("=" * 60)
total_bars = sum(v["records"] for v in summary.values())
for t, s in summary.items():
    print(f"  {t:<6} : {s['records']:>5} bars ({s['start']} -> {s['end']})")
print(f"Total Universe Observations: {total_bars} genuine OHLCV bars!")
