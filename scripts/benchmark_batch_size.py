#!/usr/bin/env python3
"""
Benchmark PostgreSQL load throughput at different batch sizes.

Measures only the LOAD stage (no API calls). Writes results to stdout and
logs/benchmark_batch_size_<timestamp>.txt for PERFORMANCE.md updates.

Usage:
    python scripts/benchmark_batch_size.py
"""

import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

load_dotenv()

from load import load_to_database  # noqa: E402


def generate_benchmark_df(rows: int = 2500) -> pd.DataFrame:
    """Synthetic transformed stock data (~17 symbols × ~150 days)."""
    symbols = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "JPM", "BAC",
        "GS", "V", "MA", "JNJ", "UNH", "PFE", "WMT", "PG", "KO",
    ]
    dates = pd.date_range(end=datetime.now(), periods=rows // len(symbols) + 1, freq="D")
    records = []
    for symbol in symbols:
        base = 100 + abs(hash(symbol)) % 200
        for i, dt in enumerate(dates):
            close = base + i * 0.1
            records.append(
                {
                    "symbol": symbol,
                    "date": dt.date(),
                    "open": close - 0.5,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                    "volume": 40_000_000 + i * 10_000,
                    "daily_change": 0.2,
                    "daily_change_percent": 0.15,
                    "price_range": 2.0,
                    "price_range_percent": 1.5,
                    "ma_5": close - 0.3,
                    "ma_20": close - 0.8,
                    "volatility_indicator": 1.2,
                    "volatility_category": "Medium",
                    "volume_category": "High",
                    "is_positive_day": True,
                    "is_negative_day": False,
                    "year": dt.year,
                    "month": dt.month,
                    "quarter": dt.quarter,
                    "day_of_week": dt.dayofweek,
                    "week_of_year": int(dt.isocalendar().week),
                    "price_vs_ma5": 0.5,
                    "price_vs_ma20": 1.0,
                    "extracted_at": datetime.utcnow(),
                    "data_source": "benchmark",
                }
            )
            if len(records) >= rows:
                break
        if len(records) >= rows:
            break
    return pd.DataFrame(records[:rows])


def run_benchmark(batch_sizes=None, table_name: str = "stock_data_benchmark") -> list:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not set")

    batch_sizes = batch_sizes or [500, 1000, 2000, 5000]
    df = generate_benchmark_df(rows=2500)
    engine = create_engine(database_url, pool_pre_ping=True)
    results = []

    print("=" * 72)
    print("LOAD BATCH SIZE BENCHMARK")
    print("=" * 72)
    print(f"Records: {len(df):,}")
    print(f"Batch sizes: {batch_sizes}")
    print()

    for size in batch_sizes:
        with engine.connect() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
            conn.execute(
                text(f"CREATE TABLE {table_name} (LIKE stock_data INCLUDING ALL)")
            )
            conn.commit()

        start = time.perf_counter()
        ok = load_to_database(
            df,
            engine=engine,
            table_name=table_name,
            chunksize=size,
        )
        elapsed = time.perf_counter() - start
        throughput = len(df) / elapsed if elapsed > 0 and ok else 0

        row = {
            "batch_size": size,
            "duration_sec": round(elapsed, 3),
            "records": len(df),
            "records_per_sec": round(throughput, 1),
            "success": ok,
        }
        results.append(row)
        print(
            f"chunksize={size:>5} | {elapsed:7.3f}s | "
            f"{throughput:8,.1f} rec/s | ok={ok}"
        )

    logs_dir = ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = logs_dir / f"benchmark_batch_size_{stamp}.txt"
    with open(out_file, "w", encoding="utf-8") as handle:
        handle.write("batch_size,duration_sec,records,records_per_sec,success\n")
        for row in results:
            handle.write(
                f"{row['batch_size']},{row['duration_sec']},"
                f"{row['records']},{row['records_per_sec']},{row['success']}\n"
            )

    print()
    print(f"Results saved: {out_file}")
    best = max(results, key=lambda r: r["records_per_sec"])
    print(f"Best throughput: chunksize={best['batch_size']} ({best['records_per_sec']:,.1f} rec/s)")
    return results


if __name__ == "__main__":
    run_benchmark()
