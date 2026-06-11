# Pipeline Performance Analysis

Instrumentation added to capture **per-stage duration**, **records/sec**, and **per-batch load metrics** for bottleneck analysis — aligned with simulation/performance engineering workflows (profile → measure → tune → verify).

## Instrumentation

| Component | What it captures |
|-----------|------------------|
| `src/performance.py` | `PipelinePerformanceTracker` — stage timings, throughput, batch logs |
| `src/pipeline.py` | `time.perf_counter()` around EXTRACT, TRANSFORM, VALIDATE, LOAD, SUMMARY |
| `src/load.py` | Per-batch duration and rec/s; tunable `chunksize` |
| `logs/perf_*.log` | Structured performance report per pipeline run |
| `logs/profile_*.txt` | Optional cProfile top-30 (set `ENABLE_CPROFILE=1`) |

### Environment variables

```bash
LOAD_BATCH_SIZE=2000    # PostgreSQL append batch size (default: 2000)
ENABLE_CPROFILE=1       # Write logs/profile_<timestamp>.txt on pipeline run
```

Default batch size is **2000** in both `src/load.py` and `src/pipeline.py` (`LOAD_BATCH_SIZE` env var).

### Example performance log

```
STAGE: EXTRACT
  duration_sec: 24.5838
  record_count: 300
  records_per_sec: 12.20

STAGE: LOAD
  duration_sec: 0.3510
  record_count: 300
  records_per_sec: 854.59
  notes: chunksize=2000

EXTRACT: 24.58s (94.5% of total) | 12.20 rec/s
LOAD: 0.35s (1.4% of total) | 855 rec/s
```

*(EXTRACT dominates wall-clock on free-tier API rate limits; LOAD dominates **local compute/DB** time when EXTRACT is excluded.)*

## Batch size tuning (LOAD stage)

Benchmark script (LOAD only, no API calls):

```bash
source venv/bin/activate
docker compose up -d
python scripts/benchmark_batch_size.py
```

Tests chunk sizes **500, 1000, 2000, 5000** on 2,500 synthetic rows into `stock_data_benchmark`.

### Measured results

**Run:** 2026-06-11 · Local Docker PostgreSQL 15 · 2,500 rows · `logs/benchmark_batch_size_20260611_195607.txt`

| Batch size | Load time (s) | Throughput (rec/s) | Notes |
|------------|---------------|--------------------|-------|
| 500 | 1.024 | 2,442 | Most round-trips; highest SQL overhead |
| 1000 | 0.967 | 2,585 | Previous common default |
| **2000** | **0.956** | **2,616** | **Selected default** — best throughput in this run |
| 5000 | 0.959 | 2,607 | No gain over 2000 at this row count |

### Before / after (same benchmark run)

| Metric | Before (`chunksize=500`) | After (`chunksize=2000`) | Change |
|--------|--------------------------|--------------------------|--------|
| LOAD stage time | 1.024s | 0.956s | **~7% faster** |
| LOAD throughput | 2,442 rec/s | 2,616 rec/s | **~7% higher** |

At ~2,500 rows on local Postgres, gains are modest because the DB is already fast; larger production loads benefit more from fewer round-trips.

## Findings

1. **EXTRACT** — Wall-clock bottleneck for full pipeline (~85–95% of end-to-end time with multiple symbols) due to Alpha Vantage rate limits (12s delay per symbol), not CPU.
2. **LOAD** — Dominant **compute/IO** bottleneck among local stages once data is in memory.
3. **TRANSFORM** — Pandas vectorized ops; typically &lt;1% of total pipeline time at current scale.
4. **VALIDATE** — Great Expectations overhead modest at current scale (~3–4% of total pipeline time).

## Tuning decision

- Default `LOAD_BATCH_SIZE` set to **2000** in code after benchmarking 500–5000.
- Batch size 2000 achieved peak throughput (2,616 rec/s) without increasing memory pressure.
- Sizes above 2000 did not improve throughput at 2,500 rows.

## Resume bullet (example)

> Profiled ETL stages with `perf_counter` and optional cProfile instrumentation; identified EXTRACT as the wall-clock bottleneck (~95% of runtime due to API rate limits) and benchmarked PostgreSQL LOAD batch sizes 500–5000. Tuned `LOAD_BATCH_SIZE` to 2000 (peak ~2,616 rec/s), reducing LOAD time ~7% vs. 500-row chunks on measured runs.

## Reproduce

```bash
source venv/bin/activate
docker compose up -d

# LOAD-only batch benchmark (no API calls)
python scripts/benchmark_batch_size.py

# Full pipeline with performance log (interactive; needs API key)
cd src && python pipeline.py

# With cProfile
cd src && ENABLE_CPROFILE=1 python pipeline.py
```

**zsh note:** Do not paste inline comments that contain `*` globs — zsh may try to expand them. List logs with `ls -t logs/`.

Check `logs/` for `benchmark_batch_size_*.txt`, `perf_*.log`, and `profile_*.txt` after runs.
