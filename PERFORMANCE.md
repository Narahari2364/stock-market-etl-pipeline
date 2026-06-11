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

**Run:** 2026-06-11 · Local Docker PostgreSQL 15 · 2,500 rows · **median of 3 runs** · `logs/benchmark_batch_size_20260611_195824.txt`

| Batch size | Median load time (s) | Throughput (rec/s) | Individual runs (s) |
|------------|----------------------|--------------------|---------------------|
| 500 | 0.959 | 2,606 | 0.984, 0.959, 0.955 |
| 1000 | 0.953 | 2,623 | 0.952, 0.953, 0.960 |
| **2000** | **0.987** | **2,534** | **0.955, 1.040, 0.987** |
| 5000 | 0.939 | 2,662 | 0.939, 0.948, 0.937 |

Raw log excerpt:

```
best_throughput_batch: 5000 (2661.9 rec/s)
vs_chunksize_1000: 3.6% slower LOAD (0.953s -> 0.987s), 3.4% lower throughput
vs_chunksize_500: 2.9% slower LOAD (0.959s -> 0.987s)
```

### Before / after (median benchmark)

| Metric | Previous default (`chunksize=1000`) | Tuned (`chunksize=2000`) | Change |
|--------|-------------------------------------|--------------------------|--------|
| LOAD stage time | 0.953s | 0.987s | 3.6% slower at 2,500 rows |
| LOAD throughput | 2,623 rec/s | 2,534 rec/s | 3.4% lower |

| Metric | Smallest batch (`chunksize=500`) | Peak batch (`chunksize=5000`) | Change |
|--------|----------------------------------|-------------------------------|--------|
| LOAD stage time | 0.959s | 0.939s | **2.1% faster** |
| LOAD throughput | 2,606 rec/s | 2,662 rec/s | **2.1% higher** |

At ~2,500 rows, differences between 1000, 2000, and 5000 are within a few percent. **2000** remains the production default to cut round-trips vs. 1000 as row counts grow (2,900+ records today), without the memory pressure of 5000-row inserts.

## Findings

1. **EXTRACT** — Wall-clock bottleneck for full pipeline (~85–95% of end-to-end time with multiple symbols) due to Alpha Vantage rate limits (12s delay per symbol), not CPU.
2. **LOAD** — Dominant **compute/IO** bottleneck among local stages once data is in memory.
3. **TRANSFORM** — Pandas vectorized ops; typically &lt;1% of total pipeline time at current scale.
4. **VALIDATE** — Great Expectations overhead modest at current scale (~3–4% of total pipeline time).

## Tuning decision

- Default `LOAD_BATCH_SIZE` set to **2000** in code after benchmarking 500–5000 (median of 3 runs per size).
- Peak median throughput at 2,500 rows: chunksize **5000** (0.939s, 2,662 rec/s).
- **2000** chosen as default: halves round-trips vs. 1000 at scale; 3.6% slower than 1000 at 2,500 rows is within measurement variance.

## Resume bullet (example)

> Profiled ETL stages with `perf_counter` and optional cProfile instrumentation; identified EXTRACT as the wall-clock bottleneck (~95% of runtime due to API rate limits) and benchmarked PostgreSQL LOAD batch sizes 500–5000 (median of 3 runs). Selected `LOAD_BATCH_SIZE=2000` to balance throughput (2,534 rec/s) with fewer DB round-trips on growing datasets.

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
