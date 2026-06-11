"""
Pipeline performance instrumentation.

Captures per-stage duration, throughput (records/sec), and batch-level
load metrics. Writes structured reports to logs/perf_*.log.
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class PipelinePerformanceTracker:
    """Track ETL stage timings and write performance logs."""

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"perf_{self.timestamp}.log"
        self.stages: List[Dict] = []
        self.batch_metrics: List[Dict] = []
        self.pipeline_start = time.perf_counter()
        self._write_header()

    def _append(self, line: str) -> None:
        with open(self.log_file, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def _write_header(self) -> None:
        self._append("=" * 72)
        self._append("ETL PIPELINE PERFORMANCE REPORT")
        self._append("=" * 72)
        self._append(f"Started: {datetime.now().isoformat()}")
        self._append(f"Log file: {self.log_file}")
        self._append("")

    @staticmethod
    def _records_per_sec(record_count: int, duration_sec: float) -> float:
        if duration_sec <= 0 or record_count <= 0:
            return 0.0
        return record_count / duration_sec

    def start_stage(self) -> float:
        """Return a perf_counter timestamp for stage timing."""
        return time.perf_counter()

    def end_stage(
        self,
        stage_name: str,
        start_time: float,
        record_count: int = 0,
        notes: str = "",
    ) -> Dict:
        """Record stage completion and append to the performance log."""
        duration_sec = time.perf_counter() - start_time
        throughput = self._records_per_sec(record_count, duration_sec)
        entry = {
            "stage": stage_name,
            "duration_sec": duration_sec,
            "record_count": record_count,
            "records_per_sec": throughput,
            "notes": notes,
        }
        self.stages.append(entry)

        self._append(f"STAGE: {stage_name}")
        self._append(f"  duration_sec: {duration_sec:.4f}")
        self._append(f"  record_count: {record_count:,}")
        self._append(f"  records_per_sec: {throughput:,.2f}")
        if notes:
            self._append(f"  notes: {notes}")
        self._append("")

        return entry

    def log_batch(
        self,
        batch_number: int,
        batch_size: int,
        duration_sec: float,
        records_loaded: int,
        chunksize: int,
    ) -> None:
        """Log per-batch load performance."""
        throughput = self._records_per_sec(records_loaded, duration_sec)
        entry = {
            "batch_number": batch_number,
            "batch_size": batch_size,
            "duration_sec": duration_sec,
            "records_loaded": records_loaded,
            "records_per_sec": throughput,
            "chunksize": chunksize,
        }
        self.batch_metrics.append(entry)

        self._append(
            f"BATCH_LOAD batch={batch_number} size={batch_size} "
            f"chunksize={chunksize} duration_sec={duration_sec:.4f} "
            f"records_per_sec={throughput:,.2f}"
        )

    def finalize(self, total_records: int = 0) -> Dict:
        """Write summary with stage share of total runtime."""
        total_duration = time.perf_counter() - self.pipeline_start
        self._append("=" * 72)
        self._append("SUMMARY")
        self._append("=" * 72)
        self._append(f"total_duration_sec: {total_duration:.4f}")
        self._append(f"total_records: {total_records:,}")
        if total_duration > 0 and total_records > 0:
            self._append(
                f"overall_throughput_records_per_sec: "
                f"{self._records_per_sec(total_records, total_duration):,.2f}"
            )
        self._append("")

        for entry in self.stages:
            pct = (
                (entry["duration_sec"] / total_duration) * 100
                if total_duration > 0
                else 0.0
            )
            entry["pct_of_total"] = pct
            self._append(
                f"{entry['stage']}: {entry['duration_sec']:.4f}s "
                f"({pct:.1f}% of total) | "
                f"{entry['records_per_sec']:,.2f} rec/s"
            )

        if self.batch_metrics:
            load_batches = [b for b in self.batch_metrics]
            avg_batch = sum(b["duration_sec"] for b in load_batches) / len(load_batches)
            self._append("")
            self._append(f"load_batches: {len(load_batches)}")
            self._append(f"avg_batch_duration_sec: {avg_batch:.4f}")

        self._append("")
        self._append(f"Completed: {datetime.now().isoformat()}")
        self._append("=" * 72)

        return {
            "log_file": str(self.log_file),
            "total_duration_sec": total_duration,
            "stages": self.stages,
            "batch_metrics": self.batch_metrics,
        }
