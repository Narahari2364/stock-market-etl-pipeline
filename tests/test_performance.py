import pytest
from pathlib import Path
from src.performance import PipelinePerformanceTracker


class TestPipelinePerformanceTracker:
    def test_stage_timing_and_log_file(self, tmp_path):
        tracker = PipelinePerformanceTracker(log_dir=str(tmp_path))
        start = tracker.start_stage()
        tracker.end_stage('EXTRACT', start, record_count=100, notes='test')
        summary = tracker.finalize(total_records=100)

        assert summary['total_duration_sec'] > 0
        assert len(summary['stages']) == 1
        assert summary['stages'][0]['stage'] == 'EXTRACT'
        assert Path(summary['log_file']).exists()

    def test_batch_metrics_logged(self, tmp_path):
        tracker = PipelinePerformanceTracker(log_dir=str(tmp_path))
        tracker.log_batch(
            batch_number=1,
            batch_size=500,
            duration_sec=0.5,
            records_loaded=500,
            chunksize=500,
        )
        assert len(tracker.batch_metrics) == 1
        assert tracker.batch_metrics[0]['records_per_sec'] == 1000.0

    def test_zero_duration_throughput(self, tmp_path):
        tracker = PipelinePerformanceTracker(log_dir=str(tmp_path))
        start = tracker.start_stage()
        tracker.end_stage('LOAD', start, record_count=0)
        assert tracker.stages[0]['records_per_sec'] == 0.0

    def test_finalize_with_batches(self, tmp_path):
        tracker = PipelinePerformanceTracker(log_dir=str(tmp_path))
        start = tracker.start_stage()
        tracker.end_stage('LOAD', start, record_count=200)
        tracker.log_batch(1, 100, 0.2, 100, 100)
        tracker.log_batch(2, 100, 0.3, 100, 100)
        summary = tracker.finalize(total_records=200)
        assert len(summary['batch_metrics']) == 2
        content = Path(summary['log_file']).read_text()
        assert 'load_batches: 2' in content
