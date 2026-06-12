import os
from pathlib import Path
import pandas as pd
from src.data_quality import (
    create_stock_data_expectations,
    validate_stock_data,
    save_validation_report,
)


class TestDataQuality:
    def test_create_stock_data_expectations(self):
        expectations = create_stock_data_expectations()
        assert len(expectations) > 0
        types = {e['expectation_type'] for e in expectations}
        assert 'expect_column_to_exist' in types
        assert 'expect_compound_columns_to_be_unique' in types

    def test_validate_stock_data_success(self, sample_transformed_df):
        results = validate_stock_data(sample_transformed_df, log_results=False)
        assert 'success_rate' in results
        assert results['total_checks'] > 0
        assert results['success_rate'] >= 90

    def test_validate_stock_data_with_failures(self, sample_transformed_df):
        bad_df = sample_transformed_df.copy()
        bad_df.loc[0, 'close'] = -5
        results = validate_stock_data(bad_df, log_results=True)
        assert results['success_rate'] < 100

    def test_validate_logs_quality_tiers(self, sample_transformed_df):
        results = validate_stock_data(sample_transformed_df, log_results=True)
        assert results['success_rate'] >= 95

    def test_validate_many_failures_logs_truncation(self, sample_transformed_df):
        bad_df = sample_transformed_df.copy()
        bad_df['close'] = -1
        bad_df['open'] = -1
        bad_df['high'] = -1
        bad_df['low'] = -1
        results = validate_stock_data(bad_df, log_results=True)
        assert results['success'] is False
        assert len(results.get('failed_expectations', [])) > 0

    def test_validate_empty_dataframe(self):
        results = validate_stock_data(pd.DataFrame(), log_results=False)
        assert results['success'] is False

    def test_validate_exception_returns_error(self, monkeypatch):
        monkeypatch.setattr(
            'src.data_quality.gx.get_context',
            lambda: (_ for _ in ()).throw(RuntimeError("gx broken")),
        )
        results = validate_stock_data(pd.DataFrame({'x': [1]}), log_results=True)
        assert results['success'] is False
        assert 'error' in results

    def test_save_validation_report(self, tmp_path):
        results = {
            'timestamp': '2026-01-01T00:00:00',
            'success_rate': 95.0,
            'passed_checks': 19,
            'total_checks': 20,
            'success': True,
            'failed_expectations': [],
        }
        path = save_validation_report(results, output_dir=str(tmp_path))
        assert Path(path).exists()
        content = Path(path).read_text()
        assert 'DATA QUALITY VALIDATION REPORT' in content

    def test_save_validation_report_with_failures(self, tmp_path):
        results = {
            'timestamp': '2026-01-01T00:00:00',
            'success_rate': 80.0,
            'passed_checks': 8,
            'total_checks': 10,
            'success': False,
            'failed_expectations': [{
                'expectation': 'expect_column_values_to_be_between',
                'column': 'close',
                'details': 'bad value',
            }],
        }
        path = save_validation_report(results, output_dir=str(tmp_path))
        assert 'FAILED EXPECTATIONS' in Path(path).read_text()
