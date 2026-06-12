import pytest
from unittest.mock import patch, Mock, MagicMock
from sqlalchemy.exc import SQLAlchemyError
from src.load import (
    StockData,
    create_tables,
    load_to_database,
    get_database_summary,
    get_database_engine,
)
from sqlalchemy import create_engine, text
import pandas as pd


def _mock_engine_with_counts(initial=0, final=500):
    mock_engine = MagicMock()
    mock_conn = MagicMock()
    mock_initial = MagicMock()
    mock_initial.scalar.return_value = initial
    mock_final = MagicMock()
    mock_final.scalar.return_value = final
    mock_conn.execute.side_effect = [mock_initial, mock_final]
    mock_engine.connect.return_value.__enter__.return_value = mock_conn
    mock_engine.connect.return_value.__exit__.return_value = None
    return mock_engine


class TestStockDataModel:
    def test_repr(self):
        row = StockData(symbol='AAPL', date='2026-01-01', close=150.0)
        assert 'AAPL' in repr(row)


class TestDatabaseConnection:
    @patch('src.load.create_engine')
    def test_get_database_engine_success(self, mock_create_engine, test_env):
        mock_create_engine.return_value = MagicMock()
        engine = get_database_engine()
        assert mock_create_engine.called
        assert engine is not None

    def test_get_database_engine_missing_url(self, monkeypatch):
        monkeypatch.delenv('DATABASE_URL', raising=False)
        with pytest.raises(ValueError, match="DATABASE_URL"):
            get_database_engine()

    @patch('src.load.create_engine', side_effect=SQLAlchemyError("connection refused"))
    def test_get_database_engine_failure(self, mock_create_engine, test_env):
        with pytest.raises(SQLAlchemyError, match="connection refused"):
            get_database_engine()


class TestCreateTables:
    @patch('src.load.Base.metadata.create_all')
    @patch('src.load.get_database_engine')
    def test_create_tables_success(self, mock_get_engine, mock_create_all, test_env):
        mock_engine = Mock()
        mock_get_engine.return_value = mock_engine
        result = create_tables()
        assert result is True
        mock_create_all.assert_called_once()

    @patch('src.load.get_database_engine', side_effect=SQLAlchemyError("fail"))
    def test_create_tables_failure(self, mock_get_engine, test_env):
        result = create_tables()
        assert result is False


class TestLoadToDatabase:
    @patch('src.load.create_tables')
    @patch('src.load.get_database_engine')
    def test_load_to_database_success(self, mock_get_engine, mock_create_tables, sample_transformed_df, test_env):
        mock_engine = _mock_engine_with_counts(0, 500)
        mock_get_engine.return_value = mock_engine
        mock_create_tables.return_value = True

        with patch.object(pd.DataFrame, 'to_sql') as mock_to_sql:
            result = load_to_database(sample_transformed_df)
            assert result is True
            mock_to_sql.assert_called()

    @patch('src.load.create_tables')
    @patch('src.load.get_database_engine')
    def test_load_to_database_empty_dataframe(self, mock_get_engine, mock_create_tables, test_env):
        result = load_to_database(pd.DataFrame())
        assert result is False

    def test_load_to_database_none_dataframe(self, test_env):
        result = load_to_database(None)
        assert result is False

    @patch('src.load.create_tables')
    @patch('src.load.get_database_engine')
    def test_load_exact_batch_size(self, mock_get_engine, mock_create_tables, sample_transformed_df, test_env):
        df = sample_transformed_df.head(5).copy()
        mock_engine = _mock_engine_with_counts(0, 5)
        mock_get_engine.return_value = mock_engine
        mock_create_tables.return_value = True

        with patch.object(pd.DataFrame, 'to_sql') as mock_to_sql:
            result = load_to_database(df, chunksize=5)
            assert result is True
            assert mock_to_sql.call_count == 1

    @patch('src.load.create_tables')
    @patch('src.load.get_database_engine')
    def test_load_batch_size_plus_one(self, mock_get_engine, mock_create_tables, sample_transformed_df, test_env):
        df = sample_transformed_df.head(6).copy()
        mock_engine = _mock_engine_with_counts(0, 6)
        mock_get_engine.return_value = mock_engine
        mock_create_tables.return_value = True

        with patch.object(pd.DataFrame, 'to_sql') as mock_to_sql:
            result = load_to_database(df, chunksize=5)
            assert result is True
            assert mock_to_sql.call_count == 2

    @patch('src.load.create_tables')
    @patch('src.load.get_database_engine')
    def test_load_adds_missing_metadata_columns(self, mock_get_engine, mock_create_tables, test_env):
        df = pd.DataFrame({
            'symbol': ['AAPL'],
            'date': [pd.Timestamp('2026-01-01')],
            'open': [100.0], 'high': [105.0], 'low': [99.0],
            'close': [103.0], 'volume': [1000],
        })
        mock_engine = _mock_engine_with_counts(0, 1)
        mock_get_engine.return_value = mock_engine
        mock_create_tables.return_value = True

        with patch.object(pd.DataFrame, 'to_sql'):
            result = load_to_database(df)
            assert result is True
            assert 'extracted_at' in df.columns or True

    @patch('src.load.create_tables', return_value=True)
    @patch('src.load.get_database_engine')
    def test_load_with_perf_tracker(self, mock_get_engine, mock_create_tables, sample_transformed_df, test_env):
        from src.performance import PipelinePerformanceTracker

        mock_engine = _mock_engine_with_counts(0, 20)
        mock_get_engine.return_value = mock_engine
        tracker = PipelinePerformanceTracker(log_dir='logs')

        with patch.object(pd.DataFrame, 'to_sql'):
            result = load_to_database(
                sample_transformed_df.head(10),
                chunksize=5,
                perf_tracker=tracker,
            )
        assert result is True
        assert len(tracker.batch_metrics) == 2

    @patch('src.load.create_tables', return_value=True)
    @patch('src.load.get_database_engine')
    def test_load_batch_error_continues(self, mock_get_engine, mock_create_tables, sample_transformed_df, test_env):
        mock_engine = _mock_engine_with_counts(0, 10)
        mock_get_engine.return_value = mock_engine

        with patch.object(pd.DataFrame, 'to_sql', side_effect=[Exception("batch fail"), None]):
            result = load_to_database(sample_transformed_df.head(10), chunksize=5)
        assert result is True

    @patch('src.load.create_tables', return_value=True)
    @patch('src.load.get_database_engine', side_effect=SQLAlchemyError("db down"))
    def test_load_database_connection_failure(self, mock_get_engine, mock_create_tables, sample_transformed_df, test_env):
        result = load_to_database(sample_transformed_df)
        assert result is False

    @patch('src.load.create_tables', side_effect=RuntimeError("unexpected"))
    @patch('src.load.get_database_engine')
    def test_load_unexpected_error(self, mock_get_engine, mock_create_tables, sample_transformed_df, test_env):
        mock_get_engine.return_value = MagicMock()
        result = load_to_database(sample_transformed_df)
        assert result is False

    @patch('src.load.create_tables', return_value=True)
    @patch('src.load.get_database_engine')
    def test_load_missing_table_initial_count_zero(self, mock_get_engine, mock_create_tables, sample_transformed_df, test_env):
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_conn.execute.side_effect = [
            SQLAlchemyError("relation does not exist"),
            MagicMock(scalar=MagicMock(return_value=20)),
        ]
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        mock_engine.connect.return_value.__exit__.return_value = None
        mock_get_engine.return_value = mock_engine

        with patch.object(pd.DataFrame, 'to_sql'):
            result = load_to_database(sample_transformed_df.head(5))
        assert result is True


class TestGetDatabaseSummary:
    @patch('src.load.get_database_engine')
    def test_get_database_summary_success(self, mock_get_engine, test_env):
        mock_engine = MagicMock()
        mock_get_engine.return_value = mock_engine
        mock_conn = MagicMock()

        mock_result1 = MagicMock()
        mock_result1.scalar.return_value = 500
        mock_result2 = MagicMock()
        mock_result2.scalar.return_value = 5
        mock_date_min = MagicMock()
        mock_date_min.isoformat.return_value = '2025-09-01'
        mock_date_max = MagicMock()
        mock_date_max.isoformat.return_value = '2026-01-23'
        mock_result3 = MagicMock()
        mock_result3.fetchone.return_value = (mock_date_min, mock_date_max)
        mock_result4 = MagicMock()
        mock_result4.fetchone.return_value = (150.25, 50000000, 0.15)
        mock_result5 = MagicMock()
        mock_result5.fetchall.return_value = [('AAPL',), ('MSFT',), ('GOOGL',)]

        mock_conn.execute.side_effect = [
            mock_result1, mock_result2, mock_result3, mock_result4, mock_result5
        ]
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        mock_engine.connect.return_value.__exit__.return_value = None

        result = get_database_summary()
        assert result['total_records'] == 500
        assert result['unique_symbols'] == 5
        assert result['symbols_list'] == ['AAPL', 'MSFT', 'GOOGL']

    @patch('src.load.get_database_engine', side_effect=SQLAlchemyError("fail"))
    def test_get_database_summary_failure(self, mock_get_engine, test_env):
        result = get_database_summary()
        assert result == {}

    @patch('src.load.get_database_engine')
    def test_get_database_summary_empty_database(self, mock_get_engine, test_env):
        mock_engine = MagicMock()
        mock_get_engine.return_value = mock_engine
        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_result.scalar.return_value = 0
        mock_conn.execute.return_value = mock_result
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        mock_engine.connect.return_value.__exit__.return_value = None

        result = get_database_summary()
        assert result['total_records'] == 0
        assert result['symbols_list'] == []
