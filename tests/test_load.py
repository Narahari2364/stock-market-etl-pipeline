import pytest
from unittest.mock import patch, Mock, MagicMock
from src.load import create_tables, load_to_database, get_database_summary
from sqlalchemy import create_engine, text
import pandas as pd


class TestDatabaseConnection:
    """Tests for database connection"""

    @patch('src.load.create_engine')
    def test_get_database_engine_success(self, mock_create_engine, test_env):
        """Test successful database engine creation"""
        from src.load import get_database_engine

        mock_create_engine.return_value = MagicMock()

        engine = get_database_engine()

        assert mock_create_engine.called
        assert engine is not None


class TestCreateTables:
    """Tests for table creation"""

    @patch('src.load.Base.metadata.create_all')
    @patch('src.load.get_database_engine')
    def test_create_tables_success(self, mock_get_engine, mock_create_all, test_env):
        """Test table creation succeeds"""
        mock_engine = Mock()
        mock_get_engine.return_value = mock_engine

        result = create_tables()

        assert result is True
        mock_create_all.assert_called_once()


class TestLoadToDatabase:
    """Tests for loading data to database"""

    @patch('src.load.create_tables')
    @patch('src.load.get_database_engine')
    def test_load_to_database_success(self, mock_get_engine, mock_create_tables, sample_transformed_df, test_env):
        """Test successful data loading"""
        mock_engine = MagicMock()
        mock_get_engine.return_value = mock_engine
        mock_create_tables.return_value = True

        mock_conn = MagicMock()
        mock_initial = MagicMock()
        mock_initial.scalar.return_value = 0
        mock_final = MagicMock()
        mock_final.scalar.return_value = 500
        mock_conn.execute.side_effect = [mock_initial, mock_final]

        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        mock_engine.connect.return_value.__exit__.return_value = None

        with patch.object(pd.DataFrame, 'to_sql') as mock_to_sql:
            result = load_to_database(sample_transformed_df)

            assert result is True
            mock_to_sql.assert_called()

    @patch('src.load.create_tables')
    @patch('src.load.get_database_engine')
    def test_load_to_database_empty_dataframe(self, mock_get_engine, mock_create_tables, test_env):
        """Test loading empty DataFrame"""
        empty_df = pd.DataFrame()
        mock_engine = MagicMock()
        mock_get_engine.return_value = mock_engine

        result = load_to_database(empty_df)

        assert result is False


class TestGetDatabaseSummary:
    """Tests for database summary function"""

    @patch('src.load.get_database_engine')
    def test_get_database_summary_success(self, mock_get_engine, test_env):
        """Test retrieving database summary"""
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

        assert result is not None
        assert result['total_records'] == 500
        assert result['unique_symbols'] == 5
        assert result['date_range']['min_date'] == '2025-09-01'
        assert result['date_range']['max_date'] == '2026-01-23'
        assert result['average_metrics']['avg_close'] == 150.25
        assert result['symbols_list'] == ['AAPL', 'MSFT', 'GOOGL']
