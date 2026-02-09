import pytest
from unittest.mock import patch, Mock
from src.extract import fetch_stock_data, fetch_multiple_stocks
import requests


class TestFetchStockData:
    """Tests for fetch_stock_data function"""

    @patch('src.extract.requests.get')
    def test_fetch_stock_data_success(self, mock_get, mock_api_response, test_env):
        """Test successful API call returns correct data structure"""
        # Setup mock
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_api_response
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        # Test
        result = fetch_stock_data('AAPL')

        # Assertions
        assert result is not None
        assert result['symbol'] == 'AAPL'
        assert 'time_series' in result
        assert len(result['time_series']) > 0
        assert 'data_points' in result

    @patch('src.extract.requests.get')
    def test_fetch_stock_data_invalid_symbol(self, mock_get, test_env):
        """Test handling of invalid stock symbol"""
        # Setup mock for error response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'Error Message': 'Invalid API call. Please check your symbol.'
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        # Test - extract raises ValueError for Error Message
        with pytest.raises(ValueError):
            fetch_stock_data('INVALID')

    @patch('src.extract.requests.get')
    def test_fetch_stock_data_rate_limit(self, mock_get, test_env):
        """Test handling of API rate limit"""
        # Setup mock - "Information" key returns None
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'Information': 'Thank you for using Alpha Vantage! Our standard API rate limit is 5 requests per minute.'
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        # Test
        result = fetch_stock_data('AAPL')

        # Assertion - should return None when rate limited (Information key)
        assert result is None

    @patch('src.extract.requests.get')
    def test_fetch_stock_data_network_error(self, mock_get, test_env):
        """Test handling of network errors"""
        # Setup mock to raise exception
        mock_get.side_effect = requests.exceptions.ConnectionError("Network error")

        # Test - extract raises RequestException
        with pytest.raises(requests.RequestException):
            fetch_stock_data('AAPL')

    @patch('src.extract.requests.get')
    def test_fetch_stock_data_timeout(self, mock_get, test_env):
        """Test handling of timeout errors"""
        # Setup mock to raise timeout
        mock_get.side_effect = requests.exceptions.Timeout("Request timeout")

        # Test - extract raises RequestException
        with pytest.raises(requests.RequestException):
            fetch_stock_data('AAPL')


class TestFetchMultipleStocks:
    """Tests for fetch_multiple_stocks function"""

    @patch('src.extract.fetch_stock_data')
    @patch('src.extract.time.sleep')
    def test_fetch_multiple_stocks_success(self, mock_sleep, mock_fetch):
        """Test fetching multiple stocks successfully"""
        # Setup mock
        mock_fetch.return_value = {
            'symbol': 'AAPL',
            'time_series': {'2026-01-23': {}},
            'data_points': 1
        }

        # Test
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        result = fetch_multiple_stocks(symbols, delay=0)

        # Assertions
        assert len(result) == 3
        assert mock_fetch.call_count == 3

    @patch('src.extract.fetch_stock_data')
    @patch('src.extract.time.sleep')
    def test_fetch_multiple_stocks_with_failures(self, mock_sleep, mock_fetch):
        """Test fetching multiple stocks with some failures"""
        # Setup mock - one success, one returns None (failure), one success
        mock_fetch.side_effect = [
            {'symbol': 'AAPL', 'time_series': {}, 'data_points': 0},
            None,  # Failed
            {'symbol': 'GOOGL', 'time_series': {}, 'data_points': 0}
        ]

        # Test
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        result = fetch_multiple_stocks(symbols, delay=0)

        # Assertions - list has 3 items, middle one is None
        assert len(result) == 3
        assert result[0] is not None
        assert result[1] is None
        assert result[2] is not None
